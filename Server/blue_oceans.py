from builtins import print
from datetime import datetime, timedelta, time

from flask import request
from flask_restx import Resource, Api, Namespace, fields
from flask import request, jsonify

import  json
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import holidays
import random
import os

import logging
from logger_setup import setup_logger  # 로거 설정 모듈 가져오기
from apscheduler.schedulers.background import BackgroundScheduler

BlueOceans = Namespace(
    name="BlueOceans",
    description="웹 플랫폼에서 주식정보를 받기위해서 사용하는 API.",
)

# 로거 설정을 가져와서 사용(처음 시작할때 로그 모듈 샛팅)
logger = setup_logger()

# APScheduler 인스턴스 생성
scheduler = BackgroundScheduler()
# 매일 정해진 시간에 setup_logger() 함수 호출
scheduler.add_job(setup_logger, 'cron', hour='0')  # 매일 자정에 실행
# 스케줄러 실행
scheduler.start()

########################################################################################################################
today_ticker_fields = BlueOceans.model('SearchTodayTickerData', {  # Model 객체 생성
    'ticker': fields.String(description='주식코드값', required=True, example="005930"), #주식 코드
    'date': fields.String(description='날자', required=True, example="2023-08-22"), #날자
})

@BlueOceans.route('/search-today-tickers',methods=['POST'], doc={"description": "ticker(6자리 숫자) 값을 이용하여 일정 기간의 종목 정보를 검색"})
class SearchTodayTickerPost(Resource):
    @BlueOceans.expect(today_ticker_fields)
    @BlueOceans.response(200, 'Success', today_ticker_fields)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-today-tickers 호출')
        ticker = request.json.get('ticker')
        date = request.json.get('date')

        # 주말일때 이전 날짜로
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        previous_weekday = get_previous_weekday(date_obj)
        # 날짜 객체를 다시 문자열로 변환합니다.
        date = previous_weekday.strftime("%Y-%m-%d")

        print('ticker : ',ticker)
        print('date : ',date)

        # 변수가 정수인지 확인
        if isinstance(ticker, int):
            ticker = str(ticker)  # 정수를 문자열로 변환

        logger.info('ticker : %s', ticker)
        logger.info('date : %s', date)

        if isinstance(ticker, str) and (  # .isdigit()는 모두 숫자인지 확인
                (len(ticker) == 6 and ticker.isdigit()) or (
                len(ticker) == 6 and ticker[:5].isdigit() and ticker[-1].isalpha())):
            try:
                data = call_ticker_data(ticker, date, date)
                stock_name = get_stock_name(ticker)
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                # 날짜 계산을 위해 1일을 뺍니다.
                date_obj = date_obj - timedelta(days=1)
                previous_weekday = get_previous_weekday(date_obj)
                # 날짜 객체를 다시 문자열로 변환합니다.
                yesterday_date = previous_weekday.strftime("%Y-%m-%d")

                data2 = call_ticker_data(ticker, yesterday_date, yesterday_date)
                yesterday_price = data2[0]['Close']

                return {
                    'stock_name': stock_name,
                    'ticker': ticker,
                    'date': date,
                    'yesterdayDate': yesterday_date,
                    'yesterdayClose': yesterday_price,
                    'data': data
                }, 200
            except :
                return {"message": "검색 결과가 없습니다."}, 404
        else:
            return {"message": "검색어를 제대로 입력해 주세요"}, 404

########################################################################################################################
ticker_fields = BlueOceans.model('SearchTickerData', {  # Model 객체 생성
    'ticker': fields.String(description='주식코드값', required=True, example="005930"), #주식 코드
    'startDate': fields.String(description='시작날자', required=True, example="2023-01-01"), #시작날자
    'endtDate': fields.String(description='종료날자', required=True, example="2023-01-31") #종료날자
})


@BlueOceans.route('/search-tickers',methods=['POST'], doc={"description": "ticker(6자리 숫자) 값을 이용하여 일정 기간의 종목 정보를 검색"})
class SearchTickerPost(Resource):
    @BlueOceans.expect(ticker_fields)
    @BlueOceans.response(200, 'Success', ticker_fields)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-tickers 호출')
        ticker = request.json.get('ticker')
        startDate = request.json.get('startDate')
        endtDate = request.json.get('endtDate')

        # 주말일때 이전 날짜로
        date_obj = datetime.strptime(startDate, "%Y-%m-%d")
        previous_weekday = get_previous_weekday(date_obj)
        # 날짜 객체를 다시 문자열로 변환합니다.
        startDate = previous_weekday.strftime("%Y-%m-%d")

        print('ticker : ',ticker)
        print('startDate : ',startDate)
        print('endtDate : ',endtDate)
        logger.info('ticker : %s', ticker)
        logger.info('startDate : %s', startDate)
        logger.info('endtDate : %s', endtDate)

        if isinstance(ticker, str) and (  # .isdigit()는 모두 숫자인지 확인
                (len(ticker) == 6 and ticker.isdigit()) or (
                len(ticker) == 6 and ticker[:5].isdigit() and ticker[-1].isalpha())):
            try:
                data = call_ticker_data(ticker, startDate, endtDate)
                stock_name = get_stock_name(ticker)

                return {
                    'stock_name': stock_name,
                    'ticker': ticker,
                    'startDate': startDate,
                    'endtDate': endtDate,
                    'data': data
                }, 200
            except :
                return {"message": "검색 결과가 없습니다."}, 404
        else:
            return {"message": "검색어를 제대로 입력해 주세요"}, 404

#종목 ticker을 활용한 주식데이터 검색
def call_ticker_data(ticker = '005930', stt = '2023-01-01', end = '2023-01-31', history_points = 50):
    data = fdr.DataReader(ticker, stt, end)

    # DataFrame의 인덱스를 8자리로 변환하여 새로운 컬럼 'Date' 추가
    data.insert(0,'Date',data.index.strftime('%Y%m%d'))

    # JSON으로 변환하면서 'Date' 컬럼 사용하도록 지정
    # json_data = data.to_json(orient='records', date_format='epoch', date_unit='s')
    json_data = data.to_json(orient='records', date_format='epoch', date_unit='s', force_ascii=False)
    json_data = json.loads(json_data)

    logger.info('json_data : %s', json_data)
    return json_data

# KRX 종목 코드를 사용하여 종목명 얻기
def get_stock_name(ticker):
    # CSV 파일명 생성
    filename = 'stock_list.csv'
    stock_name = ticker.upper()

    # 이미 파일이 존재할 경우, 해당 파일에서 주식명 검색
    # (기존의 stock_list.csv 파일이 없으면 생성하는 부분 삭제 <-144건으로 픽스)
    df = pd.read_csv(filename, encoding='utf-8-sig')
    stock_name = df[df['Code'] == ticker]['Name'].iloc[0]

    return stock_name
########################################################################################################################
# 종목명orTicker 결과를 리스트로 받는 함수의 모델에 표시될 문구
args_name_or_ticker = BlueOceans.model('SearchNameData', {  # Model 객체 생성
    'name': fields.String(description='종목명 or Ticker', required=True, example="삼성 or 005930")  # 주식 이름
})

    # 종목명orTicker 결과를 리스트로 받는 함수
@BlueOceans.route('/search-stocks-list', doc={"description": "주식 이름 또는 ticher을 이용하여 일정 기간의 종목 정보를 검색"})
class SearchStockList(Resource):
    @BlueOceans.expect(args_name_or_ticker)
    @BlueOceans.response(200, 'Success')
    @BlueOceans.response(202, 'No Data')
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-stocks-list 호출')

        search_stock = request.json.get('name')
        search_stock = search_stock.strip()
        print('stock_name : ', search_stock)
        logger.info('stock_name : %s',search_stock)

        if len(search_stock) > 0:
            # 입력이 Ticker인지 주식이름인지 판별 (isinstance()를 사용해서 str 형식인지 확인,
            # isalpha()는 해당 문자열이 모두 알파벳(영어 알파벳 대소문자)로 이루진지 확인)
            # .isdigit()는 모두 숫자인지 확인
            if isinstance(search_stock, str) and search_stock.isalpha():
                # 입력이 주식이름인 경우
                data = push_name_pull_list(search_stock)
                if len(data) > 0 :
                    # 데이터가 있을경우
                    return {
                        'data': data
                    }, 200
                else:
                    return {"message": "검색 결과가 없습니다"}, 404
            elif isinstance(search_stock, str) and ((len(search_stock) == 6 and search_stock.isdigit()) or(len(search_stock) == 6 and search_stock[:5].isdigit() and search_stock[-1].isalpha())):
                # 입력이 Ticker인 경우
                data = push_ticker_pull_list(search_stock)
                if len(data) > 0 :
                    # 데이터가 있을경우
                    return {
                        'data': data
                    }, 200
                else:
                    return {"message": "검색 결과가 없습니다"}, 404
            else:
                return {"message": "검색 결과가 없습니다"}, 404
        else:
            return {"message": "검색어를 입력해 주세요"}, 404


# 종목 이름으로 주식의 ticker와 이름 리스트를 반환한다.
def push_name_pull_list(stock_name = '삼성'):
    # CSV 파일명 생성
    filename = 'stock_list.csv'

    stock_name = stock_name.upper()

    # 이미 파일이 존재할 경우, 해당 파일에서 주식명 검색
    # (기존의 stock_list.csv 파일이 없으면 생성하는 부분 삭제 <-144건으로 픽스)
    df = pd.read_csv(filename, encoding='utf-8-sig')
    filtered_rows = df[df['Name'].str.contains(stock_name)]

    # force_ascii=False로 설정하여 한글이 깨지지 않도록 처리
    json_data = filtered_rows.to_json(orient='records', force_ascii=False)
    json_data = json.loads(json_data)

    print(json_data)
    return json_data

# ticker로 주식의 ticker와 이름 리스트를 반환한다.
def push_ticker_pull_list(stock_ticker = '005930'):
    # CSV 파일명 생성
    filename = 'stock_list.csv'

    stock_ticker = stock_ticker.upper()

    # 이미 파일이 존재할 경우, 해당 파일에서 주식명 검색
    # (기존의 stock_list.csv 파일이 없으면 생성하는 부분 삭제 <-144건으로 픽스)
    df = pd.read_csv(filename, encoding='utf-8-sig')
    filtered_rows = df[df['Code'].str.contains(stock_ticker)]

    # force_ascii=False로 설정하여 한글이 깨지지 않도록 처리
    json_data = filtered_rows.to_json(orient='records', force_ascii=False)
    json_data = json.loads(json_data)

    print(json_data)
    logger.info('JSON Data : %s',json_data)

    return json_data
########################################################################################################################
ticker_fields_getinterval = BlueOceans.model('SearchTickerIntervalData', {  # Model 객체 생성
    'ticker': fields.String(description='주식코드', required=True, example="005930"), # 주식 코드
    'date': fields.String(description='날자', required=True, example="2023-08-02"),    # 날자
    'interval': fields.String(description='(분)간격', required=True, example="10")      # 몇분 간격 데이터를 받을지
})

@BlueOceans.route('/search-tickers-getinterval',doc={"description": "ticker(6자리 숫자) 값을 이용하여 해당 날짜의 분단위 종목 값 검색(2023-08-02부터 검색 가능합니다) \n (1, 5, 10, 30, 60, 180 분 단위 입력)"})
class SearchTickerIntervalPost(Resource):
    @BlueOceans.expect(ticker_fields_getinterval)
    @BlueOceans.response(200, 'Success', ticker_fields_getinterval)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-tickers-getinterval 호출')

        ticker = request.json.get('ticker')
        date = request.json.get('date')
        interval = request.json.get('interval')

        # 주말일때 이전 날짜로
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        previous_weekday = get_previous_weekday(date_obj)
        # 날짜 객체를 다시 문자열로 변환합니다.
        date = previous_weekday.strftime("%Y-%m-%d")

        print('ticker : ',ticker)
        print('date : ',date)
        print('interval : ',interval)
        logger.info('ticker : %s',ticker)
        logger.info('date : %s',date)
        logger.info('interval : %s',interval)


        if isinstance(ticker, str) and (  # .isdigit()는 모두 숫자인지 확인
                (len(ticker) == 6 and ticker.isdigit()) or (
                len(ticker) == 6 and ticker[:5].isdigit() and ticker[-1].isalpha())):
            try:
                data = call_ticker_interval_data(ticker, date, interval)
                stock_name = get_stock_name(ticker)

                return {
                    'stock_name': stock_name,
                    'ticker': ticker,
                    'date': date,
                    'interval': interval,
                    'data': data
                }, 200
            except :
                return {"message": "검색 결과가 없습니다."}, 404
        else:
            return {"message": "검색어를 제대로 입력해 주세요"}, 404

def call_ticker_interval_data(ticker='005930', date='2023-08-02', interval='2'):
    interval = int(interval)
    # CSV 파일 경로
    file_path = './stock_data/' + date + '.csv'
    data = pd.read_csv(file_path)

    # 문자열 데이터의 앞뒤 공백 제거
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 시간 데이터를 다양한 간격으로 추출하고 JSON으로 변환
    time_data = data.iloc[:, 0]  # 시간 데이터가 있는 열의 데이터 (1번 열부터)
    json_data = {}

    extracted_data = {}
    start_extract = False  # 추출 시작 여부 플래그
    for i, time in enumerate(time_data, start=1):
        if not start_extract:
            if time == '09:01':
                start_extract = True
            else:
                continue

        hour, minute = map(int, time.split(':'))
        total_minutes = (hour - 9) * 60 + minute
        if total_minutes % interval == 0:
            extracted_data[time] = int(data.loc[i - 1, ticker])

    json_data[f'{interval}min'] = extracted_data

    # JSON 데이터 출력
    print('JSON Data:', json_data)
    logger.info('JSON Data : %s',json_data)

    return json_data

########################################################################################################################
stock_market_fields = BlueOceans.model('SearchstockMarketData', {  # Model 객체 생성
    'symbol': fields.String(description='심볼', required=True, example="KS11"),             # 심볼
    'startDate': fields.String(description='시작날자', required=True, example="2023-01-01"), # 시작날자
    'endtDate': fields.String(description='종료날자', required=True, example="2023-01-31")   # 종료날자
})

# 주식시장에 관한 정보를 가져오는 엔드포인트 (코스피, 코스닥, 코스피100 등)
@BlueOceans.route('/search-stock-market',methods=['POST'], doc={"description": "심볼을 이용하여 일정 기간의 주식시장 및 원/달러 값을 검색 "
            "\n ( KS11: 코스피, KQ11: 코스닥, KS200: 코스피200, USD/KRW: 원/달러 환율, JPY/KRW: 원/엔화 환율, BTC/KRW: 원/비트코인 ) \n "})
class SearchStockMarketPost(Resource):
    @BlueOceans.expect(stock_market_fields)
    @BlueOceans.response(200, 'Success', stock_market_fields)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-stock-market 호출')

        symbol = request.json.get('symbol')
        startDate = request.json.get('startDate')
        endtDate = request.json.get('endtDate')
        symbol =symbol.upper()

        date_obj = datetime.strptime(startDate, "%Y-%m-%d")
        previous_weekday = get_previous_weekday(date_obj)
        # 날짜 객체를 다시 문자열로 변환합니다.
        startDate = previous_weekday.strftime("%Y-%m-%d")

        print('symbol : ',symbol)
        print('startDate : ',startDate)
        print('endtDate : ',endtDate)
        logger.info('symbol : %s',symbol)
        logger.info('startDate : %s',startDate)
        logger.info('endtDate : %s',endtDate)

        try:
            data = call_symbol_data(symbol, startDate, endtDate)
            symbol_name = get_symbol_name(symbol) # 만들기
            return {
                'symbol_name': symbol_name,
                'symbol': symbol,
                'startDate': startDate,
                'endtDate': endtDate,
                'data': data
            }, 200
        except:
            return {"message": "검색 결과가 없습니다."}, 404

#종목 ticker을 활용한 주식데이터 검색
def call_symbol_data(symbol = 'KS100', stt = '2023-01-01', end = '2023-01-31'):
    data = fdr.DataReader(symbol, stt, end)

    # 데이터가 None인 경우에 대한 처리
    if data is None:
        response = {
            "status": "error",
            "message": "No data available for the specified symbol and date range."
        }
        json_data = json.dumps(response)
        return json_data


        # DataFrame의 인덱스를 8자리로 변환하여 새로운 컬럼 'Date' 추가
    # data.insert(0,'Date',data.index.strftime('%Y%m%d'))

    # JSON으로 변환하면서 'Date' 컬럼 사용하도록 지정
    # json_data = data.to_json(orient='records', date_format='epoch', date_unit='s')
    json_data = data.to_json(orient='records', date_format='epoch', date_unit='s', force_ascii=False)
    json_data = json.loads(json_data)

    logger.info('json_data : %s', json_data)
    return json_data

# KRX 종목 코드를 사용하여 종목명 얻기
def get_symbol_name(symbol):

    if symbol=='KS11':      # 코스피
        return 'KOSPI'
    elif symbol=='KQ11':    # 코스닥
        return 'KOSDAQ'
    elif symbol=='KS200':   # 코스피200
        return 'KOSPI 200'
    elif symbol=='USD/KRW': # 원/달러 환율
        return '원/달러 환율'
    elif symbol=='JPY/KRW': # 원/엔화 환율
        return '원/엔화 환율'
    elif symbol=='BTC/KRW': # 원/비트코인 환율
        return '원/비트코인 환율'
    else:
        return symbol

########################################################################################################################
stock_market_fields_getinterval = BlueOceans.model('SearchstockMarketIntervalData', {             # Model 객체 생성
    'symbol': fields.String(description='심볼', required=True, example="KS11"),       # 심볼
    'date': fields.String(description='날자', required=True, example="2023-08-09"),   # 날자
    'interval': fields.String(description='(분)간격', required=True, example="2")     # 몇분 간격 데이터를 받을지
})

@BlueOceans.route('/search-stock-market-getinterval',doc={"description": "심볼을 이용하여 분단위 주식시장 및 원/달러 값을 검색 "
            "\n ( KS11: 코스피, KQ11: 코스닥, KS200: 코스피200, USD/KRW: 원/달러 환율, JPY/KRW: 원/엔화 환율, BTC/KRW: 원/비트코인 ) \n "})
class SearchStockMarketIntervalPost(Resource):
    @BlueOceans.expect(stock_market_fields_getinterval)
    @BlueOceans.response(200, 'Success', stock_market_fields_getinterval)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-stock-market-getinterval 호출')

        symbol = request.json.get('symbol')
        date = request.json.get('date')
        interval = request.json.get('interval')
        # symbol =symbol.upper()

        #주말일때 이전 날짜로
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        previous_weekday = get_previous_weekday(date_obj)
        # 날짜 객체를 다시 문자열로 변환합니다.
        date = previous_weekday.strftime("%Y-%m-%d")

        print('symbol : ',symbol)
        print('date : ',date)
        print('interval : ',interval)
        logger.info('symbol : %s',symbol)
        logger.info('date : %s',date)
        logger.info('interval : %s',interval)

        # 현재 시간 가져오기
        current_time = datetime.now().time()
        # 비교할 시간 설정 (9시 25분)
        comparison_time = time(9, 25)

        # 현재 시간이 9시 30분 이전인지 비교
        if current_time < comparison_time:
            # 어제 날짜로 보여주도록 설정
            print("현재 시간은 9시 25분 이전입니다.")
            date_obj2 = datetime.strptime(date, "%Y-%m-%d")

            # 날짜 계산을 위해 1일을 뺍니다.
            date_obj2 = date_obj2 - timedelta(days=1)
            set_date_obj = get_previous_weekday(date_obj2)
            date = set_date_obj.strftime("%Y-%m-%d")
        else:
            print("현재 시간은 9시 25분 이후입니다.")


        if symbol in ["KS11", "KQ11", "KS200"]:
            try:
                data = call_stock_market_interval_data(symbol, date, interval)
                symbol_name = get_stock_market_name(symbol)

                date_obj = datetime.strptime(date, "%Y-%m-%d")

                # 날짜 계산을 위해 1일을 뺍니다.
                date_obj = date_obj - timedelta(days=1)
                previous_weekday = get_previous_weekday(date_obj)
                previous_weekday_str = previous_weekday.strftime("%Y-%m-%d")

                # 날짜 객체를 다시 문자열로 변환합니다.
                date2 = previous_weekday.strftime("%Y-%m-%d")
                data2 = get_index_data(symbol, date2)
                print('date2 : ',date2)

                return {
                    'symbol_name': symbol_name,
                    'ticker': symbol,
                    'date': date,
                    'interval': interval,
                    'yesterdayData': data2,
                    'data': data
                }, 200
            except:
                return {"message": "검색 결과가 없습니다."}, 404
        elif symbol in ["USD/KRW", "JPY/KRW", "BTC/KRW"]:
            try:
                data = call_stock_market_interval_data(symbol, date, interval)
                symbol_name = get_stock_market_name(symbol)

                data2 = call_symbol_data(symbol, date, date)
                open_price = data2[0]['Open']

                return {
                    'symbol_name': symbol_name,
                    'ticker': symbol,
                    'date': date,
                    'interval': interval,
                    'openData': open_price,
                    'data': data
                }, 200
            except:
                return {"message": "검색 결과가 없습니다."}, 404

def get_index_data(symbol = 'KS11', date = '2023-08-23' ):
    # CSV 파일을 pandas DataFrame으로 읽어옵니다

    file_path = './stock_index_data/' + date + '.csv'
    data = pd.read_csv(file_path)

    #  열의 마지막 값을 가져옵니다
    last_kq11_value = data[symbol].iloc[-1]

    # 소수점 둘째 자리까지 포맷팅하여 출력합니다
    formatted_kq11_value = "{:.2f}".format(last_kq11_value)
    print("Last value:", formatted_kq11_value)
    return formatted_kq11_value

# 한국의 공휴일 정보를 가져오는 함수
def get_korean_holidays(year):
    return holidays.Korea(years=year)

# 평일인지 확인하는 함수 (토요일: 5, 일요일: 6)
def is_weekday(today):
    return today.weekday() < 5 and today not in get_korean_holidays(today.year)

def get_previous_weekday(date):
    while not is_weekday(date):
        date -= timedelta(days=1)
    return date

# 심볼을 사용하여 증권 및 환율이름 얻기
def get_stock_market_name(symbol):
    # CSV 파일명 생성
    filename = 'stock_index_list.csv'
    symbol_name = symbol.upper()

    # 이미 파일이 존재할 경우, 해당 파일에서 주식명 검색
    # (기존의 stock_list.csv 파일이 없으면 생성하는 부분 삭제 <-144건으로 픽스)
    df = pd.read_csv(filename, encoding='utf-8-sig')
    symbol_name = df[df['Code'] == symbol]['Name'].iloc[0]

    return symbol_name


# symbol 활용한 증권 및 환윫 분단위 데이터 검색
def call_stock_market_interval_data(symbol='KS11', date='2023-08-02', interval='2'):

    interval = int(interval)
    # CSV 파일 경로
    file_path = './stock_index_data/' + date + '.csv'
    # print('file_path :', file_path)

    # CSV 파일을 DataFrame으로 읽어오기
    data = pd.read_csv(file_path)

    # 문자열 데이터의 앞뒤 공백 제거
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 시간 데이터를 다양한 간격으로 추출하고 JSON으로 변환
    time_data = data.iloc[:, 0]  # 시간 데이터가 있는 열의 데이터 (1번 열부터)
    json_data = {}

    extracted_data = {}
    first_hour = 0
    start_extract = False  # 추출 시작 여부 플래그
    for i, time in enumerate(time_data, start=1):
        if not start_extract:
            if time >= '09:21':
                first_hour, first_minute = map(int, time.split(':'))
                start_extract = True
            else:
                continue

        hour, minute = map(int, time.split(':'))
        total_minutes = (hour - first_hour) * 60 + minute
        if total_minutes % interval == 0:
            extracted_data[time] = data.loc[i - 1, symbol]

    json_data[f'{interval}min'] = extracted_data

    # JSON 데이터 출력
    print('JSON Data:', json_data)
    logger.info('JSON Data : %s',json_data)

    return json_data
########################################################################################################################

top10_stock_fields_getinterval = BlueOceans.model('SearchTop10Stock', { })

@BlueOceans.route('/search-top10-stock',doc={"description": "거래량 top 10 종목 호출 "
            "\n Code: 종목 코드를 나타냅니다. 주식 시장에서 종목을 고유하게 식별하는 코드입니다."
            "\n ISU_CD: 종목 코드와 동일한 열로 보이며, 종목 코드를 나타냅니다."
            "\n Name: 종목 이름을 나타냅니다. 주식 종목의 이름이 표시됩니다."
            "\n Sector: 종목이 속한 업종을 나타냅니다. 종목이 어떤 산업이나 업종에 속해 있는지를 나타냅니다."
            "\n Code: 종목 코드를 나타냅니다. 주식 시장에서 종목을 고유하게 식별하는 코드입니다."
            "\n Industry: 종목이 속한 산업 분류를 나타냅니다. 업종보다 더 구체적인 분류 정보를 제공합니다."
            "\n ListingDate: 종목이 상장된 날짜를 나타냅니다. 해당 종목이 주식 시장에 상장된 날짜를 나타냅니다."
            "\n MaturityDate: 만기일을 나타냅니다. 주식 종목이 만기가 있는 경우 해당 날짜를 나타냅니다."
            "\n MarketType: 거래소 유형을 나타냅니다. KRX는 'KOSPI', 'KOSDAQ' 등 다양한 시장 유형을 가지고 있습니다."
            "\n Marcap: 시가 총액을 나타냅니다. 해당 종목의 모든 주식을 현재 시장 가격으로 평가한 총액입니다."
            "\n Stocks: 발행 주식 수를 나타냅니다. 해당 종목의 총 발행 주식 수를 나타냅니다."
            "\n MarketId: 시장 ID를 나타냅니다. 종목이 어떤 시장에 속해 있는지를 식별하는 ID입니다."
            "\n ( STK: 주식 시장 (KOSPI 및 KOSDAQ), KSQ: 코스닥 시장 (KOSDAQ), KNX: 코넥스 시장 (KONEX)) \n "})
class SearchTop10StockPost(Resource):
    @BlueOceans.expect(top10_stock_fields_getinterval)
    @BlueOceans.response(200, 'Success', top10_stock_fields_getinterval)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('search-top10-stock 호출')

        try:
            data = call_top10_stock_data()
            return {
                'data': data
            }, 200
        except:
            return {"message": "검색 결과가 없습니다."}, 404

# symbol 활용한 증권 및 환윫 분단위 데이터 검색
def call_top10_stock_data(marketId='KRX'):

    # get_tickers() 함수를 사용하여 모든 종목의 심볼(symbol) 리스트를 가져옵니다
    all_tickers = fdr.StockListing('KRX')

    # 상위 거래량 종목을 가져오기 위해 'Volume' 기준으로 정렬합니다
    top_volume_tickers = all_tickers.sort_values(by='Volume', ascending=False)

    # 상위 N개의 거래량 종목을 선택합니다 (여기서는 상위 10개)
    top_n_tickers = top_volume_tickers.head(10)
    # top_n_tickers = pd.read_csv('top_10_tickers.csv')

    columns_to_check = ['Changes', 'ChagesRatio', 'Open', 'High', 'Low', 'Volume', 'Amount', 'Marcap']

    # 각 컬럼에 대해 NaN 값을 확인하고, any() 함수로 하나라도 True 값이 있는지 확인합니다
    nan_check_per_column = top_n_tickers[columns_to_check].isna().any()
    # print("nan_check_per_column: ", nan_check_per_column)
    # 어떤 컬럼에서 NaN 값이 있는지 출력합니다
    columns_with_nan = nan_check_per_column[nan_check_per_column].index.tolist()
    # print("columns_with_nan: ", columns_with_nan)

    if columns_with_nan:
        top_n_tickers = pd.read_csv('./top_10_data/today/top_10_tickers.csv')
    else:
        print("CSV 파일에는 None 또는 NaN 값이 없습니다.")

    # JSON으로 변환하면서 'Date' 컬럼 사용하도록 지정
    json_data = top_n_tickers.to_json(orient='records', date_format='epoch', date_unit='s', force_ascii=False)
    json_data = json.loads(json_data)

    # JSON 데이터 출력
    print('JSON Data:', json_data)
    logger.info('JSON Data : %s',json_data)

    return json_data
########################################################################################################################
# 주식 코드값으로 AI 주식 받기

ai_stock_recommend_fields_getinterval = BlueOceans.model('AIStockRecommend', {
    'ticker': fields.String(description='주식코드값', required=True, example="005930, 078340, 002020, 035720, 089590") #주식 코드
})
@BlueOceans.route('/ai-stock-recommend',doc={"description": "해당 주식 ticher를 받아서 AI 주식 추천 처리 "})
class AIStockRecommendPost(Resource):
    @BlueOceans.expect(ai_stock_recommend_fields_getinterval)
    @BlueOceans.response(200, 'Success', ai_stock_recommend_fields_getinterval)
    @BlueOceans.response(202, 'No Data' )
    @BlueOceans.response(404, 'Not found')
    @BlueOceans.response(500, 'Internal Error')
    def post(self):
        logger.info('ai-stock-recommend 호출')
        print('ai-stock-recommend 호출')

        get_ticker = request.json.get('ticker')
        get_ticker = get_ticker.strip()

        names = []
        tickers = []
        datas = []

        try:
            csv_filename = 'today_ai_list.csv'
            data_df = pd.read_csv(csv_filename)

            for index, row in data_df.iterrows():
                names.append(row['Name'])
                tickers.append(str(row['Code']).zfill(6))

            print('names - ',names)
            print('tickers - ',tickers)
            print('datas - ',datas)

            datas = call_ai_stock_recommend(tickers)

            if len(datas) ==5 :
                return {
                    'names': names,
                    'tickers': tickers,
                    'datas': datas
                }, 200
            else :
                return {"message": "다음날의 딥러닝 훈련 중 입니다."}, 404

        except:
            return {"message": "검색 결과가 없습니다."}, 404

def get_today():
    # 오늘 날짜 정보
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    return current_date

# ticker를 이용한 딥러닝 추천값 받기
def call_ai_stock_recommend(tickers):

    datas = []
    for ticker in tickers:
        # CSV 파일이 들어 있는 디렉토리 경로
        directory_path = './models/today/'  # 실제 디렉토리 경로로 변경해주세요

        # 디렉토리 내의 파일 목록 조회
        file_list = os.listdir(directory_path)

        # 확장자가 '.csv'이고 파일명이 '035720'로 시작하는 파일 찾기
        matching_files = [filename for filename in file_list if filename.startswith(ticker) and filename.endswith('.csv')]

        json_data = {}

        # 첫 번째 매칭 파일 선택 (해당 디렉토리 내에 하나의 파일만 있어야 함)
        if len(matching_files) == 1:
            selected_filename = matching_files[0]

            # CSV 파일 읽기
            csv_filepath = os.path.join(directory_path, selected_filename)
            data_df = pd.read_csv(csv_filepath)

            # 데이터를 원하는 형식의 JSON으로 변환
            for index, row in data_df.iterrows():
                json_data[row['Date']] = row['Price']

            # JSON 데이터 출력 또는 파일로 저장
            # print(json.dumps(json_data, indent=4))
        else:
            current_date = get_today()
            # CSV 파일이 들어 있는 디렉토리 경로
            directory_path = f'./models/{current_date}/'  # 실제 디렉토리 경로로 변경해주세요

            # 디렉토리 내의 파일 목록 조회
            file_list = os.listdir(directory_path)

            # 확장자가 '.csv'이고 파일명이 '035720'로 시작하는 파일 찾기
            matching_files = [filename for filename in file_list if filename.startswith(ticker) and filename.endswith('.csv')]

            # 첫 번째 매칭 파일 선택 (해당 디렉토리 내에 하나의 파일만 있어야 함)
            if len(matching_files) == 1:
                selected_filename = matching_files[0]

                # CSV 파일 읽기
                csv_filepath = os.path.join(directory_path, selected_filename)
                data_df = pd.read_csv(csv_filepath)

                # 데이터를 원하는 형식의 JSON으로 변환
                for index, row in data_df.iterrows():
                    json_data[row['Date']] = row['Price']

                # JSON 데이터 출력 또는 파일로 저장
                # print(json.dumps(json_data, indent=4))
            else:
                print("해당 조건을 만족하는 파일이 없거나 여러 개입니다.")

        # JSON 데이터 출력
        # print('JSON Data:', json_data)
        logger.info('JSON Data : %s',json_data)
        datas.append(json_data)

    return datas
########################################################################################################################




