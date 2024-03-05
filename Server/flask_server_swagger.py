
from flask import Flask
from flask_cors import CORS
from flask_restx import Resource, Api
from blue_oceans import BlueOceans
from flask_cors import CORS
from dateutil import parser
from datetime import datetime, time, date, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from blue_oceans import logger  # 로거 설정 모듈 가져오기

import FinanceDataReader as fdr
import time as time_class
import pandas as pd
import numpy as np
import holidays
import random
import csv
import sys
import os

import tensorflow
import time as time_class
from sklearn import preprocessing
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers


app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version='1.0.0',
    title="BlueOceans API Server",
    description="블루오션 주식추천 플랫폼 사용을 위한  API 제공 Server 입니다.",
)
api.add_namespace(BlueOceans, '/blue-oceans')

#####################################################################################################

def get_stock_name():
    # CSV 파일명 생성
    filename = 'stock_list.csv'

    # CSV 파일이 존재하지 않으면
    if os.path.exists(filename)==False:
        # 파일이 존재하지 않을 경우, StockListing 함수를 사용하여 주식 데이터 검색
        krx_tickers = fdr.StockListing('KRX')[['Code', 'Name']]

        # DataFrame을 CSV 파일로 저장
        krx_tickers.to_csv(filename, index=False, encoding='utf-8-sig')

# 매일 9시 10분에 주식 항목 확인 및 수정 함수
def job_function():
    # CSV 파일명 생성
    filename = 'stock_list_all.csv'

    # CSV 파일이 존재하지 않으면
    if not os.path.exists(filename):
        # 파일이 존재하지 않을 경우, StockListing 함수를 사용하여 주식 데이터 검색
        krx_tickers = fdr.StockListing('KRX')[['Code', 'Name']]

        # DataFrame을 CSV 파일로 저장
        krx_tickers.to_csv(filename, index=False, encoding='utf-8-sig')
    else:
        # 기존 종목 리스트를 불러옴 (stock_list.csv 파일로 가정)
        previous_stock_list = pd.read_csv(filename)

        # FinanceDataReader를 이용하여 현재 종목 리스트를 불러옴
        current_stock_list = fdr.StockListing('KRX')[['Code', 'Name']]

        # 이전 리스트와 현재 리스트를 비교하여 추가된 종목을 찾음
        new_stocks = current_stock_list[~current_stock_list['Code'].isin(previous_stock_list['Code'])]

        if not new_stocks.empty:
            print("새로 추가된 종목이 있음")
            print(new_stocks)
            # 추가된 종목을 처리하는 로직을 추가

            # 새로 추가된 종목을 파일에 추가로 저장 (다음 비교를 위해)
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                new_stocks.to_csv(f, header=False, index=False)

        # 현재 종목 리스트를 파일로 저장 (다음 비교를 위해)
        current_stock_list.to_csv(filename, index=False, encoding='utf-8-sig')

#####################################################################################################
#  csv 파일을 튜플형태의 배열로 받기
def read_csv_to_tuple_array(file_path):
    # csv 파일을 데이터프레임으로 읽어옴
    df = pd.read_csv(file_path)

    # 데이터프레임을 튜플 배열로 변환하여 반환
    tuple_array = [tuple(row) for row in df.values]

    return tuple_array

# 주식의 현재가격을 저장하는 함수
def save_stock_data():
    # 코드 실행 시작 시간 기록
    start_time = time_class.time()

    # csv 파일의 경로
    file_path = './save_stock_list.csv'

    # 튜플 배열로 변환 (주식 종목 리스트 (이름, 코드))
    dataList = read_csv_to_tuple_array(file_path)

    # 저장할 위치 설정 (하위 폴더명)
    save_dir = './stock_data'

    # 주식 종목 리스트 (이름, 코드)
    # stock_list = [('삼성전자', '005930'), ('SK하이닉스', '000660'), ('LG화학', '051910')]

    # 현재 시간 정보 가져오기
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M')
    print(current_time)
    # 저장할 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # 오늘 날짜의 CSV 파일이 있는지 확인
    file_path = os.path.join(save_dir, f'{current_date}.csv')
    if os.path.exists(file_path):
        # 이미 파일이 존재하면 CSV 파일을 읽어와 데이터프레임 생성
        df = pd.read_csv(file_path)
    else:
        # 파일이 존재하지 않으면 빈 데이터프레임 생성
        df = pd.DataFrame()

    # 주식 데이터 받아오기
    df_list = [fdr.DataReader(code, current_date, current_date)['Close'] for code, name in dataList]

    # 받아온 데이터를 현재 시간과 함께 데이터프레임으로 만들기
    df_new = pd.concat(df_list, axis=1)
    df_new.columns = [code for code,name in dataList]
    df_new.insert(0, 'Time', current_time)

    # 기존 데이터프레임에 추가하기
    df = pd.concat([df, df_new])

    # 데이터프레임을 CSV 파일로 저장(float_format='%.0f' 를 함으로 float와 같이있는 int 형식의 데이터를강제로 float로 안바꾸게 처리)
    df.to_csv(file_path, index=False, float_format='%.0f')

    # 코드 실행 종료 시간 기록
    end_time = time_class.time()

    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"주식 정보 저장 코드 실행 시간: {execution_time:.5f} 초")
    logger.info(f"주식 정보 저장 코드 실행 시간: {execution_time:.5f} 초")

# 한국의 공휴일 정보를 가져오는 함수
def get_korean_holidays(year):
    return holidays.Korea(years=year)

# 평일인지 확인하는 함수 (토요일: 5, 일요일: 6)
def is_weekday():
    today = datetime.today()
    return today.weekday() < 5 and today not in get_korean_holidays(today.year)

# 1분 단위로 호출할 함수
def my_function():
    if is_weekday():
        now = datetime.now().time()
        start_time = time(8, 59)
        end_time = time(15, 31)
        if now >= start_time and now <= end_time:
            # print("평일 오전 9시 00분에서 오후 3시 30분까지 1분 단위로 호출되는 함수입니다.")
            save_stock_data()

#####################################################################################################

# 1분 단위로 호출할 함수
# (KS11:코스피, KQ11:코스닥, KS200:코스피200, USD/KRW:원/달러 환율, JPY/KRW:원/엔화 환율, BTC/KRW:원/비트코인)을 저장하는 함수
def save_stock_index_data():
    # 코드 실행 시작 시간 기록
    start_time = time_class.time()
    # csv 파일의 경로
    file_path = './stock_index_list.csv'
    # 튜플 배열로 변환 (주식 종목 리스트 (이름, 코드))
    dataList = read_csv_to_tuple_array(file_path)
    # 저장할 위치 설정 (하위 폴더명)
    save_dir = './stock_index_data'

    # 현재 시간 정보 가져오기
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M')
    print(current_time)
    # 저장할 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # 오늘 날짜의 CSV 파일이 있는지 확인
    file_path = os.path.join(save_dir, f'{current_date}.csv')
    if os.path.exists(file_path):
        # 이미 파일이 존재하면 CSV 파일을 읽어와 데이터프레임 생성
        df = pd.read_csv(file_path)
    else:
        # 파일이 존재하지 않으면 빈 데이터프레임 생성
        df = pd.DataFrame()

    # 주식 데이터 받아오기
    df_list = [fdr.DataReader(code, current_date, current_date)['Close'] for code, name in dataList]

    # 받아온 데이터를 현재 시간과 함께 데이터프레임으로 만들기
    df_new = pd.concat(df_list, axis=1)
    df_new.columns = [code for code,name in dataList]
    df_new.insert(0, 'Time', current_time)

    # 기존 데이터프레임에 추가하기
    df = pd.concat([df, df_new])

    # 데이터프레임을 CSV 파일로 저장(float_format='%.0f' 를 함으로 float와 같이있는 int 형식의 데이터를강제로 float로 안바꾸게 처리)
    df.to_csv(file_path, index=False )#, float_format='%.0f')

    # 코드 실행 종료 시간 기록
    end_time = time_class.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    logger.info(f"시장,환율 정보 저장 실행 시간: {execution_time:.5f} 초")
    print(f"시장,환율 정보 저장 실행 시간: : {execution_time:.5f} 초")

    # 1분 단위로 호출할 함수

def my_function2():
    if is_weekday():
        now = datetime.now().time()
        start_time = time(9, 19)
        if now >= start_time:
            # print("평일 오전 9시 20분에서 오후 23시 59분까지 1분 단위로 호출되는 함수입니다.")
            save_stock_index_data()

#####################################################################################################

# 평일 5시에 딥러닝 모델을 새로 만들고 다음날 주가를 예측해서 파일로 남기는 함수
def save_model_deep():
    # CSV 파일을 DataFrame으로 읽어옴
    stock_list_csv_name = 'stock_list.csv'
    # 오늘 날자 받기
    current_date = get_today()

    # CSV 파일을 DataFrame으로 읽어옴
    data = pd.read_csv(stock_list_csv_name)

    # 105560 코드에 해당하는 행을 필터링
    # filtered_row = data[data['Code'] == '005930']

    # Name과 ListingDate 값을 가져옴
    name = data['Name'].values[0]
    listing_date = data['ListingDate'].values[0]

    print("Name:", name)
    print("ListingDate:", listing_date)

    # 폴더 새로만들기
    check_file_name()

    ############################################################
    #######     이 부분은 반복문을 사용하여 144개의 모델을 생성해야 한다
    ############################################################

    # CSV 파일 읽기
    with open(stock_list_csv_name, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            start_date = '2010-01-01'
            end_date = current_date
            date = row['ListingDate']

            date1 = parser.parse(start_date)
            date2 = parser.parse(date)

            if date1 < date2:
                print("2010-01-01 이후부터 데이터가 있어서 데이터가 있는 날부터 훈련 데이터로 지정")
                logger.info("2010-01-01 이후부터 데이터가 있어서 데이터가 있는 날부터 훈련 데이터로 지정")
                start_date = date
            elif date1 > date2:
                print("2010-01-01 이전 데이터부터 있어서 2010-01-01부터 훈련 데이터로 지정")
                logger.info("2010-01-01 이후부터 데이터가 있어서 데이터가 있는 날부터 훈련 데이터로 지정")

            # 모델 생성하기
            make_model(row['Code'], start_date, end_date)

            # if row['Code'] == target_code:
            #     target_name = row['Name']
            #     target_listing_date = row['ListingDate']
            #     break  # 종목을 찾았으면 루프 종료
    #########################################################################################
    #######    이 부분은 반복문을 사용하여 144개의 모델을 사용해서 오늘값 + 내일값 + 100일치 데이터를 저장하는 부분이다.
    #########################################################################################
    # CSV 파일 읽기
    with open(stock_list_csv_name, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            last_date = datetime.strptime(current_date, '%Y-%m-%d')
            start_date = (last_date + timedelta(days=-100))
            end_date = current_date
            ticker = row['Code']
            history_points = 50

            model_path = f"./models/today/{ticker}_{current_date}.h5"

            # 훈련된 모델 불러오기
            model = load_model(model_path)

            # 오늘까지의 데이터 가져오기
            today_data = fdr.DataReader(ticker, start_date, end_date)
            today_data = today_data.iloc[:, 0:-1]

            # 데이터프레임의 인덱스 값을 받기
            index_dates = today_data.index.values
            today_data = today_data.values

            # 데이터 정규화
            data_normalizer = preprocessing.MinMaxScaler()
            today_data_normalized = data_normalizer.fit_transform(today_data)

            # 예측 데이터 준비
            prediction_data = today_data_normalized[-history_points:].copy()
            predicted_normalized = model.predict(np.expand_dims(prediction_data, axis=0))

            # 역정규화하여 예측 가격 얻기
            y_normalizer = preprocessing.MinMaxScaler()
            y_normalizer.fit(today_data[:, 0].reshape(-1, 1))
            predicted_price = y_normalizer.inverse_transform(predicted_normalized)
            print(predicted_price)

            # 예측 날짜 계산
            last_date = datetime.strptime(end_date, '%Y-%m-%d')
            predicted_date = np.datetime64((last_date + timedelta(days=1)).strftime('%Y-%m-%d'))

            # 전체 날짜와 가격 데이터 생성
            all_dates = np.append(index_dates, [predicted_date])
            all_prices = np.append(today_data[:, 3], [predicted_price[0, 0]])

            # 데이터프레임 생성
            df = pd.DataFrame({
                'Date': all_dates,
                'Price': all_prices
            })

            # CSV 파일로 저장
            df.to_csv(f"./models/today/{ticker}_{current_date}.csv", index=False)

    # 오늘의 AI 랜덤값을 만든다.
    # CSV 파일 읽기
    csv_filename = 'stock_list.csv'  # 파일 이름을 적절하게 변경해주세요
    data_df = pd.read_csv(csv_filename)

    target_codes = []
    target_names = []

    while len(target_names) < 5:
        # 랜덤한 행 번호 생성 (중복 없이)
        random_number = random.choice(range(1, len(data_df) + 1))

        # 선택된 행 번호로부터 데이터 가져오기
        selected_row = data_df.iloc[random_number - 1]

        # 'Code' 열의 값이 target_codes와 일치하면 다시 랜덤한 행 번호를 선택
        if selected_row['Code'] in target_codes:
            continue

        target_names.append(selected_row['Name'])
        target_codes.append(selected_row['Code'])

    today_ai_csv_filename = 'today_ai_list.csv'
    # CSV 파일 생성 및 데이터 쓰기
    with open(today_ai_csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Code', 'Name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # 헤더(필드명) 쓰기

        for code, name in zip(target_codes, target_names):
            writer.writerow({'Code': code, 'Name': name})

    # 주말에 보여줄 거래량 top10 데이터 파일 생성
    save_top10_stock_data_file()

def save_top10_stock_data_file(marketId='KRX'):

    # get_tickers() 함수를 사용하여 모든 종목의 심볼(symbol) 리스트를 가져옵니다
    all_tickers = fdr.StockListing('KRX')
    print("all: ", all_tickers)

    # 상위 거래량 종목을 가져오기 위해 'Volume' 기준으로 정렬합니다
    top_volume_tickers = all_tickers.sort_values(by='Volume', ascending=False)
    print("top_volume_tickers: ", top_volume_tickers)

    # 상위 N개의 거래량 종목을 선택합니다 (여기서는 상위 10개)
    top_n_tickers = top_volume_tickers.head(10)
    print("top_n_tickers: ", top_n_tickers)

    # 오늘 날짜 가져오기
    today_date = datetime.now().strftime('%Y-%m-%d')

    top_10_data_path = './top_10_data'
    today_path = os.path.join(top_10_data_path, 'today')
    today_backup_path = os.path.join(top_10_data_path, today_date)

    if os.path.exists(today_path):
        os.rename(today_path, today_backup_path)  # 폴더명을 yyyy-mm-dd 형태로 변경

    # 데이터프레임을 CSV 파일로 저장합니다
    # top_n_tickers.to_csv('./top_10/top_10_tickers.csv', index=False)
    save_path = './top_10_data/today/'
    os.makedirs(save_path, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)
    save_file = os.path.join(save_path, 'top_10_tickers.csv')
    top_n_tickers.to_csv(save_file, index=False)

    if os.path.exists(save_file):
        loaded_df = pd.read_csv(save_file)





# 데이터 정규화
def call_dataset(ticker='005930', stt='1999-05-07', end='2023-08-16', history_points=50):

    data = fdr.DataReader(ticker, stt, end)
    dates = data.index  # 날짜 데이터 가져오기
    data = data.iloc[:, 0:-1]
    data = data.values  # 값만 갖고온다

    data_normalizer = preprocessing.MinMaxScaler()  # 데이터를 0~1 범위로 점철되게 하는 함수 call
    data_normalized = data_normalizer.fit_transform(data)  # 데이터를 0~1 범위로 점철되게 함수 수행
    ohlcv_histories_normalized = np.array([data_normalized[i:i + history_points].copy() for i in range(
        len(data_normalized) - history_points)])  # ohlcv를 가지고 오되, 관찰일수 만큼 누적해서 쌓는다. (열방향으로)

    next_day_open_values_normalized = np.array(
        [data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)  # 1XN 벡터 -> NX1 벡터로

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)  # 1XN 벡터 -> NX1 벡터로

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # 인풋 X : 그 이전의 OHLCV (from T = -50 to T = -1)
    # 아웃풋 y : 예측하고자 하는 주가 T = 0

    def calc_ema(values, time_period):
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalized:
        sma = np.mean(his[:, 3])  # 각 데이터포인트별 Close Price 평균
        macd = calc_ema(his, 12) - calc_ema(his, 26)  # 12일 EMA - 26일 EMA
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == \
           technical_indicators_normalized.shape[0]

    return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, dates


def make_model(ticker='005930', stt='2010-01-01', end='2023-08-16'):
    # 오늘 날자 받기
    current_date = get_today()

    history_points = 50
    np.random.seed(4)
    tensorflow.random.set_seed(44)
    # from util import csv_to_dataset, history_points

    # 코드 실행 시작 시간 기록
    start_time = time_class.time()

    # dataset
    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser, dates = call_dataset(ticker, stt, end)
    train_ratio = 0.7
    n = int(ohlcv_histories.shape[0] * train_ratio)

    # 코드 실행 종료 시간 기록
    end_time = time_class.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"정규화 처리 시간: {execution_time:.5f} 초")
    logger.info(f"정규화 처리 시간: {execution_time:.5f} 초")

    # 코드 실행 시작 시간 기록
    start_time = time_class.time()

    ohlcv_train = ohlcv_histories[-n:-1]
    y_train = next_day_open_values[-n:-1]

    ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0] - n]
    y_test = next_day_open_values[:ohlcv_histories.shape[0] - n]

    unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0] - n]
    ohlcv_train

    # model architecture
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # 예측 결과 역정규화
    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape
    model.save(f'./models/today/{ticker}_{current_date}.h5')

    # 코드 실행 종료 시간 기록
    end_time = time_class.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"모델 훈련 및 파일 생성 시간: {execution_time:.5f} 초")
    logger.info(f"모델 훈련 및 파일 생성 시간: {execution_time:.5f} 초")


# 오늘 날자 받는 함수
def get_today():
    # 오늘 날짜 정보
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    return current_date


# today 폴더가 있으면 이름을 바꿔주고, 없으면 폴더를 만들어 준다.
def check_file_name():
    current_date = get_today()

    original_path = './models/today'
    new_path = f'./models/{current_date}'

    if os.path.exists(original_path):
        try:
            os.rename(original_path, new_path)
            print(f"폴더 이름 변경 완료: 'today' -> '{current_date}'")
            logger.info(f"폴더 이름 변경 완료: 'today' -> '{current_date}'")
        except Exception as e:
            print(f"폴더 이름 변경 실패: {e}")
            logger.info(f"폴더 생성 실패: {e}")
    else:
        try:
            os.makedirs(original_path)
            print(f"폴더 생성 완료: '{original_path}'")
            logger.info(f"폴더 생성 완료: '{original_path}'")
        except Exception as e:
            print(f"폴더 생성 실패: {e}")
            logger.info(f"폴더 생성 실패: {e}")


def my_function3():
    if is_weekday():
        save_model_deep()

#####################################################################################################
def schedule_job():
    scheduler = BackgroundScheduler()
    # 주기를 매일로 설정하고 평일에만 실행하도록 설정합니다.
    scheduler.add_job(my_function, 'interval', minutes=1)  # 1분마다 함수 호출
    scheduler.add_job(my_function2, 'interval', minutes=1)  # 1분마다 함수 호출
    #scheduler.add_job(my_function3, 'cron', day_of_week='mon-fri', hour=17)
    # scheduler.add_job(job_function, 'cron', day_of_week='mon-fri', hour=9, minute=10)
    scheduler.start()


if __name__ == "__main__":
    schedule_job()
    logger.info(f"서버 시작")
    # app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
    app.run(port=5001, debug=False, use_reloader=False) #, host='192.168.0.74')