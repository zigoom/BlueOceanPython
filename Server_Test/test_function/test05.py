from pykrx import stock

import numpy as np
import pandas as pd

def get_listing_date(stock_code):
    # stock_info = stock.get_market_ticker_list()
    # listing_date = stock_info['상장일']
    market_info = stock.get_market_ticker_list()
    for item in market_info:
        if item[0] == stock_code:
            listing_date = item[2]  # 상장일 정보가 포함된 항목의 세 번째 요소
            return listing_date
def main():
    ticker = '005930'  # 삼성전자

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    listing_date = get_listing_date(ticker)

    if listing_date:
        print(f"Listing Date for {ticker}: {listing_date}")
    else:
        print(f"Listing Date for {ticker} not found.")


main()