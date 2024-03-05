import numpy as np
import FinanceDataReader as fdr
from sklearn import preprocessing
from keras.models import load_model

import numpy as np
import pandas as pd

def get_listing_date(ticker):
    stock_list = fdr.StockListing('KRX')
    listing_date = stock_list[stock_list['Code'] == ticker]['ListingDate'].values[0]
    return listing_date

def main():
    ticker = '005930'  # 삼성전자
    # stt = '2015-01-01'  # 데이터 시작 날짜
    # end = '2023-08-18'  # 데이터 종료 날짜
    # history_points = 50
    #
    # # 상장 날짜 확인
    # listing_date = get_listing_date(ticker)

    stocks = fdr.StockListing('KRX')
    stock_list = fdr.StockListing('KRX')

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    print(f"Listing Date of {ticker}: {stock_list}")

main()