import FinanceDataReader as fdr
import pandas as pd
import  json
import random
import csv
import os
from datetime import datetime

def main():
    call_top10_stock_data()

def save_top10_stock_data_file(marketId='KRX'):

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

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
    today_path = os.path.join(top_10_data_path, 'today_Te')
    today_backup_path = os.path.join(top_10_data_path, today_date)

    if os.path.exists(today_path):
        os.rename(today_path, today_backup_path)  # 폴더명을 yyyy-mm-dd 형태로 변경

    # 데이터프레임을 CSV 파일로 저장합니다
    #top_n_tickers.to_csv('./top_10/top_10_tickers.csv', index=False)
    save_path = './top_10_data/today_Te/'
    os.makedirs(save_path, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)
    save_file = os.path.join(save_path, 'top_10_tickers.csv')
    top_n_tickers.to_csv(save_file, index=False)

    if os.path.exists(save_file):
        loaded_df = pd.read_csv(save_file)
        print(loaded_df)

main()