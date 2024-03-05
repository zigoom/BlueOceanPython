import FinanceDataReader as fdr
import pandas as pd
import  json
import random
import csv
def main():
    call_top10_stock_data()

def call_top10_stock_data(marketId='KRX'):

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    # top_n_tickers = pd.read_csv('top_10_tickers.csv')
    # print("top_n_tickers: ", top_n_tickers)

    # get_tickers() 함수를 사용하여 모든 종목의 심볼(symbol) 리스트를 가져옵니다
    all_tickers = fdr.StockListing('KRX')

    # 상위 거래량 종목을 가져오기 위해 'Volume' 기준으로 정렬합니다
    top_volume_tickers = all_tickers.sort_values(by='Volume', ascending=False)

    # 상위 N개의 거래량 종목을 선택합니다 (여기서는 상위 10개)
    top_n_tickers = top_volume_tickers.head(10)

    # 필요한 컬럼들
    columns_to_check = ['Changes', 'ChagesRatio', 'Open', 'High', 'Low', 'Volume', 'Amount', 'Marcap']

    # 각 컬럼에 대해 NaN 값을 확인하고, any() 함수로 하나라도 True 값이 있는지 확인합니다
    nan_check_per_column = top_n_tickers[columns_to_check].isna().any()
    print("nan_check_per_column: ", nan_check_per_column)

    # 어떤 컬럼에서 NaN 값이 있는지 출력합니다
    columns_with_nan = nan_check_per_column[nan_check_per_column].index.tolist()
    print("columns_with_nan: ", columns_with_nan)

    # NaN 값을 포함한 행들을 찾습니다
    # rows_with_nan = top_n_tickers[top_n_tickers.isna().any(axis=1)]
    # print(rows_with_nan)

    if columns_with_nan:
        print("NaN 값이 있는 경우, 저장된 CSV 파일을 불러옵니다.")
        # loaded_df = pd.read_csv('./top_10_data/today_Te/top_10_tickers.csv')
        # print(loaded_df)
    else:
        print("NaN 값이 없읍니다.")




main()