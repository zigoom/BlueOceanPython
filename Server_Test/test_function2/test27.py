import FinanceDataReader as fdr
import pandas as pd
import  json
import random
import csv
def main():
    # CSV 파일을 읽어옵니다
    #data = pd.read_csv('top_10_tickers.csv', na_values=['None', 'none', 'NaN', 'nan', 'N/A', 'n/a'])

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    columns_to_check = ['Changes', 'ChagesRatio', 'Open', 'High', 'Low', 'Volume', 'Amount', 'Marcap']

    # CSV 파일을 읽어옵니다
    data = pd.read_csv('top_10_tickers.csv')

    # 공백을 제거합니다
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # NaN 및 None 값을 포함한 행들을 찾습니다
    nan_check_per_column = data[columns_to_check].isna().any()

    # 결과 출력
    if nan_check_per_column.empty:
        print("CSV 파일에 None 또는 NaN 값이 없습니다.")
    else:
        print("CSV 파일에 None 또는 NaN 값을 포함한 행들:")

main()