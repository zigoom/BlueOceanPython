import FinanceDataReader as fdr
import pandas as pd
import  json
import random
import csv
def main():
    # CSV 파일을 읽어와서 데이터프레임으로 저장합니다
    loaded_df = pd.read_csv('./top_10_data/today/top_10_tickers.csv')

    # 읽어온 데이터프레임 출력
    print(loaded_df)


main()