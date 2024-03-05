import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

def get_listing_date(ticker):
    df = fdr.DataReader(ticker)

    listing_date = df.index[0].date()
    return listing_date


def main():
    # 주식 코드를 입력하세요
    stock_ticker = '000660'#하이닉스 #'005930'  # 삼성전자

    listing_date = get_listing_date(stock_ticker)
    print(f"상장일: {listing_date}")


main()