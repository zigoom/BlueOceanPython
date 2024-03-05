import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

def get_listing_date(ticker):
    url = f'https://finance.naver.com/item/main.nhn?code={ticker}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'no_info'})
    rows = table.find_all('tr')

    for row in rows:
        th = row.find('th')
        if th and th.get_text() == '상장일':
            listing_date = row.find('td').get_text()
            return listing_date.strip()

    return None

def main():
    ticker = '005930'  # 삼성전자
    listing_date = get_listing_date(ticker)

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    if listing_date:
        print(f"Listing Date for {ticker}: {listing_date}")
    else:
        print(f"Listing Date for {ticker} not found.")


main()