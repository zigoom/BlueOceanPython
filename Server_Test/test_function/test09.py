import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def get_market_open_dates(start_date, num_dates):
    url = f"https://www.krx.co.kr/por_kor/popup/JHPKOR18101.jsp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {
        "indexTme": "1",
        "isu_cdnm": "전체",
        "isu_cd": "",
        "isu_nm": "",
        "isu_srt_cd": "",
        "strt_dd": start_date.strftime('%Y%m%d'),
        "end_dd": (start_date + timedelta(days=num_dates - 1)).strftime('%Y%m%d'),
        "pagePath": "/por_kor/popup/JHPKOR18101.jsp"
    }

    response = requests.get(url, headers=headers, params=params)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="board_list")

    market_open_dates = []

    if table:
        rows = table.find_all("tr")[1:]
        for row in rows:
            date_str = row.find_all("td")[0].get_text(strip=True)
            date = datetime.strptime(date_str, "%Y%m%d").date()
            market_open_dates.append(date.strftime('%Y-%m-%d'))

    return market_open_dates


def get_stock_data():
    current_date = datetime.now().date()

    # 최대 100건의 데이터를 가져옴
    stock_data = get_market_open_dates(start_date=current_date, num_dates=100)

    return stock_data

def main():
    stock_data = get_stock_data()

    # 결과 출력
    for date in stock_data:
        print(date)


main()
