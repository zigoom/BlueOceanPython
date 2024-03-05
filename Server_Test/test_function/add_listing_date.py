import pandas as pd
import FinanceDataReader as fdr

def get_listing_date(code):
    df = fdr.DataReader(code)
    listing_date = df.index[0].date()
    return listing_date
def main():
    # CSV 파일 경로
    csv_file = 'save_stock_list.csv'

    # CSV 파일 읽어오기
    data = pd.read_csv(csv_file)

    # 새로운 열 추가
    data['ListingDate'] = data['Code'].apply(get_listing_date)

    # 수정된 데이터를 CSV 파일에 저장
    data.to_csv('stock_list_with_listing_date.csv', index=False)


main()
