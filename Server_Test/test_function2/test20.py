
import FinanceDataReader as fdr
import  json
def main():
    call_top10_stock_data()


def call_top10_stock_data(marketId='KRX'):
    all_tickers = fdr.StockListing(marketId)
    print(all_tickers)

    top_n_tickers = all_tickers.nlargest(10, 'Volume')

    json_data = top_n_tickers.to_dict(orient='records')

    # JSON 데이터 출력
    print('JSON Data:', json_data)

    return json_data

main()