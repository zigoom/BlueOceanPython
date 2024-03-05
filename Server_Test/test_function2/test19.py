
import FinanceDataReader as fdr
import  json
def main():
    call_symbol_data()
def call_symbol_data(symbol='KQ11', stt='2023-08-23', end='2023-08-23'):
    data = fdr.DataReader(symbol, stt, end)

    # DataFrame의 인덱스를 8자리로 변환하여 새로운 컬럼 'Date' 추가
    # data.insert(0, 'Date', data.index.strftime('%Y%m%d'))

    # JSON으로 변환하면서 'Date' 컬럼 사용하도록 지정
    # json_data = data.to_json(orient='records', date_format='epoch', date_unit='s')
    json_data = data.to_json(orient='records', date_format='epoch', date_unit='s', force_ascii=False)
    json_data = json.loads(json_data)

    print('json_data : %s', json_data)
    return json_data

main()