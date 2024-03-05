import FinanceDataReader as fdr
import pandas as pd
import  json
def main():
    symbol = "KQ11"  # 예시로 "KQ11" 사용
    start_date = "2023-08-21"
    end_date = "2023-08-21"

    print("0000000")
    # 데이터 가져오기 시도
    data = fdr.DataReader(symbol, start_date, end_date)
    print("333333333")

    # 데이터가 None인 경우에 대한 처리
    if data is None:
        print("11111111")
        response = {
            "status": "error",
            "message": "No data available for the specified symbol and date range."
        }
        json_response = json.dumps(response)
        print(json_response)
    else:
        print("222222")
        # 여기에 데이터 처리 및 분석 코드를 작성
        # JSON으로 변환 후 반환하는 코드 추가
        data_json = data.to_json(orient="records")
        print(data_json)


main()