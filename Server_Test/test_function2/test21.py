
import FinanceDataReader as fdr
import pandas as pd
import  json
def main():
    # CSV 파일을 pandas DataFrame으로 읽어옵니다
    data = pd.read_csv("../stock_index_data/2023-08-22.csv")

    # "KQ11" 열의 마지막 값을 가져옵니다
    last_kq11_value = data["KQ11"].iloc[-1]

    # 소수점 둘째 자리까지 포맷팅하여 출력합니다
    formatted_kq11_value = "{:.2f}".format(last_kq11_value)
    print("Last KQ11 value:", formatted_kq11_value)


main()