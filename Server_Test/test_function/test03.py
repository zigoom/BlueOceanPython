import numpy as np
import FinanceDataReader as fdr
from sklearn import preprocessing
from keras.models import load_model

import numpy as np
import pandas as pd

def main():
    df = fdr.StockListing('KRX')

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    # 데이터프레임 출력
    print("Recent Real and Predicted Values:")
    print(df)

main()