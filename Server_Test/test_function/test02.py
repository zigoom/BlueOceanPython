import numpy as np
import FinanceDataReader as fdr
from sklearn import preprocessing
from keras.models import load_model
from datetime import datetime


def main():
    # 내일 예측을 위한 입력 데이터 준비
    ticker = '005930'  # 삼성전자
    stt = '2023-01-18'  # 시작 날짜
    end = '2023-08-10'  # 종료 날짜
    history_points = 50

    # 현재 시간 정보 가져오기
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')

    # 훈련된 모델 불러오기
    model = load_model(f"{ticker}_{current_date}.h5")

    data = fdr.DataReader(ticker, stt, end)
    data = data.iloc[:, 0:-1]
    data = data.values

    # 데이터가 충분한지 확인
    if len(data) < history_points:
        print("Not enough data for prediction")
    else:
        data_normalizer = preprocessing.MinMaxScaler()
        data_normalized = data_normalizer.fit_transform(data)
        ohlcv_histories_normalized = np.array(
            [data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])

        # 내일 예측 수행
        predicted_normalized = model.predict(np.expand_dims(ohlcv_histories_normalized[-1], axis=0))  # 가장 최근의 데이터로 예측
        y_normalizer = preprocessing.MinMaxScaler()
        y_normalizer.fit(data[:, 0].reshape(-1, 1))
        predicted_price = y_normalizer.inverse_transform(predicted_normalized)  # 역정규화

        print("Predicted Next Day Opening Price:", predicted_price)



main()