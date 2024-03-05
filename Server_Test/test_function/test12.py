import numpy as np
import FinanceDataReader as fdr
from sklearn import preprocessing
from keras.models import load_model
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import time as time_class

def main():
    # 내일 예측을 위한 입력 데이터 준비
    ticker = '005930'  # 삼성전자
    stt = '2023-01-18'  # 시작 날짜
    end = '2023-08-16'  # 종료 날짜
    history_points = 50


    data = fdr.DataReader(ticker, end, end)
    print(data)
    data = fdr.DataReader(ticker, '2023-08-17', '2023-08-17')
    print(data)


    model_path = '../test_function2/models/2023-08-18/005930_2023-08-17.h5'

    # 코드 실행 시작 시간 기록
    start_time = time_class.time()

    # 현재 시간 정보 가져오기
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')

    # 훈련된 모델 불러오기
    model = load_model(model_path)

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

        # 예측 날짜 계산
        last_date = datetime.strptime(end, '%Y-%m-%d')
        predicted_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(predicted_price.shape[1])]

        print('111111 : ', data[-len(ohlcv_histories_normalized):, 0])
        print('222222 : ', predicted_price.flatten())

        # 그래프 그리기
        plt.figure(figsize=(12, 6))
        plt.plot(data[-len(ohlcv_histories_normalized):, 0], label='Past Data', color='blue')
        plt.plot(predicted_price.flatten(), marker='o', markersize=8, color='red', label='Predicted Next Day')
        plt.xticks(np.arange(len(data), len(data) + len(predicted_dates)), predicted_dates, rotation=45)
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 다음 날의 예측 가격과 현재 날짜의 주식 가격 비교
        current_price = data[-1, 0]  # 현재 날짜의 주식 가격
        next_day_predicted_price = predicted_price[0, 0]  # 다음 날의 예측 가격

        if next_day_predicted_price > current_price:
            movement = "상승"
        elif next_day_predicted_price < current_price:
            movement = "하락"
        else:
            movement = "변동 없음"

        print(f"Predicted Next Day Opening Price: {next_day_predicted_price:.2f}")
        print(f"Current Day Closing Price: {current_price:.2f}")
        print(f"Price Movement: {movement}")

        # 코드 실행 종료 시간 기록
        end_time = time_class.time()
        # 실행 시간 계산
        execution_time = end_time - start_time
        print(f"정규화 처리 시간: {execution_time:.5f} 초")

main()
