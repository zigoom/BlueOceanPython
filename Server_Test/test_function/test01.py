import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return np.array(ema)
def main():
    # 주식 데이터 가져오기 (삼성전자)
    stock_data = fdr.DataReader('005930', '2021-08-07', '2023-08-07')

    # 주식 가격 데이터 추출
    stock_price = stock_data['Close'].values

    # 지수이동평균을 사용하여 데이터 정규화
    alpha = 0.15  # 지수이동평균의 smoothing factor
    ema_normalized = exponential_moving_average(stock_price, alpha)

    # 데이터 전처리
    history_points = 50  # 과거 주식 데이터 개수
    X = np.array([ema_normalized[i:i + history_points] for i in range(len(ema_normalized) - history_points)])
    y = ema_normalized[history_points:]

    # 훈련 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # LSTM 모델 구축
    model = Sequential()
    model.add(LSTM(50, input_shape=(history_points, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 모델 학습
    model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)

    # 예측 결과 생성
    y_pred = model.predict(X_test)

    # 예측 결과 역정규화
    y_pred_denormalized = y_pred * (1 - alpha) + ema_normalized[history_points:]
    y_test_denormalized = y_test * (1 - alpha) + ema_normalized[history_points:]

    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_denormalized, label='Real')
    plt.plot(y_pred_denormalized, label='Predicted')
    plt.legend()
    plt.title('Samsung Electronics Stock Price Prediction with Exponential Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()


main()
