
import numpy as np
import FinanceDataReader as fdr

from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def load_and_predict(model_path, recent_data, scaler):
    # 모델 불러오기
    model = keras.models.load_model(model_path)

    # 최근 50일 데이터 정규화 및 전처리
    recent_data_normalized = scaler.transform(recent_data)

    # 최대 50건의 데이터로 맞추기 위해 0으로 패딩
    padded_data = np.zeros((50, recent_data.shape[1]))
    padded_data[-recent_data.shape[0]:, :] = recent_data_normalized

    # 최근 50일 데이터를 3D 형태로 변환 (시계열 데이터)
    recent_data_reshaped = np.reshape(padded_data, (1, padded_data.shape[0], padded_data.shape[1]))

    # 주가 예측
    predicted_normalized = model.predict(recent_data_reshaped)

    # 역정규화하여 실제 주가 예측값 얻기
    predicted = scaler.inverse_transform(predicted_normalized)[0]  # 1D 배열로 변환

    return predicted

def main():
    # 모델 파일 경로
    model_path = './models/2023-08-18/005930_2023-08-17.h5'

    # 주식 코드 (삼성전자)
    ticker = '005930'

    # 최근 일정 기간 동안의 OHLCV 데이터 가져오기 (주말, 공휴일 제외)
    end_date = datetime.now().date() - timedelta(days=1)  # 어제까지의 데이터
    start_date = end_date - timedelta(days=50)  # 최대 50일 데이터로 맞추기 위해

    recent_data = fdr.DataReader(ticker, start_date, end_date)
    recent_data = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # 데이터 정규화
    scaler = MinMaxScaler()
    scaler.fit(recent_data)

    # 주가 예측
    predicted_price = load_and_predict(model_path, recent_data, scaler)

    print(f"Predicted Price for Tomorrow: {predicted_price:.2f}")


main()
