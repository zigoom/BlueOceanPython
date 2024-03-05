import numpy as np
import FinanceDataReader as fdr
from sklearn import preprocessing
from keras.models import load_model
from datetime import datetime, timedelta
import pandas as pd

def main():
    # 내일 예측을 위한 입력 데이터 준비
    ticker = '005930'  # 삼성전자
    stt = '2023-01-18'  # 시작 날짜
    end = '2023-08-16'  # 종료 날짜
    history_points = 50

    model_path = '../test_function2/models/2023-08-18/005930_2023-08-17.h5'

    # 훈련된 모델 불러오기
    model = load_model(model_path)

    # 오늘까지의 데이터 가져오기
    today_data = fdr.DataReader(ticker, stt, end)
    today_data = today_data.iloc[:, 0:-1]

    # 데이터프레임의 인덱스 값을 받기
    index_dates = today_data.index.values
    today_data = today_data.values

    # 예측 데이터 준비
    prediction_data = today_data[-history_points:].copy()
    data_normalizer = preprocessing.MinMaxScaler()
    prediction_data_normalized = data_normalizer.fit_transform(prediction_data)
    predicted_normalized = model.predict(np.expand_dims(prediction_data_normalized, axis=0))

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(today_data[:, 0].reshape(-1, 1))
    predicted_price = y_normalizer.inverse_transform(predicted_normalized)

    print('222222 : ', predicted_price.flatten())


    # 예측 날짜 계산
    last_date = datetime.strptime(end, '%Y-%m-%d')
    predicted_date = np.datetime64((last_date + timedelta(days=1)).strftime('%Y-%m-%d'))

    # 전체 날짜와 가격 데이터 생성
    all_dates = np.append(index_dates, [predicted_date])
    all_prices = np.append(today_data[:, 0], [predicted_price[0, 0]])

    # 데이터프레임 생성
    df = pd.DataFrame({
        'Date': all_dates,
        'Price': all_prices
    })

    # CSV 파일로 저장
    df.to_csv('predicted_prices.csv', index=False)

main()
