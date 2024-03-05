import pandas as pd
import numpy as np
import csv
import os

import tensorflow
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

import time as time_class
from datetime import datetime, timedelta
from dateutil import parser

# CSV 파일을 DataFrame으로 읽어옴
stock_list_csv_name = 'stock_list.csv'

# 데이터 정규화
def call_dataset(ticker = '005930', stt = '1999-05-07', end = '2023-08-16', history_points = 50):
    print('ticker',ticker)
    print('stt',stt)
    print('end',end)

    data = fdr.DataReader(ticker, stt, end)
    dates = data.index  # 날짜 데이터 가져오기
    data = data.iloc[:,0:-1]
    data = data.values # 값만 갖고온다

    data_normalizer = preprocessing.MinMaxScaler() # 데이터를 0~1 범위로 점철되게 하는 함수 call
    data_normalized = data_normalizer.fit_transform(data) # 데이터를 0~1 범위로 점철되게 함수 수행
    ohlcv_histories_normalized = np.array([data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)]) # ohlcv를 가지고 오되, 관찰일수 만큼 누적해서 쌓는다. (열방향으로)

    next_day_open_values_normalized = np.array([data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1) # 1XN 벡터 -> NX1 벡터로

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1) # 1XN 벡터 -> NX1 벡터로

    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

    # 인풋 X : 그 이전의 OHLCV (from T = -50 to T = -1)
    # 아웃풋 y : 예측하고자 하는 주가 T = 0

    def calc_ema(values, time_period):
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalized:
        sma = np.mean(his[:, 3]) # 각 데이터포인트별 Close Price 평균
        macd = calc_ema(his, 12) - calc_ema(his, 26) # 12일 EMA - 26일 EMA
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalized.shape[0]

    return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, dates

def make_model(ticker='005930', stt = '2010-01-01', end = '2023-08-16'):
    #오늘 날자 받기
    current_date = get_today()

    history_points = 50
    np.random.seed(4)
    print(tensorflow.__version__)
    tensorflow.random.set_seed(44)
    # from util import csv_to_dataset, history_points

    # 코드 실행 시작 시간 기록
    start_time = time_class.time()

    # dataset
    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser, dates = call_dataset(ticker, stt, end)
    train_ratio = 0.7
    n = int(ohlcv_histories.shape[0] * train_ratio)

    # 코드 실행 종료 시간 기록
    end_time = time_class.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"정규화 처리 시간: {execution_time:.5f} 초")

    # 코드 실행 시작 시간 기록
    start_time = time_class.time()

    ohlcv_train = ohlcv_histories[-n:-1]
    y_train = next_day_open_values[-n:-1]

    ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0] - n]
    y_test = next_day_open_values[:ohlcv_histories.shape[0] - n]

    unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0] - n]
    ohlcv_train

    # model architecture
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # 예측 결과 역정규화
    y_test_predicted = model.predict(ohlcv_test)
    print('y_test_predicted : ', y_test_predicted)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    print('y_test_predicted : ', y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    print('y_predicted : ', y_predicted)
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    print('y_predicted : ', y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape

    model.save(f'./models/today_Te/{ticker}_{current_date}.h5')

    # 코드 실행 종료 시간 기록
    end_time = time_class.time()
    # 실행 시간 계산
    execution_time = end_time - start_time
    print(f"모델 훈련 및 파일 생성 시간: {execution_time:.5f} 초")

    # 예측 결과 시각화
    # plt.gcf().set_size_inches(22, 15, forward=True)
    # recent_dates = dates[-100:]  # 최근 100건의 날짜 데이터
    # real = plt.plot(recent_dates, unscaled_y_test[-100:], label='real')
    # pred = plt.plot(recent_dates, y_test_predicted[-100:], label='predicted')
    #
    # # 최근 100건의 날짜 데이터 생성
    # recent_dates = dates[-100:]
    #
    # plt.legend(['Real', 'Predicted'])
    # plt.title('SK Hynix Using LSTM by TGG')
    # plt.xticks(rotation=45)
    # plt.show()

#오늘 날자 받는 함수
def get_today():
    # 오늘 날짜 정보
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    return current_date

#today_Te 폴더가 있으면 이름을 바꿔주고, 없으면 폴더를 만들어 준다.
def check_file_name():
    current_date = get_today()

    original_path = './models/today'
    new_path = f'./models/{current_date}'

    if os.path.exists(original_path):
        try:
            os.rename(original_path, new_path)
            print(f"폴더 이름 변경 완료: 'today_Te' -> '{current_date}'")
        except Exception as e:
            print(f"폴더 이름 변경 실패: {e}")
    else:
        try:
            os.makedirs(original_path)
            print(f"폴더 생성 완료: '{original_path}'")
        except Exception as e:
            print(f"폴더 생성 실패: {e}")

def main():
    #오늘 날자 받기
    current_date = get_today()

    # CSV 파일을 DataFrame으로 읽어옴
    data = pd.read_csv(stock_list_csv_name)

    # 105560 코드에 해당하는 행을 필터링
    filtered_row = data[data['Code'] == '005930']

    # Name과 ListingDate 값을 가져옴
    name = filtered_row['Name'].values[0]
    listing_date = filtered_row['ListingDate'].values[0]

    print("Name:", name)
    print("ListingDate:", listing_date)

    # 폴더 새로만들기
    check_file_name()

    ####################################################################################################################
    #######     이 부분은 반복문을 사용하여 144개의 모델을 생성해야 한다
    ####################################################################################################################

    # CSV 파일 읽기
    with open(stock_list_csv_name, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            start_date = '2010-01-01'
            end_date = current_date
            date = row['ListingDate']

            date1 = parser.parse(start_date)
            date2 = parser.parse(date)

            if date1 < date2:
                print("2010-01-01 이후부터 데이터가 있어서 데이터가 있는 날부터 훈련 데이터로 지정")
                start_date = date
            elif date1 > date2:
                print("2010-01-01 이전 데이터부터 있어서 2010-01-01부터 훈련 데이터로 지정")

            # 모델 생성하기
            make_model(row['Code'], start_date, end_date)



            # if row['Code'] == target_code:
            #     target_name = row['Name']
            #     target_listing_date = row['ListingDate']
            #     break  # 종목을 찾았으면 루프 종료
    ####################################################################################################################

    ####################################################################################################################
    #######     이 부분은 반복문을 사용하여 144개의 모델을 사용해서 오늘값 + 내일값 + 상승,하략여부 + 50일치 데이터를 저장하는 부분이다.
    ####################################################################################################################
    # CSV 파일 읽기
    with open(stock_list_csv_name, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        for row in csv_reader:
            last_date = datetime.strptime(current_date, '%Y-%m-%d')
            start_date = (last_date + timedelta(days=-100))
            end_date = current_date
            ticker = row['Code']
            history_points = 50

            model_path = f"./models/today/{ticker}_{current_date}.h5"

            # 훈련된 모델 불러오기
            model = load_model(model_path)

            # 오늘까지의 데이터 가져오기
            today_data = fdr.DataReader(ticker, start_date, end_date)
            today_data = today_data.iloc[:, 0:-1]

            # 데이터프레임의 인덱스 값을 받기
            index_dates = today_data.index.values
            today_data = today_data.values

            # 데이터 정규화
            data_normalizer = preprocessing.MinMaxScaler()
            today_data_normalized = data_normalizer.fit_transform(today_data)

            # 예측 데이터 준비
            prediction_data = today_data_normalized[-history_points:].copy()
            predicted_normalized = model.predict(np.expand_dims(prediction_data, axis=0))

            # 역정규화하여 예측 가격 얻기
            y_normalizer = preprocessing.MinMaxScaler()
            y_normalizer.fit(today_data[:, 0].reshape(-1, 1))
            predicted_price = y_normalizer.inverse_transform(predicted_normalized)
            print(predicted_price)

            # 예측 날짜 계산
            last_date = datetime.strptime(end_date, '%Y-%m-%d')
            predicted_date = np.datetime64((last_date + timedelta(days=1)).strftime('%Y-%m-%d'))

            # 전체 날짜와 가격 데이터 생성
            all_dates = np.append(index_dates, [predicted_date])
            all_prices = np.append(today_data[:, 3], [predicted_price[0, 0]])

            # 데이터프레임 생성
            df = pd.DataFrame({
                'Date': all_dates,
                'Price': all_prices
            })

            # CSV 파일로 저장
            df.to_csv(f"./models/today_Te/{ticker}_{current_date}.csv", index=False)



    ####################################################################################################################





main()
