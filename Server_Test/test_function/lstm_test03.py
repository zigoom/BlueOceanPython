# pip install -U finance-datareader

import numpy as np
import pandas as pd
import FinanceDataReader as fdr     # 버전 문제로 오류가 날수 있다 아래 내용을 터미널에서 처리해 준다.
from sklearn import preprocessing

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

import tensorflow
import matplotlib.pyplot as plt

def call_dataset(ticker='005930', stt='2010-01-01', end='2023-08-25', history_points = 50):
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


history_points = 50
ticker = '005930' # 삼성
def main():
    np.random.seed(4)
    print(tensorflow.__version__)
    tensorflow.random.set_seed(44)
    # from util import csv_to_dataset, history_points

    # dataset
    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser, dates = call_dataset(ticker=ticker)
    train_ratio = 0.7
    n = int(ohlcv_histories.shape[0] * train_ratio)

    ohlcv_train = ohlcv_histories[-n:-1]
    y_train = next_day_open_values[-n:-1]

    ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0]-n]
    y_test = next_day_open_values[:ohlcv_histories.shape[0]-n]

    unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0]-n]
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
    print('y_test_predicted : ',y_test_predicted)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    print('y_test_predicted : ',y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    print('y_predicted : ',y_predicted)
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    print('y_predicted : ',y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape

    model.save(f'basic_model.h5')
    # 예측 결과 시각화
    plt.gcf().set_size_inches(22, 15, forward=True)
    recent_dates = dates[-100:]  # 최근 100건의 날짜 데이터
    real = plt.plot(recent_dates, unscaled_y_test[-100:], label='real')
    pred = plt.plot(recent_dates, y_test_predicted[-100:], label='predicted')

    # 최근 100건의 날짜 데이터 생성
    recent_dates = dates[-100:]

    plt.legend(['Real', 'Predicted'])
    plt.title('Samsung Using LSTM by TGG')
    plt.xticks(rotation=45)
    plt.show()

    col_name = ['real', 'pred']
    real, pred = pd.DataFrame(unscaled_y_test), pd.DataFrame(y_test_predicted)
    foo = pd.concat([real, pred], axis=1)
    foo.columns = col_name

    foo.corr()
    foo['real+1'] = foo['real'].shift(periods=1)
    foo[['real+1', 'pred']].corr()

    # 예측 결과 시각화
    # Create dataframes for real and predicted values
    real_df = pd.DataFrame({'Date': dates[:ohlcv_histories.shape[0] - n], 'Real': unscaled_y_test.flatten()})
    pred_df = pd.DataFrame({'Date': dates[:ohlcv_histories.shape[0] - n], 'Predicted': y_test_predicted.flatten()})

    # 최근 100건의 실제 값과 예측 값 생성
    recent_real_values = unscaled_y_test[-100:].flatten()  # 1차원으로 변경
    recent_predicted_values = y_test_predicted[-100:].flatten()  # 1차원으로 변경

    # 데이터프레임 생성
    recent_data_df = pd.DataFrame({
        'Date': recent_dates,
        'Real': pd.Series(recent_real_values),
        'Predicted': pd.Series(recent_predicted_values)
    })

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화

    # 데이터프레임 출력
    print("Recent Real and Predicted Values:")
    print(recent_data_df)

    # 예측 결과 역정규화
    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    # 실제 값과 예측 값 비교를 위한 역정규화
    unscaled_y_test = y_normaliser.inverse_transform(unscaled_y_test)

    # 예측 결과 평가 및 저장
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100

    print(f"Mean Squared Error (MSE): {real_mse}")
    print(f"Scaled MSE: {scaled_mse}")

main()




