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

def call_dataset(ticker = '000660', stt='2021-08-07', end='2023-08-07', history_points = 50):
    data = fdr.DataReader(ticker, stt, end)
    dates = data.index  # 날짜 데이터 가져오기
    data = data.iloc[:,0:-1]
    data = data.values

    data_normalizer = preprocessing.MinMaxScaler()
    data_normalized = data_normalizer.fit_transform(data)

    ohlcv_histories_normalized = np.array([data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])

    next_day_open_values_normalized = np.array([data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1)
    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)

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
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    technical_indicators = np.array(technical_indicators)
    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalized.shape[0]

    return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, dates


np.random.seed(4)
history_points = 50
ticker = '000660' #삼성전자

def main():
    # 데이터셋 호출 및 전처리 등의 코드
    start_date = '2021-08-07'
    end_date = '2023-08-07'
    ticker = '000660'

    print("tensorflow ver: "+tensorflow.__version__)
    tensorflow.random.set_seed(44)

    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser, dates = call_dataset(ticker=ticker, stt=start_date, end=end_date, history_points=history_points)

    train_ratio = 0.7
    n = int(ohlcv_histories.shape[0] * train_ratio)

    ohlcv_train = ohlcv_histories[-n:-1]
    y_train = next_day_open_values[-n:-1]

    ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0] - n]
    y_test = next_day_open_values[:ohlcv_histories.shape[0] - n]

    unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0] - n]

    # LSTM 모델 정의 및 컴파일
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)

    # 모델 훈련
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)











########################################################################################################################
    # 예측 결과 역정규화
    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape

    # 예측 결과 평가 및 저장
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100

    print(f"Mean Squared Error (MSE): {real_mse}")
    print(f"Scaled MSE: {scaled_mse}")
    model.save(f'basic_model.h5')
    # 예측 결과 시각화
    plt.gcf().set_size_inches(22, 15, forward=True)

    real = plt.plot(dates[:ohlcv_histories.shape[0] - n], unscaled_y_test, label='real')  # 수정된 부분
    pred = plt.plot(dates[:ohlcv_histories.shape[0] - n], y_test_predicted, label='predicted')  # 수정된 부분

    plt.legend(['Real', 'Predicted'])
    plt.title('SK Hynix Using LSTM by TGG')
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

    # 데이터프레임 출력 설정 및 출력
    pd.set_option('display.max_rows', None)  # 모든 행 출력
    pd.set_option('display.max_columns', None)  # 모든 열 출력
    pd.set_option('display.width', None)  # 너비 설정 해제
    pd.set_option('display.expand_frame_repr', False)  # 줄 바꿈 비활성화
    # Print dataframes
    print("Real Values:")
    print(real_df)
    print("\nPredicted Values:")
    print(pred_df)


main()




