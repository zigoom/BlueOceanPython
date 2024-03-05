# pip install -U finance-datareader

import numpy as np
import pandas as pd
import FinanceDataReader as fdr     # 버전 문제로 오류가 날수 있다 아래 내용을 터미널에서 처리해 준다.
# pip install -U finance-datareader : 해당 라이브러리 설치 해 주고
# pip install --uphrade pandas      : 오류가 나면 판다스 버전을 올려주고
# pip install bs4                   : bs4 오류가 나면 설치해 준다

from sklearn import preprocessing

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

import tensorflow

# FinanceDataReader.DataReader : 주가 데이터, 거래량, 시가, 고가, 저가, 수정주가, 주식 분할, 배당금 등을 제공.
# 또한 지수 데이터, 환율 데이터, 암호화폐 데이터 등 다양한 종류의 금융 데이터도 가져올 수 있으며, DataFrame 형테임

def call_dataset(ticker = '005930', stt = '2022-01-01', end = '2023-08-16', history_points = 50):
    data = fdr.DataReader(ticker, stt, end)
    print('1. data: ', data)
    data = data.iloc[:,0:-1]    # 행 번호 기준으로 값 읽기(모든 행에 대해서, 처음부터 ~ 마지막 열 전까지 가져오기)
                                # 열을 한줄 제외하고 값을 가져오는것 (주식의 급락율 제외함)
    print('data: ', data.shape) # DataFrame의 행의개수, 열의 개수로 반환한다.
    data = data.values          # 필드와 인덱스를 제외하고 값만 갖는다
    print('2. data: ', data) # DataFrame의 행의개수, 열의 개수로 반환한다.

    # sklearn.Preprocessing.MinMaxScaler() : sklearn에서 지원하는 전처리기 데이터를 0과 1 사이의 범위로 변환 (최소-최대 스케일링)

    data_normalizer = preprocessing.MinMaxScaler()          # 데이터를 0~1 범위로 점철되게 하는 함수 call
    data_normalized = data_normalizer.fit_transform(data)   # 데이터를 0~1 범위로 점철되게 함수 수행(데이터를 이용하여 훈련)
    print('data_normalized: ', data_normalized.shape)
    print('3. data_normalized: ', data_normalized)

    # using the last {history_points} open close high low volume data points, predict the next open value
    # (번역) 마지막 {history_points} 오픈 클로즈 하이 로우 볼륨 데이터 포인트를 사용하여 다음 시가를 예측합니다.
    # ohlcv를 가지고 오되, 관찰 일수 만큼 누적해서 쌓는다. (열방향으로)
    # 380 라인에서 -50을 한 후에 330 를 50개를 복사하는데 이때 값을 0부터 1씩 올라가게 한다. (0 ~ 330, 1~331, 2~332 .. 이렇게 들어간다)
    ohlcv_histories_normalized = np.array([data_normalized[i:i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    print('7. ohlcv_histories_normalized : ',ohlcv_histories_normalized)

    # print('2. ohlcv_histories_normalized: ', ohlcv_histories_normalized)
    # print('ohlcv_histories_normalized: ', ohlcv_histories_normalized.shape)

    # 1차원 배열로 전달한다. np.arraty([data_normalized[][]])
    # data_normalized[:, 0] : 해당 2차원 배열에서 0번째 열에 있는 데이터를 반환한다(즉 2차원 배열에서 0번쨰 값만 남긴상태).
    # data_normalized[:, 0][i + history_points] : i는 0~(380-50)까지 for 문으로 반환하여 i+history_points 는 50~380 값을 반환한다.
    #                                             그러면 2차원 배열에서 0번째 열의 데이터만 가진 2차원 배열에서 50~380 배열 값을 가지고 np,array로 변환하여 1차원 배열로 반환한다.
    next_day_open_values_normalized = np.array([data_normalized[:, 0][i + history_points].copy() for i in range(len(data_normalized) - history_points)])
    # 1XN 벡터는 2차원 배열로 표현된 행 벡터이며, NX1 벡터는 2차원 배열로 표현된 열 벡터입니다.
    # 즉, 1XN 벡터를 NX1 벡터로 변환하는 작업은 해당 벡터를 전치(Transpose)하는 것과 동일합니다.
    print('8. next_day_open_values_normalized : ',next_day_open_values_normalized)
    next_day_open_values_normalized = np.expand_dims(next_day_open_values_normalized, -1) # 1XN 벡터 -> NX1 벡터로
    print('9. next_day_open_values_normalized : ',next_day_open_values_normalized)
    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1) # 1XN 벡터 -> NX1 벡터로
    print('10. next_day_open_values : ',next_day_open_values)
    y_normalizer = preprocessing.MinMaxScaler()
    y_normalizer.fit(next_day_open_values)
    print('11. y_normalizer : ',y_normalizer)


    # 인풋 X : 그 이전의 OHLCV (from T = -50 to T = -1)
    # 아웃풋 y : 예측하고자 하는 주가 T = 0

    def calc_ema(values, time_period): #정규화된
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        # 지수이동평균법(EMA) 사용
        # 공식 : EMA =(Price(t)*k)+(EMA(t-1)*(1-k))
        # Price(t) : 현재 가격
        # EMA(t-1) : 바로 전 기간의 EMA
        # k = 2/(n+1) : n은 MA의 숫자 (예: MA-5,MA-7,MA-20,MA-100 등)
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalized:
        # note since we are using his[3] we are taking the SMA of the closing price
        # print('his: ', his)
        # print('his[:, 3]: ', his[:, 3])
        sma = np.mean(his[:, 3]) # 각 데이터포인트별 Close Price 평균 (4번쨰 열에 대한 평균을 구한다.)
        print('3. sma: ', sma)
        '''# 위의 예시의 의미 
        주어진 his는 여전히 리스트의 리스트 구조로 이루어진 3D 리스트입니다. 각 원소가 2D 리스트인 세 개의 서브 리스트를 포함하고 있습니다.
        his = [
            [[1, 2, 3, 1, 5], [1, 2, 3, 1, 5], [1, 2, 3, 1, 5]],
            [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]],
            [[3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [3, 4, 5, 6, 7]]
        ]
        이제 np.mean(his[:, 3])을 계산하겠습니다. his[:, 3]는 3D 리스트 his의 첫 번째 차원에 해당하는 모든 서브 리스트들이 선택되며,
        각 서브 리스트에서 3번째 열(인덱스 3에 해당하는 열)에 대한 값을 반환하게 됩니다.        
        먼저, 주어진 his에서 각 서브 리스트들의 3번째 열을 추출하겠습니다:        
        [1, 1, 1]
        [5, 5, 5]
        [6, 6, 6]
        이제 np.mean(his[:, 3])은 위에서 추출한 열에 대한 평균을 계산합니다:        
        np.mean([1, 1, 1, 5, 5, 5, 6, 6, 6])
        따라서 결과는 (1 + 1 + 1 + 5 + 5 + 5 + 6 + 6 + 6) / 9 ≈ 3.2222가 됩니다                
        '''
        # his 는 380에서 앞에 50개를 한개씩 더한 330, 331, 332 ...
        print('4. EMA-12 : ',calc_ema(his, 12))
        print('4. EMA-26 : ',calc_ema(his, 26))
        macd = calc_ema(his, 12) - calc_ema(his, 26) # 12일 EMA - 26일 EMA (지수이동평균 12일 기준 - 지수이동평균 26일 로 처리함)
        print('5. sma : ',sma)
        print('5. macd : ',macd)
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)   # for 문으로 돌린 종가의 이동평균값 배열을 넘파이로 변환한다.

    tech_ind_scaler = preprocessing.MinMaxScaler()          # 데이터를 0~1사이의 값으로 바꾼다.
    # fit_transform() 메서드는 두 단계를 합친 것으로,
    # 우선 데이터를 스케일링하기 위한 변환(Scaling)에 필요한 평균과 분산을 계산하고,
    # 그 후에 실제로 데이터를 스케일링하여 변환된 값을 반환합니다.
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    # ??위에서 같은 작업을 하는데 중간에 값이 수정되는지 찍어봐야 할것 같음
    technical_indicators = np.array(technical_indicators)
    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalized = tech_ind_scaler.fit_transform(technical_indicators)

    # assert는 뒤에 나오는 조건문의 ture값을 보장하는 것으로 여기에는 안나왔지만 false가 될때는 따로 오류메시지를 줄 수 있다.
    # 참조 : https://blockdmask.tistory.com/553
    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalized.shape[0]

    print('ohlcv_histories_normalized.shape[0]: ', ohlcv_histories_normalized.shape[0])

    return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer




np.random.seed(4) #고정된 난수 발생을 위해 설정

history_points = 50
ticker = '000660' # sk hynix

def main():
    print("tensorflow ver: "+tensorflow.__version__)   # 텐서플로 버전 확인
    tensorflow.random.set_seed(44)  # 텐서플로에서 시드 44로 렌덤생성

    # dataset 생성
    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = call_dataset(ticker=ticker)
    print('6. ohlcv_histories : ',ohlcv_histories)          # 시그모이드 함수로 값들을 바꾼 후에 330건씩 50일로 나눠서 3차원 배열로 만든값
    print('6. _ : ',_)                                      # 330개를 가지고 있는 50묶음의 배열에 대한 종가의 평균 배열
    print('6. next_day_open_values : ',next_day_open_values)# 시그모이드함수로 정규화된 주식 시작가 2차원 배열 (대신에 1차원에 값이 하나씩만 들어있다)
    print('6. unscaled_y : ',unscaled_y)                    # 시작가만 있는 (정규화 X) 2차원 배열 (기존에 받아온 데이터에서 시작가격만 남기고 삭제)
    print('6. y_normaliser : ',y_normaliser)                # MinMaxScaler()


    train_ratio = 0.7
    n = int(ohlcv_histories.shape[0] * train_ratio) # 330 * 0.7의 값을 가진다. = 230
    print('13. n : ',ohlcv_histories.shape[0])
    print('14. n : ',n)
    # print('13. ohlcv_histories.shape : ',ohlcv_histories.shape)

    ohlcv_train = ohlcv_histories[-n:-1] # 330 중에서 100 ~ 330이전까지 229개
    y_train = next_day_open_values[-n:-1]

    ohlcv_test = ohlcv_histories[:ohlcv_histories.shape[0] - n]
    y_test = next_day_open_values[:ohlcv_histories.shape[0] - n]

    unscaled_y_test = unscaled_y[:ohlcv_histories.shape[0] - n]

    print('ohlcv_train.shape: ', ohlcv_train.shape)
    print('ohlcv_test.shape: ', ohlcv_test.shape)

    ohlcv_train

    # model architecture
    # Keras의 함수형 API를 사용하여 LSTM 네트워크에 입력을 정의하는 부분입니다.
    # Input(shape=(50, 5), name='lstm_input')는 LSTM 네트워크에 들어갈 입력 데이터의 형태를 정의하는 부분입니다.
    # Input: Keras 함수형 API에서 모델의 입력을 정의하는 클래스입니다.
    # shape=(50, 5): 입력 데이터의 형태를 나타내며, (시퀀스 길이, 특성 개수)로 구성됩니다. 이 경우, 시퀀스 길이는 50이며, 각 시퀀스는 5개의 특성으로 구성되어 있음을 의미합니다.
    # name='lstm_input': 입력 레이어의 이름을 지정합니다. 이를 통해 모델을 시각화하거나 모델의 구조를 살펴볼 때 레이어를 식별할 수 있습니다.
    # 이를 통해 입력 데이터의 형태가 (None, 50, 5)인 LSTM 모델을 구성할 수 있습니다. 여기서 None은 입력 데이터의 배치 크기를 의미하며, 실제 모델을 컴파일할 때 배치 크기가 지정됩니다.
    # 입력 데이터의 형태는 (시퀀스 길이, 특성 개수)로 50개의 시퀀스 길이와 각 시퀀스에 5개의 특성으로 구성된 3D 텐서입니다.
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)         # 50개의 유닛(뉴런)을 가진 LSTM 샐을 생성, 이름은 'lstm_0'으로하고, lstm_input로 출력을 생성
    x = Dropout(0.2, name='lstm_dropout_0')(x)      # 과적합을 방지하기 위한 드롭아웃 레이어 랜덤하게 일부 뉴런 비활성화 (0.2)는 20%를 뜻한다, lstm_dropout_0는 레이어 이름
    x = Dense(64, name='dense_0')(x)                # 64개의 유닛을 가진 완전 연결 레이어를 생성합니다, dense_0는 레이어 이름
    x = Activation('sigmoid', name='sigmoid_0')(x)  # 활성화 함수를 정의합니다. sigmoid 활성화 함수를 적용하여 출력을 생성,
    x = Dense(1, name='dense_1')(x)                 # 마지막 완전 연결(Dense) 레이어를 정의합니다. 이 레이어는 1개의 유닛을 가지며, 최종적인 출력을 생성한다, 입력 데이터 x에 마지막 완전 연결 레이어를 적용하여 출력을 생성한다
    # 최종 출력 레이어를 정의합니다. 활성화 함수로 linear를 사용하여 선형 함수를 적용합니다. 선형 함수는 입력과 동일한 값을 출력하는 함수이며, 회귀 문제에서 주로 사용한다. 입력 데이터 x에 최종 출력 레이어를 적용하여 최종 출력을 생성합니다.
    output = Activation('linear', name='linear_output')(x)
    # 위와 같이 정의된 모델은 LSTM 기반의 시계열 예측 모델이며, 입력으로 50개의 시퀀스 길이와 5개의 특성을 가진 데이터를 받고, 출력으로 단일 값을 예측합니다.



    model = Model(inputs=lstm_input, outputs=output)    # 입력과 출력을 지정하여 최종 모델을 생성합니다.
    adam = optimizers.Adam(lr=0.0005)                   # Adam 옵티마이저를 생성합니다. 학습률(learning rate)은 0.0005로 설정되어 있습니다.
    model.compile(optimizer=adam, loss='mse')           # 모델을 컴파일합니다. 손실 함수로는 평균 제곱 오차(Mean Squared Error, MSE)를 사용하고, Adam 옵티마이저를 지정합니다
    # 모델을 학습합니다. 입력 데이터로 ohlcv_train, 출력 데이터로 y_train을 사용하며, 한 번의 배치 크기는 32입니다.
    # 총 50 에포크(epoch) 동안 학습하며, 데이터를 무작위로 섞어가며 학습합니다.
    # 학습 데이터의 10%를 검증 데이터로 사용하도록 지정되어 있습니다.
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    # evaluation

    y_test_predicted = model.predict(ohlcv_test)                        # 학습된 모델을 사용하여 ohlcv_test 데이터를 예측합니다.
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted) # 정규화된 예측값을 다시 역정규화하여 원래의 주가 단위로 변환합니다.
    y_predicted = model.predict(ohlcv_histories)                        # 학습된 모델을 사용하여 ohlcv_histories 데이터를 예측합니다.
    y_predicted = y_normaliser.inverse_transform(y_predicted)           # 정규화된 예측값을 다시 역정규화하여 원래의 주가 단위로 변환합니다.

    assert unscaled_y_test.shape == y_test_predicted.shape              # assert는 해당값의 true라는 것을 보장
    print('15. y_test_predicted : ',y_test_predicted)
    print('16. unscaled_y_test : ',unscaled_y_test)

    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))   #공통된 값을 빼고, 값 값을 제곱한 다음에 배열의 평균을 구한다.
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    from datetime import datetime
    model.save(f'basic_model.h5')

    import matplotlib.pyplot as plt

    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    # real = plt.plot(unscaled_y_test[start:end], label='real')
    # pred = plt.plot(y_test_predicted[start:end], label='predicted')

    real = plt.plot(unscaled_y[start:end], label='real')
    pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])
    plt.title('SK Hynix Using LSTM by TGG')
    plt.show()

    col_name = ['real', 'pred']
    real, pred = pd.DataFrame(unscaled_y[start:end]), pd.DataFrame(y_predicted[start:end])
    foo = pd.concat([real, pred], axis=1)
    foo.columns = col_name

    foo

    foo.corr()

    foo['real+1'] = foo['real'].shift(periods=1)
    foo[['real+1', 'pred']].corr()


main()




