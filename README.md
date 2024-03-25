# 주식 데이터 수집 AI 예측 RestApi 서버  
![img8](https://github.com/zigoom/PortfolioPage/assets/24885296/5ae8340e-0578-4ba9-bc9c-57adef5fda92)  
<br/>  
### 1. 개발 목표  
&nbsp;&nbsp;&nbsp;   웹사이트에서 사용할 데이터를 가공하여 제공하며, 일부 AI 모델에 의해서 데이터를 만들어낸다.       
<br/>
### 2. 개발환경 및 도구  
  - **소스 관리 -**  Github  
  - **Python -** 3.8 Ver 
  - **IDE -** PyCham (ver. 2023.1.3)  
  - **서버 환경 -** Flask 1.1.0+Swagger  
  - **파이썬 라이브러리 -** keras(2.13.1v), numpy(1.23.5v), pandas(2.0.3v), FinanceDataReader(0.9.50v)   
<br/>

![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/a077d018-09a9-4d82-96c8-345f0c11259b)  
<br/>  
### 3. 제공하는 데이터 및 사용방법
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/6f346429-5253-4675-95f8-09bf2d56b71d)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/f7d716e9-b187-46f7-b76e-1558e8ef3342)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/c1d29c16-d971-4dc2-93f8-c3a6b13f2c10)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/0fc4522a-8b59-42b5-89fa-d71cdf3d1d33)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/bb05691a-964d-4a6e-b565-e595f55a002c)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/5404b434-fb61-45b2-99c8-49c7200d7258)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/2dd30fd2-64f3-4588-beda-78223e2c1355)  
![image](https://github.com/zigoom/BlueOceanPython/assets/24885296/124fe42a-6834-4a1e-bbe6-befed57564a2)  
<br/>  

### 4. LSTM 모델의 아키텍처를 정의 및 훈련 과정

1. 모델 아키텍처 정의:
    
    ```python
    pythonCopy code
    lstm_input = Input(shape=(50, 5), name='lstm_input')  # LSTM 입력 데이터의 크기 (50 타임스텝, 5 특성)
    x = LSTM(50, name='lstm_0')(lstm_input)  # LSTM 레이어를 정의하고 입력 데이터를 전달
    x = Dropout(0.2, name='lstm_dropout_0')(x)  # 드롭아웃 레이어를 추가하여 과적합을 줄임
    x = Dense(64, name='dense_0')(x)  # 64개의 유닛을 가지는 fully connected (dense) 레이어
    x = Activation('sigmoid', name='sigmoid_0')(x)  # 시그모이드 활성화 함수를 적용
    x = Dense(1, name='dense_1')(x)  # 1개의 유닛을 가지는 fully connected 레이어
    output = Activation('linear', name='linear_output')(x)  # 선형 활성화 함수를 적용하여 최종 출력 생성
    
    ```
    
2. 모델 컴파일:
    
    ```python
    pythonCopy code
    model = Model(inputs=lstm_input, outputs=output)  # 입력과 출력을 지정하여 모델 생성
    adam = optimizers.Adam(lr=0.0005)  # Adam 옵티마이저를 생성하고 학습률을 설정
    model.compile(optimizer=adam, loss='mse')  # 모델을 컴파일하고 손실 함수로 평균 제곱 오차(Mean Squared Error)를 사용
    
    ```
    
3. 모델 훈련:
    
    ```python
    pythonCopy code
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    
    ```
    
    - **`x=ohlcv_train, y=y_train`**: 훈련 데이터로 모델을 훈련시킴. **`ohlcv_train`**은 입력 데이터, **`y_train`**은 실제 타겟 값(출력)을 나타냄.
    - **`batch_size=32`**: 배치 크기를 32로 설정하여 한 번의 반복에서 처리되는 데이터의 수를 지정.
    - **`epochs=50`**: 전체 훈련 데이터셋을 50번 반복하여 모델을 훈련시킴.
    - **`shuffle=True`**: 매 에포크마다 데이터를 섞어서 훈련을 더 효과적으로 진행.
    - **`validation_split=0.1`**: 훈련 데이터 중 10%를 검증 데이터로 사용하여 모델의 성능을 평가.  
<br/>  

### 5. 작업시에 고려사항
  - 장고/플라스크 선택 이유 :  
    장고는 수업과정에서 배울예정이라 배워보지 못한 플라스크를 접해보는 기회라고 생각이 되었고,
    단순히 요청과 결과 정보를 전달해주는 역활을 할 예정이라서 프로젝트가 작은 플라스크를 선택  
  
  - 플라스크에 Swagger를 사용한 이유 :  
    REST API를 구성함으로서 웹에서 해당 요청값을 쉽게 반환하고 확인 할 수 이게 만들어서 팀원들이 편하게 사용하였으면 하는 생각에서 선택
  
  - 실시간 주식 데이터 수집 라이브러리 선택 과정 :  
    대신증권 API : 회원 계정이 필요하며, 초당 호출에 대한 제한사항이 있으며, Python3.7(32bit)에서 사용해야 한다.
    키움증권 API : 회원 계정이 필요하며, 초당 호출에 대한 제한사항이 있다
    yfinance (야후) : 국내주식에 대한 내용이 그렇게 상세하지 않아서 사용하기에 불현하다.
    위의 결과로 FinanceDataReader 를 사용해서 데이터를 수집하고 가공해서 적용하게 되었다.
    
