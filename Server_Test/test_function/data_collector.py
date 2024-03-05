import numpy as np
from sklearn import preprocessing
from keras.models import load_model
import FinanceDataReader as fdr
from datetime import datetime, timedelta


def call_dataset(ticker = '005930', stt = '2015-01-01', end = '2023-08-18', history_points = 50):
    data = fdr.DataReader(ticker, stt, end)
    dates = data.index
    data = data.iloc[:, 0:-1]
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
        for i in range(len(values) - time_period, len(values)):
            close = values[i][3]
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

    assert ohlcv_histories_normalized.shape[0] == next_day_open_values_normalized.shape[0] == technical_indicators_normalized.shape[0]

    return ohlcv_histories_normalized, technical_indicators_normalized, next_day_open_values_normalized, next_day_open_values, y_normalizer, dates

def get_stock_listing_date(ticker):
    try:
        df = fdr.StockListing('KRX')
        listing_date = df[df['Symbol'] == ticker]['ListingDate'].values[0]
        return listing_date
    except:
        return None

def main():
    ticker = '005930'  # 삼성
    today = datetime.today()
    listing_date = get_stock_listing_date(ticker)

    if listing_date is None:
        print(f"Could not retrieve listing date for {ticker}")
        return

    if today - listing_date < timedelta(days=365 * 8):
        start_date = listing_date.strftime('%Y-%m-%d')
    else:
        start_date = (today - timedelta(days=365 * 8)).strftime('%Y-%m-%d')

    end_date = today.strftime('%Y-%m-%d')

    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser, dates = call_dataset(ticker=ticker, stt=start_date, end=end_date, history_points=50)

    if len(ohlcv_histories) < 1:
        print("Not enough data for prediction")
    else:
        model = load_model('basic_model.h5')
        # 예측 수행
        predicted_normalized = model.predict(np.expand_dims(ohlcv_histories[-1], axis=0))  # 가장 최근의 데이터로 예측
        predicted_price = y_normaliser.inverse_transform(predicted_normalized)  # 역정규화

        print("Predicted Next Day Opening Price:", predicted_price)


main()