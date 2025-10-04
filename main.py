from data_processing import *
from catboost_main_model import MultiStockTimeSeriesCatBoost

def main():
    data = load_data()
    data = mark_data(data)

    data_train = data[data['begin'] < '2025-08-20']
    data_test = data[data['begin'] >= '2025-08-20']

    stock_predictor = MultiStockTimeSeriesCatBoost(
            target_col='target_return_i',
            time_col='begin',
            stock_col='ticker',
            max_lags=20,
            rolling_windows=[5, 10, 20],
            seasonal_periods=[5, 10, 20],
            pct_change_periods=[2, 3, 5, 10, 15, 20],
            ewm_spans = [2, 3, 5, 10, 15, 20],
            ewm_alphas = [0.1, 0.2, 0.3, 0.5, 0.8],
            cross_stock_features=False,
            catboost_params={
                'iterations': 100,
                'learning_rate':0.03,
                'depth': 3,
                'verbose': 50
            }
        )
    stock_predictor.fit(data_train, validation_split=0.1)

    pred = stock_predictor.predict(data_test, data_train)

    create_sub(pred, 'submission.csv')

if __main__():
    