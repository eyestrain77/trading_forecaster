# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import math
import numpy as np
import pandas as pd

from datetime import date, datetime, timedelta

from tqdm import tqdm

import catboost as cb

from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error, mean_squared_error

# %%
import os
# import torch
import random
import numpy as np

def seed_everything(seed):
    global SEED
    SEED = seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

seed_everything(42)

# dtype = torch.bfloat16
# torch.set_default_dtype(dtype)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
candles = pd.read_parquet('candles_with_embs.parquet')
candles['begin'] = pd.to_datetime(candles['begin'])
candles = candles.drop_duplicates().sort_values(by=['ticker', 'begin']).reset_index(drop=True)

# %%
candles

# %%
def preprocess(df):
    df = df.copy()
    df = df.sort_values(by='begin').reset_index(drop=True)

    use_features = ['open', 'close', 'high', 'low', 'volume'] + [f'emb_{i}' for i in range(32)]

    df['volume'] = np.log(df['volume'])

    for i in range(1, 32 + 1):
        df[f'close_shift_{i}'] = df['close'].shift(i)
        use_features.append(f'close_shift_{i}')

    for i in range(4, 32 + 1, 4):
        for j in range(32 + 1):
            df[f'close_diff_{i}_shift_{j}'] = df['close'].diff(i).shift(j)
            use_features.append(f'close_diff_{i}_shift_{j}')

    for i in range(4, 32 + 1, 4):
        for j in range(32 + 1):
            df[f'close_rolling_{i}_mean_shift_{j}'] = df['close'].rolling(i).mean().shift(j)
            use_features.append(f'close_rolling_{i}_mean_shift_{j}')
            df[f'close_rolling_{i}_std_shift_{j}'] = df['close'].rolling(i).std().shift(j)
            use_features.append(f'close_rolling_{i}_std_shift_{j}')

    for i in range(32 + 1):
        df[f'volatility_shift_{i}'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        use_features.append(f'volatility_shift_{i}')
        df[f'abs_volatility_shift_{i}'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
        use_features.append(f'abs_volatility_shift_{i}')

    dfs = []
    for i in range(1, 20 + 1):
        df1 = df.copy()
        df1['i'] = i
        df1['target'] = candles['close'].transform(lambda x: x.shift(-i) / x - 1)
        dfs.append(df1)
    use_features.append('i')
    dfs = pd.concat(dfs).reset_index(drop=True)
    return dfs, use_features

# %%
def make_model(subset, use_features):
    params = {
        'iterations': 5000,
        'learning_rate': 0.005,
        'depth': 2,
        'subsample': 0.8,
        'colsample_bylevel': 0.5,
        'l2_leaf_reg': 1.0,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'early_stopping_rounds': 1000,
        'verbose': 500,
        'random_seed': SEED,
        # 'boosting_type': 'Ordered',
        # 'bootstrap_type': 'Bayesian',
        # 'bagging_temperature': 1,
    }

    model = cb.CatBoostRegressor(**params)

    subset = subset[subset['i'] <= 14]

    train_subset = subset.dropna()

    val_size = 16
    train, val, test = [], [], []
    for i in subset['i'].unique():
        train.append(train_subset[train_subset['i'] == i].iloc[:-val_size - 1])
        val.append(train_subset[train_subset['i'] == i].iloc[-val_size - 1:-1])
        test.append(train_subset[train_subset['i'] == i].iloc[-1:])
    train = pd.concat(train).reset_index(drop=True)
    val = pd.concat(val).reset_index(drop=True)
    test = pd.concat(test).reset_index(drop=True)

    X_train = train[use_features]
    y_train = train['target']

    X_val = val[use_features]
    y_val = val['target']

    X_test = test[use_features]

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )

    y_test_pred = model.predict(X_test)

    return y_test_pred

# %%
res = []
for ticker in candles['ticker'].unique():
    print('-' * 20, ticker, '-' * 20)
    subset = candles[candles['ticker'] == ticker]
    subset, use_features = preprocess(subset)
    preds = make_model(subset, use_features)
    for p in preds:
        res.append(pd.DataFrame({'i': list(range(14)), 'ticker': [ticker for i in range(14)], 'predicted_value': p}))
    print(preds.shape)
    print()
res = pd.concat(res).reset_index(drop=True)

# %%
res

# %%



