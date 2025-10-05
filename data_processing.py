import pandas as pd
from tqdm import tqdm
import numpy as np

def load_data():
    candles = pd.read_csv('data/candles.csv')
    candles2 = pd.read_csv('data/candles_2.csv')

    merged_candles = pd.concat([candles, candles2], ignore_index=True)
    merged_candles = merged_candles.sort_values(['ticker', 'begin'])

    return merged_candles

def mark_data(df):
    marked_test = df.copy()
    stocks = []
    for stock in tqdm(df['ticker'].unique()):
        nigga_return_i = []
        i_list = []
        dates = []
        tickers = []
        for i in range(1, 21):
            nigga_test = marked_test[marked_test['ticker']==stock]
            for idx in range(nigga_test.shape[0]-i):
                nigga_return_i += [(nigga_test.iloc[idx+i]['close']/nigga_test.iloc[idx]['close'])-1]
                i_list += [i]
                dates += [nigga_test.iloc[idx]['begin']]
                tickers += [stock]
            nigga_return_i.extend(list([np.nan]*i)) 
            i_list.extend(list([i]*i)) 
            for idx in range(nigga_test.shape[0]-i, nigga_test.shape[0]):
                dates += [nigga_test.iloc[idx]['begin']]
            tickers.extend(list([stock])*i)
        new_df_stock = {'target_return_i': nigga_return_i, 'i': i_list, 'ticker': tickers, 'begin': dates}
        stocks += [pd.DataFrame(new_df_stock)]
    
    new_df = pd.concat(stocks, axis=0)
    predict_all = pd.merge(new_df, df, on=['ticker', 'begin'], how='inner').sort_values(['ticker', 'i', 'begin'])
    return predict_all

def create_sub(preds):
    weekends = [3, 4, 3+7, 4+7, 3+14, 4+14]
    shift = 0
    sub = pd.DataFrame()
    for i in range(1, 21):
        if i not in weekends:
            s = preds[preds['i'] == i-shift].groupby('ticker')['predicted_value'].mean()
        else:
            s = np.nan
            shift+=1
        sub[f'p{i}'] = s
    sub = sub.reset_index()
    return sub

def create_sub2(preds):
    weekends = [1, 1+5, 1+10]
    shift = 0
    sub = pd.DataFrame()
    for i in range(0, 14):
        s = preds[preds['i'] == i].groupby('ticker')['predicted_value'].mean()
        sub[f'p{i+shift}'] = s
        if i in weekends:
            s1 = np.nan
            sub[f'p{i+shift+1}'] = s1
            sub[f'p{i+shift+2}'] = s1
            shift+=2
    sub = sub.reset_index()
    return sub