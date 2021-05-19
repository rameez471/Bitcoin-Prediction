import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
CURR_TO_PREDICT = 'BTC-USD'


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess(df):
    df = df.drop('future',1)

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]])

    random.shuffle(sequential_data)


main_df = pd.DataFrame()

curr_list = ['BTC-USD','LTC-USD','ETH-USD','BCH-USD']

for curr in curr_list:
    dataset = f'crypto_data/{curr}.csv'

    df = pd.read_csv(dataset ,names=['time','low','high','open','close','volume'])
    df.rename(columns={'close':f'{curr}_close',
                      'volume':f'{curr}_volume'},inplace=True)
    df.set_index('time',inplace=True)
    df = df[[f'{curr}_close',f'{curr}_volume']]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f'{CURR_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{CURR_TO_PREDICT}_close'], main_df['future']))


times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

# train_x, train_y = preprocess(main_df)
# validation_x, validation_y = preprocess(validation_main_df)