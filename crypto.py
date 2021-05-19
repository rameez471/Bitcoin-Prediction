import pandas as pd

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
CURR_TO_PREDICT = 'BTC-USD'


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


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

print(main_df.head(10))