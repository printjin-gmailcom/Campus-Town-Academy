import pandas as pd

raw = pd.DataFrame()

temp = pd.read_excel('./data/아파트(매매)_실거래가_202201.xlsx',
                     header = 16,
                    thousands=','
                   )

raw = raw.append(temp)

import os

raw = pd.DataFrame()

for filename in os.listdir('./data'):
    temp = pd.read_excel(f'./data/{filename}',header = 16, thousands=',')
    raw = raw.append(temp)
    print(len(raw))

raw.info()

raw['평'] = raw['전용면적(㎡)'] / 3.3

raw['평당금액(만원)'] = raw['거래금액(만원)'] / raw['평']

raw.head(1)

raw.to_csv('./아파트.csv', encoding = 'cp949')

