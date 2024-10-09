import pandas as pd

raw = pd.read_excel('./data/아파트(매매)_실거래가_202201.xlsx',
                     header = 16, thousands=','
                   )

raw['평'] = raw['전용면적(㎡)'] / 3.3
raw['평당금액(만원)'] = raw['거래금액(만원)'] / raw['평']

raw.pivot_table(index = '중개사소재지',
               values = '평당금액(만원)',
               aggfunc = 'mean')



df = raw.pivot_table(index = '중개사소재지',
               values = '평당금액(만원)',
               aggfunc = ['count', 'mean'])
df.columns = ['거래건수', '평균평당금액(만원)']
df

df.sort_values(by='평균평당금액(만원)', ascending=False).head(3)

cond = df['거래건수'] > 2
df[cond].sort_values(by='평균평당금액(만원)', ascending=False).head(3)
df[cond].sort_values(by='평균평당금액(만원)', ascending=False).head(3).index

raw.info()
df.info()



