import pandas as pd

raw = pd.read_excel('./data/아파트(매매)_실거래가_202201.xlsx',
                     header = 16,
                    thousands=',')

raw.info()

# raw.head()
# raw.tail()
# raw.sample()
raw.describe()



명령어()
그룹(선택)

raw['거래금액(만원)'] / 7

b = raw[ [ '시군구','단지명', '거래금액(만원)'] ]
b

raw['거래금액(만원)'].hist(bins=30)

# raw['중개사소재지'].unique()
raw['중개사소재지'].value_counts()

#raw.sort_value(by='컬럼명')
raw.sort_values(by='거래금액(만원)', ascending = False)

raw['평'] = raw['전용면적(㎡)'] / 3.30579

raw['평당금액(만원)'] = raw['거래금액(만원)'] / raw['평']
b = raw.sort_values(by='평당금액(만원)', ascending = False)
b

raw['평당금액(만원)'] = raw['거래금액(만원)'] / raw['평']
b = raw.sort_values(by='평당금액(만원)', ascending = False).head(10)[['시군구','단지명','평','거래금액(만원)']]
b



cond = (raw['거래금액(만원)'] < 20000) & (raw['시군구'] == '강원도 강릉시 교동')
raw[cond]

cond = [ ]
for value in raw ['거래금액(만원)']:
    check = value < 20000
    cond.append(check)
raw[cond]



cond = [ ]
for value in raw['시군구']:
    check = '서울' in value
    cond.append(check)
raw[cond]

