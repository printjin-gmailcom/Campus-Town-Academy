import pandas as pd

raw = pd.read_csv('./아파트.csv', encoding = 'cp949')

cond = [ ]
for sigungu in raw['시군구']:
    check = '서울' in sigungu
    cond.append(check)
seoul = raw[cond]

t = seoul.pivot_table(index = '단지명',
                 values = '평당금액(만원)',
                 aggfunc = 'mean')
t.sort_values(by = '평당금액(만원)', ascending = False)

df = seoul.pivot_table(index = ['시군구', '단지명'],
                values = '평당금액(만원)',
                 aggfunc = 'mean',
                  columns = '계약년월')
df

df.info()

df.fillna(method = 'backfill')

df.dropna().to_excel('./서울아파트거래액변화.xlsx')

df.dropna

