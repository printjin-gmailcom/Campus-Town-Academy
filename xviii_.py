"""
# DM 반응 예측모형

## *1*. 환경 설정

### 1.1 분석에 필요한 library 호출 및 google drive 연결
"""

# 분석에 사용할 패키지 로딩
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt

from sklearn.model_selection import train_test_split # 데이터 세트 분할
from sklearn.ensemble import RandomForestClassifier # 모델링
from sklearn.metrics import accuracy_score # 정확도 평가
from sklearn.model_selection  import GridSearchCV # 하이퍼 파라메터 튜닝용  (훈련데이터 세트에서 valdiation set 생성  )

## pandas 옵션 모든 컬럼 표시
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
# pd.set_option('display.max_rows', None)

# 그래프 스타일 선택
plt.style.use('ggplot')

# 구글드라이브에 있는 데이터셋 연결을 위한 구글드라이브 세팅
from google.colab import drive
drive.mount('/content/drive')

"""## *2*. 데이터 불러오기 및 확인

### 2.1 데이터 불러오기 및 확인
"""

data = pd.read_csv('/content/pm_customer_train1.csv')

data.head() # 처음 5개 데이터 확인

data.tail() # 마지막 5개 데이터 확인

# 각 컬럼(변수)의 유형 확인
data.info()

# 연속형 분포의 변수 확인
data.describe()

"""### 2.2 데이터형변환

"""

# 데이터형변환 : 날짜 계산을 위해 InvoiceDate의 Data type을 object type -> datetime64 로 변환
data['response_date'] = pd.to_datetime(data['response_date'])
data['purchase_date'] = pd.to_datetime(data['purchase_date'])

data.info()

"""### 2.3 변수명 수정"""

# 변수명이 특이한 column 확인 'average#balance#feed#index' '#'을 '_'로 수정
data.rename(columns = {'average#balance#feed#index' : 'average_balance_feed_index'},inplace= True)

data.info()

"""### 2.4 데이터 확인 및 선택"""

# 캠페인 별 반응(target)비율 확인
data.groupby(['campaign','response']).customer_id.count()

# 캠페인 2번 데이터만 선택
dataset = data[data['campaign'] == 2]

# campaign 2만 있는지 확인
dataset.head()

"""### 2.5 데이터에서 target 및 contol 의비율 확인 및 반응 유저의 구매율 확인"""

# 반응 이후 구매 비율 확인
print(dataset.groupby(['response','purchase']).customer_id.count())
print(dataset.groupby(['response','purchase']).customer_id.count() / dataset.shape[0])
print(dataset[dataset['response']==1].groupby(['response','purchase']).customer_id.count() / dataset[dataset['response']==1].shape[0])

dataset['customer_id'].duplicated().sum()

dataset.info()

"""### 2.6 의미없는 변수 삭제

> 목적이 응답 여부이기 때문에 응답이후 구매 정보 등은 모델 학습에 사용하지 않음
"""

# 필요 없는 response_date, purchase, purchase_date, product_id, Rowid 제거
dataset = dataset.drop(['response_date', 'purchase', 'purchase_date', 'product_id', 'Rowid'], axis = 1)
dataset.head()

# X_random 데이터 확인
dataset[['response','X_random']].value_counts()



"""## *3*. Null값 확인 및 처리

### 3.1 Null 값 확인
"""

# 3.1 Null 값이 있는 전체 case 수 확인
dataset.isnull().sum().sum()

"""## *4*. 데이터 탐색 및 이상치 제거

### 4.1 이상치 확인
"""

# Quantity 이상치 case 확인
dataset.describe()

len(dataset[dataset['months_current_account']< 0])

dataset[dataset['months_current_account']< 0]

# 음수인값 제거
dataset = dataset[dataset['months_current_account']>= 0]

dataset.shape

"""### 4.2 EDA 형식

### 데이터 탐색을 위한 시각화
"""

# 기본적인 히스토그램
bin = 10
dataset[dataset['response'] == 0 ].age.plot.hist(bins = bin)
dataset[dataset['response'] == 1 ].age.plot.hist(bins = bin)
plt.legend([0,1])

"""### 구간별 상대비율 히스토그램"""

# 각 구간별 상대 비율 히스토 그램
dataset['age_bin'] = pd.cut(dataset['age'],bins = 10)

temp_1 = pd.DataFrame(dataset.groupby(['response','age_bin']).customer_id.count()).reset_index()
temp_1_pivot = temp_1.pivot(index = 'age_bin',
                            columns= 'response',
                            values = 'customer_id')
temp_1_pivot['row_sum'] = temp_1_pivot.sum(axis=1)
temp_1_pivot[0] = temp_1_pivot[0]/temp_1_pivot['row_sum']
temp_1_pivot[1] = temp_1_pivot[1]/temp_1_pivot['row_sum']
temp_1_pivot[[1,0]].plot.bar(stacked = True )

#  문자형 변수
print(dataset.groupby(['marital','response']).customer_id.count())
print(dataset.groupby(['marital','response']).customer_id.count()/dataset.shape[0])

pd.concat([dataset.groupby(['marital','response']).customer_id.count(), dataset.groupby(['marital','response']).customer_id.count()/dataset.shape[0]], axis =1)

dataset['marital'].value_counts(normalize=True).sort_index()

pd.crosstab(index = dataset['marital'], columns = dataset['response'],margins = True, margins_name ="total",normalize = 'index').style.format('{:.2%}')

"""### 함수화 (연속형 및 명목형)"""

## 연속형 변수 그래프 생성 함수
def hist_plot (colname, taget_colname, n_bin = 10) : # colname : 그래프 그릴 컬럼명 , taget_colname: n_bin : 구간갯수

  n_bins = int(n_bin)
  data['bin'] = pd.cut(data[colname],bins = n_bins)

  temp = data.reset_index().groupby([taget_colname,'bin']).index.count().reset_index() # 구간화
  temp = temp.pivot(index = 'bin',columns = taget_colname, values = 'index')
  temp['row_sum'] = temp.sum(axis = 1) # 각각의 비율을 만들기 위해 구간 합계 생성
  temp['0_rate'] = temp[0]/temp['row_sum'] #각 row의 0의 비율
  temp['1_rate'] = temp[1]/temp['row_sum'] #각 row의 1의 비율

  ## 그래프
  fig = plt.figure()
  aig, ax = plt.subplots(ncols = 2 , figsize =(12,5))

  ## 첫번째  히스토그램
  sns.histplot(data = data, x= colname, hue= taget_colname, bins = n_bins, color=['orange','blue'], alpha  = 0.5, hue_order= (0,1), ax= ax[0])

  ## 구간별 타겟비율 그래프
  temp[['0_rate','1_rate']].plot.bar(stacked = True ,ax= ax[1])
  ax[1].legend(labels=[0,1],title =taget_colname) # 범례 순서 바꾸기
  plt.xlabel(colname)
  plt.show()

hist_plot('age','response',25 )

# 명속형 변수 그래프 생성 함수
def bar_plot (colname, taget_colname): # colname : 그래프 그릴 컬럼명 , taget_colname: target 컬럼명

## 그래프용 데이터 생성
  temp = data.reset_index().groupby([colname,taget_colname]).index.count()
  temp = temp.reset_index()
  temp = temp.pivot(index = colname, columns = taget_colname, values ='index')
  temp['row_sum'] = temp.sum(axis=1)
  temp['0_rate'] = temp[0]/temp['row_sum']
  temp['1_rate'] = temp[1]/temp['row_sum']

## 그래프
  fig = plt.figure()
  aig, ax = plt.subplots(ncols = 2 , figsize =(12,5))
## 빈도 막대 그래프
  temp[[0,1]].plot.bar(stacked = True ,ax= ax[0])
  ax[0].legend(labels=[0,1],title =taget_colname) # 범례 순서 바꾸기
  plt.xlabel(colname)
## 구간별 타겟 비율 막대 그래프
  temp[['0_rate','1_rate']].plot.bar(stacked = True ,ax= ax[1])
  ax[1].legend(labels=[0,1],title =taget_colname) # 범례 순서 바꾸기
  plt.xlabel(colname)
  plt.show()

  print(dataset[colname].value_counts(normalize=True).sort_index())
  print(" ")
  print(pd.crosstab(index = dataset[taget_colname], columns = dataset[colname],margins = True, margins_name ="total",normalize = 'columns'))

bar_plot('marital','response')

"""### 4.3 EDA"""

# age_youngest_child
hist_plot('age_youngest_child','response', 25)

# average_balance_feed_index
hist_plot('average_balance_feed_index','response')

# debt_equity

hist_plot('debt_equity','response')

# bad_payment

#hist_plot('bad_payment','response')
bar_plot('bad_payment','response')

# gold_card

bar_plot('gold_card','response')

# pension_plan

bar_plot('pension_plan','response')

# household_debt_to_equity_ratio

hist_plot('household_debt_to_equity_ratio','response')

# income

hist_plot('income','response')

# members_in_household

bar_plot('members_in_household','response')

# months_current_account

hist_plot('months_current_account','response')

# months_customer

bar_plot('months_customer','response')

# call_center_contacts

hist_plot('call_center_contacts','response')

# loan_accounts

bar_plot('loan_accounts','response')

# number_products

bar_plot('number_products','response')

# number_transactions

bar_plot('number_transactions','response')

#  non_worker_percentage

hist_plot('non_worker_percentage','response')

#  white_collar_percentage

hist_plot('white_collar_percentage','response')

#  rfm_score

hist_plot('rfm_score','response')

# marital

bar_plot('marital','response')

"""## *6*. 모델링

### 6.1 모델링을 의한 데이터 선택
"""

# 모델에 사용할 데이터 선택
dataset.reset_index(inplace =True)
dataset_for_ml =dataset[['response','age','age_youngest_child','average_balance_feed_index','debt_equity','gender','bad_payment',
                         'gold_card','pension_plan','household_debt_to_equity_ratio','income','marital','members_in_household',
                         'months_current_account','months_current_account','months_customer','call_center_contacts','loan_accounts',
                         'number_products','number_transactions','non_worker_percentage','white_collar_percentage','X_random']]



"""### 6.2 문자형 변수 더미변수로 변환"""

#one hot encoding

dataset_for_ml_dm = pd.get_dummies(dataset_for_ml)

dataset_for_ml_dm.info()

"""### 6.3 학습 확인 데이터 셋 분리"""

# 전체를 랜덤하게 나눈법
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# 변수로 들어있는 X_random 사용
X_train = dataset_for_ml_dm[dataset_for_ml_dm['X_random'] <= 2].drop(['X_random','response'], axis =1 )
X_test = dataset_for_ml_dm[dataset_for_ml_dm['X_random'] > 2].drop(['X_random','response'], axis =1 )
y_train = dataset_for_ml_dm[dataset_for_ml_dm['X_random']<= 2].response
y_test = dataset_for_ml_dm[dataset_for_ml_dm['X_random']> 2].response

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

"""### 6.4 모델 학습 및 평가"""

model = RandomForestClassifier(n_estimators = 100, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

result = pd.DataFrame({'pred' : y_pred, 'real' : y_test})

result.head(10)

# 정확도 확인
print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

"""### 6.5 하이퍼 마라메터 튜닝"""

# GridSearchCV의 param_grid 설정
# 찾고자 하는 parameter 정의
params = {
    'n_estimators': [100, 150, 200, 250],
    #'max_depth': [6, 9]
    #'min_samples_split': [0.01, 0.05, 0.1],
    'max_features': ['auto', 'log2'],
}

# 사용하고자 하는 모델 정의
rtc = RandomForestClassifier()

# n_jobs=-1 -> 모든 코어 사용 (속도빨라짐)
# verbose로 log 출력의 level을 조정 (숫자가 클 수록 많은 log 출력)
grid_tree = GridSearchCV(rtc, param_grid=params, cv=3,
                           n_jobs=-1,verbose=2)
grid_tree.fit(X_train, y_train)

grid_tree.best_params_

print('best parameters : ', grid_tree.best_params_)
print('best score : ', grid_tree.best_score_)

"""### 6.6 최종 모델 선택"""

model = RandomForestClassifier(n_estimators = 100,max_features= 'log2', random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

result = pd.DataFrame({'pred' : y_pred, 'real' : y_test})

"""### 6.7 최종 지표 확인"""

# confusion matrix 확인
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# roc cruve 확인
from sklearn import metrics

metrics.plot_roc_curve(model, X_test, y_test)
plt.plot([0,1], [0,1], "k--", "r+")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest ROC curve')
plt.show()
print("test_AUC : ", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Feature importance 확인
feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores

