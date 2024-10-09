"""
# 이탈 예측모형

## *1*. 환경 설정

### 1.1 분석에 필요한 library 호출 및 google drive 연결
"""

# 분석에 사용할 패키지 로딩
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt

import sklearn as skz

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import xgboost as xgb

# Commented out IPython magic to ensure Python compatibility.
# pandas 옵션 모든 컬럼 표시
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
# pd.set_option('display.max_rows', None)

# 그래프 스타일 선택
plt.style.use('ggplot')

# 그래프 바로 보기
# %matplotlib inline

# 구글 드라이브에 있는  데이터셋 연결을 위한 구글드라이브 세팅
from google.colab import drive
drive.mount('/content/drive')

"""## *2*. 데이터 불러오기 및 확인

### 2.1 데이터 불러오기 및 데이터 확인
"""

data = pd.read_csv('/content/Cell2Cell_NA.csv')

data.head() # 처음 5개 데이터 확인

data.tail() # 마지막 5개 데이터 확인

# 전체 데이터 모양 확인
data.shape

# 각 컬럼별 유형 확인
type(data.info())

"""### 2.2 데이터셋트 분리 및 타겟 확인"""

# 학습 및 테스트 데이터 세트 확인
pd.concat ([data['CHURNDEP'].value_counts(dropna= False),
            data['CHURNDEP'].value_counts(dropna= False, normalize= True)*100],
            axis = 1)

# Target 비율 확인
pd.concat([data['CHURN'].value_counts(dropna= False),
           data['CHURN'].value_counts(dropna= False, normalize = True) *100],
          axis = 1)

# 데이터 세트 구분 및 target 비율 같이 보기
pd.concat(
    [data[['CHURNDEP','CHURN']].value_counts(dropna= False, sort = False),
     data[['CHURNDEP','CHURN']].value_counts(dropna= False, sort = False, normalize =True)*100],
     axis = 1)

# 테스트 데이터 셋트의 CHURN 비율 확인

data[pd.isnull(data['CHURNDEP'])].CHURN.value_counts(normalize =True)

"""## *3*. Null값 확인 및 처리

### 3.1 Null 값 확인
"""

# data 에 포함된  Null 값 수 확인
print(data.isnull().sum().sum())

print(data.isnull().sum().sum()/data.shape[0])

# Null 값이 있는 변수 확인
pd.set_option('display.max_rows', None)
data.isnull().sum()

# Null 값이 있는 변수 확인
data.columns[data.isnull().sum()>0]

# 학습과 테스트를 구분하는 변수를 제외한 Null 값수
data.iloc[:,:77].isnull().sum().sum()

# REVENUE 가 NULL 값인것 확인해보기
data[pd.isnull(data['REVENUE'])].iloc[:,:77].isnull().sum()

data[pd.isnull(data['AGE1'])].iloc[:,:77].isnull().sum()

data[pd.isnull(data['CSA'])].iloc[:,:77].isnull().sum()

data[pd.isnull(data['PHONES'])].iloc[:,:77].isnull().sum()]

pd.set_option('display.max_rows', 30)  # pandas 시각화 옵션 원상태로

"""### 3.2 Null 값 제거 및 대체"""

# AGE 와 PHONES CSA NULL 값 제거
null_index = data[(data['AGE1'].isnull()) | (data['PHONES'].isnull()) | (data['CSA'].isnull()) ].index

print(len(null_index))

data_na_treat = data.drop(null_index)
data_na_treat.shape

# null 값 대체  사용 요금 및 금액 부분 변수는 0으로 대체

data_na_treat['REVENUE'] = data_na_treat['REVENUE'].fillna(0)
data_na_treat['MOU'] = data_na_treat['MOU'].fillna(0)
data_na_treat['RECCHRGE'] = data_na_treat['RECCHRGE'].fillna(0)
data_na_treat['DIRECTAS'] = data_na_treat['DIRECTAS'].fillna(0)
data_na_treat['OVERAGE'] = data_na_treat['OVERAGE'].fillna(0)
data_na_treat['ROAM'] = data_na_treat['ROAM'].fillna(0)
data_na_treat['CHANGEM'] = data_na_treat['CHANGEM'].fillna(0)
data_na_treat['CHANGER'] = data_na_treat['CHANGER'].fillna(0)

data_na_treat.iloc[:,0:77].isnull().sum().sum()

"""## 4.EDA

### 4.1 연속형 데이터 탐색
"""

n_bins = 10
data_na_treat['bin'] = pd.cut(data_na_treat['REVENUE'],bins = n_bins)

temp_1 = pd.DataFrame(data_na_treat.groupby(['CHURN','bin']).CUSTOMER.count()).reset_index()
temp_1_pivot = temp_1.pivot(index = 'bin',columns = 'CHURN',values = 'CUSTOMER')
temp_1_pivot['row_sum'] = temp_1_pivot.sum(axis=1)
temp_1_pivot[0] = temp_1_pivot[0]/temp_1_pivot['row_sum']
temp_1_pivot[1] = temp_1_pivot[1]/temp_1_pivot['row_sum']

temp_1_pivot

## 연속형 변수 그래프 생성 함수
def hist_plot (colname, taget_colname, n_bin = 10) : # colname : 그래프 그릴 컬럼명 , taget_colname: n_bin : 구간갯수

  n_bins = int(n_bin)
  data_na_treat['bin'] = pd.cut(data_na_treat[colname],bins = n_bins)

  temp = data_na_treat.reset_index().groupby([taget_colname,'bin']).index.count().reset_index() # 구간화
  temp = temp.pivot(index = 'bin',columns = taget_colname, values = 'index')
  temp['row_sum'] = temp.sum(axis = 1) # 각각의 비율을 만들기 위해 구간 합계 생성
  temp['0_rate'] = temp[0]/temp['row_sum'] #각 row의 0의 비율
  temp['1_rate'] = temp[1]/temp['row_sum'] #각 row의 1의 비율

  ## 그래프
  fig = plt.figure()
  aig, ax = plt.subplots(ncols = 2 , figsize =(12,5))

  ## 첫번째  히스토그램
  sns.histplot(data = data_na_treat, x= colname, hue= taget_colname, bins = n_bins, color=['orange','blue'], alpha  = 0.5, hue_order= (0,1), ax= ax[0])

  ## 구간별 타겟비율 그래프
  temp[['0_rate','1_rate']].plot.bar(stacked = True ,ax= ax[1])
  ax[1].legend(labels=[0,1],title =taget_colname) # 범례 순서 바꾸기
  plt.xlabel(colname)
  plt.show()

# 명속형 변수 그래프 생성 함수
def bar_plot (colname, taget_colname): # colname : 그래프 그릴 컬럼명 , taget_colname: target 컬럼명

## 그래프용 데이터 생성
  temp = data_na_treat.reset_index().groupby([colname,taget_colname]).index.count()
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

  print(data_na_treat[colname].value_counts(normalize=True).sort_index())
  print(" ")
  print(pd.crosstab(index = data_na_treat[taget_colname], columns = data_na_treat[colname],margins = True, margins_name ="total",normalize = 'columns'))

hist_plot('MOU','CHURN',25)

## 연속형 변수 및 명목형 변수 리스 생성
int_colname = []
cat_colname = []
for i in data_na_treat.columns :
  if pd.api.types.is_numeric_dtype(data_na_treat[i]) == True :
    int_colname.append(i)
  elif pd.api.types.is_string_dtype(data_na_treat[i]) == True :
    cat_colname.append(i)
print(int_colname)
print(cat_colname)

## 연속형 변수 그래프 생성
for i in int_colname[0:10] :
  hist_plot(i,'CHURN',10)

"""### 4.2 명목형 변수 탐색"""

bar_plot('CREDITA', 'CHURN')



# 더미변수를 단일 명목형 변수 만들기
data_na_treat.head()

CREDIT =  data_na_treat[['CREDITA','CREDITAA','CREDITB','CREDITC','CREDITDE','CREDITGY','CREDITZ']].stack() #하나의 Series로 쌓기
CREDIT = pd.Series(CREDIT[CREDIT != 0].index.get_level_values(1)) # 0 인값들 제거후 각 인덱스를 값으로
data_na_treat['CREDIT'] = CREDIT.str.replace('CREDIT','')  # 값의 앞에 'CREDIT을 삭제하여 신용등급만 값으로 만들기

data_na_treat['CREDIT'].head(10)  # 데이터 확인

data_na_treat['CREDIT'].value_counts() # 분포 확인

bar_plot('CREDIT', 'CHURN')

# 지역정보 확인
pd.crosstab(data_na_treat['CSA'].str.slice(0,3),data_na_treat['CHURN'],margins= True, normalize= 'columns')

"""## *5*. 변수 파생 (feature engineering)"""

# 숫자값을 0과 1로 구분 바꾸기 위한 함수 생성
def over0 (x):
  if x > 0:
    return 1
  else :
    return 0

# 가족수
data_na_treat['FAMILY_CNT']= data_na_treat['AGE1'].apply(over0) + data_na_treat['AGE2'].apply(over0) + data_na_treat['CHILDREN']

# 전체 통화건수
data_na_treat['TOTAL_CALLS'] = data_na_treat['INCALLS'] +data_na_treat['OUTCALLS']

# 추가비용
data_na_treat['OVER_CHARGE'] = data_na_treat['REVENUE'] - data_na_treat['RECCHRGE']

# 추가서비스 사용갯수
data_na_treat['ADDED_SERVICE_CNT']= data_na_treat['THREEWAY'].apply(over0) + data_na_treat['CALLFWDV'].apply(over0) + data_na_treat['CALLWAIT'].apply(over0)

data_na_treat[['FAMILY_CNT','TOTAL_CALLS','OVER_CHARGE','ADDED_SERVICE_CNT']].head()

hist_plot('OVER_CHARGE','CHURN')

"""## *6*. 데이터 세트 분할 및 미사용 변수 제거


"""

X_train = data_na_treat[data_na_treat['CHURNDEP'].isnull() == False].drop(['CUSTOMER','CHURN','CALIBRAT','CHURNDEP','CSA','bin','CREDIT','NEWCELLN'],axis= 1)
X_test = data_na_treat[data_na_treat['CHURNDEP'].isnull() ].drop(['CUSTOMER','CHURN','CALIBRAT','CHURNDEP','CSA','bin','CREDIT','NEWCELLN'],axis= 1)
y_train = data_na_treat[data_na_treat['CHURNDEP'].isnull() == False].CHURN
y_test = data_na_treat[data_na_treat['CHURNDEP'].isnull() ].CHURN

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

X_train.head()

print(y_train.head(10))
print(y_train.value_counts())

# 변수간의 상관분석
colormap = plt.cm.PuBu
plt.figure(figsize=(40,40))
plt.title("Person Correlation of Features", y = 1.05, size = 15)
sns.heatmap(X_train.astype(float).corr(), linewidths = 0.1, vmax = 1.0,square = True, cmap = colormap, linecolor = "white", annot = True)

"""## *7*. 모델링 및 평가

### 7.1 RandomForest
"""

X_train

# Random Forest 모델적용

model_RF = RandomForestClassifier(n_estimators = 1000, random_state=0) # 기본 반복횟수 및 랜덤 시드 등 옵션 선택하여 분류기 생성
model_RF.fit(X_train, y_train) # 학습용 데이터 지정하여 학습

RF_y_pred = model_RF.predict(X_test) # 학습된 모형에 확인용 데이터 (test_datase)를 넣어 예측값 생성

RF_result = pd.DataFrame({'pred' : RF_y_pred, 'real' : y_test}) #정확도 평가를 위해 예측값과 실제값을 하나의 dataframe 에 생성

RF_result.head(10) #확인

# 정확도 확인
# confusion matrix 확인
print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, RF_y_pred)))
print(confusion_matrix(y_test, RF_y_pred))
print(classification_report(y_test, RF_y_pred))
print("test_AUC : ", metrics.roc_auc_score(y_test, model_RF.predict_proba(X_test)[:,1]))

# ROC 그래프
metrics.plot_roc_curve(model_RF, X_test, y_test)
plt.plot([0,1], [0,1], "k--", "r+")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest ROC curve')
plt.show()

feature_scores = pd.Series(model_RF.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores.head(10)

"""### 7.2 Logstic Regession"""

# from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# from sklearn.linear_model import LogisticRegression

model_LR = LogisticRegression()
model_LR.fit(X_train_scaled, y_train)

LR_y_pred = model_LR.predict(X_test)

LR_result = pd.DataFrame({'pred' : LR_y_pred, 'real' : y_test})

LR_result.head(5)

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, LR_y_pred)))
print(confusion_matrix(y_test, LR_y_pred))
print(classification_report(y_test, LR_y_pred))
print("test_AUC : ", metrics.roc_auc_score(y_test, model_LR.predict_proba(X_test)[:,1]))

# ROC 그래프
metrics.plot_roc_curve(model_LR, X_test, y_test)
plt.plot([0,1], [0,1], "k--", "r+")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest ROC curve')
plt.show()

# 입력 변수의 Coefficent (변수별 영향력)
model_LR.coef_

range(0,X_train.shape[0])

"""### 7.3 XGBoos"""

#import xgboost as xgb
XGB_model = xgb.XGBClassifier(booster='gbtree',
                              colsample_bylevel=0.9,
                              colsample_bytree=0.8,
                              gamma=0,
                              max_depth=8,
                              min_child_weight=3,
                              n_estimators=50,
                              nthread=8,
                              objective='binary:logistic',
                              random_state=2,
                              silent= True)



XGB_model.fit(X_train,y_train, eval_set=[(X_test,y_test)],early_stopping_rounds=50, eval_metric ='auc')

XGB_y_pred  = XGB_model.predict(X_test)

XGB_result = pd.DataFrame({'pred' : XGB_y_pred, 'real' : y_test})

XGB_result.head(5)

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, XGB_y_pred)))
print(confusion_matrix(y_test, XGB_y_pred))
print(classification_report(y_test, XGB_y_pred))
print("test_AUC : ", metrics.roc_auc_score(y_test, XGB_model.predict_proba(X_test)[:,1]))

# ROC 그래프
metrics.plot_roc_curve(XGB_model, X_test, y_test)
plt.plot([0,1], [0,1], "k--", "r+")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest ROC curve')
plt.show()

plt.figure(figsize=(20,20))
sorted_idx = XGB_model.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], XGB_model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")

