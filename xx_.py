"""
# 평점표 모형

##1.환경 설정

### 1.1 분석에 필요한 library 호출 및 google drive 연결
"""

## 분석에 사용할 패키지 로딩

# 데이터 핸들링
import numpy as np
import pandas as pd

#시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 학습 훈련 데이터 분리
from sklearn.model_selection import train_test_split

#로지스틱 회귀 알고리즘
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# 정확도 확인
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve

## pandas 옵션 모든 컬럼 표시
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
# pd.set_option('display.max_rows', None)

# 그래프 스타일 선택
plt.style.use('ggplot')

## 구글 드라이브에 있는  데이터셋 연결을 위한 구글드라이브 세팅
from google.colab import drive
drive.mount('/content/drive')

"""##2.데이터 불러오기 및 확인

### 2.1 데이터 불러오기 및 데이터 확인
"""

data = pd.read_csv('/content/german.data',
                   delimiter=' ',
                   names= ['checking','duration','history','purpose','amount',
                           'savings','employed','installp','marital','coapp',
                           'resident','property','age','other','housing',
                           'existrc','job', 'depends', 'telephon','foreign','good_bad'])
data.head()

data.tail()

## 전체 데이터 모양 확인
data.shape

## 각 컬럼별 유형 확인
data.info()

## Target 비율 확인
pd.concat(
    [data['good_bad'].value_counts(dropna= False, sort = False),
     data['good_bad'].value_counts(dropna= False, sort = False, normalize =True)*100]
     , axis = 1)

## 2를 0으로 바꾸기
data['good_bad'] = data['good_bad'].apply(lambda x: 1 if x == 1 else 0)

## 1은 정상 2가 부도  0과 1로 바꾸기
data['good_bad'].value_counts()

"""##3.Null값 확인 및 처리

### 3.1 Null 값 확인
"""

## data 에 포함된  Null 값 수 확인
print(data.isnull().sum().sum())

print(data.isnull().sum().sum()/data.shape[0])

## 0건

"""## 4.EDA

### 4.1 시각화 함수 생성
"""

# 명속형 변수 그래프 생성 함수
def bar_plot (colname, taget_colname): # colname : 그래프 그릴 컬럼명 , taget_colname: target 컬럼명

## 그래프용 데이터 생성
  temp = data.reset_index().groupby([colname,taget_colname]).index.count()
  temp = temp.reset_index()
  temp = temp.pivot(index = colname, columns = taget_colname, values ='index')
  temp['row_sum'] = temp.sum(axis=1)
  temp['0_rate'] = temp[0]/temp['row_sum']
  temp['1_rate'] = temp[1]/temp['row_sum']

 # IV 계산을 위한 데이터
  temp['col_0_rate']= temp[0]/temp.sum(axis =0)[0]
  temp['col_1_rate']= temp[1]/temp.sum(axis =0)[1]
  temp['woe'] =np.log(temp['col_1_rate']/temp['col_0_rate'])
  temp['diff_rate'] = temp['col_1_rate'] - temp['col_0_rate']
  temp['iv'] = temp['diff_rate'] *temp['woe']

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
  print(colname," IV : ",temp.sum(axis =0)['iv'])

bar_plot('marital','good_bad')

## 연속형 변수 그래프 생성 함수
def hist_plot (colname, taget_colname, n_bin = 10) : # colname : 그래프 그릴 컬럼명 , taget_colname: n_bin : 구간갯수

  n_bins = int(n_bin)
  data['bin'] = pd.cut(data[colname],bins = n_bins)

  temp = data.reset_index().groupby([taget_colname,'bin']).index.count().reset_index() # 구간화
  temp = temp.pivot(index = 'bin',columns = taget_colname, values = 'index')
  temp['row_sum'] = temp.sum(axis = 1) # 각각의 비율을 만들기 위해 구간 합계 생성
  temp['0_rate'] = temp[0]/temp['row_sum'] #각 row의 0의 비율
  temp['1_rate'] = temp[1]/temp['row_sum'] #각 row의 1의 비율

 # IV 계산을 위한 데이터
  temp['col_0_rate']= temp[0]/temp.sum(axis =0)[0]
  temp['col_1_rate']= temp[1]/temp.sum(axis =0)[1]
  temp['woe'] =np.log(temp['col_1_rate']/temp['col_0_rate'])
  temp['diff_rate'] = temp['col_1_rate'] - temp['col_0_rate']
  temp['iv'] = temp['diff_rate'] *temp['woe']

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

  print(colname," IV : ",temp.sum(axis =0)['iv'])

hist_plot('duration','good_bad')

## 연속형 변수 및 명목형 변수 리스 생성
int_colname = []
cat_colname = []
for i in data.columns :
  if pd.api.types.is_numeric_dtype(data[i]) == True :
    int_colname.append(i)
  elif pd.api.types.is_string_dtype(data[i]) == True :
    cat_colname.append(i)
print(int_colname)
print(cat_colname)

"""### 4.2 연속형 변수 탐색"""

## 연속형 변수 그래프 생성
for i in int_colname :
  hist_plot(i,'good_bad',10)

"""### 4.3 명목형 변수 탐색"""

## 명목형 변수 그래프 생성
for i in cat_colname :
  bar_plot(i,'good_bad')

"""# 5.변수 파생 (feature engineering)

### 5.1 변수 구간화 설명
* 로지스틱 회귀분석의 특성상
* 연속형 변수는 값이 1단위 증가함에따라 오즈비의 증감이 베타값으로 표현됨
* 명목형 변수는 첫번째 카테고리 대비 각 카테고리의 오즈비의 증감을 베타값(변수별 가중치)으로 표현됨
* 그렇기 때문에 각변수의 점수와 스코어를 한눈에 보기 위해서
* bad 비율이 가장 높은 쪽이 첫 카테고리가 되도록 변수를 서열화 함
* 또한 이렇게 만들어야 카테고리 변수의 다중 공선성 확인을 하기 쉬움.

###5.2 변수 서열화 및 구간화  
- checking (당좌예금 계좌 상태) :그대로 사용
- duration (신용거래 개월수): 17개월 이하, 44개월 이하, 44개월 초과로 구간화
- history (신용이력) : A30~ A31 , A32 ~ A33, A34 로 구간화
- purpose (대출 목적): A40, A46 ~ A410, A42 ~ A45, A41
- amount (신용대출 금액): 11000초과, 11000 이하, 4000 이하
- savings (보통예금 계좌 평잔): A65, A61,A62
- employed (현직장 재직기간): A71 ~ A72, A73, A74 ~ A75
- installp (소득대비 대출금비율): ~3.5, ~ 2.5, ~ 1.5, 1.5 ~
- marital (결혼여부 및 성별): A91, A92 & A95, A93, A94
- coapp (채무관계): A101, A102 ~ A103
- resident (현 거주지 거주기간): <= 1 ,  >1  
- property (재산): A124, A122 ~ A123, A121
- age (나이):  >52 & <= 60  , <= 33 , >33 & <= 52, >60
- other (기타 할부 거래): A141 ~ A142 , A143 ~
- housing (주거형태): A153, A151, A152
- existrc (당행 현재 대출건수): <= 1, > 1
- job (직업): A174 , A173, A171 ~ A172
- depends (부양가족수): 0 , <= 1 , >1
- telephon (전화소유): A191, A192
- foreign (외국인여부): A201, A202
"""

# 머신러닝용 데이터셋 생성 및 변수 변환
dataset = data.copy()
dataset['checking'] = dataset['checking'].str.replace('A','B')
dataset['duration'] = dataset['duration'].apply(lambda x: 'B23' if x <= 17 else ('B22' if x <= 44 else 'B21'))
dataset['history'] = dataset['history'].apply(lambda x: 'B31' if (x == 'A30') | (x == 'A31') else ('B32' if x == 'A32' else ('B33' if  x == 'A33' else 'B34')))
dataset['purpose'] = dataset['purpose'].apply(lambda x: 'B40' if (x == 'A40') else ('B43' if x == 'A41' else ('B42' if  (x == 'A42') | (x == 'A43') | (x == 'A44') | (x == 'A45') else 'B41')))
dataset['amount'] = dataset['amount'].apply(lambda x: 'B53' if x <= 4000 else ('B52' if x <= 11000 else 'B51'))
dataset['savings'] = dataset['savings'].apply(lambda x: 'B61' if (x == 'A65')|(x == 'A61')|(x == 'A62') else ('B62' if x == 'A63' else 'B63'))
dataset['employed'] = dataset['employed'].apply(lambda x: 'B71' if (x == 'A71')|(x == 'A72')else ('B72' if x == 'A73' else 'B73'))
dataset['installp'] = dataset['installp'].apply(lambda x: 'B84' if x <= 1.5 else ('B83' if x <= 2.5 else ('B82' if x <= 3.5 else 'B81')))
dataset['marital'] = dataset['marital'].apply(lambda x: 'B91' if x == 'A91' else ('B92' if (x == 'A92')|(x == 'A95') else ('B94' if x == 'A93' else 'B93')))
dataset['coapp'] = dataset['coapp'].apply(lambda x: 'B101' if x == 'A101' else 'B102')
dataset['resident'] = dataset['resident'].apply(lambda x: 'B112' if x <= 1 else 'B111')
dataset['property'] = dataset['property'].apply(lambda x: 'B123' if x == 'A121' else ('B122' if (x == 'A122')|(x == 'A123') else 'B121'))
dataset['age'] = dataset['age'].apply(lambda x: 'B132' if x <= 33 else ('B133' if x <= 52 else ('B131' if x <= 60 else 'B134' )))
dataset['other'] = dataset['other'].apply(lambda x: 'B141' if  (x == 'A141')|(x == 'A142') else 'B142')
dataset['housing'] = dataset['housing'].apply(lambda x: 'B151' if x == 'A153' else ('B152' if x == 'A151' else 'B153' ))
dataset['existrc'] = dataset['existrc'].apply(lambda x: 'B161' if x <= 1 else 'B162')
dataset['job'] = dataset['job'].apply(lambda x: 'B173' if  (x == 'A171')|(x == 'A172') else ('B172' if x == 'A173' else 'B171' ))
dataset['depends'] = dataset['depends'].apply(lambda x: 'B181' if x == 0 else ('B182' if x <= 1 else 'B83'))
dataset['telephon'] = dataset['telephon'].str.replace('A','B')
dataset['foreign'] = dataset['foreign'].str.replace('A','B')
dataset.drop('bin', axis= 1, inplace = True)
dataset.head()

"""###5.3 변수의 서열화 확인"""

# 명속형 변수 그래프 생성 함수
def bar_plot (colname, taget_colname): # colname : 그래프 그릴 컬럼명 , taget_colname: target 컬럼명

## 그래프용 데이터 생성
  temp = dataset.reset_index().groupby([colname,taget_colname]).index.count()
  temp = temp.reset_index()
  temp = temp.pivot(index = colname, columns = taget_colname, values ='index')
  temp['row_sum'] = temp.sum(axis=1)
  temp['0_rate'] = temp[0]/temp['row_sum']
  temp['1_rate'] = temp[1]/temp['row_sum']
# IV 계산을 위한 데이터
  temp['col_0_rate']= temp[0]/temp.sum(axis =0)[0]
  temp['col_1_rate']= temp[1]/temp.sum(axis =0)[1]
  temp['woe'] =np.log(temp['col_1_rate']/temp['col_0_rate'])
  temp['diff_rate'] = temp['col_1_rate'] - temp['col_0_rate']
  temp['iv'] = temp['diff_rate'] *temp['woe']
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

  print(colname," IV : ",temp.sum(axis =0)['iv'])

lm_col_names = dataset.iloc[:,0:20].columns

for i in lm_col_names :
  bar_plot(i,'good_bad')

"""#6.모델링

##6.1 데이터 세트 분할 및 미사용 변수 제거
"""

# IV 값이 0.04 이하인 installp	coapp	resident existrc	job	depends	telephon	foreign	제거

ML_dataset = dataset.drop(['installp','coapp','resident','existrc','job','depends','telephon','foreign','housing'],axis= 1)
ML_dataset.head()

"""##6.2 명목형 변수 더미화"""

ML_dataset = pd.get_dummies(ML_dataset)
ML_dataset.head()

# 변수간의 상관분석
colormap = plt.cm.PuBu
plt.figure(figsize=(20,20))
plt.title("Person Correlation of Features", y = 1.05, size = 15)
sns.heatmap(ML_dataset.corr(), linewidths = 0.1, vmax = 1.0,square = True, cmap = colormap, linecolor = "white", annot = True)

"""##6.3 데이터셋 분리"""

# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (ML_dataset.iloc[:,1:],ML_dataset.iloc[:,0], test_size =0.2, random_state=0)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

y_train.value_counts(normalize = True)

"""## 6.4 학습

####  사이킷런에 있는 로지스틱 회귀분석을 이용
"""

#from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

print(model.coef_)

print(model.intercept_)

"""#### stats model 의 로지스틱 회귀분석 이용"""

# import statsmodels.api as sm
X_train = sm.add_constant(X_train)
X_train.head()

# stats model 의 로지시틱 회귀 모형 사용
# import statsmodels.api as sm
X_train = sm.add_constant(X_train)

logit = sm.Logit(y_train, X_train)

result = logit.fit()

print(result.summary())

LR_y_pred = model.predict(X_test)

LR_result = pd.DataFrame({'pred' : LR_y_pred, 'real' : y_test})

LR_result.head()

"""## *6.5* 평가

## 6.6 점수화
"""

# 정확도 확인
# confusion matrix 확인
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import plot_roc_curve

print('Model accuracy : {0:0.4f}'. format(accuracy_score(y_test, LR_y_pred)))
print(confusion_matrix(y_test, LR_y_pred))
print(classification_report(y_test, LR_y_pred))
print("test_AUC : ", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# ROC 그래프
plot_roc_curve(model, X_test, y_test)
plt.plot([0,1], [0,1], "k--", "r+")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RandomForest ROC curve')
plt.show()

X_train.columns[1:]

# 점수화
model_coef =pd.DataFrame( pd.Series(model.coef_[0], X_train.columns[1:]))
model_coef['score']= model_coef[0]*40/np.log(2)
model_coef['score']
