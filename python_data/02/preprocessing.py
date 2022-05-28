# 라이브러리 불러오기
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

# 데이터 불러오기
train=pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# 결측치가 있는 피쳐 살펴보기
print(train.isnull().sum())
print(test.isnull().sum())

# 결측치 평균으로 대체
train.fillna({'hour_bef_temperature':int(train['hour_bef_temperature'].mean())}, inplace=True)
train.fillna({'hour_bef_precipitation':int(train['hour_bef_precipitation'].mean())}, inplace=True)
train.fillna({'hour_bef_windspeed':int(train['hour_bef_windspeed'].mean())}, inplace=True)
train.fillna({'hour_bef_humidity':int(train['hour_bef_windspeed'].mean())}, inplace=True)
train.fillna({'hour_bef_cisibility':int(train['hour_bef_visibility'].mean())}, inplace=True)
train.fillna({'hour_bef_ozone':int(train['hour_bef_ozone'].mean())},inplace=True)
train.fillna({'hour_bef_pm10':int(train['hour_bef_pm10'].mean())}, inplace=True)
train.fillna({'hour_bef_pm2.5':int(train['hour_bef_pm2.5'].mean())},inplace=True)

# 결과 확인
print(train.isnull().sum())

# 결측치 보간법으로 대체
train.interpolate(inplace=True)
test.interpolate(inplace=True)

# 결과 확인
print(train.isnull().sum())
print(test.isnull().sum())

