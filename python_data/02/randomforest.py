# 랜덤포레스트는 여러 개의 의사결정나무를 만들어서 이들의 평균으로
# 예측의 성능을 높이는 방법(앙상블 기법)


# 랜덤포레스트회귀 모델 선언
from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()

# 라이브러리 불러오기
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# csv 파일 불러오기
train=pd.read_csv('./train.csv')
test=pd.read_csv('./test.csv')

# 결측치 전처리(보간)
train.interpolate(inplace=True)

# 모델 훈련
X_train = train.drop(['count'], axis=1)
Y_train = train['count']

# 랜덤포레스트 모듈의 옵션 중 criterion 옵션을 통해 어떤
# 평가 척도를 기준으로 훈련할 것인지 정할 수 있음
model = RandomForestRegressor(criterion='squared_error')
# 모델 훈련
model.fit(X_train, Y_train)

# 랜덤포레스트모델 예측변수의중요도를 출력
print(model.feature_importances_)

# X_train에서 drop할 피쳐
X_train_1 = train.drop(['count','id'], axis=1)
X_train_2 = train.drop(['count','id','hour_bef_windspeed'], axis=1)
X_train_3 = train.drop(['count','id','hour_bef_windspeed','hour_bef_pm2.5'], axis=1)

# 각 트레인에 따라 동일하게 피쳐를 drop한 test셋들 생성
test_1 = test.drop(['id'], axis=1)
test_2 = test.drop(['id','hour_bef_windspeed'], axis=1)
test_3 = test.drop(['id','hour_bef_windspeed','hour_bef_pm2.5'], axis=1)

# 각 X_train에 대해 모델 훈련 
model_input_var1 = RandomForestRegressor(criterion='squared_error')
model_input_var1.fit(X_train_1, Y_train)

# 각 모델로 test셋들 예측
y_pred_1 = model_input_var1.predict(test_1)

# 각 결과들을 submission 파일로 저장
submission_1 = pd.read_csv('./submission.csv')
submission_1['count']=y_pred_1
submission_1.to_csv('suv_1.csv', index=False)



