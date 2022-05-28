import pandas as pd
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
test = test.fillna(0)
train=train.dropna()

print(train.isnull().sum())

X_train = train.drop(['count'], axis=1)
Y_train = train['count']

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

pred=model.predict(test)

submission = pd.read_csv('./submission.csv')
submission['count'] = pred
submission.to_csv('sub.csv', index=False)