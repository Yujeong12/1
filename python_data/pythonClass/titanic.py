import pandas as pd
import seaborn as sns
import numpy as nuumpy
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('c:/python/pythonClass/datasample/train.csv')
print(titanic_df)
print(titanic_df.head(3))
print(titanic_df.tail(3))
print(titanic_df.info())
print(titanic_df.describe())

value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)

print(titanic_df.sort_values('Fare', ascending=False))

titanic_df['Age_0']=0
print(titanic_df)

titanic_df['Age_by_10']=titanic_df['Age']*10
print(titanic_df)

titanic_df.drop(['Age_0'], axis=1, inplace=True)
print(titanic_df.columns)

print(titanic_df[titanic_df['Pclass']==3].head(3))

print(titanic_df['Age'].max())

print(titanic_df.groupby('Pclass')['Age'].agg([max,min]))

sns.heatmap(data=titanic_df.corr(), annot=True, cmap='Blues')
plt.show()

