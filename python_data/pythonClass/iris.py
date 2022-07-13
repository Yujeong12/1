import pandas as pd
import seaborn as sns
import numpy as nuumpy
import matplotlib.pyplot as plt

iris_df=pd.read_csv('c:/python/pythonClass/datasample/IRIS.csv')
print(iris_df)
print(iris_df.info())
print(iris_df[iris_df['species']=='Iris-setosa']['petal_length'].mean())
print(iris_df[iris_df['species']=='Iris-setosa']['petal_width'].mean())
