import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('Bodyfat.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isna().sum())
# print(df.duplicated().sum())
# df=df.drop_duplicates()
# print(df.corr())
# Dropping the 'cheating' column and the most redundant ones
df = df.drop(columns=['Density', 'Weight'])

y=df['bodyfat']
X=df.drop(columns=['bodyfat'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)