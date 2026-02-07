import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('houses.csv')
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.info())
# print(df.isna().sum())
# print(df[].describe())
df['ZipCode'] = df['ZipCode'].astype(str)
df = pd.get_dummies(df, columns=['ZipCode'], drop_first=True)
df.replace(r'^\s*$', np.nan, regex=True,inplace=True)
df.dropna(inplace=True)
df=df.apply(pd.to_numeric,errors='coerce')
df.dropna(inplace=True)

X = df.drop(columns=['Price'])
y=df['Price']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_hat=model.predict(X_test)

# print(f'{"Prediction:":20} {y_hat}')
# print(f'{"Actual:":20} {y_test.values}')
mse=mean_squared_error(y_test,y_hat)
r2=r2_score(y_test,y_hat)

print(f'{"Mean squared error is:":20}{mse}')
print(f'{"R2 is:":20}{r2}')