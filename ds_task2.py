import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Ice Cream.csv")


mean_temp = df['Temperature'].mean()
df.loc[df['Temperature'] < 5, 'Temperature'] = mean_temp


df['Revenue'] = df['Revenue'].fillna(df['Revenue'].mean())


x = df[['Temperature']]
y = df[['Revenue']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
accuracy = model.score(x_test, y_test)
rmse = mean_squared_error(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"Root Mean Squared Error: {rmse:.2f}")


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
