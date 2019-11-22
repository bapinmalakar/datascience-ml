#here we going to find that given dataset is Linera in nature or not
#Steps
#1. Create linneRegrassor model on the data
#2. Then find  least square error(r2_score)
#3. if Square value show high accuracy, the Linear
#4. Else non linear

#LinerRegressor can apply only on Numaric value

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

fileIs= './query_result.csv'
dataFrame = pd.read_csv(fileIs, index_col='id')

dataFrame.dropna(axis=0)

print(dataFrame.head())

Y = dataFrame.likes
X = dataFrame[['format_name', 'verified', 'views', 'shares']]

#find categorial column and replace with unique number
cols = [col for col in X.columns if X[col].dtype == 'object']
print('cols are: ', cols)

newX = X.copy()
labelEncoder = LabelEncoder()

for c in cols:
    newX[c] = labelEncoder.fit_transform(X[c])

print('newX ', newX.head())

linerModel = LinearRegression()
randomForesetModel = RandomForestRegressor()

linerModel.fit(newX, Y)
randomForesetModel.fit(newX, Y)

linearPredicResult = linerModel.predict(newX)
randomPredictResult = randomForesetModel.predict(newX)

linear_accuracy_score = r2_score(linearPredicResult, Y) #if score is negative
random_accuracy_score = r2_score(randomPredictResult, Y)

print('Square score is: ', linear_accuracy_score, random_accuracy_score)

#linear_accuracy_score score is negative means non-linear data and this file have non-linear data