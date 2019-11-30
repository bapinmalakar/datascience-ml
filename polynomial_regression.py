# Polynomial regression is form of linear regression which will modeled the relationship betwwen X and Y by nth Degree polynomial
# Its good fit for non-linear
# All suppose you want for a features your target should always increase, as X is increase then we can use polynomial

#h(Q)= Q0 + Q1*X + Q2(X^2) + Q3(X^3), this ensure taht your Y never decrease with X


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

dataFrame = pd.read_csv('./query_result.csv', index_col="id")

X = dataFrame[['shares', 'views']]
Y = dataFrame.likes

print(X.head())
cols = [col for col in X.columns if X[col].dtype=='object']
print(cols)

#BUILD POLYNOMIAL FOR MODEL
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

#Create Linear model
polyModel = LinearRegression()
nonPolyModel = LinearRegression()

#trai or Fit model
polyModel.fit(X_poly, Y)
nonPolyModel.fit(X, Y)

#prediction
polyPredict = polyModel.predict(X_poly)
nonPolyPredict = nonPolyModel.predict(X)

print('\n\n', polyPredict)
print('\n\n', nonPolyPredict)

#MAE error
print('\n\n', mean_absolute_error(Y, polyPredict), r2_score(Y, polyPredict))
print('\n\n', mean_absolute_error(Y, nonPolyPredict), r2_score(Y, nonPolyPredict))