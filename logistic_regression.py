# Logistic Regression
# Is statistic, classification model for binary classification
# Binary because it can only two distinct class
# Its Supervised learning model

# Odds
# Number event occure on something / number of event not occure on same place
# range from negative to infinity positive

# Probabilty
# Number of evens occure on something/ total number of events
# range from 0 to 1

# Predict whether a people have dibates or not.
# Where we want probabilty between 0 to 1.
# 1 means: guaranteed to pass
# 0 means guranteed to fail

# In mathmathics this can done by formula
# P = (1/ (1 + (e^-y)))
# Known as Sigmoid function

from sklearn.datasets import make_classification
# to generate data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

fileIs = "./diabetes.csv"


dataFrame = pd.read_csv(fileIs)

print(dataFrame.head())

feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = dataFrame[feature_cols] # features
Y = dataFrame.Outcome # target

stringCols = [col for col in dataFrame.columns if dataFrame[col].dtype == 'object']

print('stringcols: ', stringCols)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)

lr_model = LogisticRegression()

lr_model.fit(x_train, y_train)

predict_result = lr_model.predict(x_test)

print(predict_result)

#evalute model accuracy
accuracy = confusion_matrix(y_test, predict_result)
print('Accuracy is: ', accuracy)

#return 2D array with 2 list
# left diagonal element syas, accurate prediction
# right diagonal element syas, in-accurate prediction

#we can also visulize confusion_metrix using HeatMap




