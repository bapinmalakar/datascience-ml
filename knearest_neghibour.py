#K-nearest neighbors algorithm
# It is used for classification and regresssion
# It is non-parametric method
# Here a object is classified by majority vote of its neighbor
# And object assign to a class, which is most commom with its K nearest neighbor 
# K is integer and small number
# if k=1 then object is assign to the class of that single nearset neighbor
# KNN is lazy algorithm
#1. It doesn't use traning data for generalization. 
#2. Other word no explicit training phase or if present then very minimal
#3. Lack of generalization means KNN keep al data training data
#4. Training data use during the testing phase

#Its high expensive algorithm, because its need all data for test, need high memory, prediction stage also slow O(N^2)

# How we choose neighbors
# 1. Brute Force
# 2. K-D Tree

# Steps
# 1. Defined positive integer K with new sample
# 2. Select K entries in our database are closest to the new sample
# 3. Find most common classification of this entries
# 4. Then give this classification to the new sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

fileIs = './input/Social_Network_Ads.csv'
dataFrame = pd.read_csv(fileIs)

print(dataFrame.head())

#SELECT X AND Y
#X: Age, Estimate Salary
#Y: Purchase, 0= yse and 1= no

X= dataFrame.iloc[:,[2,3]].values
Y = dataFrame.iloc[:,[4]].values



#Divide data to taring and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=0)

#Now normalize the data or distribute the data such that MEAN is 0 and Standard Deviation =1 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting model
knerestClassifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knerestClassifier.fit(X_train, Y_train)

#predict the result
predictResult = knerestClassifier.predict(X_test)

# print(len(Y_test), len(predictResult))

#check accuracy
acc = accuracy_score(Y_test, predictResult)
print('Accuracy score is: ', acc)

#check confusion metrics
confusion = confusion_matrix(predictResult, Y_test)
print(confusion)
