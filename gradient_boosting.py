#another essemble model(RandomForest also) is "Gradient Boosting"

#Innitial Model(naive model)=>Calulate prediction=>Loose function=>add new model to ensemble<=Train new model based on loose function=> repeat cycle
#XGBoost stands for extreme gradient boosting
#XGBoost is best for data in the from of table

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

filePath = './query_result.csv'
dataFrame = pd.read_csv(filePath)

y= dataFrame.likes
X = dataFrame[['featured', 'views', 'shares']]

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

myModel = XGBRegressor()

myModel.fit(X_train, y_train)

predictionIs = myModel.predict(X_valid)

print('Without parameter MAE Score is: ', mean_absolute_error(predictionIs, y_valid))

#gradien boosting parameters
#n_estimators: how may times cycle should be repeat, n_estimators=number of model in enssemble(normally 100-1000)
myModel= XGBRegressor(n_estimators=300)
myModel.fit(X_train, y_train)
predictionIs = myModel.predict(X_valid)

print('With 300 model MAE Score is: ', mean_absolute_error(y_valid, predictionIs))


#early_stopping_rounds: allow automacally find ideal n_estimators value, it will stop iterating when model stop improving
#good to use n_estimators with high value and the early_stopping_round for automacaaly stop when optimal value found
#early_stopping_rounds=10, means atleast 5 cycle should be done
#eval_set, define evalute score based on what data

myModel= XGBRegressor(n_estimators=300)
myModel.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)], verbose=False)
predictionIs = myModel.predict(X_valid)

print('With early_stopping_round model MAE Score is: ', mean_absolute_error(y_valid, predictionIs))

#learning_rate: directly added prediction we can multiply prediction with small value befor adding
#small learning rate and high n_estimators lead accuracy
#by default learnig_rate=0.1
myModel= XGBRegressor(n_estimators=1000, learning_rate=0.05)
myModel.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
predictionIs = myModel.predict(X_valid)

print('With learning rate model MAE Score is: ', mean_absolute_error(y_valid, predictionIs))

#for lager data set we can use paraller tasking to create models for fasteest processing, n_jobs
#n_jobs=4, no_of_cores in machine
#nothing to do with model imporvment, only make fast processing
myModel= XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
myModel.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
predictionIs = myModel.predict(X_valid)

print('With Prallel model creation model MAE Score is: ', mean_absolute_error(y_valid, predictionIs))