#use for better messurement of model performance

#cross-validation we run our model on different set of data, to get model quality
#it is useful, when you have very small data set, then try your model by different different data set to find quality your model
#cross_val_score from sklern.model_selection

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

filePath = './query_result.csv'
dataFrame = pd.read_csv(filePath)

X= dataFrame.likes
Y = dataFrame[['featured', 'views', 'shares']]

myPipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

#cv is number of flods
scores =1 * cross_val_score(myPipeline, X, Y,cv=3)

print('Score is: ', scores)

