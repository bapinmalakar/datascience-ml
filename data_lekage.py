#data lekage:  happen when traing data contain information about target but prediction data does not contain
#it will lead high performance in traing but very poor performance in production
#two type of lekage: targetleakage and train-test contamination

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

filePath = './week_day.csv'
dataFrame = pd.read_csv(filePath)
print('Now info: ', dataFrame.shape)

y = dataFrame.likes

#find empty columns
colmnss = [col for col in dataFrame.columns if dataFrame[col].isnull().any()]

colmnss.append('likes')
colmnss.append('days')

print('Columns for delete: ', colmnss)


X= dataFrame.drop(colmnss, axis=1)

#shape contain information about no.of. rows and columns in tuple
print('Number of rows and columns ', X.shape[0], X.shape[1])

# # Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=2,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())