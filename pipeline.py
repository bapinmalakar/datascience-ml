#simple way to bundle data preprocessing and modeling code, to run your code in single step
#clear code, fewer bugs, easier to productionized, model validation
#package is "from sklearn.pipeline import Pipeline"

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

filePath = './query_result.csv'

dataFrame = pd.read_csv(filePath)
Y = dataFrame.likes
X = dataFrame[['featured', 'views', 'shares']]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size= 0.2, train_size= 0.8, random_state= 1)

#find numarical columns
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
print('numarical column: ', numerical_cols)

#find categorial columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype=='object']
print('categorical_cols: ', categorical_cols)


#step 1, define preprocessing steps
#for bundels preprocessing data and modeling step we use ColumnTransformer
#impute numarical missing data
#impute missing values and one-hot encoding for categorial data

#preprocessing for numarical missing value
numarical_transformer = SimpleImputer(strategy="constant")


#preprocessing for categorial data and missing value
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#bundle preprocessing numarical and categorial data
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numarical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

#create model
model = RandomForestRegressor(n_estimators=20, random_state=0)

#bundle preprocessing and model
myPipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ('model', model)
])


#fit training data

myPipeline.fit(X_train, Y_train)

predict = myPipeline.predict(X_valid)

mae = mean_absolute_error(Y_valid, predict)

print('MAE Score is: ', mae)

#save result in file
output = pd.DataFrame({'id': X_valid.index, 'predict': predict})
print(output.head())

#save in excel file
output.to_csv('output_data.csv', index=False)