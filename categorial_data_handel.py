# handeling categorial data

# 1. Remove colume which have categorial data, if data is not important for informartion
# 2. Lable, label each categorial with unique value, like(how hard is question) easy:1, very easy: 2, hard: 3, very hard: 4, dont no: 0
# 3. One-hot encoding, create new columns to indicate present and absent of each value
# if categorial data is easy, good, hard then 3 column will create for each row to indicate their present and absent
# one-hot, normally not use whent categorial data is very large

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def scoreOfAModel(model, train_x, train_y, val_x, val_y):
    model.fit(train_x, train_y)
    predictResult = model.predict(val_x)
    return mean_absolute_error(val_y, predictResult)


filePathIs = './query_result.csv'
dataFrame = pd.read_csv(filePathIs)
dataFrame.dropna(axis=0)  # drop row which have empty value

columns = [col for col in dataFrame.columns if dataFrame[col].isnull().any()]
print('empty columns : ', columns)

Y = dataFrame.likes
features = ['format_name', 'verified', 'views', 'shares']
X = dataFrame[features]
modelIs = RandomForestRegressor(n_estimators=20, max_depth=10)

x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=1)

print(x_train.head())

# get list of columns have  categorical values
s = (x_train.dtypes == 'object')
columnsAre = list(s[s].index)
print('columns contain categorical values: ', columnsAre)

# apply approach 1, remove column have categorial value select_dtypes()
drop_x_train = x_train.select_dtypes(exclude=['object'])
drop_x_val = x_val.select_dtypes(exclude=['object'])
print('MAE with delete columns: ', scoreOfAModel(
    modelIs, drop_x_train, y_train, drop_x_val, y_val))

# apply approach 2, add unique value for each categorical value,  LabelEncoder()
train_x2 = x_train.copy()
val_x2 = x_val.copy()
label_encoder = LabelEncoder()

for col in columnsAre:
    train_x2[col] = label_encoder.fit_transform(x_train[col])
    val_x2[col] = label_encoder.transform(x_val[col])
print('Lable data are: ', train_x2.head())

print('MAE with approach 2: ', scoreOfAModel(modelIs, train_x2, y_train, val_x2, y_val))

#apply approach 3
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[columnsAre]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(x_val[columnsAre]))

# One-hot encoding removed index; put it back
OH_cols_train.index = x_train.index
OH_cols_valid.index = x_val.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = x_train.drop(columnsAre, axis=1)
num_X_valid = x_val.drop(columnsAre, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print('Approach 3 data: \n', OH_X_train.head())

print('MAE with approach 3 is: ', scoreOfAModel(modelIs, OH_X_train, y_train, OH_X_valid, y_val))