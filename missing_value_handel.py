# delete column, have missing value
# Imputation, best approach fill the missing value cell with minimum value
# an extension of Imputation, add another column which will indeciate that this raw has missing value and fill up with default value

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


def scoreOfAModel(model, t_x, t_y, v_x, v_y):
    model.fit(t_x, t_y)
    prediction_values = model.predict(v_x)

    mae_error = mean_absolute_error(v_y, prediction_values)
    return mae_error


dataFile = './query_result.csv'
dataFrame = pd.read_csv(dataFile)
dataFrame.dropna(axis=0)

Y = dataFrame.likes  # target
features = ['featured', 'views', 'shares']
X = dataFrame[features]

train_x, val_x, train_y, val_y = train_test_split(
    X, Y, test_size=0.2, train_size=0.8, random_state=3)

random_forest_modal = RandomForestRegressor(
    n_estimators=20, max_depth=4, random_state=0)

# random_forest_modal.fit(train_x, train_y)
# predict_result = random_forest_modal.predict(val_x)

# print(predict_result[:10])

# get name of the columns with missing value
columns_to_drop = [
    col for col in train_x.columns if train_x[col].isnull().any()]
print(columns_to_drop)

# #drop column have missing value
new_train_x = train_x.drop(columns_to_drop, axis=1)
new_val_x = val_x.drop(columns_to_drop, axis=1)
print('delete columne contain emty value: then MAE=>  ', scoreOfAModel(random_forest_modal, new_train_x, train_y, new_val_x, val_y))

# simple imputation, replce empty ceil with mean value
my_imputation = SimpleImputer()
impute_train_x = pd.DataFrame(my_imputation.fit_transform(train_x))
impute_val_x = pd.DataFrame(my_imputation.transform(val_x))

#imputation solved it, put back
impute_train_x.columns = train_x.columns
impute_val_x.columns = val_x.columns

print('with imputation MAE:  ', scoreOfAModel( random_forest_modal, impute_train_x, train_y, impute_val_x, val_y))

#impute and add new column to identify which row impute
X_train_plus = train_x.copy()
X_valid_plus = val_x.copy()

# Make new columns indicating what will be imputed
for col in columns_to_drop:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.fit_transform(X_valid_plus))

imputed_X_train_plus.colmns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print('Imputation with new column: MAE: ', scoreOfAModel(random_forest_modal,imputed_X_train_plus, train_y, imputed_X_valid_plus, val_y))