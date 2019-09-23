import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

##RandomForest try different tree component and then find best tree and give that predict

# create data frame
dataFrame = pd.read_csv('query_result.csv',  header=0, index_col=0)

# drop row have misvalues
dataFrame.dropna(axis=0)

# select target
y = dataFrame.likes

# select fetures
features_columnes = ['featured', 'views', 'shares']
x = dataFrame[features_columnes]

# split data into train and validate data set
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# create RandomForest model
new_modal = RandomForestRegressor(random_state=1)

# train modal with train data
new_modal.fit(train_x, train_y)

# predict modal with validate data
predict_result = new_modal.predict(val_x)
print(predict_result[:10])

#check MAE
mean_error_val = mean_absolute_error(val_y, predict_result)
print('MAE is:  ', mean_error_val) 


#evalute random forest model
#test_size = some_value(amount of data used for test if, they are in flot, then define the poprtion)
#train_size = 1 - some_value (amount of data used for test)
#train_size = 1-0.2
#train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=0)
#n_estimators: number of tree want to make, befor take decision
modal_1 = RandomForestRegressor(n_estimators=50)

#criterion: Measures the quality of each split criterion='mae', default gini
model_2 = RandomForestRegressor(n_estimators=100, criterion='mae')

#max_depth: define how depth you want to make your tree max_depth=20, default none
model_3 = RandomForestRegressor(n_estimators=200, max_depth=10)

#min_samples_spli: minimum number of sample must be present from your data, by default 2
model_4 = RandomForestRegressor(n_estimators=300, max_depth=10, criterion='mae', min_samples_split=4)

# write predicted data result into csv file
#output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
#output.to_csv('submission.csv', index=False)


