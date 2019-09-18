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

