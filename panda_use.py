import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print('Hello, Panad and SkLearn Demo!')

def getMAE(max_leaf_node, train_x, train_y, val_x, val_y):
    model_with_leaf = DecisionTreeRegressor(max_leaf_nodes=max_leaf_node, random_state=0)
    model_with_leaf.fit(train_x, train_y)
    predict_val = model_with_leaf.predict(val_x)
    return mean_absolute_error(val_y, predict_val)

dataFrame = pd.read_csv('query_result.csv',  header=0,index_col=0)

#drop row contain misvalues
dataFrame.dropna(axis=0)

descibeData = dataFrame.describe() #descibe data in count, mean, std,max

columns = dataFrame.columns #list all columns of the data frame

#select target, which one want to predict(ie: like)
y = dataFrame.likes

#select features, set of columns which use to predict target
features_columnes = ['featured', 'views', 'shares']
x=dataFrame[features_columnes]

#create decisiontree modal
decisiontree_modal = DecisionTreeRegressor(random_state=1)

#fit the modal
decisiontree_modal.fit(x,y) #train your model with some data

#now predict top 5 record only
predict5Values = decisiontree_modal.predict(x.head()) #evalute predict on train data

print('Predict result of each row is: ', predict5Values)

#Now we want to evalute modal, to determind how mouch acuurate. For that
#1. Train data and predict data sould be different, so we can split data from two dataFrame
#2. Calculate MAE(Mean absolute err), error=Actual_value-predictvalue and then find mean

#split data into two part, one is for train and another one from validate modal
train_x, val_x, train_y, val_y = train_test_split(x,y, random_state=1)

#create new modal
new_modal = DecisionTreeRegressor(random_state=1)

#train modal with tarin data
new_modal.fit(train_x, train_y)

#predict with validate data
predict_val = new_modal.predict(val_x)

#find MAE (avarage errors)
mae_error = mean_absolute_error(val_y, predict_val)

print('Actual val: ', val_y.head())
print('Predict value: ', predict_val[:5])
print('Modal MAE is: ', mae_error)

#now we have MAE, but for best case of underfit and overfit means find tree size which will give best result
#now find best tree size
consider_leaf_amount = [5, 50, 500, 1500, 5000]

scores = {leaf_node: getMAE(leaf_node, train_x, train_y, val_x, val_y) for leaf_node in consider_leaf_amount}

print('scores are ', scores)

#find best leaf size, means find min MAE which one give
best_leaf_size = min(scores, key=scores.get)
print('Best tree size is: ', best_leaf_size)

#now we get bset tree size means best fit between under and over fit, now find the predict
best_leaf_modal = DecisionTreeRegressor(max_leaf_nodes=best_leaf_size, random_state=0)
best_leaf_modal.fit(x,y)
best_predict = best_leaf_modal.predict(x)
print(best_predict[:5])

