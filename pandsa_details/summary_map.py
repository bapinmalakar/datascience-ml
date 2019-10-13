import pandas as pd

filePath= './../query_result.csv'
dataFrame = pd.read_csv(filePath, index_col=0)

#describe(), return highlevel summary of a column, like count, mean, min value, max value, dtype etc.
print(dataFrame.likes.describe())

#mean(), return mean value of a column
print('Avg value of likes: ', dataFrame.likes.mean())

#median(), get median value of a column
print('Median likes is: ', dataFrame.likes.median())

#max(), min(), return max and min value of a column
print('Max and Min like is: ', dataFrame.likes.max(), dataFrame.likes.min())

#count(), shape
print('Total number of likes value are : ', dataFrame.likes.count(), dataFrame.shape[0])

#unique(), list of unique value
print('All unique values are: ', dataFrame.format_name.unique())

#value_counts(), to see list of unique value how offens they occure
print('All format how mouch occured: ', dataFrame.format_name.value_counts())

#map, take one set of values and then transfer it to another set of values

review_likes = dataFrame.likes.mean()
remean_like = dataFrame.likes.map(lambda p: p - review_likes) # we can also do this by  dataFrame.likes - review_likes
print('Re mean likes is: \n')
print(remean_like)

def remeainCalc(row):
    row.likes = row.likes - review_likes
    return row

#apply, same as map but its allow to pass custom method to treat each row individually
valIs = dataFrame.apply(remeainCalc, axis="columns")
print(valIs)
#map and apply return new Series or DataFrame, they never update original data

#concat two field
print(dataFrame.format_name + '--' + dataFrame.verified)