import pandas as pd

dataFrame = pd.read_csv('./../query_result.csv', index_col=0)

#know the datatype of filed likes
print(dataFrame.likes.dtype)

#know the datatype of all the fileds in dataFrame
print(dataFrame.dtypes)

#change a columnt type(i.e likes from int64 to float64) here type is not changed but value changed
print(dataFrame.likes.astype('float64'))
print('Now type is:  ', dataFrame.likes.dtype)

#if we allow index in dataFrame then it also have datatype
print(dataFrame.index.dtype)

#get all values those are null, or NaN in dataFrame
print(dataFrame[pd.isnull(dataFrame.views)])

#get all values or row does not contatin null values
print(dataFrame[pd.notnull(dataFrame.views)])

#fill all null or NaN value with a value of a field
print('Replace values: \n', dataFrame.views.fillna('no value'))
#also replace null value with first non-null value, known as backfill strategy

#replace some values with another values
#i.e replace 140 with value 'value change'
changedDataFrame = dataFrame.format_name.replace('140','value change')
print('chnaged value: \n', changedDataFrame)


#get total number of rows have null value in filed shares
totalNumber = dataFrame[pd.notnull(dataFrame.shares)].shape[0]
print(totalNumber)

#or
totalNumber = len(dataFrame[pd.notnull(dataFrame.shares)])
print(totalNumber)

#or
totalNumber = dataFrame.shares.notnull().sum()
print(totalNumber)