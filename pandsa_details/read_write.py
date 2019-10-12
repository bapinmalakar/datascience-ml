import pandas as pd

#DataFrame, Series

#DataFrame is a table, its contains array of individual entries and each entry correspondes to a roe and a column

dataFrameDemo = pd.DataFrame({'name': ['Biplab', 'Sukanta', 'Puja'], 'age': ['27', '25', '20']})

#name and age are column names, each item in array respect to each row of a column and index start with default value 0
#DataFrame() constructor is used to create new DataFrame and its accept dictionary

print(dataFrameDemo)

#we can also defind our own index by paasing value to the 'index' parameter
dataFrameDemo = pd.DataFrame({'name': ['Biplab', 'Sukanta', 'Puja'], 'age': ['27', '25', '20']}, index=['123b', '345s', '567p'])
print(dataFrameDemo)
#another way to create
dataFrameDemo=pd.DataFrame([['biplab', 27], ['sukanta', 25], ['puja', 20]], columns=['name', 'age'])
print('another way to craete \n ', dataFrameDemo)
print('name, age only are: \n', dataFrameDemo[['name', 'age']])
#series
#a series is a list, 
seriesDemo = pd.Series(['Biplab', 'Sukanta', 'Puja']) #index will start with default value 0
print('\n', seriesDemo)
#we can also say series is the single column of DataFrame, series does not have column name, it only contaion one colume
#we can also assing name to the series

seriesDemo = pd.Series(['Biplab', 'Sukanta', 'Puja'], index=['name_1', 'name_2', 'name_3'], name="children_names")
print(seriesDemo)

#craete series from dataFrame
seriesDemo = pd.Series(dataFrameDemo['age'].values, index=dataFrameDemo['name'], name="age_details")
print('\n\n',seriesDemo)

#get dataFrame row and columns information
print('dataFrame contain rows and columns: \n',dataFrameDemo.shape)


#when we rad csv file, then csv file by default have own index. And panda not read this index automatically
#to read index index_col=0, index is column 0

csvDataFrame =pd.read_csv('./../query_result.csv', index_col=0)
print(csvDataFrame.head())

#write dataframe to csv or save in disk
dataFrameDemo.to_csv('dataFrameDemo.csv')