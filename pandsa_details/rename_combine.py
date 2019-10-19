import pandas as pd

dataFrame = pd.read_csv('./../query_result.csv', index_col=0)

#rename views to reader, we can do this by column name or by  index

#column name
readerFrame = dataFrame.rename(columns={'views': 'reader'})
print(readerFrame.head())

#by index field value, 
readerFrame = dataFrame.rename(index={3: 'language', 5: 'reader'})
print(readerFrame.head())

#givae a own name to index filed
newIndexName = dataFrame.rename_axis('index_filed', axis="columns")
print(newIndexName)

#give row name to index filed
newIndexName = dataFrame.rename_axis("wines", axis='rows')
print('newIndexName: \n', newIndexName)

#give both column and row name to index filed
newIndexName = dataFrame.rename_axis('index_fileld', axis="columns").rename_axis("wines", axis='rows')

#for combine two or more datafrmae together we can use

#leftFrame.set_index('column_name').join(rightFrame.set_index('column_name'), lsuffix='lf', rsuffix='rf')
#lsuffix means add this value to all columns of left table and rsuffix same for right table

#concat([leftTable, rightTable])

#and we can also use merge()