import pandas as pd

fileIs = './../query_result.csv'

dataFrameIs = pd.read_csv(fileIs, index_col=0)
print(dataFrameIs.head())

#select specific column
print('By array notation \n', dataFrameIs['likes'])
print('By dot notation \n', dataFrameIs.likes)

#get first value of likes
print('First likes value: ', dataFrameIs['likes'].values[0])

#indexing, loc(label based selectiom) and iloc(index based selection)
#row first and then column

#select first row
print('\n\n', dataFrameIs.iloc[0])

#select first row and column format_name
index_one = dataFrameIs.iloc[[0], 1]
print('first and colun 1\n', index_one)

#select all rows and column will be format_name
index_one = dataFrameIs.iloc[:, 1]
print('\n\n', index_one)

#select first 3 row and column format_name
index_one = dataFrameIs.iloc[:3, 1]
print(index_one)

print('Done')

#negative number start from last

ind_lbl = dataFrameIs.loc[:, 'likes']
print(ind_lbl) #return all row with like column value

#first row only
ind_lbl = dataFrameIs.loc[:1, ['likes', 'format_name']]
print(ind_lbl)

#set index, means i don't want default index
dataFrameIs.set_index('likes')
print(dataFrameIs.head())


#conditional selection
poemDataAre = dataFrameIs.format_name == 'poem'
print(poemDataAre) #return all ids with true and false
#false, format_name != poem and true, format_name == poem

#get all data where format_name is poem
print('only poem data loc \n\n', dataFrameIs.loc[dataFrameIs.format_name == 'poem'])

#get all data where format_name is poem and liks is <50
print('only poem data loc \n\n', dataFrameIs.loc[(dataFrameIs.format_name == 'poem') & (dataFrameIs.likes<50)])

#isin(), get all data where format_name is poem or 140
print('only poem data isin \n\n', dataFrameIs.loc[dataFrameIs.format_name.isin(['poem', '140'])])
#notnull also

#assigning value or add new field with value
dataFrameIs['add_vale'] = 'demo value'
print(dataFrameIs.head())
