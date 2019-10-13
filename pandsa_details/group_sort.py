import pandas as pd

filePath = './../query_result.csv'
dataFrame = pd.read_csv(filePath, index_col=0)

#groupby, group dat based on the value
print(dataFrame.groupby('likes').likes.count())

#find the repetation of each format
print(dataFrame.groupby('format_name').format_name.count())

#value_counts() is the shortcut of groupby()

#get min value of each identical like
print(dataFrame.groupby('likes').likes.min())

#groupby(), using multiple field
print(dataFrame.groupby(['likes', 'format_name']).likes.count())

#agg(), allow to execute multiple function in groupby, like build statistics
dataAre= dataFrame.groupby(['likes', 'format_name']).likes.agg([len, min, max])
print(dataAre)

#print its index, it is multi-index
indexAre = dataAre.index
print(indexAre)
print(type(indexAre))

#reset index
dataAre = dataAre.reset_index()

#sort_values(by=column_name),sort data if data im multi_index then reset first
print(type(dataAre.index))
print('Sort bt length(No.of time repeat): \n', dataAre.sort_values(by='len')) #by length
print('Sort by format name\n', dataAre.sort_values(by='format_name')) #by format name

#by default sort_ alues() work on accending order, we want decending order
print('Sort by length in Descending order\n', dataAre.sort_values(by='len', ascending=False))

#sort by index value, sort_index()
print('Sort by index, default is ascending order: \n', dataAre.sort_index())

#sort by more tahn one column
print('Sort by format and likes: \n', dataAre.sort_values(by=['likes','format_name']))