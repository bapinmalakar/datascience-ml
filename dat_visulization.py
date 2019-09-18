import pandas as pd
from IPython import get_ipython
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

print('setup complete')
warnings.filterwarnings('ignore')

# after set the setup
filePath = './like_share_days_wise.csv'
fileData = pd.read_csv(filePath, index_col='days', parse_dates=True)
print(fileData.head())


#now plot data data
#set height and width
plt.figure(figsize=(16,10))
plt.title('Yearly like and share graph')

#draw line graph
sns.lineplot(data=fileData)

#set label for horizontal axis
plt.xlabel('Day usages')

#select specific field show on graph
plt.figure(figsize=(16,10))
plt.title('Yearly like and share graph')

sns.lineplot(data=fileData['shares'])  #only disply share column data in graph

#want give label for selected field in graph
sns.lineplot(data=['reads'], label='viewer')

#bar graph
#show the day wise like getting
dataForBarChart = pd.read_csv(filePath, index_col='days') #based on days
plt.figure(figsize=(10,10)) #width, height
plt.title('Day wise like getting')
plt.ylim(10, 40000) #y-axis interval is fro 10 to 40000
sns.barplot(x=dataForBarChart.index, y=dataForBarChart['likes'])

plt.ylabel('like getting')
plt.xlabel('days')

#heat map
plt.figure(figsize=(14,7))
plt.title('heat map for likes, shares and views day wise')
sns.heatmap(data=dataForBarChart, annot=True)
plt.xlabel('Analyze data')

#scatter plot
scatterData = pd.read_csv(filePath)
plt.title('scatter graph x=share, y=like')
sns.scatterplot(x=scatterData['shares'], y=scatterData['likes'])
plt.xlabel('Shares')
plt.ylabel('Likes')

#regression plot describe more scatter (create lines)
plt.title('Regression plot of same share and like')
sns.regplot(x=scatterData['shares'], y=scatterData['likes'])
plt.xlabel('Shares')
plt.ylabel('Likes')



