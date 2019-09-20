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

#color code scatter graph which will explin day of week analyze

weekFilePath='./week_day.csv'
weekShareLikeData = pd.read_csv(weekFilePath)
plt.title('Scatter graph with color week day explain')
sns.scatterplot(x=weekShareLikeData['shares'], y=weekShareLikeData['likes'], hue=weekShareLikeData['week_day'])
plt.xlabel('Shares')
plt.ylabel('Likes')

#add more regression line corresponding to the week day 1,2,3,4,5,6,7

sns.lmplot(x='shares', y="likes", hue="week_day", data=weekShareLikeData)

#catgorize scatter graph based on week_day likes and share relationship

sns.swarmplot(x=weekShareLikeData['week_day'],y=weekShareLikeData['likes'])

#histogram graph, display histograp chart of shares(group by numbers)
sns.distplot(a=weekShareLikeData['shares'], ked=False) # a, select column want to plot and kde is False for histogram data
#if kde ie true, then mix of density and histogram graph
#KDE kernal density estimate, smotth histogram
sns.kdeplot(data=weekShareLikeData['shares'], shade=True) # shade true, color the space below line

#we plot only one column in kde, we have 2 column also to see their effet and relationship
#we call this 2D-KDE(jointplot)
sns.jointplot(x=weekShareLikeData['shares'], y=weekShareLikeData['likes'], kind='kde') #kind, kde means density graph

#color coded histogram for mix data from different file
iris_set_filepath = "../input/iris_setosa.csv"
iris_ver_filepath = "../input/iris_versicolor.csv"
iris_vir_filepath = "../input/iris_virginica.csv"

# Read the files into variables 
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

#graph for each file data
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

#day going to combine together in single graph

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()

#we can do same for kde also


#design your graph
sns.set_style('dark') #the display plane will be colored as dark, darkgrid, whitegrid, white, ticks
plt.figure()
sns.lineplot()
