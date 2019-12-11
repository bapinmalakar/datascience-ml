#Make netflix recommendation system using Correlations/CF
# Recommend with Collaborative Filtering
# Recommend with Pearsons'R correlation

#Here we have 4 files
# Each file have 4-columnes, 
# movieId, customerId, rating(1,5), Date when they give rate
# another file have details of movie like name, year of release etc

import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plot
import seaborn as sns
from surprise import Reader, Dataset, SVD

from sklearn.model_selection import cross_validate # evalute deprecate and functinality replace in cross_validate

sns.set_style('darkgrid')

print('Step1 Setpup done')

#now load data, are txt file
#In txt file each line consider as row and each ',' seperated words are columns
dataFolderPath = './input/netflix_data/'
dataFrame1 = pd.read_csv(dataFolderPath+'combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
#give name to my columns names=['Cust_Id', 'Rating']
# usecols=[0,1]// consider only 2 columns and 2 columns are 0 and 1 indexes

print(dataFrame1.head())

#convert Rating field to float
dataFrame1['Rating'] = dataFrame1['Rating'].astype(float)
#Print table description, means number of row and columns
print('Description of DataFrame1 {}'.format(dataFrame1.shape))

#Now load other 3 files and and convert ration to float
dataFrame2=pd.read_csv(dataFolderPath+'combined_data_2.txt',header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
dataFrame3=pd.read_csv(dataFolderPath+'combined_data_3.txt',header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])
dataFrame4=pd.read_csv(dataFolderPath+'combined_data_4.txt',header=None, names=['Cust_Id', 'Rating'], usecols=[0,1])

dataFrame2['Rating'] = dataFrame2['Rating'].astype(float)
dataFrame3['Rating'] = dataFrame3['Rating'].astype(float)
dataFrame4['Rating'] = dataFrame4['Rating'].astype(float)

print('Description of DataFrame2 {}'.format(dataFrame2.shape))
print('Description of DataFrame3 {}'.format(dataFrame3.shape))
print('Description of DataFrame4 {}'.format(dataFrame4.shape))

#Combine all dataset and make one dataset
dataFrame = dataFrame1
dataFrame = dataFrame.append(dataFrame2)
#for loading issue consider only 2 dataset
# dataFrame = dataFrame.append(dataFrame3)
# dataFrame = dataFrame.append(dataFrame4)
print('Now total number of records {} and Shape is: {}'.format(len(dataFrame), dataFrame.shape))

print('Step 2 Loading file and DataFrame creation done')

#See total number of total count of each rating
# like Rating 200 user give rating 1, Rating 100 user give rating 5
p = dataFrame.groupby('Rating')['Rating'].agg(['count'])
print(p)

#get movie count, row which doesnot have rating means its movie
# find number of row have empty rating filed
movie_count = dataFrame.isnull().sum()[1] #rating field is second column
print(movie_count)

#get total number of customers
cust_count = dataFrame['Cust_Id'].nunique()
print('Total number of customer are: ', cust_count)

#get total number of rating
rating_count = dataFrame['Rating'].notnull().sum()
print(rating_count)

#Now plot rating distributions
ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plot.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plot.axis('off')

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')