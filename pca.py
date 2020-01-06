#Principal Component Analysis
#Used to reduce the dimension of the data set
# In data set dimension are your features(X)
# In given dataset we have n number of interrelated features(dimension/Variables) which describe all the variation
# But we don't need all the dimension, we can acehive the by reducing the features
# If we ignore, then we may lost some important data.
# So, PCA will help us to do this. It will transform your dataset into new dimension
# By transform in PC's (Principal Component).
# And this component are in order, the First PC is the dimension which have largest variance
# PCA is Unsupervised learning algorithm

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print('Library imported')

#load data now
df = pd.read_csv('./input/pca_data.csv');
print(df.head())

#list all the column name to see
print(df.columns.tolist())

#Print total number of row and column
print(df.shape) # we have in total 11 columns/variable means 11 dimension dataset

#select x all the  and y
X=df.iloc[:, 2:]# all field accept number_people
y=df.iloc[:, 0]# only number_people
print(X.head())
print(y.head())

#see the correlation among the call olumns or variables by heatmap
correlation = X.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation between different fearures')


#Now scale the 
scale = StandardScaler()
X= scale.fit_transform(X)
print(X)

#Now find PC using PCA
pca = PCA()
new_x = pca.fit_transform(X)

#now get variance explanation
explained_variance=pca.explained_variance_ratio_
print(explained_variance)

#View the variance of each PC's
with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 10))

    plt.bar(range(9), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

#now we can use new_x for predict our data with y

