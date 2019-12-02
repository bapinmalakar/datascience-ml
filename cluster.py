#cluster by K-means
#cluster is unsupervised learning
#try to investigate structure of the data by grouping the data points into distinct subgroups
#K-means is iterative algorithm which try to partition the dataset into Kpre defined distinct non-overlaping subgroups(cluster)
#Where each data point belong to only one group
#steps
#1. Define number of clusters K
#2. Initialized centroid(avarage position of all points) by firts suffling and then randomly select K-data points for centroid without replacement
#3. Keep iterating untill no change in centroids
#4. Compute sum of he squared distance between data points and all centroids
#5. Assign each data points to closest cluster(centroid)
#6. Compuet centroid for the clusters by taking the avarage of all data points that belongs to each cluster

#K-means use the approach Expectation-Maximization
#Where
#E-steps to assign data points to the closest cluster
#M-steps to compute centroid of the each cluster

#Here we try to find the user_segemnt of a mall, which have details of a customers
#Customer id, age, gender, annual_income(in $), spending_score in 1-100

import pandas as pd
import numpy as np #for linear algebra
import matplotlib.pyplot as plt # for data visulization 
import seaborn as sn # library for visulization
from sklearn.cluster import KMeans # KMeans model

dataFile = './input/Mall_Customers.csv'

dataFrame = pd.read_csv(dataFile, index_col="CustomerID")

print(dataFrame.head())
print('Total data: ', dataFrame.shape) #no of row, columns

#get details of dataFrame like missing value, type if each column, memory usages
print(dataFrame.info())

##get total missing values of each column, if not then each column have 0
dataFrame.isnull().sum()

#for check each columne individual
dataFrame.Gender.isnull().sum() # no missing value in Gender then 0

#select features Annual imcome and spending score
# X= dataFrame[['Annual_Income', 'Spending_Score']]
#columne start from index 0
X= dataFrame.iloc[:, [2,3]].values
print(X[:10])

#craete model and process
#allow KMeans to select optimum K(k number of cluster)
#KMeans++ use Elbow() method to figure out K for KMeans
wcss = []

for i in range(1,11): #assume maximum number of cluster is 10
    kmeans = KMeans(n_clusters=1, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # methos for assign data point to cluster

#visulize the ELBO method to get optimal valu of K
plt.plot(range(1,11), wcss)
plt.title('ELBOW METHOD')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()
#in this plot we can see first ELOB in 2.5 and last Elob is 5
#No matter if increase range
# if we select big range, then processing power increased also not become visible

#from that graph we find our K, and K=5
#craete final model
finalKMeans = KMeans(n_clusters=5, init="k-means++", random_state=0)
y_kmeans = finalKMeans.fit_predict(X)
#for unsupervised learning we used fit_predict()
# and for supervised used fit_transform()
print(y_kmeans) #it will give cluster name for each data( cluster name are: 1,2,3,4,5 becuase we have 5 cluster) in array
print(finalKMeans.cluster_centers_)
# print centroid of each cluster in the form of array
# where we have 5 clister, so length of the array is 5

#plot the cluster for view
#first for X-axis value:  X[y_kmeans == 0, 0], first columne of features(Annual_Income) those are in cluster 1
# second for Y-axis value X[y_kmeans == 0, 1], second columne of features(Spending_Score) those are in cluster 1
# s=50, means marker size 50
# Repeat five time, because we have 5 cluster
# last plot to view center(centroid) of each cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label="cluster 2")
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label="cluster 3")
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label="cluster 4")
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label="cluster 5")
plt.scatter(finalKMeans.cluster_centers_[:, 0], finalKMeans.cluster_centers_[:, 1], s=300, c="yellow", label="Centroids")
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

