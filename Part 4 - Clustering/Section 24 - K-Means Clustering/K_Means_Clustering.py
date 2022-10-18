# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:24:19 2022

@author: Wasim Xaman
"""

# Importing the required libraries.....

import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset for clustering
dataset = pd.read_csv('Mall_Customers.csv')

'''
We know that on the basis of last two columns, the company needs to 
cluster all the customers into different cluster, so grab these two columns
from the dataset....
'''

X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

'''
Now moving forward with the K-Mean, but
Since we do not know how many clusters would be made, for that of course we 
will use the {elbow method} for optimal number of clusters of our problem 
'''
# Finding the optimal number of clusters using elbow method...
'''
For that, first we need to import the KMean class form the sklearn..
'''

from sklearn.cluster import KMeans

# Let's plot the elbow graph for selecing optimal number of clusters
'''
In order to do that, we will find the {with in cluster sum of square} 
distences for 10 clusters, so that from that we can find the optimal number 
of cluseters. in order to find WCSS for 10 clusters, 
we will loop through 10 iteration and in each iteration we will need
to do two things.

1. Fit the KMean algotithm to data X
2. WE will compute the WCSS and append it to the list we create for ploting
later on

But before that, keep in mind that in sklearn the {WCSS} is also called 
Inertia
'''

wcss = []
for m in range (1,11):
    kmeans = KMeans(n_clusters= m, init= 'k-means++', n_init= 10,
                    max_iter= 300, random_state=0)
    '''
    The {init} variable show the random initialization method of the centroid
    since we want to avoid random initialization trap, that is why we are 
    using k-means++
    
    The max_iter parameter:- The max number of iterations of the algo when 
    it is running...
    
    The n_init parameter shows the number of times the algo runs with diff
    initial centroid.    
    '''
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)

# Fitting the dataset on the correct model with optimal No of centroid...
'''
Since we have now identified that the optimal number of centriod must be 
5, so now creating a model having 5 centroid....
'''

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,
                random_state=0)

'''
This time when we know the optimal number of clusters, we not only fit the 
model but also predict for each value of X, the y value (which is nothing
but the cluster number). the value of y shows which data point belongs to 
which cluster, as here we have 5 clusters, so the number starts from 0 on the 
way upto 4.                                                    
'''
y_kmeans = kmeans.fit_predict(X)

# visualize the clusters.....

'''
As the cluster numbers starts from zero, it means that our first cluster 
corresponds to ykmeans == 0 and so on
'''
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red',
            label='Cluster-1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='green',
            label='Cluster-2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='cyan',
            label='Cluster-3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='magenta',
            label='Cluster-4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='yellow',
            label='Cluster-5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]
            , s=300, c='orange', label='Clusters')

