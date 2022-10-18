# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:05:56 2022

@author: Wasim Xaman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset from the file
dataset = pd.read_csv('MALL_Customers.csv')
'''
We want to find the clusters or similar groups in this dataset, for that
we need to find the X so that we can farther apply the algorithm Un-supervised
algorithm of the dataset
'''

X = dataset[['Annual Income (k$)','Spending Score (1-100)']].values

# Using the dendogram to find the optimal number of clusters.
'''
Same action plane is used as k-means to find the optimal number of clusters,
unlike k-means, here we are using the dendogram from scipy for finding the
optimal number of clusters.
'''

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward', metric='euclidean'))
'''
Linkage is the algorith for clustering itself, we specify X in its arguments 
so that we can apply this linkage algorithm of the dataset we want to find
dendogram of.

On the other hand ward is same as with in cluster sum of square, but here 
we minimize the varience with in each cluster, on the other hand in wcss,
we minimize the with in cluster sum of square, here we minimize with in cluster
varience.


unlike k-means, it does not require to use for loop for finding the optimal
number of clusters, the above line of code does everything for us, we only need
to specify the labels of the plots.
'''
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean-Distance')
plt.legend()
plt.show();

'''
Using the standard rule, we can now easily specify where to set the threshold
and how many clusters would be made later on....
The answer is 5....
'''

# Fitting the hierarchical clustering algorithm
'''
Since we have find the optimal number of clusters which is 5, now we are 
going to fit the hierarchical clustering algorithm with the right number of
clusters and this step is same as the k-means..

The first thing that we need to do is to import the right tool that is 
going to help in fitting the HC algorithm, 
but remember this tool was the k-mean class and here it is hierarchical 
clustering class.
As we know that there are two types, agglomerative and divisive.
'''

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', 
                             linkage='ward')
y_hc = hc.fit_predict(X)

'''
the affinity attribute shows the distance to do the linkage, here we are using
the euclidean distance, and the linkage we know is the ward which minimizez
the with in cluster varience.

Just like k-means, we use the fit_predict method to fit the algorithm and then
use the same code to plot the clusters and all data points on the screen
'''

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='green')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='orange')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
