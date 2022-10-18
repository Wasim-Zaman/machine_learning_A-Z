# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:43:53 2022

@author: Wasim Xaman
"""

# Importing all the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading the cvs file data set
dataset = pd.read_csv('CC GENERAL.csv')
X = dataset.drop('CUST_ID', axis=1).values
#X = X[:3000,:]
X = X[:500,:]


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 14:15])
X[:, 14:15] = imputer.transform(X[:, 14:15])


# Using Elbow method to find the oprimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 31):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 31), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 15, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters

