# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 18:40:17 2022

@author: Wasim Xaman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion= 'mse', random_state=(0))
regressor.fit(X, y)

regressor.predict(np.array([[6.5]]))

plt.scatter(X, y, color='black')
plt.plot(X, regressor.predict(X), color='blue')

'''
AS the decision tree algorithm splits the data into many number of slices, 
and for the prediction of dependent variable, this algorithm takes the 
average of all the points lie in that slice where the x is matching.

That is why in this case, we have total 10 data points but in 1 dimension,
so this algorithm divides the dataset into 10 slices but as every slice
contains only one point, that is why this algorithm cannot take average 
of different slices, which means that some is wrong in the above model.

If there were average, there should have vertical line in each slice because
the y value (average) must be same for the whole slice, but we are seeing
horizontal lines which means that this one is not the correct graph

In order to correct it, we increase the number of points by increasing its 
step so that we can take average at every slice and so that the model 
can easily predict y value depending upon the area or the slice in which 
it comes.
'''

X_increment = np.arange(min(X), max(X), .01).reshape(-1, 1)
plt.scatter(X, y, color='black')
plt.plot(X_increment, regressor.predict(X_increment), color='blue')

