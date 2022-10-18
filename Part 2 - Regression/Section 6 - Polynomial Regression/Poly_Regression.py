# TODO-1: Import all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
# TODO-2: Load the dataset
dataset = pd.read_csv("Position_Salaries.csv")

# TODO-3: Set the dependent vector and independent matrics of features...
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# TODO-4: Splitting the dataset into the training and test set 
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)'''

# TODO-5: Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""
    
# TODO-6: Fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

''' The lin_reg object is being fitted on the X and y values because 
we did not split the data into the training and test data '''
    
# TODO-7: Fit the polynomial regression to the dataset

'''
In order to create this polynomial regression model, we need to import 
new class so that we can add polynomial features to the dataset, that class
is located under the sklearn.preprocessing library and the name of the class
is polynomialfeatures '''
from sklearn.preprocessing import PolynomialFeatures

'''poly_features = PolynomialFeatures(degree=2)'''
'''poly_features = PolynomialFeatures(degree=3)'''
poly_features = PolynomialFeatures(degree=4)
'''
This poly_feature regressor is used to transform X into a new matrics of 
features called X_poly that will contain not only x1 independent variable 
but also polynomial features like x1^2, x1^3 etc depending upon the degree
we pass in the argument of the constructor 
'''
X_poly = poly_features.fit_transform(X)
    
# TODO-8: Fitting the multiple linear regression model on X_poly
'''
For that we need another linear regression model so that we do not confuse
ourselves with the first linear regression object
'''

lin_reg2 = LinearRegression().fit(X_poly, y)

# TODO-9: Visualizing the results of both two linear regression models...
    
## TODO-9.1: Visualizing the linear regression result
plt.scatter(X, y, color = 'black')

'''
to see the prediction of our model
'''
plt.plot(X, lin_reg.predict(X), color='red')

''' 
WE can conclude that this is not a good prediction except from some
one or two points or obsevations.
but from the other observations, our model linear regressin line is so far.
That is why we need to make better model (in this case a polynomial model)
that best fits the line
'''
    
## TODO-9.2: Visualizing the polynomial regression result
plt.scatter(X, y, color = 'red')
'''plt.plot(X, lin_reg2.predict(X_poly), color='black')'''
plt.plot(X, lin_reg2.predict(poly_features.fit_transform(X)), color='black')
'''
Instead of putting X_poly, we will put poly_features.fit_transform(X) in 
order to make our model generalized for new predictions, consider if new 
variable instead of X comes, then our model will predict incorrect result
for that variable if we put X_poly
'''

# TODO-12: Improving the plot by increasing step size of the prediction of X
X_increment = np.arange(min(X), max(X), 0.1)
X_increment = X_increment.reshape((len(X_increment), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_increment, lin_reg2.predict(poly_features.fit_transform(X_increment))
         , color='black')
plt.show()

# TODO-13: Predicting a new result.
'''
# Suppose we want to know about the a person who is currently in level 6,
but has spent 2 years in level 6 and it tooks 4 years to reach level 7 
which means that he is nearly in 6.5 level in the company, 
now what will be the salary of the person having level = 6.5
'''

## TODO-13.1: Predicting a new result with linear regression
'''
Just like the lin_reg object can predict 10 observations of X independent 
variable, simmilarly it can predict single dependent variable for one 
observation value of independent variable
But as the lin_reg object is being fitted by the matrics (2D array) of
features, for that while passing single value we need to pass it as 
matrics.
'''

'''lin_reg.predict(np.array([6.5]).reshape(1,1))'''
lin_reg.predict([[6.5]])
'''
We can see that this is not the relevent value from the graph
'''
## TODO-13.2: Predicting a new result with linear regression
'''lin_reg2.predict(poly_features.fit_transform(np.array([6.5]).reshape(1,1)))'''
lin_reg2.predict(poly_features.fit_transform([[6.5]]))