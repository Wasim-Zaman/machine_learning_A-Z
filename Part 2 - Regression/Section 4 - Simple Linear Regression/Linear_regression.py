import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The very first thing that we need is to import the dataset
dataset = pd.read_csv('Salary_Data.csv')

''' in order to find the correlation between the variables in the dataset,
 first we need to check graphically what are actually they look like '''
 
import seaborn as sns

#sns.pairplot(dataset)
#sns.heatmap(dataset.corr(), annot = True)
sns.distplot(dataset['YearsExperience'])

# Separating feature matrix and target vector
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]
# Train or Fit the regression model
''' In order to train the regression model, first we need library then we 
need to split the data into test and training dataset so that from the 
training dataset, we can fit our model and then from the test dataset,
we can test the correct behavior of the predicted values of the model '''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size = 1/3, random_state = 0)

# Feature scaling....
''' Keep in mind that the regression model autometically does the feature 
scaling for us, that is why we do not need to scale our variable into a 
specific range '''

# Fitting the linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

''' Fitting the model using training dataset because our model will learn
using training data in order to predict salary, the test data is used to 
check the accuracy of the same model '''

regressor.fit(X_train, y_train)

# Model Evaluation....

''' checking the slope and interception points '''
coeff = regressor.coef_
intercept = regressor.intercept_

''' Predicting the X_test results in order to compare whith the y_text '''
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

''' Now let's analyze the difference between the y_test and y_pred
to check the accuracy of our model.... and see how closer predictions 
to the actual values can our model made '''

plt.scatter(X_train, y_train, color='green') # actual observations

''' drawing a regression line (on the predicted values) '''
plt.plot(X_train, y_pred_train, color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()