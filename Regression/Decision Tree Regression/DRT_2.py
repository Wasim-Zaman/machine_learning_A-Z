'''
It is a tree-structured classifier with three types of nodes. 
The Root Node is the initial node which represents the entire sample and 
may get split further into further nodes. The Interior Nodes 
represent the features of a data set and the branches represent the decision 
rules. Finally, the Leaf Nodes represent the outcome. 
This algorithm is very useful for solving decision-related problems.


You might used decision trees in case when we talk with the helpline
and they say press 1 for that, 2 for that and so on. This is nothing but a
decision tree example, the intelligent machine let you know about the 
final correct person based on some tree of the decisions
'''

'''
With a particular data point, it is run completely through the entirely tree 
by answering True/False questions till it reaches the leaf node. 
The final prediction is the average of the value of the dependent variable in 
that particular leaf node. Through multiple iterations, 
the Tree is able to predict a proper value for the data point.
'''

# Step 1: Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Step 2: Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')

X = dataset['Temperature'].values.reshape(-1, 1)
y = dataset['Revenue'].values.reshape(-1)

# Step 3: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.05, random_state=42)

# Step 4: Training the Decision Tree Regression model on the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'mse', random_state= 0)
regressor.fit(X_train, y_train)

# Step 5: Predicting the Results
y_pred = regressor.predict(X_test)

# Step 6: Comparing the Real Values with Predicted Values

comparison_df = pd.DataFrame({"Real Values": y_test, 
                              'Predicted Values':y_pred})

# Visualising the Decision Tree Regression Results

plt.scatter (X_test, y_test, color = 'red')
plt.scatter (X_test, regressor.predict(X_test), color = 'blue')
plt.plot (X_test, regressor.predict(X_test), color = 'black')



X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color = 'black')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()


