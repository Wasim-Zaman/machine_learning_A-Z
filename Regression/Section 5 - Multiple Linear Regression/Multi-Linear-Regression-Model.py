# Multiple linear regression......
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the categorical data into numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
encoder = LabelEncoder()
X[:, 3] = encoder.fit_transform(X[:, 3])

# Creating dummy variables....
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)

X = ct.fit_transform(X)
# Avoid dummy variable trap
''' This step is aurometically done by the sklearn linear regression model,
but we can also manually do this step 
if we do not do this step, the machine learning linear regression model in the
sklearn will autometically do this step'''
X = X[:, 1:]

# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Creating linear regression model.....
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(X_train)






#############################################################################

''' How do we know that the above model is the optimal model that we can make
with the dataset that we have here.
WE know that when we create a model we create it using independent variable,
but what if there are some independent variables that are highly statisticaly
significant ----> that has great impact on the dependent variable and what if 
there are some independent variable that are not highly statisticaly 
significant at all.

Which means if we remove those non-statistically significant variable, still
we get some amazing predictions

our goal is to find the best group of independent variables that greatly 
impact the result of dependent varaible.

This impact can be positive or negative'''

# Building optimal model using backward elimination 

''' Again we will need the linear regression formula for this which is 
y = b0 + b1x1 + .... bnxn

in the above formula, b is not associated with the indepenedent variable x,
while it is associated, x0^0 = 1.

most of the machine learning libraries like linear regression model itself 
taken b0 into account but the library that we are using now does not take
this b0 into account, we will explicitly add the x0 independent variable.
'''

''' we are adding the column of x0 ^ 0 = 1, so that the library which we are
using for building optimal model can understand that there exist value for 
x0 also like other independent variable '''


X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis=1)

import statsmodels.regression.linear_model as sm

''' The first thing in order to build optimal model using backward
elimination is to create matrics of features and that would be our optimal
matrics of features '''

# X_opt = X[:, [0,1,2,3,4,5]]    #statistically significant

''' This matrics of features will contian only those xi, which will be 
having great impact on the value of y later on, and we will use this 
matrics of features to make our model optimal for the predictions '''

# Steps for backword elimination method for optimal model creation......


# TODO-1:- Select the significance level to stay in the model (SL = 0.05)

SL =0.05

# TODO-2:- Fit the full model with all possible predictors

X_opt = X[:, [0,1,2,3,4,5]].astype(float)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# TODO-3:- Consider the predictor with highest p-value
        # If p > SL then goto step 4 otherwise your model is ready

''' In order to check for the value of p, we have an awesome method 
known as summary which will tell us all the things '''

summary = regressor_OLS.summary()

# TODO-4:- Remove the predictor

X_opt = X[:, [0,3]].astype(float)


# TODO-5:- Fit your model without this variable.
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
summary = regressor_OLS.summary()

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
summary = regressor_OLS.summary()


''' Then you can plot graphs or compare values etc....'''
