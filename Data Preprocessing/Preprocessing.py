# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 19:42:43 2022

@author: Wasim Xaman
"""

# TODO-1: import all the suitable packages
import numpy as np
import pandas as pd


# TODO-2: importing the appropriate dataset
dataset = pd.read_csv('Data.csv')

# TODO-3: Creating X, and y where X is the features of independent variables
    # and y is the vector of dependent variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# TODO-4: Encoding cathegorical data
from sklearn.preprocessing import LabelEncoder
encoder_x = LabelEncoder()
X[:, 0] = encoder_x.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto', drop='first'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)

# TODO-5: Changing the dtype of the arrays
X = np.array(X, dtype=np.float32)

# TODO-6: Taking care of missing data....
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# TODO-7: Splitting the data into training and test datasets....
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)
