import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X = scale_X.fit_transform(X)
y = scale_y.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

plt.scatter(X, y, color = 'orange')
plt.plot(X, regressor.predict(X), color = 'pink')
plt.show()

pred_new_level = regressor.predict(scale_X.transform(np.array([[6.5]])))
pred_new_level = scale_y.inverse_transform(pred_new_level)
