#!/usr/bin/env python
# coding: utf-8

# # Solving classification problem using K-NN

# In[2]:


# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Grab the dataset 
dataset = pd.read_csv('Classified Data')

# checking the head of the dataset
dataset.head()


# # Scaling Features

# In[4]:


# importing libraries
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(dataset.drop('TARGET CLASS', axis=1))
scaled_features


# In[5]:


# Creating dataframe off of the scaled features
scaled_features_df = pd.DataFrame(data=scaled_features, columns=dataset.columns[:-1])
scaled_features_df.head()


# In[6]:


# Spliting the data into X and y
X = scaled_features

y = dataset['TARGET CLASS']


# In[115]:


# Spliting the data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Creating a model

# In[116]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# In[117]:


# Do some predictions on it
y_preds = knn.predict(X_test)


# In[118]:


# Making a confusion matrix for evaluation
from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_preds)
cr = classification_report(y_test, y_preds)


# In[119]:


# Evaluating the model using above metrices 
cm


# In[120]:


print(f'{cr}')


# # finding the ideal k-value using elbow method

# In[126]:


loss_rate = []

for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_preds = knn.predict(X_test)
    loss_rate.append(np.mean(y_preds != y_test))


# In[127]:


# creating dataframe for loss rate
loss_rate_df = pd.DataFrame(data=loss_rate, columns=['lOSS RATE'])


# In[128]:


loss_rate_df.tail()


# In[129]:


# Visualizing the loss rate in Elbow method to choose best k-value
plt.figure(figsize=(7,7))

plt.plot(range(1, 40), loss_rate_df, color='r', ls='--', marker='o', markerfacecolor='blue', markersize=13)
plt.xlabel('k-values')
plt.ylabel('loss rate')
plt.title('Elbow method')
plt.show();


# Since we can see that when the value for k is greater than 40, it has the lowest error rate

# In[133]:


# Creating a model again with best k value

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)

# Predicting the model for the sack of evaluation
y_preds = knn.predict(X_test)


# In[134]:


# Evaluation of knn using confusion matrix and comparing the old confusion matrix with the new model for new k value
sns.heatmap(cm, annot=(cm/cm.sum()*100))


# In[135]:


new_cm = confusion_matrix(y_test, y_preds)
sns.heatmap(new_cm, annot=new_cm/new_cm.sum() * 100)


# In[ ]:





# In[ ]:




