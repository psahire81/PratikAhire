#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire
# In[16]:


# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[18]:


# Load datasets
data = pd.read_csv("C:\\Users\\Admin\\Downloads\\glass.csv")
data


# In[20]:


data.info()


# In[21]:



array = data.values
X = array[:, 0:9]
X


# In[22]:



Y = array[:, 9]
Y


# In[23]:


kfold = KFold(n_splits=10)


# In[24]:



model = KNeighborsClassifier(n_neighbors=18)
results = cross_val_score(model, X, Y, cv=kfold)


# In[25]:



print(results.mean())
# # Grid Search for Algorithm Tuning


# In[26]:


import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[27]:



n_neighbors1 = numpy.array(range(1,80))
param_grid = dict(n_neighbors=n_neighbors1)


# In[28]:



model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[29]:


print(grid.best_score_)


# In[30]:


print(grid.best_params_)


# In[31]:



import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 80
k_range = range(1, 80)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[ ]:




