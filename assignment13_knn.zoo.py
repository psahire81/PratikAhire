#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


zoo=pd.read_csv('C:\\Users\\Admin\\Downloads\\Zoo.csv')
zoo


# In[3]:


zoo.info()


# In[4]:



from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
zoo["animal name"] = label_encoder.fit_transform(zoo["animal name"])


# In[5]:


zoo.head()


# In[6]:



array = zoo.values
X = array[:, 1:17]
X


# In[7]:


Y = array[:, -1]
Y


# In[8]:


kfold = KFold(n_splits=4)


# In[9]:



model = KNeighborsClassifier(n_neighbors=13)
results = cross_val_score(model, X, Y, cv=kfold)


# In[10]:


print(results.mean())
# # Grid Search for Algorithm Tuning


# In[11]:



import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[12]:



n_neighbors1 = numpy.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors1)


# In[13]:



model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[14]:



print(grid.best_score_)


# In[15]:


print(grid.best_params_)
# # visualize the CV results


# In[16]:



import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 70
k_range = range(1, 70)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=4)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[ ]:




