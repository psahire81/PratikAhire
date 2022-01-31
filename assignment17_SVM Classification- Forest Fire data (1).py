#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[2]:



# Load data sets
fire = pd.read_csv("C:\\Users\\Admin\\Downloads\\forestfires.csv")
fire.head()


# In[3]:



fire.info()


# In[4]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[5]:



fire["month"] = label_encoder.fit_transform(fire["month"])
fire["day"] = label_encoder.fit_transform(fire["day"])
fire["size_category"] = label_encoder.fit_transform(fire["size_category"])


# In[6]:


fire.head()


# In[7]:


# Define X & y


# In[8]:



X=fire.iloc[:,:11]
X.head()


# In[9]:



y=fire["size_category"]
y.head


# In[10]:


# Split the Data intp Training Data and Test Data


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[12]:



clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10],'C':[15,14,13,12] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[13]:


gsv.best_params_ , gsv.best_score_


# In[14]:



clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[15]:



clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# In[16]:



clf2 = SVC()
param_grid = [{'kernel':['poly'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[17]:


gsv.best_params_ , gsv.best_score_


# In[18]:



clf3 = SVC()
param_grid = [{'kernel':['sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[19]:


gsv.best_params_ , gsv.best_score_


# In[ ]:




