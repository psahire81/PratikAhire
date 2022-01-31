#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire
# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


# load datasets
company = pd.read_csv("C:\\Users\\Admin\\Downloads\\Company_Data.csv")
company


# In[4]:


company.info()


# In[5]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[6]:


company["ShelveLoc"] = label_encoder.fit_transform(company["ShelveLoc"])
company["Urban"] = label_encoder.fit_transform(company["Urban"])
company["US"] = label_encoder.fit_transform(company["US"])


# In[7]:


company


# In[8]:


feature_cols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']


# In[9]:


company['High'] = company.Sales.map(lambda x: 1 if x>8 else 0)


# In[10]:


company


# In[11]:


x = company.drop(['Sales', 'High'], axis = 1)


# In[12]:


x = company[feature_cols]


# In[13]:


y = company.High


# In[14]:


print(x)


# In[15]:


print(y)


# In[16]:


# Splitting the data into the Training data and Test data
from sklearn.model_selection import train_test_split


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 40)


# In[18]:


print(x_train)


# In[19]:


print(y_train)


# In[20]:


print(x_test)


# In[21]:


print(y_test)


# In[22]:


# # Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[23]:


sc = StandardScaler()


# In[24]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[25]:


# # Training the Random Forest Classification model on the Training data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 40)
classifier.fit(x_train, y_train)


# In[26]:


classifier.fit(x_train, y_train)


# In[27]:


classifier.score(x_test, y_test)


# In[28]:


# # Predicting the Test set results
y_pred = classifier.predict(x_test)


# In[29]:


y_pred


# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[31]:


cm = confusion_matrix(y_test, y_pred)


# In[32]:


print(cm)


# In[33]:


accuracy_score(y_test, y_pred)


# In[34]:


classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(x_train, y_train)


# In[35]:


classifier.score(x_test, y_test)


# In[ ]:




