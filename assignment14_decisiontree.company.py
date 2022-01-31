#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
import seaborn as sns


# In[2]:


company=pd.read_csv('C:\\Users\\Admin\\Downloads\\Company_Data.csv')
company.head()


# In[3]:


company.info()


# In[4]:


company.corr()


# In[5]:


sns.jointplot(company['Sales'],company['Income'])


# In[6]:


company.loc[company["Sales"] <= 10.00,"Sales1"]="Not High"
company.loc[company["Sales"] >= 10.01,"Sales1"]="High"


# In[7]:


company


# In[8]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[9]:


company["ShelveLoc"] = label_encoder.fit_transform(company["ShelveLoc"])
company["Urban"] = label_encoder.fit_transform(company["Urban"])
company["US"] = label_encoder.fit_transform(company["US"])
company["Sales1"] = label_encoder.fit_transform(company["Sales1"])


# In[10]:


company


# In[11]:


# Define x & y
x = company.iloc[:,1:11]
y = company['Sales1']


# In[12]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=50)


# In[13]:


# # Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[14]:


model.get_n_leaves()


# In[15]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[16]:


preds


# In[17]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[18]:


# Accuracy 
np.mean(preds==y_test)


# In[19]:


print(classification_report(preds,y_test))


# In[20]:


# # Building Decision Tree Classifier (CART) using Gini Criteria
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[21]:


model_gini.fit(x_train, y_train)


# In[22]:


model_gini.get_n_leaves()


# In[23]:


preds = model_gini.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[24]:


preds


# In[25]:


# Accuracy 
np.mean(preds==y_test)


# In[26]:


print(classification_report(preds,y_test))


# In[27]:


# # Building Decision Tree Regression 
from sklearn.tree import DecisionTreeRegressor


# In[28]:


model_R = DecisionTreeRegressor()
model_R.fit(x_train, y_train)


# In[29]:


preds = model_R.predict(x_test) 


# In[30]:


np.mean(preds==y_test)


# In[31]:


# # Plot Tree Diagram
# Decision Tree Classifier using Entropy Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model)


# In[32]:


# Decision Tree Classifier (CART) using Gini Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_gini)


# In[33]:


# Decision Tree Regression
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_R)


# In[ ]:




