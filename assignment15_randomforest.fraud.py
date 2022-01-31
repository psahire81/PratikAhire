#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[33]:


# Load datasets
fraud = pd.read_csv("C:\\Users\\Admin\\Downloads\\Fraud_check.csv")
fraud


# In[34]:


fraud.info()


# In[35]:


fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
fraud["TaxInc"]


# In[36]:


fraudcheck = fraud.drop(columns=["Taxable.Income"])
fraudcheck


# In[37]:


FC = pd.get_dummies(fraudcheck .drop(columns = ["TaxInc"]))


# In[38]:


Fraud_final = pd.concat([FC,fraudcheck ["TaxInc"]], axis = 1)


# In[39]:


colnames = list(Fraud_final.columns)
colnames


# In[40]:


predictors = colnames[:9]
predictors


# In[41]:


target = colnames[9]
target


# In[42]:


X = Fraud_final[predictors]
X.shape


# In[43]:


Y = Fraud_final[target]
Y


# In[44]:


# Splitting the data into the Training data and Test data
from sklearn.model_selection import train_test_split


# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 40)


# In[46]:


# # Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[47]:


sc = StandardScaler()


# In[48]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 40)
classifier.fit(X_train, Y_train)


# In[50]:


classifier.fit(X_train, Y_train)


# In[51]:


classifier.score(X_test, Y_test)


# In[52]:


y_pred = classifier.predict(X_test)


# In[53]:


y_pred


# In[54]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[55]:


cm = confusion_matrix(Y_test, y_pred)


# In[56]:


print(cm)


# In[57]:


accuracy_score(Y_test, y_pred)


# In[62]:


classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(X_train, Y_train)


# In[64]:


classifier.score(X_test, Y_test)


# In[65]:


y_pred = classifier.predict(X_test)


# In[66]:


y_pred


# In[67]:


cm = confusion_matrix(Y_test, y_pred)
print(cm)


# In[68]:


accuracy_score(Y_test, y_pred)


# In[ ]:




