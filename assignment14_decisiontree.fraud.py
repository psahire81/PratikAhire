#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire

# In[7]:


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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[8]:


fraud=pd.read_csv('C:\\Users\\Admin\\Downloads\\Fraud_check.csv')
fraud.head()


# In[9]:


fraud.info()


# In[10]:


fraud.corr()


# In[11]:


#Fraud_check.loc[Fraud_check["Taxable.Income"]!="Good","Taxable_Income"]="Risky"
fraud.loc[fraud["Taxable.Income"] <= 30000,"Taxable_Income"]="Good"
fraud.loc[fraud["Taxable.Income"] > 30001,"Taxable_Income"]="Risky"


# In[12]:


fraud


# In[13]:


# # Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[14]:


fraud["Undergrad"] = label_encoder.fit_transform(fraud["Undergrad"])
fraud["Marital.Status"] = label_encoder.fit_transform(fraud["Marital.Status"])
fraud["Urban"] = label_encoder.fit_transform(fraud["Urban"])
fraud["Taxable_Income"] = label_encoder.fit_transform(fraud["Taxable_Income"])


# In[15]:


fraud


# In[16]:


fraud.drop(['City.Population'],axis=1,inplace=True)
fraud.drop(['Taxable.Income'],axis=1,inplace=True)


# In[17]:


fraud["Taxable_Income"].unique()


# In[18]:


fraud


# In[19]:


# Define x 
x = fraud.iloc[:,0:4]
x


# In[20]:


# Define y
y = y = fraud["Taxable_Income"]
y


# In[21]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[22]:


# # Building Decision Tree Classifier using Entropy Criteria
model = RandomForestClassifier(n_estimators=100, max_features=3)
model.fit(x_train,y_train)


# In[23]:


model.get_n_leaves()


# In[22]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category


# In[23]:


preds


# In[24]:


# Accuracy 
np.mean(preds==y_test)


# In[25]:


print(classification_report(preds,y_test))


# In[26]:


# # Building Decision Tree Classifier (CART) using Gini Criteria
model_gini = DecisionTreeClassifier(criterion='gini')


# In[27]:


model_gini.fit(x_train, y_train)


# In[28]:


model_gini.get_n_leaves()


# In[29]:


preds = model_gini.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[30]:


preds


# In[31]:


# Accuracy 
np.mean(preds==y_test)


# In[32]:


print(classification_report(preds,y_test))


# In[33]:


# # Building Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[34]:


model_R = DecisionTreeRegressor()
model_R.fit(x_train, y_train)


# In[35]:


preds = model_R.predict(x_test) 


# In[36]:


np.mean(preds==y_test)


# In[37]:


# # Plot Tree Diagram
# Decision Tree Classifier using Entropy Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model)


# In[38]:


# Decision Tree Classifier (CART) using Gini Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_gini)


# In[39]:


# Decision Tree Regression
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_R)


# In[ ]:




