#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire

# In[2]:


# Imoprt libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[3]:


#Load the data set
bank = pd.read_csv("C:\\Users\\Admin\\Downloads\\bank-full.csv""",sep=";") #seperate parameter imprint 
bank.head()


# In[4]:


bank.info()


# In[5]:


# trasform the categrical variables into dummies 
bank_1 = pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome'])
bank_1


# In[6]:


# To see all columns
pd.set_option("display.max.columns", None)
bank_1


# In[7]:


# Custom Binary Encoding of Binary o/p variables 
bank_1['default'] = np.where(bank_1['default'].str.contains("yes"), 1, 0)
bank_1['housing'] = np.where(bank_1['housing'].str.contains("yes"), 1, 0)
bank_1['loan'] = np.where(bank_1['loan'].str.contains("yes"), 1, 0)
bank_1['y'] = np.where(bank_1['y'].str.contains("yes"), 1, 0)
bank_1


# In[5]:



# Custom Binary Encoding of Binary o/p variables 
bank_1['default'] = np.where(bank_1['default'].str.contains("yes"), 1, 0)
bank_1['housing'] = np.where(bank_1['housing'].str.contains("yes"), 1, 0)
bank_1['loan'] = np.where(bank_1['loan'].str.contains("yes"), 1, 0)
bank_1['y'] = np.where(bank_1['y'].str.contains("yes"), 1, 0)
bank_1


# In[8]:


# Find and Replace Encoding for month categorical varaible
bank_1['month'].value_counts()


# In[9]:


order = {'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}


# In[15]:


bank_1 = bank_1.replace(order)


# In[16]:


bank_1


# In[17]:



bank_1.info()


# # Model Building


# In[18]:



# Dividing our data into input and output variables
x=pd.concat([bank_1.iloc[:,0:11],bank_1.iloc[:,12:]],axis=1)
y=bank_1.iloc[:,11]


# In[19]:



classifier = LogisticRegression()
classifier.fit(x,y)


# In[20]:



# predict for x data sets
y_pred = classifier.predict(x)


# In[21]:


y_pred_df = pd.DataFrame({'actual':y,'predict_prob':y_pred})
y_pred_df


# In[22]:



# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(y,y_pred)
print(confusion_matrix)


# In[23]:


(39152+1238)/(39152+1238+4051+770)


# In[24]:



#Classification report
from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


# In[25]:



# ROC curve

fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba (x)[:,1])

auc = roc_auc_score(y, y_pred)


plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[26]:


auc

