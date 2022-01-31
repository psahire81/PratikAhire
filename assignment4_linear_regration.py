#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Q2


# In[1]:


import pandas as pd
data = pd.read_csv("C:\\Users\\Admin\\Downloads\\salary.csv")
data.head()


# In[2]:


data1=pd.get_dummies(data)


# In[3]:


data1


# In[4]:


data.info()


# In[5]:


data.corr()


# In[7]:


import seaborn as sns
sns.distplot(data['educationno'])


# In[19]:


import seaborn as sns
sns.distplot(data['capitalgain'])


# In[9]:


import statsmodels.formula.api as smf
model = smf.ols("educationno~capitalgain",data = data).fit()


# In[10]:


sns.regplot(x="educationno", y="capitalgain", data=data);


# In[11]:


model.params


# In[12]:


print(model.tvalues, '\n', model.pvalues)    


# In[13]:


(model.rsquared,model.rsquared_adj)


# In[27]:


newdata=pd.Series([10000])


# In[28]:


data_pred=pd.DataFrame(newdata,columns=['capitalgain'])


# In[29]:


model.predict(data_pred)


# In[ ]:





# In[ ]:





# In[ ]:


#Q1


# In[30]:


import pandas as pd
time=pd.read_csv("C:\\Users\\Admin\\Downloads\\delivery_time (1).csv")


# In[31]:


time.head()


# In[32]:


time2=time.rename({'Delivery Time':'Deliverytime','Sorting Time':'Sortingtime'},axis=1)


# In[51]:


time2


# In[34]:


time2.info()


# In[35]:


time2.corr()


# In[36]:


import seaborn as sns
sns.distplot(time2['Sortingtime'])


# In[37]:


import seaborn as sns
sns.distplot(time2['Deliverytime'])


# In[38]:


import statsmodels.formula.api as smf
model = smf.ols("Deliverytime~Sortingtime",data = time2).fit()


# In[39]:


sns.regplot(x="Sortingtime", y="Deliverytime", data=time2);


# In[40]:


model.params


# In[41]:


print(model.tvalues, '\n', model.pvalues)    


# In[42]:


(model.rsquared,model.rsquared_adj)


# In[53]:


newdata=pd.Series([10,4,6,9,10,6,7,3,10,9,8,4,7,3,3,4,6,7,2,7,5])


# In[54]:


time_pred=pd.DataFrame(newdata,columns=['Sortingtime'])


# In[55]:


model.predict(time_pred)


# In[ ]:




