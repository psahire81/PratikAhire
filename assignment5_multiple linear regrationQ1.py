#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


start = pd.read_csv("C:\\Users\\Admin\\Downloads\\50_Startups.csv")
start.head()


# In[3]:


start2=start.rename({'R&D Spend':'RDspend','Marketing Spend':'Marketingspend'},axis=1)


# In[4]:


start2.info()


# In[5]:


start2.isna().sum()


# In[6]:


start2.corr()


# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(start2)


# In[8]:


import statsmodels.formula.api as smf
model=smf.ols('Profit~RDspend+Administration+Marketingspend',data=start2).fit()


# In[9]:


model.params


# In[10]:


print(model.tvalues,'\n',model.pvalues)


# In[11]:


(model.rsquared,model.rsquared_adj)


# In[12]:


ml_v=smf.ols('Profit~RDspend',data = start2).fit()  
print(ml_v.tvalues, '\n', ml_v.pvalues)  


# In[13]:


ml_w=smf.ols('Profit~Administration',data = start2).fit()  
print(ml_w.tvalues, '\n', ml_w.pvalues)  


# In[14]:


ml_wv=smf.ols('Profit~RDspend+Administration',data = start2).fit()  
print(ml_wv.tvalues, '\n', ml_wv.pvalues)  


# In[17]:


rsq_rd = smf.ols('RDspend~Administration+Marketingspend',data=start2).fit().rsquared  
vif_rd = 1/(1-rsq_rd) 

rsq_ad = smf.ols('Administration~RDspend+Marketingspend',data=start2).fit().rsquared  
vif_ad = 1/(1-rsq_ad) 

rsq_ms = smf.ols('Marketingspend~RDspend+Administration',data=start2).fit().rsquared  
vif_ms = 1/(1-rsq_ms)


d1 = {'Variables':['RDspend','Administration','Marketingspend'],'VIF':[vif_rd,vif_ad,vif_ms]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[18]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[19]:


list(np.where(model.resid<-20000))


# In[20]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[21]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[23]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RDspend", fig=fig)
plt.show()


# In[24]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Administration", fig=fig)
plt.show()


# In[25]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Marketingspend", fig=fig)
plt.show()


# In[26]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[27]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(start2)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[28]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[29]:


k = start2.shape[1]
n = start2.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[31]:


start2[start2.index.isin([49, 46])]


# In[32]:


start2.head()


# In[34]:


start_new = pd.read_csv("C:\\Users\\Admin\\Downloads\\50_Startups.csv")


# In[ ]:





# In[35]:


start1=start_new.drop(cars_new.index[[49,46]],axis=0).reset_index()


# In[54]:


start1=start1.drop(['index'],axis=1)


# In[55]:


start1


# In[56]:


start3=start1.rename({'R&D Spend':'RDspend','Marketing Spend':'Marketingspend'},axis=1)


# In[57]:


final_ml_V= smf.ols('Profit~RDspend+Marketingspend',data = start3).fit()


# In[58]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[59]:


final_ml_W= smf.ols('Profit~Administration+Marketingspend',data = start3).fit()


# In[60]:


(final_ml_W.rsquared,final_ml_W.aic)


# In[61]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[62]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(start3)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[63]:


(np.argmax(c_V),np.max(c_V))


# In[64]:


start4=start3.drop(start3.index[[45,47]],axis=0)


# In[65]:


start4


# In[67]:


start5=start4.reset_index()


# In[69]:


start6=start5.drop(['index'],axis=1)


# In[71]:


start6


# In[72]:


final_ml_V= smf.ols('Profit~RDspend+Marketingspend',data = start6).fit()


# In[73]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[75]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(start6)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[76]:


(np.argmax(c_V),np.max(c_V))


# In[77]:


final_ml_V= smf.ols('Profit~RDspend+Marketingspend',data = start6).fit()


# In[78]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[81]:


new_data=pd.DataFrame({'RDspend':78389,"Administration":135495,"Marketingspend":252664},index=[1])


# In[82]:


final_ml_V.predict(new_data)


# In[84]:


final_ml_V.predict(start3.iloc[0:5,])


# In[85]:


pred_y = final_ml_V.predict(start3)


# In[86]:


pred_y


# In[ ]:




