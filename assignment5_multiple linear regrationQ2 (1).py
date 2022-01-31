#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[83]:


data=pd.read_csv('C:\\Users\\Admin\\Downloads\\ToyotaCorolla.csv')
data


# In[22]:


data.info()


# In[23]:


data.isna().sum()


# In[24]:


data.corr()


# In[25]:


data.dropna()


# In[26]:


sns.set_style(style='darkgrid')
sns.pairplot(data)


# In[28]:



import statsmodels.formula.api as smf 
model = smf.ols('Price~Age_08_04+HP+KM+Weight',data=data).fit()


# In[29]:


model.params


# In[30]:


print(model.tvalues,'\n',model.pvalues)


# In[31]:


(model.rsquared,model.rsquared_adj)


# In[32]:


ml_v=smf.ols('Price~KM',data = data).fit()  
print(ml_v.tvalues, '\n', ml_v.pvalues) 


# In[33]:


ml_v=smf.ols('Price~Age_08_04',data = data).fit()  
print(ml_v.tvalues, '\n', ml_v.pvalues)  


# In[34]:


rsq_Age_08_04 = smf.ols('Age_08_04~KM+HP+Weight',data=data).fit().rsquared  
vif_Age_08_04 = 1/(1-rsq_Age_08_04) 

rsq_KM = smf.ols('KM~Age_08_04+HP+Weight',data=data).fit().rsquared  
vif_KM = 1/(1-rsq_KM)

rsq_HP = smf.ols('HP~Age_08_04+KM+Weight',data=data).fit().rsquared  
vif_HP = 1/(1-rsq_HP)

rsq_Weight = smf.ols('Weight~Age_08_04+KM+HP',data=data).fit().rsquared  
vif_Weight = 1/(1-rsq_Weight)


d1 = {'Variables':['Age_08_04','KM','HP','Weight'],'VIF':[vif_Age_08_04,vif_KM,vif_HP,vif_Weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[35]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') 
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[36]:


list(np.where(model.resid<-4000))


# In[37]:


list(np.where(model.resid>5000))


# In[38]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[39]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[41]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[42]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age_08_04", fig=fig)
plt.show()


# In[43]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[44]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Weight", fig=fig)
plt.show()


# In[45]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[47]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[48]:


(np.argmax(c),np.max(c))


# In[49]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[50]:


k = data.shape[1]
n = data.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[52]:


data[data.index.isin([960, 221])]


# In[53]:


data.head()


# In[55]:


data1=data.drop(data.index[[960,221]],axis=0).reset_index()


# In[56]:


data1=data1.drop(['index'],axis=1)


# In[57]:


data1


# In[60]:


final_ml_V= smf.ols('Price~HP+KM+Weight',data = data1).fit()


# In[61]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[63]:


final_ml_W= smf.ols('Price~HP+Age_08_04+Weight',data = data1).fit()


# In[64]:


(final_ml_W.rsquared,final_ml_W.aic)


# In[65]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[67]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[68]:


(np.argmax(c_V),np.max(c_V))


# In[69]:


data2=data1.drop(data1.index[[600,650]],axis=0)


# In[70]:


data2


# In[71]:


data3=data2.reset_index()


# In[72]:


data4=data3.drop(['index'],axis=1)


# In[73]:


data4


# In[75]:


final_ml_V= smf.ols('Price~HP+KM+Weight',data = data4).fit()


# In[76]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[78]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data4)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[79]:


(np.argmax(c_V),np.max(c_V))


# In[81]:


final_ml_V= smf.ols('Price~HP+KM+Weight',data = data4).fit()


# In[82]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[85]:


new_data=pd.DataFrame({'HP':90,"KM":95,"Age_08_04":24,"Weight":35},index=[1])


# In[86]:


final_ml_V.predict(new_data)


# In[88]:


final_ml_V.predict(data.iloc[0:5,])


# In[89]:


pred_y = final_ml_V.predict(data)


# In[90]:


pred_y


# In[ ]:




