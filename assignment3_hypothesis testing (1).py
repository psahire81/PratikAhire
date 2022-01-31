#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q1


# In[2]:


import pandas as pd
import scipy 
import numpy as np
from scipy import stats


# In[3]:


cutlet=pd.read_csv("C:\\Users\\Admin\\Downloads\\Cutlets.csv")


# In[4]:


cutlet.head()


# In[5]:


cutlet_UnitA=cutlet['Unit A']
cutlet_UnitB=cutlet['Unit B']


# In[6]:


p_value=stats.ttest_ind(cutlet_UnitA,cutlet_UnitB)


# In[7]:


p_value


# In[8]:


##
    # ANS- We have to conduct 2-sample ,2- tail test for this .
           #   Assumption= Ho->uA=uB  ,Hi-> uA ≠ uB
#Conclusion- As per the test the p value is greater than the α ,hence we have fail to  reject Ho(null hypothesis) .There is no significant difference in the diameter of the cutlet between two units.  


# In[ ]:





# In[9]:


#Q2-


# In[10]:


import pandas as pd
import scipy 
import numpy as np
from scipy import stats


# In[11]:


lab=pd.read_csv("C:\\Users\\Admin\\Downloads\\LabTAT.csv")


# In[12]:


lab.head()


# In[13]:


import scipy.stats as stats
stats.f_oneway(lab.iloc[:,0], lab.iloc[:,1],lab.iloc[:,2],lab.iloc[:,3])
    


# In[14]:


#Ans- We have to use Anova here 
         # Assumption- Ho-µL1=µL2=µL3=µL4                      Hi- difference in avg. TAT among the lab
#Conclusion- p value is less than the α, hence reject Ho(null hypothesis)


# In[ ]:





# In[15]:


#Q3


# In[16]:


import pandas as pd
import scipy 
import numpy as np
from scipy import stats
import scipy.stats as stats


# In[17]:


buyer=pd.read_csv("C:\\Users\\Admin\\Downloads\\BuyerRatio.csv")


# In[18]:


buyer


# In[19]:


stats.f_oneway(buyer.iloc[:,1], buyer.iloc[:,2],buyer.iloc[:,3],buyer.iloc[:,4])
    


# In[20]:


# here the p value is greater than α, hence fail to reject null hypothesis Ho


# In[ ]:





# In[21]:


#Q4-


# In[22]:


form=pd.read_csv("C:\\Users\\Admin\\Downloads\\Costomer+OrderForm.csv")


# In[23]:


form.head()


# In[24]:


form2=pd.get_dummies(form)


# In[25]:


form2.head()


# In[26]:


obs = np.array([[271,267,269,280],[29,33,31,20]])


# In[27]:


obs


# In[29]:


from scipy.stats import chi2_contingency


# In[30]:


chi2_contingency(obs)


# In[31]:


pValue =  0.2771


# In[32]:


#Assume Null Hypothesis as Ho: Independence of categorical variables (customer order forms defective % does not varies by centre) Thus, Alternative hypothesis as Ha Dependence of categorical variables (customer order forms defective % varies by centre)


# In[33]:


# Compare p_value with α = 0.05


# In[34]:


if pValue< 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[35]:



# Inference: As (p_value = 0.2771) > (α = 0.05); Accept Null Hypthesis i.e. Independence of categorical variables Thus, customer order forms defective % does not varies by centre


# In[ ]:




