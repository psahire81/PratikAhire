#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[10]:


book = pd.read_csv("C:\\Users\\Admin\\Downloads\\book (1).csv")
book.head()


# In[15]:


book.info()    # data has clear


# In[16]:



# # Apriori Algorithm
# ## 1. Association rules with 10% Support & 70% confidence


# In[17]:


# With 10% Support
frequent_itemsets = apriori(book, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[18]:


# With 70% confidence 
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[19]:



#  An leverage value of 0 indicates independence. Range will be [-1 1]
# High conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


# In[ ]:





# In[20]:


rules.sort_values('lift',ascending = False)


# In[21]:


rules[rules.lift>1]


# In[22]:



# visualization of obtained rule
plt.scatter(rules.support,rules.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[23]:


# ## 2. Association rules with 20% Support & 60% confidence


# In[24]:


# With 20% Support
frequent_itemsets2 = apriori(book,min_support=0.2,use_colnames=True)
frequent_itemsets2


# In[25]:



# With 60% confidence 
rules2 = association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2


# In[26]:



# visualization of obtained rule
plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[27]:


## 3. Association rules with 5% Support and 80% confidence


# In[28]:


# With 5% Support
frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
frequent_itemsets3


# In[29]:



# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3


# In[30]:


rules3[rules3.lift>1]


# In[31]:


# visualization of obtained rule
plt.scatter(rules3.support,rules3.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[ ]:




