#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import pivot


# In[3]:


book_df=pd.read_csv("C:\\Users\\Admin\\Downloads\\book.csv")


# In[4]:


book_df[0:5]


# In[5]:


book1_df=book_df.iloc[:,1:]


# In[6]:


book1_df


# In[7]:


book1_df.info()


# In[8]:


book2_df=book1_df.rename({'User.ID':'userId','Book.Title':'booktitle','Book.Rating':'rating'},axis=1)


# In[9]:


book2_df


# In[10]:


len(book2_df.userId.unique())


# In[11]:


len(book2_df.booktitle.unique())


# In[12]:


book2_df[book2_df.duplicated()]


# In[17]:


book_cleaned=book2_df.drop_duplicates()


# In[18]:


book2_df.drop_duplicates(subset='userId',inplace=True)


# In[19]:


user_book_df = book_cleaned.pivot(index='userId',
                                 columns='booktitle',
                                 values='rating').reset_index(drop=True)


# In[20]:


user_book_df 


# In[22]:


user_book_df.fillna(0, inplace=True)


# In[23]:


user_book_df 


# In[24]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[25]:


user_sim = 1 - pairwise_distances( user_book_df.values,metric='cosine')


# In[26]:


user_sim


# In[27]:


user_sim_df = pd.DataFrame(user_sim)


# In[30]:


user_sim_df.index = book2_df.userId.unique()
user_sim_df.columns = book2_df.userId.unique()


# In[31]:


user_sim_df.iloc[0:5, 0:5]


# In[32]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[33]:


user_sim_df.idxmax(axis=1)[0:5]


# In[ ]:




