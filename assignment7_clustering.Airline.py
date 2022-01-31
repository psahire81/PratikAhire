#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[17]:


Air=pd.read_csv("C:\\Users\\Admin\\Downloads\\EastWestAirlines.csv")


# In[18]:


Air.head()


# In[19]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[21]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Air.iloc[:,1:])


# In[26]:


dendrogram = sch.dendrogram(sch.linkage(df_norm, method='complete'))


# In[27]:


hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'complete')


# In[28]:


y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[29]:


Clusters


# In[ ]:





# In[30]:


# import K-mean libraries


# In[31]:


from sklearn.cluster import KMeans


# In[33]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Air_df = scaler.fit_transform(Air.iloc[:,1:])


# In[34]:


#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:


# In[35]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Air_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[36]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(4, random_state=42)
clusters_new.fit(scaled_Air_df)


# In[37]:


clusters_new.labels_


# In[38]:


#Assign clusters to the data set
Air['clusterid_new'] = clusters_new.labels_


# In[39]:


#these are standardized values.
clusters_new.cluster_centers_


# In[40]:


Air.groupby('clusterid_new').agg(['mean']).reset_index()


# In[42]:


Air


# In[ ]:





# In[ ]:





# In[43]:


#import DBscan library 


# In[44]:


from sklearn.cluster import DBSCAN


# In[45]:


array=Air.values


# In[46]:


array


# In[47]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[48]:


X


# In[57]:


dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X)


# In[58]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])


# In[59]:


cl


# In[ ]:




