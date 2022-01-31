#!/usr/bin/env python
# coding: utf-8
# Pratik Ahire

# In[10]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[11]:


crime=pd.read_csv("C:\\Users\\Admin\\Downloads\\crime_data.csv")


# In[12]:


crime.head()


# In[17]:


crime.rename(columns={'Unnamed: 0': 'state'}, inplace=True)


# In[18]:


crime.head()


# In[19]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[20]:


df_norm = norm_func(crime.iloc[:,1:])


# In[21]:


dendrogram = sch.dendrogram(sch.linkage(df_norm, method='complete'))


# In[22]:


hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'complete')


# In[23]:


y_hc = hc.fit_predict(df_norm)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[24]:


Clusters


# In[ ]:





# In[ ]:





# In[25]:


# import K-mean libraries


# In[26]:


from sklearn.cluster import KMeans


# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_crime_df = scaler.fit_transform(crime.iloc[:,1:])


# In[30]:


#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:


# In[32]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_crime_df)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[35]:


#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(4, random_state=5)
clusters_new.fit(scaled_crime_df)


# In[36]:


clusters_new.labels_


# In[38]:


#Assign clusters to the data set
crime['clusterid_new'] = clusters_new.labels_


# In[39]:


#these are standardized values.
clusters_new.cluster_centers_


# In[40]:


crime.groupby('clusterid_new').agg(['mean']).reset_index()


# In[41]:


crime


# In[ ]:





# In[ ]:





# In[42]:


#import DBscan library 


# In[43]:


from sklearn.cluster import DBSCAN


# In[57]:


crime.drop(crime.columns[[0]], axis = 1, inplace = True)


# In[58]:


array=crime.values


# In[59]:


array


# In[60]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[61]:


X


# In[119]:


dbscan = DBSCAN(eps=1, min_samples=4)
dbscan.fit(X)


# In[122]:



#Noisy samples are given the label -1.
dbscan.labels_


# In[125]:



ml=pd.DataFrame(dbscan.labels_,columns=['cluster'])
pd.concat([crime,ml],axis=1)


# In[126]:


ml


# In[ ]:





# In[ ]:




