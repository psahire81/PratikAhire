#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import numpy


# In[2]:


import pandas as pd
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[4]:


#load data
dataset = pd.read_csv("C:\\Users\\Admin\\Downloads\\forestfires (1).csv")
dataset.head(20)


# In[5]:


dataset.info()


# In[6]:


# # Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset["month"] = label_encoder.fit_transform(dataset["month"])
dataset["day"] = label_encoder.fit_transform(dataset["day"])
dataset["size_category"] = label_encoder.fit_transform(dataset["size_category"])


# In[7]:


dataset.head(10)


# In[8]:


# split into input (X) and output (Y) variables
X = dataset.iloc[:,:11]
Y = dataset.iloc[:,-1]


# In[9]:


X


# In[10]:


Y


# In[11]:


# # Build ANN Model
model = Sequential()
model.add(layers.Dense(50, input_dim=11,  activation='relu'))
model.add(layers.Dense(11,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[12]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# In[13]:


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10)


# In[14]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[15]:


# # Graphical Visualization
# summarize history for accuracy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[16]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




