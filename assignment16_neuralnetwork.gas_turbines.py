#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load datasets
data = pd.read_csv("C:\\Users\\Admin\\Downloads\\gas_turbines.csv")
data


# In[3]:


# Define X & y
X = data.drop(columns = ['TEY'], axis = 1) 
y = data.iloc[:,7]


# In[4]:


X


# In[5]:


y


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
X_test_scaled


# In[7]:


input_size = len(X.columns)
output_size = 1
hidden_layer_size = 50

model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layer_size, input_dim = input_size, activation = 'relu'),
                             tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                             tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
                             tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),     
                             tf.keras.layers.Dense(output_size)])


# In[8]:


optimizer = tf.keras.optimizers.SGD(learning_rate = 0.03)
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['MeanSquaredError'])


# In[9]:


num_epochs = 50
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
model.fit(X_train_scaled, y_train, callbacks = early_stopping, validation_split = 0.1, epochs = num_epochs, verbose = 2)


# In[10]:


test_loss, mean_squared_error = model.evaluate(X_test_scaled, y_test)


# In[11]:


predictions = model.predict_on_batch(X_test_scaled)


# In[12]:


plt.scatter(y_test, predictions)


# In[13]:


predictions_df = pd.DataFrame()
predictions_df['Actual'] = y_test
predictions_df['Predicted'] = predictions
predictions_df['% Error'] = abs(predictions_df['Actual'] - predictions_df['Predicted'])/predictions_df['Actual']*100
predictions_df.reset_index(drop = True)


# In[ ]:




