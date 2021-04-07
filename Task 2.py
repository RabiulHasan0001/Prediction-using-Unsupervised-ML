#!/usr/bin/env python
# coding: utf-8

# ## Author-Md Rabiul Hasan

# ## Organization-The Sparks Foundation

# ## Task2-Prediction using Unsupervised ML
# ### Predict the optimum number of clusters and represent it visually

# ### Step1-Importing the libraries and dataset

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[21]:


df=pd.read_csv('Iris.csv')
df.shape


# ### Step2-Visualizing the data

# In[22]:


df.head()


# In[23]:


df.tail()


# In[24]:


df.describe()


# In[25]:


df.info()


# In[26]:


iris=pd.DataFrame(df)
iris_drop=iris.drop(columns=['Id','Species'])
iris_drop


# ### Step3-Finding the number of clusters

# In[27]:


from sklearn.cluster import KMeans
sse=[]
k_rng=range(1,11)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(iris_drop)
    sse.append(km.inertia_)
sse


# In[28]:


plt.xlabel('k')
plt.ylabel('sse')
plt.grid()
plt.plot(k_rng,sse)
plt.show()


# ### Step4-Apply the cluster value on the prediction

# In[29]:


from sklearn.cluster import KMeans
model=KMeans(n_clusters=3, init='k-means++',max_iter=300, n_init=10,random_state=0)
pred=model.fit_predict(iris_drop)
pred


# ### Step5-Cluster visualization

# In[30]:


x=iris_drop.iloc[:,[0,1,2]].values
plt.scatter(x[pred==0,0],x[pred==0,1],color='green',label='Iris-setosa')
plt.scatter(x[pred==1,0],x[pred==1,1],color='blue',label='Iris-versicolour')
plt.scatter(x[pred==2,0],x[pred==2,1],color='red',label='Iris-vorginica')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='orange',label='center')
plt.legend()
plt.grid()
plt.show()

