# The Sparks Foundation
# # K.Mahitha
# # Task 3: To explore unsupervised machine learning

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn.cluster import KMeans


# In[8]:


df = pd.read_csv(r"C:\Users\hp\Downloads\Iris.csv")
df.drop('Id',axis=1,inplace=True)
df.head()


# In[9]:


df['Species'].value_counts()


# In[10]:


df.info()


# In[11]:


X= df.drop('Species',axis=1)
y = df['Species']


# In[12]:


X.head()


# In[13]:


sns.set_style('darkgrid')
plt.figure(figsize=(15,10))
plt.scatter(data=X,x='SepalLengthCm',y='SepalWidthCm')
plt.xlabel('sepal length')
plt.ylabel('sepal width');


# In[14]:


X_scaled = preprocessing.scale(X)
X_scaled[:10]


# # Using k-means cluster

# In[16]:


wcss = []
cl = 10
for i in range(1,cl):
    kmeans = KMeans(i)
    kmeans.fit(X_scaled)
    x = kmeans.inertia_
    wcss.append(x)
wcss


# In[17]:


plt.figure(figsize=(15,9))
num_cl = range(1,cl)
plt.plot(num_cl,wcss);
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares');


# In[18]:


kmeans_2 = KMeans(2)
kmeans_2.fit(X_scaled)

cl_2 = X.copy()

cl_2['pred'] = kmeans_2.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(cl_2['SepalLengthCm'], cl_2['SepalWidthCm'], c= cl_2['pred'], cmap = 'rainbow');


# In[19]:


kmeans_3 = KMeans(3)
kmeans_3.fit(X_scaled)

cl_3 = X.copy()

cl_3['pred'] = kmeans_3.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(cl_3['SepalLengthCm'], cl_3['SepalWidthCm'], c= cl_3['pred'], cmap = 'rainbow');


# In[20]:


kmeans_5 = KMeans(5)
kmeans_5.fit(X_scaled)

cl_5 = X.copy()

cl_5['pred'] = kmeans_5.fit_predict(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(cl_5['SepalLengthCm'], cl_5['SepalWidthCm'], c= cl_5['pred'], cmap = 'rainbow');


# The color indicates Iris-setosa,Iris-versicolour,Iris-virginica and the green indicates the centroids.
# 
# This indicates the k-mean clustering

# In[ ]:




