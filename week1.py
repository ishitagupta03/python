#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno


# In[4]:


train = pd.read_csv('./test.csv')
test = pd.read_csv('./test.csv')


# In[5]:


train.describe()


# In[6]:


train.head()


# In[7]:


train.tail()


# In[8]:


train.shape , test.shape


# In[9]:


categorical_features = train.select_dtypes(include=[np.object])

categorical_features.columns


# In[10]:


numeric_features = train.select_dtypes(include=[np.number])

numeric_features.columns


# In[11]:


msno.matrix(train.sample(250))


# In[2]:





# In[ ]:





# In[ ]:




