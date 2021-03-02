
# coding: utf-8

# In[149]:


import pandas as pd
import os
os.getcwd()
centroid = pd.read_csv(os.getcwd()+'/../dataset/input', sep="\t", header=None)
centroid = centroid[[1,3,4]]
centroid.head()


# In[143]:


trump = pd.read_csv('../dataset/3.csv', header=None)
trump.loc[trump[2] == 'Negative', [1]] = trump[1]*-1
trump[0] = trump[0].str.split(',',n=1,expand=True)[0]
trump.head(10)



# In[144]:


group = trump.groupby(0)        .agg({ 1 : ['mean'], 2 : ['size']})        .reset_index()


# In[145]:


group.head(10)


# In[146]:


joined = group.set_index(0).join(centroid.set_index(1))
joined = joined.loc[joined[3].notnull()]


# In[147]:


joined.head(10)


# In[141]:


joined.to_csv('../dataset/4.csv')

