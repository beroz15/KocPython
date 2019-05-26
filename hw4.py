#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import os


# In[4]:


os.chdir('.')


# In[5]:


from sklearn.gaussian_process import GaussianProcessRegressor


# In[6]:


from sklearn.gaussian_process.kernels import ConstantKernel, RBF


# In[ ]:





# In[7]:


survey_file = 'https://raw.githubusercontent.com/carlson9/KocPython2019/master/12.GaussianProcesses/immSurvey.csv'


# In[8]:


tt = pd.read_csv(survey_file)


# In[9]:


tt.head()


# In[ ]:





# In[ ]:





# In[10]:


alphas = tt.stanMeansNewSysPooled


# In[11]:


sample = tt.textToSend


# In[ ]:





# In[ ]:





# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[13]:


vec = TfidfVectorizer(ngram_range=(1,2))


# In[ ]:





# In[ ]:





# In[14]:


X = vec.fit_transform(sample)


# In[15]:


pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, alphas,


# In[18]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, alphas, random_state=1)


# In[19]:


rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)


# In[20]:


gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


gpr.fit(Xtrain.toarray(), ytrain)


# In[ ]:





# In[ ]:





# In[22]:


mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)


# In[ ]:





# In[23]:


np.corrcoef(ytest, mu_s)


# In[24]:


#array([[1.        , 0.64347499],
       [0.64347499, 1.        ]])


# In[ ]:




