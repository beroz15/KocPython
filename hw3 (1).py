#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import pystan


# In[ ]:





# In[6]:


os.getcwd()


# In[7]:


os.chdir("KocPython2019/Homework/")


# In[8]:


data = pd.read_csv("trend2.csv")


# In[9]:


data.columns = data.columns.map(str.strip)


# In[10]:


data = data.dropna()


# In[11]:


data.country = data.country.map(str.strip)


# In[12]:


countries = data.country.unique()


# In[13]:


k = len(countries)


# In[14]:


country_lookup = dict(zip(countries, range(k)))


# In[15]:


country = data['country_code'] = data.country.replace(country_lookup).values


# In[16]:


religiosity = data.church2


# In[17]:


inequality = data.gini_net.values


# In[18]:


rgdpl = data.rgdpl.values


# In[19]:


data {


# In[21]:


data:
      int<lower=0> J;
    int<lower=1,upper=J> country[N]
      vector[N] x1; //inequality
        
            


# In[22]:


data {


# In[23]:


int<lower=0> J; 


# In[24]:


int<lower=0> N; 


# In[25]:


int<lower=1,upper=J> country[N]; 


# In[26]:


vector[N] x1; //inequality


# In[27]:


vector[N] x2; //rgdpl


# In[28]:


vector[N] y; //religiosity


# In[29]:


}


# In[30]:


parameters {


# In[31]:


vector[J] a;


# In[32]:


real b1;


# In[33]:


real b2;


# In[34]:


real mu_a;


# In[35]:


real mu_a;


# In[36]:


real<lower=0,upper=100> sigma_a;


# In[37]:


real<lower=0,upper=100> sigma_y;


# In[38]:


}


# In[39]:


transformed parameters {


# In[40]:


vector[N] y_hat;


# In[41]:


for (i in 1:N)


# In[42]:


y_hat[i] = a[country[i]] + x1[i] * b1 + x2[i] * b2;


# In[43]:


}


# In[44]:


model {


# In[45]:


sigma_a ~ uniform(0, 100);


# In[46]:


a ~ normal(mu_a, sigma_a);


# In[47]:


b1 ~ normal(0,1);


# In[48]:


b2 ~ normal(0,1);


# In[49]:


sigma_y ~ uniform(0, 100);


# In[50]:


y ~ normal(y_hat, sigma_y);


# In[51]:


}


# In[52]:


varying_intercept_data = {'N': len(religiosity),


# In[53]:


varying_intercept_data = {'K': len(religiosity),


# In[54]:


'J': len(countries),


# In[55]:


'J': len(countries),


# In[56]:


'country': country+1,


# In[57]:


varying_intercept_data = {'K': len(religiosity),
                          'J': len(countries),
                          'country': country+1,
                          'x1': inequality,
                          'x2': rgdpl,
                          'y': religiosity}


# In[58]:


varying_intercept_fit = pystan.stan(model_code=varying_intercept, data=varying_intercept_data, iter=1000, chains=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[59]:


a_sample = pd.DataFrame(varying_intercept_fit['a'])


# In[ ]:





# In[ ]:




