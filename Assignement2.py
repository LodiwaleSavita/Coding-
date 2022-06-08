#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


x = pd.Series([24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00])


# In[8]:


name = ['Allied Signal','Bankers Trust','General Mills','ITT Industries','P.Morgan&Co','lemon Brother','Marriott','MCI','Merrill Lynch','Microsoft','Morgon Stanley','Sun Microsystens','Travelers','US Airways','Warner-Lambert'] 


# In[9]:


#pie Plot
plt.figure(figsize=(6,8))
plt.pie(x,labels=name,autopct = '%1.0f%%')
plt.show()


# In[10]:


#Box Plot to find outliers
sns.boxplot(x)


# In[ ]:


#Mean


# In[11]:


x.mean()


# In[ ]:


#Variance


# In[12]:


x.var()


# In[ ]:


#Standard Deviation


# In[13]:


x.std()


# In[16]:


#SET 2(question 2)


# In[18]:


from scipy import stats
from scipy.stats import norm


# In[ ]:


# A.More employees at the processing center are older than 44 than between 38 and 44.


# In[19]:


# P (x>44);Employees older than 44 yrs of age
1-stats.norm.cdf(44,loc=38,scale=6)


# In[ ]:


# P(38<X<444);Employees btween 38 to 44 yrs of age


# In[20]:


stats.norm.cdf(44,38,6) - stats.norm.cdf(38,38,6)


# In[ ]:


# B.A training program for employees under the age f 30 at the center would be expected to attract about 30


# In[21]:


# P(X<30);Employees under 30 years of age
stats.norm.cdf(30,38,6)


# In[22]:


# No.of employees attending training program from 400 nos,is N=p(X<30)
400*stats.norm.cdf(30,38,6)


# In[ ]:


# SET4 (question 3)


# In[ ]:


import numpy as np
from scipy import stats
from scipy.stats import norm


# In[ ]:


# for No investigationp(45<x<55)
#for Investigation 1 - p(45<X<55)


# In[23]:


# find z-scores at x =45;z(s_mean -P_mean)/(P_SDsqrt(n))
z=(45-50)/(40/100**0.5)


# In[24]:


z


# In[25]:


#find z - scores at x = 55;z=(s_mean-P_mean)/(p_SD/sqrt(n))
z=(55-50)/(40/100**0.5)
z


# In[26]:


# for No investigation P(45<X<50) using z_score =p(x<50)-p(x<45)
stats.norm.cdf(1.25)-stats.norm.cdf(-1.25)


# In[27]:


stats.norm.interval(0.7887,loc = 50,scale = 40/(100**0.5))


# In[28]:


#for Investigation 1-P(45<x<55)
1-0.7887


# In[ ]:




