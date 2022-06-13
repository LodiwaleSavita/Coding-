#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("C:/Users/DELL/Downloads/delivery_time.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


import seaborn as sns
sns.distplot(data['Sorting Time'])


# In[ ]:


sns.distplot(data['Delivery Time'])


# In[ ]:


data = data.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)


# In[ ]:


data


# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


sns.regplot(x = data['sorting_time'],y = data['delivery_time'],data = data)
model = smf.ols("delivery_time ~ sorting_time",data = data).fit()
model


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.tvalues,'\n',model.pvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# In[ ]:


y = (6.582734)+(1.649020)*5


# In[ ]:


y


# In[ ]:


newtime = pd.Series([5,8])


# In[ ]:


pred =pd.DataFrame(newtime,columns=['sorting_time'])
pred


# In[ ]:


model.predict(pred)


# # log Transformation

# In[ ]:


x_log=np.log(data['sorting_time'])
y_log=np.log(data['delivery_time'])


# In[ ]:


model = smf.ols("y_log~ x_log",data = data).fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


model.rsquared,model.rsquared_adj


# In[ ]:


y_log = (1.741987)+(0.597522)*5


# In[ ]:


y_log


# In[ ]:


newtime = pd.Series([5,8])


# In[ ]:


pred = pd.DataFrame(newtime,columns=['x_log'])


# In[ ]:


pred


# In[ ]:


model.predict(pred)


# In[ ]:


data


# # Improving Model Using Squareroot Transformation

# In[ ]:


data.insert(len(data.columns),'a_sqrt',
           np.sqrt(data.iloc[:,0]))


# In[ ]:


data


# In[ ]:


model = smf.ols("delivery_time ~ a_sqrt",data = data").fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# In[ ]:


y_quad = (-3.930699)+(3.977225)*5


# In[ ]:


y_quad


# In[ ]:


newtime=pd.Series([5,8])


# In[ ]:


pred=pd.DataFrame(newtime,columns = ['a_sqrt'])


# # Improving model with SquareTransformation

# In[ ]:


data['Squar_del_time']=data.apply(lambda row:row.delivery_time**2,axis =1)


# In[ ]:


data


# In[ ]:


model = smf.ols('Squar_del_time ~ sorting_time',data = data).fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # Improvement Model With Reciprocol Transformation
# 

# In[ ]:


reciprocal_del_time=1/data["delivery_time"]


# In[ ]:


reciprocal_del_time


# In[ ]:


model = smf.ols('reciprocal_del_time~sorting_time',data = data).fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # Improving model using Box - cox Transformation

# In[ ]:


from scipy.stats import boxcox
bcx_target,lam = boxcox(data["delivery_time"])


# In[ ]:


model = smf.ols('bcx_target~sorting_time',data = data).fit()


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # Improving model using yeo-johson transformation

# In[ ]:


from scipy.stats import yeojohnson
yf_target,lam = yeojohnson(data["delivery_time"])


# In[ ]:


model = smf.ols('yf_target~sorting_time',data=data).fit()


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # Model.rsquared,model.rsquared_adj
# 

# # The Reciprocol transformation is best transformation for this model

# # Statement2

# In[ ]:


salary = pd.read_csv("C:/Users/DELL/Downloads/Salary_Data.csv")


# In[ ]:


salary


# In[ ]:


salary.corr()


# In[ ]:


sns.distplot(salary['YearsExperience'])


# In[ ]:


sns.displot(salary['Salary'])


# In[ ]:


salary = salary.rename({'YearsExperience':'year','Salary':'income'},axis=1)


# In[ ]:


salary


# In[ ]:


sns.regplot(x ='year',y ='income',data = salary)


# In[ ]:


model = smf.ols ("income ~ year",data = salary).fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.tvalues,'\n',model.pvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# In[ ]:


newsalary = pd.Series([200,300])


# In[ ]:


data_pred = pd.DataFrame(newsalary,columns = ['year'])


# In[ ]:


data_pred


# In[ ]:


model.predict(data_pred)


# # Improving Model Using logarithm

# In[ ]:


salary1 = np.log(salary)


# In[ ]:


salary1


# In[ ]:


sns.regplot(x ='year',y='income',data = salary1)


# In[ ]:


model =smf.ols("income ~ year",data = salary1).fit()


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared)


# # Improving Model Using Squarroot Transformation

# In[ ]:


salary.insert(len(salary.columns),'A_sqrt',
             np.sqrt(salary.iloc[:,0]))


# In[ ]:


salary


# In[ ]:


model =smf.ols('income~A_sqrt',data = salary).fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.tvalues,'\n',model.pvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # Improving Model with Square Transformation

# In[ ]:


salary['Squar_income']=salary.apply(lambda row:row.income**2,axis = 1)


# In[ ]:


salary


# In[ ]:


model=smf.ols('Squar_income~year',data=salary).fit()


# In[ ]:


model


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # Improvement Model using box - cox transformation 

# In[ ]:


from scipy.stats import boxcox
bcx_target,lam = boxcox(salary["income"])


# In[ ]:


model = smf.ols('bcx_target ~ year',data = salary).fit()


# In[ ]:


model


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)


# # improving model using yeo-johnson transformation

# In[ ]:


from scipy.stats import yeojohnson
yf_target,lam = yeojohnson(salary["income"])


# In[ ]:


model = smf.ols('yf_target~year',data=salary).fit()


# In[ ]:


model.params


# In[ ]:


print(model.pvalues,'\n',model.tvalues)


# In[ ]:


(model.rsquared,model.rsquared_adj)

