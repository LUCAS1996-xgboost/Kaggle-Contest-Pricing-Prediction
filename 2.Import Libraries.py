
# coding: utf-8

# # Kaggle竞赛P2房价数据集（自我调试）

# # 一. 调入包/Import libraries

# In[1]:

import numpy as np


# In[2]:

import pandas as pd


# In[3]:

from matplotlib import pyplot as pit


# In[21]:

pd.set_option('display.max_columns',100) #确保每行每列都在100之内，方便统计和操作
pd.set_option('display.max_rows',100)


# # 二. 读取数据/Read data

# In[22]:

df = pd.read_csv('train.csv')
type(df)                                  #用type函数验证一下这个文件的属性


# # 三. 测试一些常见功能

# In[24]:

df.shape


# In[25]:

df.shape[0]


# In[26]:

df.shape[1]   #元组的索引功能


# In[27]:

df.info()


# In[28]:

df.dtypes  # data types


# In[29]:

df[0:5]


# In[30]:

df.head(10)


# In[33]:

df['Id'].head(10)


# In[34]:

df['MSZoning'].head(10)  #objext相当于str, 字符串


# In[35]:

df[['Id','LotArea']].head(10) #两列要有两层中括号


# In[38]:

df[['Id','LotArea','RoofStyle']].head(10)    #看来多列一律使用双重中括号


# In[39]:

df.loc[0:2]


# In[40]:

df.head(3)


# In[41]:

df.iloc[0:3,1]


# In[43]:

df.select_dtypes(exclude=[np.number]).head(5)  #只选取非数字类型的列，并且保留头部5行


# In[44]:

df.select_dtypes(exclude=[np.number]).columns  #只显示列名字


# In[45]:

df.select_dtypes(include=[np.number]).head(5)   #只选取数字类型的列，并且保留头部5行


# In[46]:

df.dtypes[df.dtypes=='int64']


# In[48]:

df.dtypes=='str'  #此处object可换做str


# In[49]:

df.head()


# In[50]:

df.dtypes[df.dtypes=='int64'].index[2]   #对整数型的列求索引，取第3个数值


# In[60]:

for feature in df.dtypes[df.dtypes=='object'].index:
    print(feature) 
    print('---')   #此处的缩进符，只要超出4格就都可以识别


# In[61]:

feature     #为何此处自动输出之前的最后一个元素？？？？？


# In[64]:

df.describe()   #实际导出的都是数据类型，也就是可以分析平均数，方差，标准差的数据类型的相关参数


# In[65]:

df['Alley'].describe()   #导出单独某列的参数


# In[66]:

df['MSSubClass'].describe()


# In[67]:

df.describe(include=['object'])   #这里补充上了字符串类型数据的参数


# In[70]:

df.[['SalePrice','BldgType']].groupby('BldgType').agg(['mean','std','count'])


# In[71]:

df.groupby('BldgType').agg(['mean','std','count'])['SalePrice']


# In[79]:

df.corr()      #相关性矩阵，对角线上的数都为1，总体介于（-1，1）


# In[80]:

df.corr()


# In[83]:

correlations['SalePrice']


# In[ ]:



