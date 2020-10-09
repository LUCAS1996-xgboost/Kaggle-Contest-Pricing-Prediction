
# coding: utf-8

# In[1]:

import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.options.mode.chained_assignment = None  # default='warn'

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().magic('matplotlib inline')

# Seaborn for easier visualization
import seaborn as sns


# In[2]:

df = pd.read_csv('cleaned_input.csv')


# In[3]:

df.head()


# # 1. Start with Domain Language 从领域知识开始

# 这里的意思是，可以运用你的本专业知识进行深层特征的构建，比如从商业分析的角度，我们先以构建一个豪宅的概念为例。

# In[4]:

df['big_house'] = ((df.TotRmsAbvGrd >= 5) & (df.GarageArea >= 500)& (df.GrLivArea >= 3000)).astype(int) #地面房间数大于5， 停车场空间大于500， 居住面积大于3000， 定义为豪宅。


# In[5]:

df.big_house.mean() #满足是Bighouse的比例。


# Indicator Variable based on date
# Create a new feature called 'during_recession' to indicate if a transaction falls between 2007 and 2009.
# 
# Create a boolean mask to check if tx_year >= 2009
# Create a boolean mask to check if tx_year <= 2013
# Combine the two masks with an & operator
# Convert the resulting series to type int to map from <code

# In[6]:

df.columns


# In[7]:

#创造一个概念，就是在经济萧条期间建造的房子
df['built_during_recession'] = ((df.YearBuilt >= 2009) & (df.YearBuilt <= 2013)).astype(int)


# In[8]:

#卖出的房子
df['sold_during_recession'] = df.YrSold.between(2009, 2013).astype(int)


# In[9]:

# Print percent of transactions where built_during_recession == 1
df.built_during_recession.mean()# 经济萧条期间建造的房屋占比约百分之一。


# In[10]:

# Print percent of transactions where sold_during_recession == 1
df.sold_during_recession.mean()#经济萧条期间建造的房屋占比约33.7%


# In[11]:

df.groupby('sold_during_recession')['SalePrice'].mean()#0表示不是在经济萧条期间卖出的房子，可以看出萧条期间卖出的房子价格低于前者2%， 这里可以做一个显著性检验


# 这里我们比较的是经济萧条非萧条期间卖出房子的价格。

# # 2. Create Interaction Features

# In[12]:

df['property_age'] = df.YrSold - df.YearBuilt


# In[13]:

df.property_age.min() #会出现负数说明可能存在预售现象。


# In[14]:

df.loc[df.property_age <= 0]['property_age'] = 0#都调整为零


# In[15]:

df.property_age.max() #检查是否存在数据上的错误，比如1136年肯定是错的。


# In[16]:

df.property_age.mean() #平均代售年龄36年


# In[6]:

sum(df.property_age > 50)/len(df) #50年以上的待售房占比30%


# In[7]:

df['property_remodel_age'] = df.YearRemodAdd - df.YearBuilt


# In[8]:

df.property_remodel_age.min()


# In[9]:

df.loc[df.property_remodel_age <= 0]['property_remodel_age'] = 0    #remodel,房屋翻新


# In[10]:

df.property_remodel_age.mean()   #平均每12.97年加一次翻新


# # 3.Group sparse classes 加总稀疏类别

# 稀疏值容易影响预测的准确性。

# In[11]:

sns.countplot(y='RoofMatl', data=df)


# In[12]:

# print the values of RoofMatl        此处的稀疏特征需要合并，否则单个的数据容易引起以偏概全性偏差。
class_counts = df.groupby('RoofMatl').size()
print(class_counts)


# In[13]:

# Group small groups together. Label all of them as 'Other'. 所谓合并稀疏数据只是经验之谈，不能百分百保证提升模型效果，效率。
df.RoofMatl.replace(['ClyTile', 'Membran', 'Metal','Roll','Tar&Grv','WdShake','WdShngl'], 'Other', inplace=True)


# In[14]:

sns.countplot(y='RoofMatl', data=df)


# In[ ]:

# Overfitting -> The performance gap between sample and out-of-sample test. 
# Underfitting 


# In[ ]:

# It can reduce overfitting problem when the sample set for a specific class is super small.


# 降低过拟合的影响。

# # 4.Encode dummy variables¶

# 解码数值型变量，换为0和1的形式（用矩阵的形式），方便套入模型。

# In[15]:

df.head()


# In[16]:

df['BldgType'].unique() #以Bldgtype为例，下面有四个分支，


# In[17]:

df.select_dtypes(include=['object']).columns #选出所有非数值型变量，因为他们需要转换成数值来解码。


# In[18]:

df = pd.get_dummies(df, columns = df.select_dtypes(include=['object']).columns)


# 如下图所示，将所有columns都按小类别展开，然后用矩阵标记法将类别型变量转为数值。

# 不能为了图省事就把每个二级特征都标注为123456， 这样会忽略数值上的潜在关系特征。

# In[24]:

df.head(3)


# In[26]:

df[['BldgType_1Fam','BldgType_2fmCon','BldgType_Duplex']].tail(100)


# In[28]:

df[['RoofMatl_CompShg','RoofMatl_Other']].head(100)


# # 5. Remove unused or redundant features

# In[29]:

df = df.drop(['YearRemodAdd', 'YearBuilt'], axis=1)


# In[31]:

df.to_csv('clean_inputwithfeatures.csv', index=None)


# In[ ]:



