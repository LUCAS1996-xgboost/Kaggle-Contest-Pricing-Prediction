
# coding: utf-8

# # Muyang Niu-Kaggle Lesson3-Data Visualization(自我调试)

# # 使用Python的数据可视化

# # 一. 安装环境

# In[1]:

import numpy as np


# In[2]:

import pandas as pd


# In[3]:

from matplotlib import pyplot as plt


# In[4]:

import seaborn as sns


# In[8]:

pip install seaborn


# In[6]:

get_ipython().system('pip install -- user seaborn')


# In[7]:

get_ipython().system('pip install -- upgrade pip')


# In[9]:

get_ipython().system('pip install --user seaborn    #安装的用法同np,pd,注意--和user之间是没有空格的，和前者有空格')


# In[10]:

import seaborn as sns


# In[12]:

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[13]:

df = pd.read_csv('train.csv')
type(df)                        


# In[14]:

df.shape


# In[15]:

df.dtypes


# In[16]:

df.head()    # 使用空格默认列出头部5行


# # 二. Scatter Plot 散点图

# In[18]:

correlations=df.corr()


# In[19]:

correlations['SalePrice']    #一定要先给correlations做定义，否则一上来就使用correlations[],py不会识别
                               #先做预测，寻找有价值的参数，比如Overall quality，r=0.79,相关性很高，有研究意义。


# In[22]:

plt.scatter(df.YearBuilt,df.SalePrice,color='blue')
plt.xlabel('Year--Built')
plt.ylabel('SalePrice')
plt.show()


# In[34]:

plt.scatter(df.OverallQual,df.SalePrice,color='red')
plt.xlabel('Overall Quality')
plt.ylabel('SalePrice')
plt.show()


# # 三. 柱状图 Bar Plot (类别型变量)

# In[24]:

sns.countplot(x='BldgType',data=df)   #柱状图居然画不出来


# In[26]:

object=df.dtypes[df.dtypes=='object'].index
object


# In[29]:

int64=df.dtypes[df.dtypes=='int64'].index
int64


# In[45]:

sns.countplot(x='BldgType',data=df)                  #查看的是房屋类型
plt.show()                                          #柱状图居然画不出来///画出来啦哈哈哈，要加上plt.show()
                                                      #目前的问题是，没有按照从大到小的顺序


# In[58]:

object=df.dtypes[df.dtypes=='object'].index     #列出所有类别型的数据
object[0]                                       #可以使用索引


# In[67]:

object=df.dtypes[df.dtypes=='object'].index     #列出所有类别型的数据
object  


# In[73]:

for feature in object:      #一定要注意循环带引号！！！！！！！！！！！！！！加引号！！！！！！！！！！！！
        fig = plt.gcf()
        fig.set_size_inches(20, len(df[feature].unique())/2)
        
        sns.countplot(y=feature, data=df, order = df[feature].value_counts().index)
        plt.show()
    
   


# In[70]:

for feature in object:      #(非标准版，将xy轴做了对调)
        fig = plt.gcf()
        fig.set_size_inches(len(df[feature].unique())/2, 20)
        
        sns.countplot(x=feature, data=df, order = df[feature].value_counts().index)
        plt.show()
    
   


# In[68]:

for feature in object:      #（非标准版，没有自动调节柱子厚度哈哈哈）
        fig = plt.gcf()
        fig.set_size_inches(20, 5)
        
        sns.countplot(y=feature, data=df, order = df[feature].value_counts().index)
        plt.show()


# In[74]:

for feature in object:      #（非标准版，没有自动调节按大小排列，以及厚度哈哈哈）
        fig = plt.gcf()
        fig.set_size_inches(20, 5)
        
        sns.countplot(y=feature, data=df)
        plt.show()


# In[75]:

object=df.dtypes[df.dtypes=='object'].index     #列出所有类别型的数据
object  


# In[82]:

df[feature].value_counts().index


# # 利用Python画出连续性变量

# In[85]:

df.hist()
plt.show()


# In[86]:

df.hist(figsize=(18,20),xrot=-45,grid=True)
plt.show()


# In[87]:

df.GarageArea.hist(figsize=(8,8),xrot=-45,grid=True)
plt.show()


# # 利用Seaborn画出连续型变量

# In[88]:

type(df)


# In[89]:

df.describe()


# In[90]:

df.groupby('BldgType')['SalePrice'].mean()


# In[92]:

df['SalePrice'].plot()
plt.show()


# In[98]:

sns.relplot(x="GrLivArea", y='SalePrice', col="BldgType",
            hue="BldgType", style='BldgType', size="OverallQual",
            data=df)
plt.show()


# # Box Plot 箱线图

# In[100]:

sns.boxplot(y='BldgType', x='SalePrice', data=df)
plt.show()


# # 相关性矩阵&反热力分布图 （Correlation Matrix&Sub Plot）

# In[102]:

correlations = df.corr()


# In[103]:

correlations


# In[121]:

plt.figure(figsize=(8,8))
sns.heatmap(correlations)
plt.show()                           #颜色越浅，相关性越高


# In[120]:

plt.figure(figsize=(15,14))
sns.heatmap(correlations)
plt.show()


# In[11]:

from matplotlib import pyplot as plt   #所以看起来每次重新调用数据都要重新导入一遍数据集和安装包
import seaborn as sns
import pandas as pd
import numpy as np


# In[12]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[13]:

df = pd.read_csv('train.csv')
type(df)              


# In[14]:

plt.figure(figsize=(15,14))
sns.heatmap(correlations)
plt.show()


# In[15]:

correlations = df.corr()


# In[17]:

plt.figure(figsize=(15,14))
sns.heatmap(correlations*100, annot=True, fmt='.0f')
plt.show()


# In[18]:

mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True


# In[19]:

plt.figure(figsize=(15,14))
sns.heatmap(correlations*100, annot=True, fmt='.0f', mask=mask)
plt.show()


# In[1]:




# In[ ]:



