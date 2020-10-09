
# coding: utf-8

# #         Lesson 4 Data Cleaning 数据清洗

# In[3]:

get_ipython().system('pip install --user tqdm          #还是用老办法安装tqdm， 可以提供进度条')


# In[5]:

import pandas as pd
pd.set_option('display.max_columns', 100)

import numpy as np

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
import tqdm                           


# In[8]:

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


# In[9]:

df1.shape


# In[10]:

df2.shape


# In[13]:

df = df1.append(df2)
df


# In[14]:

df.columns


# In[17]:

df = df.reset_index().drop('index', axis = 1)
df


# In[18]:

df.head()


# # 一. Drop Duplications 去重

# In[19]:

df.shape


# In[20]:

df = df.drop_duplicates()       #检验是否有重复值，如果有的话，说明是很差的数据集了。本项目中没有重复数据。
df.shape


# # 二. Fix Missing Values 处理缺失值

# In[23]:

df.describe()
df


# In[21]:

df.isnull()  #显示全部数据中的缺失值， “TRUE”即为缺失值


# In[24]:

df.isnull().sum()   #查看每一列中缺失值的个数


# In[25]:

df[df['PoolQC'].isnull() == False].PoolQC   #查看某一特定列（例如PoolQC）中不是缺失值的的具体项目


# In[26]:

df.isnull().any()   #查看每列是否存在空值和非空值的情况


# In[27]:

df.columns[df.isnull().any()]    #查看所有存在非空值列的名字


# In[28]:

len(df.columns[df.isnull().any()])   #查看存在空值列的个数


# In[29]:

df[df.columns[df.isnull().any()]].head()    #查看所有空值列的头部，默认显示五行


# In[30]:

df[df.columns[df.isnull().any()]].dtypes  #判断所有空值列的数据类型


# Basic Rules :

# 在这里，我们要对空值进行处理，填补。
# For missing value of numerical features, we input them with average value.
# For missing value of categorical features, we input them using 'missing' / the value with higest frequency.
# 

# In[2]:

import pandas as pd
pd.set_option('display.max_columns', 100)

import numpy as np

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
import tqdm              


# In[3]:

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


# In[5]:

df = df1.append(df2)


# In[6]:

df.describe() #此处显示的都是数值型变量（int 和 float）


# In[7]:

df.describe(include=['object']) #查看类别型变量


# In[8]:

df['PoolQC']


# In[9]:

df['PoolQC'].fillna('Missing',inplace = True) #将所有缺失值换成Missing。


# In[10]:

df['PoolQC']


# In[11]:

df['PoolQC'] = df['PoolQC'].fillna('Missing') #两种表达方式


# In[13]:

df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean()) #数值型变量一般用平均数填补


# In[ ]:

df.columns


# 写一个循环，将不同数据类型的变量都进行替代。

# In[14]:

for column in df.columns:
    if df[column].isnull().any() == True:  #将空值列的名字依次打印出来。
        print(column)
        
        if (df[column].dtype == 'int64' or df[column].dtype == 'float64') and column != 'SalePrice': #注意这里要排除掉SalePrice
            df[column] = df[column].fillna(df[column].mean())
        
        if df[column].dtype == 'object':
            df[column] = df[column].fillna('Missing')


# Is there any missing value left?

# In[15]:

sum(df.isnull().sum()) #刚好和test集中的行数一致。说明没有问题。


# In[16]:

df.columns[df.isnull().any()] #说明对所有列都进行了填补缺失值的操作。


# In[17]:

sns.countplot(y='GarageType', data=df) #画出类别柱状图，可以看出Missing的出现。


# Combine Values with the same meaningb

# In[19]:

sns.countplot(y='BldgType', data=df)  #可以看出在Bldgtype中，Twnhse 和 Twnhs应为同类数据。


# In[20]:

df.BldgType.replace(['TwnhsE', 'Twnhs'], 'TownHouse', inplace=True)


# In[21]:

sns.countplot(y='BldgType', data=df)


# # 三. Remove Unwanted Outliers 移除离群值

# Outliers means the value far away from the general population.
# Outliers can be wrong data, but they can be true as well.
# Anyway, we want to exclude outliers to enhance the model performance.
# Start with a box plot of your target variable, since that's the variable that you're actually trying to predict.
# 离群值可能是正确但是离谱的数据，也可能是错误的数据，总之要去除以提升模型准确性。

# In[22]:

sns.boxplot(df.SalePrice) #使用箱状图检查离群值


# In[23]:

df['SalePrice'].mean() + 5 * df['SalePrice'].std()#房价大于578133的就是离群值


# In[2]:

import pandas as pd
pd.set_option('display.max_columns', 100)

import numpy as np

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
import tqdm                           


# In[3]:

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


# In[ ]:

df = df1.append(df2)
df


# In[5]:

sns.boxplot(df.SalePrice)
plt.xlim(0, 1000000) # setting x-axis range to be the same as in violin plot
plt.show()

# Violin plot of 'SalePrice' using the Seaborn library
sns.violinplot(df.SalePrice)
plt.show()


# 这不是一个标准的正态分布，因为定义域不是不是从负无穷开始的。

# In[6]:

df.shape


# In[7]:

DF2 = df[df.SalePrice.isnull() == True]


# In[8]:

DF1 = df[(df.SalePrice <= 578133.7103048441)]

# print length of df
DF1.shape


# In[9]:

DF = DF1.append(DF2)


# In[10]:

DF.shape  #说明除去了五个离群值。


# Finally, let's save the cleaned dataframe.
# Before we move on to the next module, let's save the new dataframe we worked hard to clean.
# 
# We'll use Pandas's .to_csv() function.
# We set index=None so that Pandas drops the indices and only stores the actual data in the CSV.

# In[12]:

DF.to_csv('cleaned_input.csv', index=None) #将清洗后的数据保存。


# In[13]:

DF = pd.read_csv('cleaned_input.csv')


# In[2]:

sum(DF.isnull().sum())


# In[ ]:



