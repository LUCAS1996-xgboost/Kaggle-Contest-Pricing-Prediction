
# coding: utf-8

# In[1]:

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().magic('matplotlib inline')

# Seaborn for easier visualization
import seaborn as sns


# In[2]:

DF = pd.read_csv('new new clean_inputwithfeatures.csv')


# In[3]:

df = DF[DF.SalePrice.isnull() == False]


# In[4]:

fig, axs = plt.subplots(1, 3, sharey=True)
df.plot(kind='scatter', x='GrLivArea', y='SalePrice', ax=axs[0], figsize=(16, 8))
df.plot(kind='scatter', x='property_age', y='SalePrice', ax=axs[1])
df.plot(kind='scatter', x='YrSold', y='SalePrice', ax=axs[2])


# In[5]:

# this is the standard import if you're using "formula notation" (similar to R)
import statsmodels.formula.api as smf   #引入统计学的包

# create a fitted model in one line
lm = smf.ols(formula='SalePrice ~ GrLivArea', data=df).fit() #用saleprice做因变量，GrlivArea做自变量，做一元线性回归。

# print the coefficients #打印出系数，看见截距为279832，斜率为100
lm.params


# In[6]:

X_new = pd.DataFrame({'GrLivArea': [df.GrLivArea.min(), df.GrLivArea.max()]})
X_new.head()#找出房屋面积的最大值和最小值，进行初步评估。


# In[7]:

preds = lm.predict(X_new)
preds


# In[8]:

# first, plot the observed data，此处有明显的outlier, 可以自己尝试移除。
df.plot(kind='scatter', x='GrLivArea', y='SalePrice')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=3)


# In[9]:

df = df[(df.GrLivArea <= 4000)]#此处我尝试移除了离群值。


# In[10]:

lm.rsquared # r平方反映的是自变量对于因变量变化的影响占比。


# In[11]:

# this is the standard import if you're using "formula notation" (similar to R)
import statsmodels.formula.api as smf

# create a fitted model in one line
lm = smf.ols(formula='SalePrice ~ GrLivArea + property_age + OverallQual + YrSold', data=df).fit()

# print the coefficients
lm.params


# In[12]:

# print a summary of the fitted model
lm.summary() #R平方会随着参数的增加而提高，所以不能仅仅用R平方去评判一个模型准确与否。
# 下图中，adj.R-squared为调整之后的R平方。


# In[13]:

df = df.drop(['YrSold'], axis=1)


# In[14]:

lm = smf.ols(formula='SalePrice ~ GrLivArea + property_age + OverallQual ', data=df).fit()


# In[15]:

lm.params


# In[16]:

lm.summary()


# In[17]:

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()

# create X and y
feature_cols = ['GrLivArea','property_age','OverallQual']
X = df[feature_cols]
y = df.SalePrice

#fit the model
lm.fit(X, y)


# In[18]:

# Print intercept and coefficient，这里的截距和用statisticmodel求出来的一样。
print( lm.intercept_ )
print( lm.coef_ )


# In[19]:

m = lm.coef_[0]
m2 = lm.coef_[1]
m3 = lm.coef_[2]

b = lm.intercept_
print(' y = {1} + ( x1 * {0} ) + ( x2 * {2} ) + ( x3 * {3} )'.format(m, b, m2, m3))


# In[20]:

df.head()


# In[21]:

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[22]:

X = df[['GrLivArea','property_age']].as_matrix()


# In[23]:

X.shape


# In[24]:

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=1234
)
y_km = km.fit_predict(X)


# In[25]:

# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


# In[ ]:



