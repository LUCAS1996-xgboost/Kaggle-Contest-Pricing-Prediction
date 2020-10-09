
# coding: utf-8

# # 1.Split-out validation dataset to Training and Testing¶

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

from sklearn.cross_validation import train_test_split


# In[6]:

# Create separate object for target variable
y = df.SalePrice

# Create separate object for input features
X = df.drop(['SalePrice','Id'], axis=1)


# In[7]:

type(y)


# # 2.Train/ Test Split¶¶

# In[8]:

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=1)#此处的0.25意为，训练集和测试集的比为3， 这个数值可以手动调节。


# In[ ]:

这里由于Lasso 和 Ridge都是进阶的Linear Regression, 我们需要将数据分为train 和test集。


# In[9]:

print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[10]:

# Summary statistics of X_train
X_train.describe()


# # 3.## b) Test options and evaluation metric

# # LASSO & Random Forrest

# In[11]:

# Linear Regressor
from sklearn.linear_model import Lasso

# Emsemble Regressor
from sklearn.ensemble import RandomForestRegressor #这里分别使用了Lasso和 Random forrest, 为了将两者预测的数据加以对比。


# In[12]:

import warnings
warnings.simplefilter("ignore", UserWarning)


# In[13]:

# Fit and tune model
Lasso = Lasso(alpha = 0.3, max_iter=100)
Lasso.fit(X_train, y_train)


# In[ ]:

#这里边，lamda为0.3， 最大迭代次数100， 


# In[14]:

Lasso.predict(X_test)[:5] #这里只是随机抽取测试集前五个点， 用Lasso进行预测。这里边显示的是用Lasso显示出的预测值。


# In[15]:

y_test[:5]


# In[16]:

# Fit and tune model
RandomForestRegressor = RandomForestRegressor(n_estimators = 100, max_depth =5) #100是树的个数，5是随机森林的层数。
RandomForestRegressor.fit(X_train, y_train)


# In[17]:

RandomForestRegressor.predict(X_test)[:5] #可以看出随机森林的预测值和Lasso相比差的远，不精确，但是只是针对这五个随机点的。


# In[18]:

y_test[:5]


# # Model Evaluation 模型评估

# 从sklearn中调出两个重要的衡量包，r方和MAE

# In[19]:

# Import r2_score and mean_absolute_error functions
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[20]:

# Predict test set using fitted Lasso
pred_lasso = Lasso.predict(X_test)


# In[21]:

# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred_lasso ))
print( 'MAE:', mean_absolute_error(y_test, pred_lasso))


# 上图是针对整体X-Test用Lasso做预测，发现R方为0.62

# In[22]:

# Predict test set using fitted GradientBoostingRegressor
pred_rf = RandomForestRegressor.predict(X_test)


# In[23]:

# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred_rf ))
print( 'MAE:', mean_absolute_error(y_test, pred_rf))


# 上图是针对整体X Test用Lasso做预测，发现R方为0.87

# In[ ]:

这里的LAsso可以继续调参，说明0.3不是一个很好的参数，而且MAE的差距远小于两者的r方差距，所以RF好于LAsso并不是一个最终的结论，还有很多值得探讨。


# It seems RF has a better performance than Lasso in this case.¶
