
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# In[2]:

DF = pd.read_csv('new new clean_inputwithfeatures.csv')


# In[3]:

df = DF[DF.SalePrice.isnull() == False]


# In[4]:

df.head()


# Multiple Linear Regression
# •Set it as a benchmark and see the performance.
# 

# In[5]:

# this is the standard import if you're using "formula notation" (similar to R)
import statsmodels.formula.api as smf

# create a fitted model in one line
lm = smf.ols(formula='SalePrice ~ GrLivArea + LotArea + TotRmsAbvGrd + property_age + YrSold + OverallQual + GarageCars + property_remodel_age', data=df).fit()

# print the coefficients
lm.params


# In[6]:

# print a summary of the fitted model
lm.summary()


# Split-out validation dataset to Training and Testing¶

# In[7]:

from sklearn.model_selection import train_test_split


# In[8]:

# Create separate object for target variable
y = df.SalePrice

# Create separate object for input features
X = df.drop(['SalePrice','Id'], axis=1)


# In[9]:

type(y)


# Train/ Test Split

# In[10]:

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=1)


# In[11]:

print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[12]:

# Summary statistics of X_train
X_train.describe()


# In[13]:

# Standardize X_train
X_train_new = (X_train - X_train.mean()) / X_train.std()


# In[14]:

# Summary statistics of X_train_new
X_train_new.describe()


# In[15]:

X_train_new = X_train_new.fillna(0)


# In[16]:

from sklearn.preprocessing import StandardScaler  #进行预处理


# In[17]:

scaler = StandardScaler()
scaler.fit(X_train)


# In[18]:

X_train_new = scaler.transform(X_train)


# In[19]:

X_train


# In[20]:

X_train_new


# Import ML Models
# 
# Linear Models: Lasso, Ridge
# 
# Emsemble Models: Random Forrest, Gradient Boosting

# In[21]:

from sklearn.pipeline import make_pipeline


# In[22]:

from sklearn.preprocessing import StandardScaler, RobustScaler


# In[23]:

# Import Ridge Regression, and Lasso Regression
from sklearn.linear_model import Ridge, Lasso

# Import Random Forest and Gradient Boosted Trees
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


# In[24]:

from sklearn.feature_selection import SelectFromModel


# # Feature Selection¶

# In[25]:

select = SelectFromModel(Lasso(alpha = 10))
select.fit(X_train_new,y_train)


# In[26]:

X_new = select.transform(X)


# In[27]:

import pickle


# In[28]:

with open('featureselection.pkl', 'wb') as f:
    pickle.dump(select, f)


# In[29]:

X.columns[select.get_support()]


# In[30]:

set(X.columns) - set(X.columns[select.get_support()])


# In[31]:

X.shape


# In[32]:

X_new.shape


# In[33]:

X_train_selected = select.transform(X_train_new)
X_test_selected = select.transform(scaler.transform(X_test))


# # Pipelines¶

# In[34]:

# Create pipelines dictionary，这里应用到了字典的索引功能。
pipelines = {
    'lasso' : make_pipeline(RobustScaler(), Lasso(random_state=1)),
    'ridge' : make_pipeline(RobustScaler(), Ridge(random_state=1)),
    'rf' : make_pipeline(RobustScaler(), RandomForestRegressor(random_state=1)) ,
    'gb' : make_pipeline(RobustScaler(), GradientBoostingRegressor(random_state=1)) ,
    'ada' : make_pipeline(RobustScaler(), AdaBoostRegressor(random_state=1)) ,
}


# In[35]:

pipelines['lasso'] #Lasso后边的一大长串都是参数，其中的alpha就是lambda,所谓的惩罚参数，最重要的参数。


# 这里的max_iter=1000 指迭代次数默认为1000.

# In[36]:

# Check that we have all 5 algorithms, and that they are all pipelines
for key, value in pipelines.items():
    print( key, type(value) )


# # c) Spot Check Algorithms

# In[37]:

# List tuneable hyperparameters of our Lasso pipeline，获得所有的参数，方便调参
pipelines['lasso'].get_params()


# In[38]:

# Lasso hyperparameters，值域不够大的话，接下来可以选择100，1000.
lasso_hyperparameters = { 
    'lasso__alpha' : [0.001, 0.01, 0.1, 0.5, 1, 5, 10] 
}

# Ridge hyperparameters
ridge_hyperparameters = { 
    'ridge__alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10]  
}


# In[39]:

#  Random forest hyperparameters
rf_hyperparameters = { 
    'randomforestregressor__n_estimators' : [100, 200, 500],
    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}


# In[40]:

# Boosted tree hyperparameters
gb_hyperparameters = { 
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate' : [0.01,0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [3, 5, 10]
}


# In[41]:

ada_hyperparameters= { 
    'adaboostregressor__loss' : ['linear', 'square', 'exponential'],
    'adaboostregressor__learning_rate': [0.05,0.1,0.2,1],
    'adaboostregressor__n_estimators': [50,100,200]
}


# In[42]:

# Create hyperparameters dictionary
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'ada' : ada_hyperparameters
}


# In[43]:

for key in ['gb', 'ridge', 'rf', 'lasso','ada', 'lr']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')


# # Model Comparison

# In[44]:

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Tuning Models

# In[45]:

model = GridSearchCV(pipelines['lasso'], hyperparameters['lasso'], cv=5, n_jobs=-1)


# In[46]:

import warnings
warnings.simplefilter("ignore", UserWarning)

# Fit and tune model
model.fit(X_train_selected, y_train)


# In[47]:

import warnings
warnings.simplefilter("ignore", UserWarning)

# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=5, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train_selected, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')


# In[48]:

# Check that we have 5 cross-validation objects
for key, value in fitted_models.items():
    print( key, type(value) )


# In[49]:

from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test_selected)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))


# # Select Model

# In[50]:

# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )


# In[51]:

# Import r2_score and mean_absolute_error functions
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# It seems GBT performs the best

# In[52]:

# Display fitted GradientBoostingRegressor object
fitted_models['gb']


# # Testing on validation (testing) dataset¶

# In[53]:

# Predict test set using fitted GradientBoostingRegressor
pred = fitted_models['gb'].predict(X_test_selected)


# In[54]:

# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))


# In[55]:

# print the performance of each model in fitted_models on the test set.
for name, model in fitted_models.items():
    pred = model.predict(X_test_selected)
    print( name )
    print( '--------' )
    print( 'R^2:', r2_score(y_test, pred))
    print( 'MAE:', mean_absolute_error(y_test, pred))
    print()


# In[56]:

gb_pred = fitted_models['gb'].predict(X_test_selected)
plt.scatter(gb_pred, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[57]:

type(fitted_models['gb'])


# In[58]:

type(fitted_models['gb'].best_estimator_)


# In[59]:

fitted_models['gb'].best_estimator_


# In[60]:

import pickle  #pickle是一个包，用来储存模型，此处将表现良好的gb等的相关参数保存起来，


# In[61]:

with open('gb.pkl', 'wb') as f:
    pickle.dump(fitted_models['gb'].best_estimator_, f)


# In[62]:

with open('rf.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


# In[ ]:



