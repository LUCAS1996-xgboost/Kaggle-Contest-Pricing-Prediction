
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
submission = DF[DF.SalePrice.isnull() == True]


# In[4]:

df.head()


# In[5]:

import statsmodels.formula.api as smf

# create a fitted model in one line
lm = smf.ols(formula='SalePrice ~ GrLivArea + LotArea + TotRmsAbvGrd + property_age + YrSold + OverallQual + GarageCars + property_remodel_age', data=df).fit()

# print the coefficients
lm.params


# In[6]:

lm.summary()


# In[7]:

from sklearn.model_selection import train_test_split


# In[8]:

df_train = df[df['GrLivArea']<4000].copy() 


# In[9]:

y = df_train.SalePrice

# Create separate object for input features
X = df_train.drop(['SalePrice','Id'], axis=1)


# In[10]:

X_sub = submission.drop(['SalePrice','Id'], axis=1)


# In[11]:

type(y)


# In[12]:

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=1)


# In[13]:

print( len(X_train), len(X_test), len(y_train), len(y_test) )


# In[14]:

X_train.describe()


# In[15]:

X_train_new = (X_train - X_train.mean()) / X_train.std()


# In[16]:

X_test_new = (X_test - X_test.mean()) / X_test.std()


# In[17]:

X_train_new.describe()


# In[18]:

X_train_new = X_train_new.fillna(0)


# In[19]:

from sklearn.pipeline import make_pipeline


# In[20]:

from sklearn.preprocessing import StandardScaler, RobustScaler


# In[21]:

from sklearn.linear_model import ElasticNet, Ridge, Lasso

# Import Random Forest and Gradient Boosted Trees
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Import Neural Network
from sklearn.neural_network import MLPRegressor


# In[22]:

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import pickle


# In[23]:

clf = ExtraTreesClassifier(n_estimators=50)


# In[24]:

clf = clf.fit(X, y)


# In[25]:

featureselection = SelectFromModel(clf, prefit=True)
X_new = featureselection.transform(X)


# In[26]:

with open('featureselection.pkl', 'wb') as f:
    pickle.dump(featureselection, f)


# In[27]:

pipelines = {
    'lasso' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), Lasso(random_state=1)),
    'ridge' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), Ridge(random_state=1)),
    'enet' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), ElasticNet(random_state=1)),
    'rf' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), RandomForestRegressor(random_state=1)),
    'gb' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), GradientBoostingRegressor(random_state=1)),
    'ada' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), AdaBoostRegressor(random_state=1)),
    'nn' : make_pipeline(RobustScaler(quantile_range=(5.0, 95.0)), MLPRegressor(random_state=1))
}


# In[28]:

for key, value in pipelines.items():
    print( key, type(value) )


# In[29]:

pipelines['lasso'].get_params()


# In[30]:

pipelines['lasso'].get_params().keys()


# In[31]:

lasso_hyperparameters = { 
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'lasso__max_iter': [500, 1000, 2000],
    'lasso__positive': [True, False]
}

# Enet hyperparameters
enet_hyperparameters = { 
    'elasticnet__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'elasticnet__l1_ratio': [0.25, 0.5, 0.75],
    'elasticnet__max_iter': [500, 1000, 2000]
}

# Ridge hyperparameters
ridge_hyperparameters = { 
    'ridge__alpha': [0.001, 0.005, 0.01,0.025, 0.05, 0.1, 0.5, 1, 5, 10]  
}


# In[32]:

rf_hyperparameters = { 
    'randomforestregressor__n_estimators' : [5, 10, 20, 100, 200, 400],
    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
    'randomforestregressor__criterion' : ['mae','mse']
}


# In[33]:

gb_hyperparameters = { 
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate' : [0.01,0.05, 0.1],
    'gradientboostingregressor__max_depth': [3, 5, 10, 20],
    'gradientboostingregressor__alpha' : [0.9,0.99,0.999]
}


# In[34]:

ada_hyperparameters= { 
    'adaboostregressor__loss' : ['linear', 'square', 'exponential'],
    'adaboostregressor__learning_rate': [0.01,0.05,0.1,0.2,1],
    'adaboostregressor__n_estimators': [50,100,200]
}


# In[35]:

nn_hyperparameters= { 
    'mlpregressor__hidden_layer_sizes': [(25,),(50,),(100,)],
    'mlpregressor__alpha': [0.0001,0.0002],
    'mlpregressor__early_stopping' : [True, False],
    'mlpregressor__learning_rate_init': [0.0005,0.001,0.005,0.01,0.1],
    'mlpregressor__learning_rate': ['constant','invscaling','adaptive'],
    'mlpregressor__max_iter': [500] 
}


# In[36]:

hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'enet' : enet_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'ada' : ada_hyperparameters,
    'nn' : nn_hyperparameters,
}


# In[37]:

for key in ['gb', 'ridge', 'rf', 'lasso', 'enet', 'ada', 'nn']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')


# In[38]:

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[39]:

import warnings
warnings.simplefilter("ignore", UserWarning)

# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_new, y)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')


# In[40]:

for key, value in fitted_models.items():
    print( key, type(value) )


# In[41]:

from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_new)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))


# In[42]:

# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )


# In[43]:

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[44]:

fitted_models['gb']


# In[45]:

for name, model in fitted_models.items():
    pred = model.predict(X_new)
    print( name )
    print( '--------' )
    print( 'R^2:', r2_score(y, pred ))
    print( 'MAE:', mean_absolute_error(y, pred))
    print( 'RMSE:', (mean_squared_error(y, pred))**(1/2))
    print()


# In[53]:

Input = pd.DataFrame({})


# In[54]:

for name, model in fitted_models.items():
    Input[name] = model.predict(X_new)


# In[55]:

get_ipython().system('pip install --user xgboost')


# In[56]:

get_ipython().system('pip install --upgrade pip')


# In[57]:

from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor


# In[61]:

get_ipython().system('pip install --user mlxtend')


# In[62]:

get_ipython().system('pip install --upgrade pip')


# In[63]:

from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor


# In[64]:

xgboost = XGBRegressor(learning_rate = 0.005,n_estimators = 10000,
                                     max_depth = 6, min_child_weight = 0.1,early_stopping_rounds = 30,
                                     gamma = 0.3, subsample = 0.65,
                                     rate_drop = 0.3,
                                     skip_drop = 0.3,
                                     colsample_bytree=0.3,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=1,
                                     reg_alpha=0.002)


# In[65]:

stacked= StackingCVRegressor(regressors=(fitted_models['lasso'].best_estimator_,
                                         fitted_models['enet'].best_estimator_,
                                         fitted_models['ridge'].best_estimator_,
                                         fitted_models['rf'].best_estimator_,
                                         fitted_models['gb'].best_estimator_,
                                         fitted_models['ada'].best_estimator_,
                                         fitted_models['nn'].best_estimator_,
                            ),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# In[66]:

stacked_model = stacked.fit(X_new,y)


# In[67]:

Pred = stacked_model.predict(X_new)


# In[68]:

print( 'R^2:', r2_score(y, Pred))
print( 'MAE:', mean_absolute_error(y, Pred))
print( 'RMSE:', (mean_squared_error(y, Pred))**(1/2))


# In[69]:

plt.scatter(Pred, y)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[70]:

type(fitted_models['gb'])


# In[71]:

type(fitted_models['gb'].best_estimator_)


# In[72]:

fitted_models['nn'].best_estimator_


# In[73]:

X_sub_new = featureselection.transform(X_sub)


# In[74]:

submission['SalePrice'] = stacked_model.predict(X_sub_new)


# In[76]:

submission[['Id','SalePrice']].to_csv('submission.csv', index=False)


# In[ ]:



