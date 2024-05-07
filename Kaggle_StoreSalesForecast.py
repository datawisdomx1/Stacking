#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:17:09 2021

@author: nitinsinghal
"""

# Kaggle Store Sales Forecast

#Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from vecstack import stacking
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import datetime
import warnings
warnings.filterwarnings('ignore')


# Load the data
holidays_events = pd.read_csv('./Kaggle/store-sales-time-series-forecasting/holidays_events.csv')
oil = pd.read_csv('./Kaggle/store-sales-time-series-forecasting/oil.csv')
stores = pd.read_csv('./Kaggle/store-sales-time-series-forecasting/stores.csv')
transactions = pd.read_csv('./Kaggle/store-sales-time-series-forecasting/transactions.csv')
train = pd.read_csv('./Kaggle/store-sales-time-series-forecasting/train.csv')
test = pd.read_csv('./Kaggle/store-sales-time-series-forecasting/test.csv')


# Perform EDA - see the data types, content, statistical properties
print(train_data.describe())
print(train_data.info())
print(train_data.head(5))
print(train_data.dtypes)
      
print(test_data.describe())
print(test_data.info())
print(test_data.head(5))
print(test_data.dtypes)

# Perform data wrangling - remove duplicate values and set null values to 0
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)
train_data.drop(['id'], axis=1, inplace=True)

# Only used fillna=0. Dropna not used as other columns/row shave useful data
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Split categorical data for one hot encoding
train_data_cat = train_data.select_dtypes(exclude=['int64','float64'])
train_data_num = train_data.select_dtypes(include=['int64','float64'])

test_data_cat = test_data.select_dtypes(exclude=['int64','float64'])
test_data_num = test_data.select_dtypes(include=['int64','float64'])

# Encode the categorical features using OneHotEncoder. Use the same encoder for train and test set
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(train_data_cat)
train_data_cat = pd.DataFrame(ohe.transform(train_data_cat))
test_data_cat = pd.DataFrame(ohe.transform(test_data_cat))

# Merge encoded categorical data with mueric data
train_data_ohe = train_data_num.join(train_data_cat)
test_data_ohe = test_data_num.join(test_data_cat)

# Setup the traing and test X, y datasets
y_train = train_data_ohe.loc[:,'loss'].values
train_data_ohe.drop(['loss'], axis=1, inplace=True)
X_train = train_data_ohe.iloc[:,:-1].values
X_test = test_data_ohe.iloc[:,1:-1].values

# Scale all the data as some features have larger range compared to the rest
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform hyperparameter tuning of the models to be used for stacking
# Create the pipeline to run gridsearchcv for hyperparameter tuning to find the best regression estimator
pipe_rf = Pipeline([('rgr', RandomForestRegressor())])
pipe_xgb = Pipeline([('rgr', XGBRegressor())])

# Set grid search params
grid_params_rf = [{'rgr__n_estimators' : [200,500],
                   'rgr__criterion' : ['mse', 'absolute_error'], 
                   'rgr__max_features' : [200,400]}]

grid_params_xgb = [{'rgr__booster' : ['dart','gbtree'],
                    'rgr__eta' : [0.2,0.3,0.4],
                    'rgr__tree_method' : ['exact','approx']}]

# Create the grid search objects
gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=grid_params_rf,
                     scoring='neg_mean_squared_error',
                     cv=10,
                     n_jobs=-1)

gs_xgb = GridSearchCV(estimator=pipe_xgb,
                      param_grid=grid_params_xgb,
                      scoring='neg_mean_squared_error',
                      cv=10,
                      n_jobs=-1)

# List of grid pipelines
grids = [gs_rf, gs_xgb] 
# Grid dictionary for pipeline/estimator
grid_dict = {0:'RandomForestRegressor', 1: 'XGBoostRegressor'}

# Fit the pipeline of estimators using gridsearchcv
print('Fitting the gridsearchcv to pipeline of estimators...')

for gsid,gs in enumerate(grids):
    print('\nEstimator: %s. Start time: %s' %(grid_dict[gsid], datetime.datetime.now()))
    gs.fit(X_train, y_train)
    print('\n Best score : %.5f' % gs.best_score_)
    print('\n Best grid params: %s' % gs.best_params_)
    y_pred = gs.predict(X_test)
    
    # Output predicted y values into csv file, submit in kaggle competition and check score
    df_result = pd.DataFrame()
    df_result.index = test_data['id']
    df_result['loss'] = y_pred
    
    df_result.to_csv('./kagglenflbigdatabowl_submission.csv')


# Use Vecstack for model stacking. Only using the models which give good results
stacking_models = [RandomForestRegressor(), XGBRegressor()]

# Hyperparameter tuning of stacking model and use best result
for i in (4,6,8):
    print('Stacking Level0 for different estimators...')
    print('\nStart time: %s' %(datetime.datetime.now()))
    
    SM_Train, SM_Test = stacking(stacking_models,
                               X_train, y_train, X_test,
                               regression=True, 
                               mode='oof_pred_bag', 
                               needs_proba=False,  
                               save_dir=None,     
                               metric=None, 
                               n_folds=i, 
                               stratified=True,
                               shuffle=True,  
                               random_state=0,    
                               verbose=2)
    
    print('Predict using stacking Train Test data prediction...')
    print('\nStart time: %s' %(datetime.datetime.now()))
    SM_model = XGBRegressor()
    SM_model = SM_model.fit(SM_Train, y_train)
    y_pred = SM_model.predict(SM_Test)

    # Output predicted y values into csv file, submit in kaggle competition and check score
    df_result = pd.DataFrame()
    df_result.index = test_data['id']
    df_result['loss'] = y_pred
    df_result.to_csv('./kagglenflbigdatabowl_stackingsubmission.csv')




