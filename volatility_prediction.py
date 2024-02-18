# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:43:41 2024

@author: dbda
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path='D:\BigData_staging\BIGDATAPROYECT'    

#Generate the list of ID
import glob
list_order_book_file_train = glob.glob(path+"/book_train.parquet/*")
list_stock_id = [path.split("=")[1] for path in list_order_book_file_train]
list_stock_id = np.sort(np.array([int(i) for i in list_stock_id]))

#%%
#Defining important function
def wap(df,bid_price,ask_price,bid_size,ask_size):
    return (df[bid_price]*df[ask_size]+df[ask_price]*df[bid_size])/(df[bid_size]+df[ask_size])

def log_return(wap):
    return np.log(wap).diff()

def volatility(log_return):
    return np.sqrt(np.sum(log_return**2))

def relative_fluctuation(data):
    return np.std(data)/np.mean(data)

def preprocess_book(file_path):
    stock_id = file_path.split("=")[1]
    book_df=pd.read_parquet(file_path)
    
    book_df['micro_price1'] = wap(book_df,'bid_price1','ask_price1','bid_size1','ask_size1')
    book_df['log_return1'] = log_return(book_df['micro_price1'])
    
    book_df['micro_price2'] = wap(book_df,'bid_price2','ask_price2','bid_size2','ask_size2')
    book_df['log_return2'] = log_return(book_df['micro_price2'])
    
    # Spread normalized by "mean price"
    book_df['spread'] = 2*(book_df['ask_price1'] - book_df['bid_price1'])/(book_df['ask_price1'] 
                                                                           + book_df['bid_price1'])
    # # Low market depth could indicate a sharp future price movement in case of aggressive buy or sell 
    book_df['ask_depth'] = book_df['ask_size1'] + book_df['ask_size2'] 
    book_df['bid_depth'] = book_df['bid_size1'] + book_df['bid_size2']
    
    book_df['volume_imbalance'] = np.abs(book_df['ask_size1'] - 
                                         book_df['bid_size1'])*2/(book_df['ask_size1']+book_df['bid_size1'])
    
    aggregate = {
        'log_return1' : volatility,
        'log_return2' : volatility,
        'spread' : 'mean',
        'ask_depth' : 'mean',
        'bid_depth' : 'mean',
        'bid_price1': relative_fluctuation,
        'ask_size1': relative_fluctuation,
        'bid_price2': relative_fluctuation,
        'ask_size2': relative_fluctuation,            
        'volume_imbalance' : 'mean'
    }

    preprocessed_df = book_df.groupby('time_id').agg(aggregate)
    preprocessed_df = preprocessed_df.rename(columns={'log_return1':'volatility1',
                                                      'log_return2':'volatility2'})
    preprocessed_df_last_300 = book_df[book_df['seconds_in_bucket']>300].groupby('time_id').agg(aggregate)
    preprocessed_df_last_300 = preprocessed_df_last_300.rename(columns={'log_return1':'realized_volatility1_last_300',
                                                                'log_return2':'realized_volatility2_last_300',
                                                                'spread':'spread_last_300',
                                                                'ask_depth':'ask_depth_last_300',
                                                                'bid_depth':'bid_depth_last_300',
                                                                'volume_imbalance':'volume_imbalance_last_300'})
    
    preprocessed_df.reset_index(inplace=True)
    preprocessed_df_last_300.reset_index(inplace=True)
    preprocessed_df['row_id'] = preprocessed_df['time_id'].apply(lambda x:f'{stock_id}-{x}')
    preprocessed_df.drop('time_id', axis=1, inplace=True)
    preprocessed_df_last_300['row_id'] = preprocessed_df_last_300['time_id'].apply(lambda x:f'{stock_id}-{x}')
    preprocessed_df_last_300.drop('time_id', axis=1, inplace=True)
    return preprocessed_df.merge(preprocessed_df_last_300, how='left', on='row_id')
    # return preprocessed_df

def preprocess_trade(file_path):
    stock_id = file_path.split("=")[1]
    trade_df = pd.read_parquet(file_path)
    trade_df['size_total']=trade_df['size']
    trade_df['order_count_total']=trade_df['order_count']
    aggregate = {
        'price': relative_fluctuation,
        'size': relative_fluctuation,
        'order_count': relative_fluctuation,
        'size_total':'sum',
        'order_count_total': 'sum'
    }
    preprocessed_df = trade_df.groupby('time_id').agg(aggregate)
    preprocessed_df = preprocessed_df.rename(columns={'price':'trade_price_fluc', 
                                                      'size':'size_fluc', 
                                                      'order_count':'orders_fluc'})
    preprocessed_df.reset_index(inplace=True)
    preprocessed_df['row_id'] = preprocessed_df['time_id'].apply(lambda x:f'{stock_id}-{x}')
    preprocessed_df.drop('time_id', axis=1, inplace=True)
    
    preprocessed_df_last_300 = trade_df[trade_df['seconds_in_bucket']>300].groupby('time_id').agg(aggregate)
    preprocessed_df_last_300 = preprocessed_df_last_300.rename(columns={'price':'trace_price_rv_last_300', 
                                                                        'size':'volume_last_300', 
                                                                        'order_count':'number_of_orders_last_300'})
    preprocessed_df_last_300.reset_index(inplace=True)
    preprocessed_df_last_300['row_id'] = preprocessed_df_last_300['time_id'].apply(lambda x:f'{stock_id}-{x}')
    preprocessed_df_last_300.drop('time_id', axis=1, inplace=True)
    return preprocessed_df.merge(preprocessed_df_last_300, how='left', on='row_id')
    # return preprocessed_df


from joblib import Parallel, delayed

def prep_merge_trade_book(list_stock_id,state='train'):
    trade_book_df = pd.DataFrame()
    def job(stock_id,state=state):
        if state=='train':
            book_path = path+"/book_train.parquet/stock_id="+str(stock_id)
            trade_path = path+"/trade_train.parquet/stock_id="+str(stock_id)
        elif state=='test':
            book_path = path+"/book_test.parquet/stock_id="+str(stock_id)
            trade_path = path+"/trade_test.parquet/stock_id="+str(stock_id)  
        else:
            return print('Insert correct state: train/test')
        book_df = preprocess_book(book_path)
        trade_df = preprocess_trade(trade_path)
        temp_df = book_df.merge(trade_df, how='left', on='row_id')
        return(pd.concat([trade_book_df, temp_df]))
    
    trade_book_df = Parallel(n_jobs=-1, verbose=1) (delayed(job)(stock_id) for stock_id in list_stock_id)
    trade_book_df = pd.concat(trade_book_df)
    
    train_df = pd.read_csv(path+'/train.csv')
    train_df['row_id'] = train_df['stock_id'].astype(str) + '-' + train_df['time_id'].astype(str)
    train_df.drop(['stock_id', 'time_id'], axis=1, inplace=True)

    return trade_book_df.merge(train_df, how='left', on='row_id').reset_index(drop=True)

trade_book_df = prep_merge_trade_book(list_stock_id)
trade_book_df.to_csv('trade_book_df3.csv')

  # trade_book_df=pd.read_csv(path+'/trade_book_df3.csv')


#ALERT
#We have 19 nan values we will deleted
trade_book_df.dropna(inplace=True)

#%%
#Now we define the train test
#In total we have 428913 Row_id so will  take 20% for test this is 85.000 row and time id

from sklearn.model_selection import train_test_split

# trade_book_df.drop(['bid_price1','bid_price2','ask_size1','ask_size2'],axis=1,inplace=True)
trade_book_df.drop(['bid_price1_x', 'ask_size1_x', 'bid_price2_x', 'ask_size2_x',
                    'bid_price1_y', 'ask_size1_y', 'bid_price2_y', 'ask_size2_y'],axis=1,inplace=True)
trade_book_df_train,trade_book_df_test= train_test_split(trade_book_df,test_size=0.2,random_state=1)


trade_book_df_train.drop('Unnamed: 0',axis=1, inplace=True)
trade_book_df_test.drop('Unnamed: 0',axis=1, inplace=True)


#%%
#We see the correlation of the differents features
import seaborn as sns
# corr = trade_book_df[['volatility1', 'volatility2', 'trade_price_fluc', 'size_fluc',
       # 'orders_fluc', 'target']].corr()
# fig = plt.figure(figsize=(5, 4))
corr = trade_book_df_train.drop('row_id',axis=1).corr()
fig = plt.figure(figsize=(12, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title('Correlation features')
plt.savefig('big correlation train30', dpi=100)
plt.show()

#%%
#Now we only considerer the important features.
# trade_book_df_train.drop(['ask_depth', 'bid_depth', 'size_total',
#                           'order_count_total','volatility1','spread','orders_fluc'],axis=1,inplace=True)

# trade_book_df_test.drop(['ask_depth', 'bid_depth', 'size_total',
#                           'order_count_total','volatility1','spread','orders_fluc'],axis=1,inplace=True)

trade_book_df_train= trade_book_df_train[['row_id','volatility1','volatility2','realized_volatility1_last_300',
                                          'realized_volatility2_last_300','volume_imbalance',
                                         'trade_price_fluc','orders_fluc','target']]

trade_book_df_test=trade_book_df_test[['row_id','volatility1','volatility2','realized_volatility1_last_300', 
                                       'realized_volatility2_last_300','volume_imbalance',
                                         'trade_price_fluc','orders_fluc','target']]

#%%
#We see the histogram of the diferents features

trade_book_df_train.hist(figsize=(10,10),bins=200) 
plt.savefig('histogram of features300', dpi=100)

#We see a huge skewees so will use the scaler to transform in a normal distribution

#%%
#QuantileTransformer

from sklearn.preprocessing import QuantileTransformer
# from sklearn.compose import ColumnTransformer

trade_book_df_train.set_index('row_id',inplace=True)
trade_book_df_test.set_index('row_id',inplace=True)

y_train=trade_book_df_train['target']
y_test=trade_book_df_test['target']
X_train=trade_book_df_train.drop('target',axis=1)
X_test=trade_book_df_test.drop('target',axis=1)

qt_x=QuantileTransformer(output_distribution='normal').set_output(transform="pandas")
qt_y=QuantileTransformer(output_distribution='normal').set_output(transform="pandas")

# ct= ColumnTransformer(
#         remainder='passthrough', #passthough features not listed
#         transformers=[
#             ('std', qt , ['volatility2', 'volume_imbalance','trade_price_fluc',
#                    'size_fluc', 'target'])
#         ])

# ct.set_output(transform="pandas")

# X_train_scaled,y_train_scaled=ct.fit_transform(X_train,y_train)
# X_test_scaled,y_test_scaled=ct.transform(X_test,y_test)

X_train_scaled=qt_x.fit_transform(X_train)
X_test_scaled=qt_x.transform(X_test)

y_train_scaled=qt_y.fit_transform(pd.DataFrame(y_train))
y_test_scaled=qt_y.transform(pd.DataFrame(y_test))

X_train_scaled.join(y_train_scaled).hist(figsize=(10,10),bins=200) 
plt.savefig('histogram of features normal300b', dpi=100)

#%%
#Now we analize the data with knn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error

def rmspe(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square((y_true-y_pred)/y_true)))
    return loss

# df = pd.read_csv("./trade_book_df.csv", index_col=0)
# y = trade_book_df['target']
# X = trade_book_df.drop(['target', 'row_id'], axis=1)

knn=KNeighborsRegressor()
params={'n_neighbors':[15,100,500,1000],
        'weights':['uniform']}
kfold = KFold(n_splits=5, shuffle= True, random_state= 1)
rgcv=GridSearchCV(knn,param_grid=params,cv=kfold,scoring='r2',verbose=3)

rgcv.fit(X_train_scaled,y_train_scaled)
print(rgcv.best_params_)
print(rgcv.best_score_)
    
# scoring r2
# {'n_neighbors': 9}
# 0.6937467574489175

y_pred_scaled=rgcv.predict(X_test_scaled)
y_pred=qt_y.inverse_transform(y_pred_scaled)
#scores:
print('r2_score:', r2_score(pd.DataFrame(y_test),y_pred))
print('mean_squared_error:', mean_squared_error(pd.DataFrame(y_test),y_pred))
print('rmspe:', rmspe(pd.DataFrame(y_test),y_pred))
    
# scoring r2
# rmspe: 0.4133285602395383
# r2_score: 0.671660878541061
# mean_squared_error: 2.8263009473929675e-06

#With all the features
# {'n_neighbors': 9}
# 0.8134632252968972
# r2_score: 0.7775509552777532
# mean_squared_error: 1.909650837788346e-06
# rmspe: 0.2962377609243982

# {'n_neighbors': 14, 'weights': 'uniform'}
# 0.8197267743691239
# r2_score: 0.7845411154777341
# mean_squared_error: 1.8496426444565388e-06
# rmspe: 0.29011896301056317

# {'n_neighbors': 100, 'weights': 'uniform'}
# 0.8280252795712435
# r2_score: 0.7944724081649561
# mean_squared_error: 1.764385809912002e-06
# rmspe: 0.28542365963948113

#%%
#Now we tray with XGBRegressor
from xgboost import XGBRegressor

# Define hyperparameter grid
param_grid = {
    "n_estimators": [250,300,350],  # Number of trees 10, 100,300
    "learning_rate": [0.075, 0.1, 0.125],  # Learning rate
    "max_depth": [4, 5, 6],  # Maximum tree depth
    "colsample_bytree": [0.8, 0.9,0.95],  # Subsample ratio of columns
    "subsample": [0.7,0.8,0.9]  # Subsample ratio of features 0.8,1
}
kfold = KFold(n_splits=3, shuffle= True, random_state= 1)
# Create the XGBoost model
xgb_model = XGBRegressor()

# Create GridSearchCV object
gs_xb = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid,cv=kfold, 
    scoring='r2', verbose=3)

# Conduct randomized search
gs_xb.fit(X_train_scaled,y_train_scaled)

print('Best parameters',gs_xb.best_params_)
print('Best score',gs_xb.best_score_)

#Prediction
y_pred_scaled=gs_xb.predict(X_test_scaled)
y_pred=qt_y.inverse_transform(pd.DataFrame(y_pred_scaled))

#scores:
print('r2_score:', r2_score(pd.DataFrame(y_test),y_pred))
print('mean_squared_error:', mean_squared_error(pd.DataFrame(y_test),y_pred))
print('rmspe:', rmspe(pd.DataFrame(y_test),y_pred))
    
# scoring r2
# Best parameters {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
# Best score 0.722527456187711
# r2_score: 0.696937860963547
# mean_squared_error: 2.6087199322203925e-06
# rmspe: 0.3833798891315117

#using last300
# Best parameters {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
# Best score 0.8234728322597545
# r2_score: 0.7908750740115169
# mean_squared_error: 1.795267723513822e-06
# rmspe: 0.28464771192490085

#last300+volatility1
# Best parameters {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
# Best score 0.8291825487762917
# r2_score: 0.7938152251514328
# mean_squared_error: 1.7700275068404924e-06
# rmspe: 0.279516465317215

#using 300 volatility2 
# Best parameters {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
# Best score 0.8301626848216914
# r2_score: 0.795384103190476
# mean_squared_error: 1.7565592122681898e-06
# rmspe: 0.2745513232796968

#using 300 volatility2 + size_fluc BYE
# Best parameters {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}
# Best score 0.8316561312588867
# r2_score: 0.7974824541795368
# mean_squared_error: 1.738545569057281e-06
# rmspe: 0.2751983156439022

#refined
# Best parameters {'colsample_bytree': 0.8, 'learning_rate': 0.075, 'max_depth': 5, 'n_estimators': 350, 'subsample': 0.9}
# Best score 0.8311720883407596
# r2_score: 0.7964928902048224
# mean_squared_error: 1.7470406456520948e-06
# rmspe: 0.27512861521773435

#%%
# #Now SVM
# from sklearn.svm import SVR

# # Define hyperparameter grid
# param_grid = {
#     "C": [0.5, 1, 10],  # Regularization parameter
#     # "kernel": ["linear","poly","rbf"]  # Kernel type
#     #"gamma": ["scale", "auto"]  # Kernel coefficient for rbf
# }
# kfold = KFold(n_splits=3, shuffle= True, random_state= 1)
# # Create the SVR model
# svr = SVR(kernel="linear")

# # Create GridSearchCV object
# gs_svm = GridSearchCV(
#     estimator=svr, param_grid=param_grid,cv=kfold, 
#     scoring='r2', verbose=3)

# # Conduct randomized search
# gs_svm.fit(X_train_scaled,y_train_scaled['target'])

# print('Best parameters',gs_svm.best_params_)
# print('Best score',gs_svm.best_score_)
    
# # scoring r2


# y_pred_scaled=gs_svm.predict(X_test_scaled)
# y_pred=qt_y.inverse_transform(pd.DataFrame(y_pred_scaled))

# #scores:
# print('r2_score:', r2_score(pd.DataFrame(y_test),y_pred))
# print('mean_squared_error:', mean_squared_error(pd.DataFrame(y_test),y_pred))
# print('rmspe:', rmspe(pd.DataFrame(y_test),y_pred))
    
# # scoring r2

#%%
#Now Elasticnet
from sklearn.linear_model import ElasticNet

# Define hyperparameter grid
param_grid={'alpha':np.linspace(0.0001,1,10),
        'l1_ratio':np.linspace(0.0001,1,10)}


kfold = KFold(n_splits=3, shuffle= True, random_state= 1)
# Create the elastic model
el = ElasticNet()

# Create GridSearchCV object
gs_el = GridSearchCV(
    estimator=el, param_grid=param_grid,cv=kfold, 
    scoring='r2', verbose=3)

# Conduct randomized search
gs_el.fit(X_train_scaled,y_train_scaled['target'])

print('Best parameters',gs_el.best_params_)
print('Best score',gs_el.best_score_)
    

y_pred_scaled=gs_el.predict(X_test_scaled)
y_pred=qt_y.inverse_transform(pd.DataFrame(y_pred_scaled))

#scores:
print('r2_score:', r2_score(pd.DataFrame(y_test),y_pred))
print('mean_squared_error:', mean_squared_error(pd.DataFrame(y_test),y_pred))
print('rmspe:', rmspe(pd.DataFrame(y_test),y_pred))
   
#With all features 
# Best parameters {'alpha': 0.001, 'l1_ratio': 0.001}
# Best score 0.8234578649982144
# r2_score: 0.7822250352536889
# mean_squared_error: 1.8695254205132102e-06
# rmspe: 0.29155588541566374

#%%
#Now Elasticnet with poly
# from sklearn.linear_model import ElasticNet
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline

# poly= PolynomialFeatures(degree=5)
# # Define hyperparameter grid
# param_grid={'EL__alpha':np.linspace(0.001,1,3),
#         'EL__l1_ratio':np.linspace(0.001,1,3),
#         'POLY__degree':np.arange(3,6)}


# kfold = KFold(n_splits=3, shuffle= True, random_state= 1)
# # Create the elastic model
# el = ElasticNet()

# pipe= Pipeline([('POLY',poly),('EL',el)])

# # Create GridSearchCV object
# gs_elp = GridSearchCV(
#     estimator=pipe, param_grid=param_grid,cv=kfold, 
#     scoring='r2', verbose=3)

# # Conduct randomized search
# gs_elp.fit(X_train_scaled,y_train_scaled['target'])

# print('Best parameters',gs_elp.best_params_)
# print('Best score',gs_elp.best_score_)
    

# y_pred_scaled=gs_elp.predict(X_test_scaled)
# y_pred=qt_y.inverse_transform(pd.DataFrame(y_pred_scaled))

# #scores:
# print('r2_score:', r2_score(pd.DataFrame(y_test),y_pred))
# print('mean_squared_error:', mean_squared_error(pd.DataFrame(y_test),y_pred))
# print('rmspe:', rmspe(pd.DataFrame(y_test),y_pred))
   

# Best parameters {'EL__alpha': 0.001, 'EL__l1_ratio': 0.5005, 'POLY__degree': 4}
# Best score 0.8308566050608938
# r2_score: 0.7903124497658961
# mean_squared_error: 1.800097664965589e-06
# rmspe: 0.2884680112832094


#%%
#Stacking
from sklearn.ensemble import StackingRegressor,RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error

def rmspe(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square((y_true-y_pred)/y_true)))
    return loss

rdf=RandomForestRegressor(random_state=1)
knn=KNeighborsRegressor()
xgb_model = XGBRegressor()
el = ElasticNet()

kfold = KFold(n_splits=3, shuffle= True, random_state= 1)

stack=StackingRegressor([('knn',knn),('XG',xgb_model),('el',el)],
                         final_estimator=rdf)
params={'knn__n_neighbors':[100],
        "XG__n_estimators": [300],  # Number of trees 10, 100,300
        "XG__learning_rate": [0.075],  # Learning rate
        "XG__max_depth": [5],  # Maximum tree depth
        "XG__colsample_bytree": [0.8],  # Subsample ratio of columns
        "XG__subsample": [0.9],
        'el__alpha':[0.001],
        'el__l1_ratio':[0.001],
        'final_estimator__max_features':[2,5],
        'passthrough':[True,False]}

# params={'knn__n_neighbors':[80,100,120],
#         "XG__n_estimators": [250,300,350],  # Number of trees 10, 100,300
#         "XG__learning_rate": [0.075, 0.1],  # Learning rate
#         "XG__max_depth": [4, 5],  # Maximum tree depth
#         "XG__colsample_bytree": [0.8, 0.9],  # Subsample ratio of columns
#         "XG__subsample": [0.8,0.9],
#         'el__alpha':[0.001, 0.01],
#         'el__l1_ratio':[0.001],
#         'final_estimator__max_features':[2,3,4,5],
#         'passthrough':[True,False]}

gcv_stack=GridSearchCV(stack, param_grid=params,
                 cv=kfold,verbose=3)

gcv_stack.fit(X_train_scaled,y_train_scaled['target'])

print('Best parameters',gcv_stack.best_params_)
print('Best score',gcv_stack.best_score_)
    
y_pred_scaled=gcv_stack.predict(X_test_scaled)
y_pred=qt_y.inverse_transform(pd.DataFrame(y_pred_scaled))

#scores:
print('r2_score:', r2_score(pd.DataFrame(y_test),y_pred))
print('mean_squared_error:', mean_squared_error(pd.DataFrame(y_test),y_pred))
print('rmspe:', rmspe(pd.DataFrame(y_test),y_pred))
    
# scoring r2
# Best parameters {'XG__colsample_bytree': 0.8, 'XG__learning_rate': 0.075, 'XG__max_depth': 5, 'XG__n_estimators': 300, 'XG__subsample': 0.9, 'el__alpha': 0.001, 'el__l1_ratio': 0.001, 'final_estimator__max_features': 2, 'knn__n_neighbors': 100, 'passthrough': True}
# Best score 0.8286396965746069
# r2_score: 0.7928918683045912
# mean_squared_error: 1.7779542173298556e-06
# rmspe: 0.2813165608396096

#%%
#Now we create the test data for submision
# list_test_id = [0]

# trade_test_df = prep_merge_trade_book(list_test_id,'test')
# trade_test_df.to_csv('trade_test_df.csv')


# X_test=trade_test_df.drop(['target', 'row_id'],axis=1)
# y_pred=rgcv.predict(X_test)

# # array([0.00187766])

# ss = pd.read_csv(path+"\sample_submission.csv")
# ss['target']=y_pred
# ss.to_csv('optiver_knn.csv',index=False)