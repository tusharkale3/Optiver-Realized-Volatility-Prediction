# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:05:26 2023

@author: Group 10
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
#Charging all
book_train = pd.read_parquet("E:/BIG DATA PROject Alexis/book_train.parquet")
book_train.info()

#%%
#Charging only 0
book_train_0 = pd.read_parquet('E:/BIG DATA PROject Alexis/book_train.parquet/stock_id=0/c439ef22282f412ba39e9137a3fdabac.parquet')
book_train_0.info()

#%% 
#Charging somes
import glob
subset_paths = glob.glob('E:/BIG DATA PROject Alexis/book_train.parquet/stock_id=11*/*')
book_train_subset = pd.read_parquet(subset_paths)
book_train_subset.info()

#%%
#Defining WAP function
def wap(df,bid_price,ask_price,bid_size,ask_size):
    return (df[bid_price]*df[ask_size]+df[ask_price]*df[bid_size])/(df[bid_size]+df[ask_size])

def log_return(wap):
    return np.log(wap).diff()

def volatility(log_return):
    return np.sqrt(np.sum(log_return**2))


df=book_train_0[book_train_0['time_id']==5]
#Calculating WAP
df['wap1']=wap(df,'bid_price1','ask_price1','bid_size1','ask_size1')


#Ploting the WAP function
plt.figure('wap')
plt.title('WAP funtion order 0 - second in bucket 5')
plt.plot(df['seconds_in_bucket'],df['wap1'])
plt.xlabel('Seconds in bucket')
plt.ylabel('WAP')

#Calculating log return
df['wapLog']=df['wap1'].agg(log_return)

#Ploting the log return
plt.figure('log')
plt.title('log returns funtion order 0 - second in bucket 5')
plt.plot(df['seconds_in_bucket'],df['wapLog'])
plt.xlabel('Seconds in bucket')
plt.ylabel('log(returns)')

#Calculating volatility
vol = volatility(df['wapLog'])
print('Volatility for stock_id 0 on time_id 5:',vol)


#%%
#Now we use a definition for plotting
path='E:/BIG DATA PROject Alexis/' 

def visualize_book_data(stock_id, time_id):
    df_book = pd.read_parquet(path+'/book_train.parquet/stock_id='+str(stock_id))
    df_book = df_book[df_book['time_id'] == time_id]
    df_book['micro_price1'] = wap(df,'bid_price1','ask_price1','bid_size1','ask_size1')
    df_book['micro_price2'] = wap(df,'bid_price2','ask_price2','bid_size2','ask_size2')
    
    rv = volatility(log_return(df_book['micro_price1']))
    df_book.set_index('seconds_in_bucket', inplace=True)
    fig, axes = plt.subplots(figsize=(40, 30), nrows=2)
    axes[0].plot(df_book['micro_price1'], label='micro price 1', color = 'blue', linestyle="--")
    axes[0].plot(df_book['micro_price2'], label='micro price 2', color = 'blue', alpha=0.4, linestyle="--")
    axes[0].plot(df_book['ask_price1'], label="ask price 1", color = "green")
    axes[0].plot(df_book['bid_price1'], label="bid price 1", color = "red")
    axes[0].plot(df_book['ask_price2'], label="ask price 2", color = "green", alpha=0.4)
    axes[0].plot(df_book['bid_price2'], label="bid price 2", color = "red", alpha=0.4)
    axes[0].set_ylabel('price', size=20)
    axes[0].set_xlabel('seconds in bucket', size=20)
    axes[0].set_title(f'Price of stock {stock_id} for time_id {time_id} with current realized volatility: {rv}', size=30)
    
    axes[1].plot(df_book['bid_size1'], label='bid size 1', color = 'red')
    axes[1].plot(df_book['ask_size1'], label='ask size 1', color = 'green')
    axes[1].plot(df_book['bid_size2'], label='bid size 2', color = 'red', alpha=0.4)
    axes[1].plot(df_book['ask_size2'], label='ask size 2', color = 'green', alpha=0.4)
    axes[1].set_ylabel('size', size=20)
    axes[1].set_xlabel('seconds in bucket', size=20)
    axes[1].set_title(f'Size of stock {stock_id} for time_id {time_id} with current realized volatility: {rv}',
                     size=30)
    for i in range(2):
        axes[i].legend(prop={'size': 18})
        axes[i].tick_params(axis='x', labelsize=20, pad=10)
        axes[i].tick_params(axis='y', labelsize=20, pad=10)
    plt.show()

visualize_book_data(70,24600)


#%%
#Charging only 0 of test
book_test_0 = pd.read_parquet(r'E:/BIG DATA PROject Alexis/book_test.parquet/stock_id=0')
book_test_0.info()
