#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic, ccf
import matplotlib.pyplot as plt


# In[13]:


df = pd.read_csv('sample_time_series_data.csv')
df.head(1)


# In[14]:


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

input_cols = ['input_1','input_2','input_3','input_4','input_5','input_6']
for i in range(len(input_cols)):
    for j in range(i+1, len(input_cols)):
        ccres = ccf(df[input_cols[i]], df[input_cols[j]], adjusted=True)
        if ccres.max()>0.3:
            rs=ccres
            offset = np.floor(len(rs)/2)-np.argmax(rs)
            f,ax=plt.subplots(figsize=(14,3))
            ax.plot(rs)
            ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
            ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
            ax.set(title='Offset = {} frames\n {} leads <> {} leads'.format(offset, input_cols[i], input_cols[j]), xlabel='Offset',ylabel='Pearson r')

            plt.legend()

