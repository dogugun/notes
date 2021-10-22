

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic, ccf
import matplotlib.pyplot as plt

```


```python
df = pd.read_csv('sample_time_series_data.csv')
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>target</th>
      <th>input_1</th>
      <th>input_2</th>
      <th>input_3</th>
      <th>input_4</th>
      <th>input_5</th>
      <th>input_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-06-01 01:00:00.000</td>
      <td>77.301848</td>
      <td>7956.306758</td>
      <td>4.504717</td>
      <td>11.588595</td>
      <td>106.005488</td>
      <td>12.806226</td>
      <td>55.120697</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


![png](plots\output_2_0.png)



![png](plots\output_2_1.png)



![png](plots\output_2_2.png)



![png](plots\output_2_3.png)



![png](plots\output_2_4.png)



![png](plots\output_2_5.png)



![png](plots\output_2_6.png)



![png](plots\output_2_7.png)



![png](plots\output_2_8.png)



![png](plots\output_2_9.png)



![png](plots\output_2_10.png)



![png](plots\output_2_11.png)



![png](plots\output_2_12.png)



![png](plots\output_2_13.png)



![png](plots\output_2_14.png)

