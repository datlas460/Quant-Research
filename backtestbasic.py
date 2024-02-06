

import pandas as pd
import matplotlib
from pylab import mpl,plt
import numpy as np


raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()

# raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
#                   index_col=0, parse_dates=True).dropna()
#
print(raw.info)

data = pd.DataFrame(raw['EUR='])
print(data)
data.rename(columns={'EUR=':'price'}, inplace=True)
print(data)
print(data.info())


data['SMA1'] = data['price'].rolling(42).mean()
data['SMA2']= data['price'].rolling(252).mean()
print(data.tail())

plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi']= 300
mpl.rcParams['font.family']='serif'

data.plot(title='EUR/USD | 42 & 252 days SMAs',figsize=(10,6))

#plt.show()

data['position']=np.where(data['SMA1']>data['SMA2'],1,-1)
data.dropna(inplace=True)
data['position'].plot(ylim=[-1.1, 1.1],
                      title='Market Positioning',
                      figsize=(10,6))
#plt.show()

data['returns']= np.log(data['price']/data['price'].shift(1))
data['returns'].hist(bins=35,figsize=(10,6))
print('returns values')
print(data['returns'])
#plt.show()

data['strategy'] = data['position'].shift(1) * data['returns']
print('sum of returns strategy')
print(data[['returns','strategy']].sum())
print('gross performance')
print(data[['returns', 'strategy']].sum().apply(np.exp))

data[['returns','strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))
#plt.show()

print('risk/return')
print(data[['returns','strategy']].mean()*252)
print('risk/return Standard Deviation')
print(data[['returns','strategy']].std()*252**0.5)

data['cumret']= data['strategy'].cumsum().apply(np.exp)
data['cummax']=data['cumret'].cummax()
data[['cumret', 'cummax']].dropna().plot(figsize=(10,6))
#plt.show()

drawdown = data['cummax']-data['cumret']
print('max drawdown')
print(drawdown.max())

temp=drawdown[drawdown==0]
print('temp')
print(temp)

print('periods')
periods=(temp.index[1:].to_pydatetime()-temp.index[:-1].to_pydatetime())
print(periods)

print('max period, longest drawdown')
print(periods.max())


