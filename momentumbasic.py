import matplotlib
from pylab import plt, mpl
import pandas as pd
import numpy as np

fn='https://hilpisch.com/pyalgo_eikon_eod_data.csv'


raw = pd.read_csv(fn,
                  index_col=0, parse_dates=True).dropna()
data = pd.DataFrame(raw['XAU='])
data.rename(columns={'XAU=': 'price'}, inplace=True)

data['returns']=np.log(data['price']/data['price'].shift(1))

data['position']= np.sign(data['returns'].rolling(3).mean())
data['strategy'] = data['position'].shift(1)*data['returns']

data[['returns','strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10,6))
plt.show()

to_plot = ['returns']
for m in [1, 3, 5, 7, 9]:
    data['position_%d' % m] = np.sign(data['returns'].rolling(m).mean())
    data['strategy_%d' % m] = (data['position_%d' % m].shift(1) *
                               data['returns'])
    to_plot.append('strategy_%d' % m)


data[to_plot].dropna().cumsum().apply(np.exp).plot(
    title='Gold  May 2020',
    figsize=(10, 6), style=['-', '--', '--', '--', '--', '--'])
plt.show()


