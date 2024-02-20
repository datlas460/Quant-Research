from pylab import plt, mpl
import pandas as pd
import numpy as np





class MeanReversionbasics(object):
    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()
    def get_data(self):
        '''Retrieves and prepares data
        '''
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return']= np.log(raw/raw.shift(1))
        self.data=raw


if __name__ == '__main__':
    mrb = MeanReversionbasics('GDX', '2010-1-1', '2020-12-31', 10000, 0.0)

    SMA = 25

    mrb.data['SMA']= mrb.data['price'].rolling(SMA).mean()

    threshold = 3.5

    mrb.data['distance'] = mrb.data['price']-mrb.data['SMA']

    mrb.data['distance'].dropna().plot(figsize=(10,6), legend=True)
    plt.axhline(threshold, color='r')
    plt.axhline(-threshold, color ='r')
    plt.axhline(0, color ='r')
    plt.show()


    mrb.data['position'] = np.where(mrb.data['distance'] > threshold, -1, np.nan)
    mrb.data['position'] = np.where(mrb.data['distance'] < -threshold, 1, mrb.data['position'])
    mrb.data['position'] = np.where(mrb.data['distance'] * mrb.data['distance'].shift(1) < 0, 0, mrb.data['position'])
    mrb.data['position'] = mrb.data['position'].ffill().fillna(0)
    mrb.data['position'].iloc[SMA:].plot(ylim=[-1.1,1.1], figsize=(10,6))
    plt.show()








