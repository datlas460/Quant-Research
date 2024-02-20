from pylab import plt, mpl
import pandas as pd
import numpy as np

class MomVectorBacktester(object):

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

    def run_strategy(self, momentum=1):
        '''backtests the trading strategy
        '''
        self.momentum = momentum
        data = self.data.copy().dropna()
        data['position']=np.sign((data['return'].rolling(momentum)).mean())
        data['strategy']=data['position'].shift(1) * data['return']

        #determine when a trade takes places
        data.dropna(inplace=True)
        trades = data['position'].diff().fillna(0) != 0

        #subtract the transaction costs
        data['strategy'][trades] -= self.tc
        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data

        #abs performance of strategy
        aperf = self.results['cstrategy'].iloc[-1]

        #over/under performance of the strategy
        operf = aperf - self.results['creturns'].iloc[-1]

        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        '''plots cum performance of strategy'''
        if self.results is None:
            print('Run strategy before plotting. No data available')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns','cstrategy']].plot(title=title, figsize=(10,6))
        plt.show()

if __name__ == '__main__':
    # mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31', 10000, 0.0)
    # print(mombt.run_strategy())
    # print(mombt.run_strategy(momentum=2))
    # mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31', 10000, 0.1)
    # print(mombt.run_strategy(2))

    mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31', 10000, 0.0)
    mombt.run_strategy(momentum=3)
    print(mombt.run_strategy(momentum=3))
    mombt.plot_results()

    mombt= MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31', 10000, 0.001)
    mombt.run_strategy(momentum=3)
    print(mombt.run_strategy(momentum=3))
    mombt.plot_results()
    plt.show()










