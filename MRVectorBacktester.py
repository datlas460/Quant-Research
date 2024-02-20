from MomVectorBacktester import *

class MRVectorBacktester(MomVectorBacktester):
    def run_strategy(self, SMA, threshold):
        #backtests trading strategy

        data = self.data.copy().dropna()
        data['sma'] = data['price'].rolling(SMA).mean()
        data['distance'] = data['price']=data['sma']
        data.dropna(inplace=True)

        #sell signals
        data['position'] = np.where(data['distance'] > threshold, -1, np.nan)

        #buy signals
        data['position'] = np.where(data['distance'] < -threshold, 1, data['position'])

        #crossing of current price and SMA (zero distance)
        data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0, 0, data['position'])

        data['position'] = data['position'].ffill().fillna(0)
        data['strategy'] = data['position'].shift(1) * data['return']

        #determine when a trade takes place
        trades = data['position'].diff().fillna(0) != 0

        #subtract transaction costs from return when trade takes place
        data['strategy'][trades] -= self.tc
        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)


