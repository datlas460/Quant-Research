import SMAVectorBacktester as SMA
import matplotlib
from pylab import plt, mpl

smabt = SMA.SMAVectorBacktester('EUR=', 42, 252, '2010-1-1', '2019-12-31')
print(smabt.run_strategy())

print(smabt.optimize_parameters((30,50,2), (200,300,2)))

smabt.plot_results()
plt.show()

