import os
import random
import numpy as np
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

#print evenly spaced grid of floats for x values between 0 and 10
x = np.linspace(0,10)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)

#fix the seed value for all random generators
set_seeds()

#generate randomized data for y values
y = x + np.random.standard_normal(len(x))

#ols regression of degree 1, which means linear regression
reg = np.polyfit(x, y, deg=1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='data')
plt.plot(x, np.polyval(reg, x), 'r', lw = 2.5, label='linear regression')
plt.legend(loc=0)


#the same as above but enlarged x domain
plt.figure(figize=(10,6))
plt.plot(x, y, 'bo', label='data')
xn = np.linspace(0, 20)
plt.plot(xn, np.polyval(reg, xn), 'r', lw=2.5, label='linear regression')
plt.legend(loc=0)
plt.show()


