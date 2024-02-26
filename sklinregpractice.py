import os
import random
import numpy as np
import matplotlib
import matplotlib_inline
from pylab import mpl, plt
import pandas as pd
from sklearn import linear_model

x = np.arange(12)
print(x)

lags = 3

m = np.zeros((lags+1, len(x) - lags))
print(m)

m[lags] = x[lags:]
for i in range(lags):
    m[i] = x[i:i-lags]

lm = linear_model.LinearRegression()

lm.fit(m[:lags].T, m[lags])

lm.predict(m[:lags].T)


hours = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 1.75, 2.,
                  2.25, 2.5, 2.75, 3., 3.25, 3.5, 4., 4.25,
                  4.5, 4.75, 5., 5.5])
success = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                    0, 1, 1, 1, 1, 1, 1])

plt.figure(figsize=(10,6))
plt.plot(hours, success, 'ro')
plt.ylim(-0.2, 1.2)
plt.show()

reg = np.polyfit(hours, success, deg=1)

plt.figure(figsize=(10, 6))
plt.plot(hours, success, 'ro')
plt.plot(hours, np.polyval(reg, hours), 'b')
plt.ylim(-0.2, 1.2)
plt.show()

lm = linear_model.LogisticRegression(solver='lbfgs')
hrs = hours.reshape(1, -1).T

lm.fit(hrs, success)

prediction = lm.predict(hrs)
plt.figure(figsize=(10, 6))
plt.plot(hours, success, 'ro', label='data')
plt.plot(hours, prediction, 'b', label='prediction')
plt.legend(loc=0)
plt.ylim(-0.2, 1.2)
plt.show()

prob = lm.predict_proba(hrs)
plt.figure(figsize=(10, 6))
plt.plot(hours, success, 'ro')
plt.plot(hours, prediction, 'b')
#probability of failure
plt.plot(hours, prob.T[0], 'm--', label='$p(h)$ for zero')
#probability of success
plt.plot(hours, prob.T[1], 'g-.', label='$p(h)$ for one')
plt.ylim(-0.2, 1.2)
plt.legend(loc=0)
plt.show()



