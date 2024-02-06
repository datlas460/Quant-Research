import numpy as np
import pandas as pd

# v = [1, 2, 3, 4, 5]
# a = np.array(v)
#
# print(a)
#
# print(type(a))
#
# print(2*a)
#
# a = np.arange(12).reshape((4,3))
# print(a)
#
# print(np.mean(a))
#
# print(np.mean(a, axis=1))

a = np.arange(15).reshape(5,3)
print(a)

columns = list('abc')
print(columns)

index = pd.date_range('2021-7-1', periods=5, freq='B')
print(index)

df = pd.DataFrame(a, columns=columns, index=index)
print(df)

print(df*2)

print(df**2)

print(df.sum())
print(df.mean())

print(df.a*0.5 + 2*df.b -df.c)

print(df[df['a']>5])
