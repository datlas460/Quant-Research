#import csv
import pandas as pd
import quandl as q



fn = 'AAPL.csv'

# with open(fn, 'r') as f:
#     for _ in range(5):
#
#         print(f.readline(), end='')

# csv_reader = csv.DictReader(open(fn, 'r'))
#
# data = list(csv_reader)
#
# print(data[:5])

data = pd.read_csv(fn, index_col=0, parse_dates=True)
print(data.tail())