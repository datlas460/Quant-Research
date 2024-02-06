import time
from sample_data import generate_sample_data
import tstables
import tables as tb
import datetime

start= time.time()
data = generate_sample_data(rows=2.5e6, cols=5, freq='1s').round(4)
end = time.time()
datagenerationtime = end-start
class desc(tb.IsDescription):
    timestamp = tb.Int64Col(pos=0)
    No0 = tb.Float64Col(pos=1)
    No1 = tb.Float64Col(pos=2)
    No2 = tb.Float64Col(pos=3)
    No3 = tb.Float64Col(pos=4)
    No4 = tb.Float64Col(pos=5)

start = time.time()
h5 = tb.open_file('data.h5ts','w')
ts = h5.create_ts('/', 'data', desc)
end=time.time()
h5creationtime = end-start
print(h5)

start = time.time()
ts.append(data)
end = time.time()
tsappendtime= end-start

starttime=time.time()
start = datetime.datetime(2021,1,25,12,30,0)
end = datetime.datetime(2021,1,30,12,30,0)
subset = ts.read_range(start, end)
endtime = time.time()
subsetreadtime = endtime-starttime

print('subset info')
print(subset.info())

print('data generation time')
print(datagenerationtime)
print('h5 creation time')
print(h5creationtime)
print('ts data insert time')
print(tsappendtime)
print('ts reading time')
print(subsetreadtime)



