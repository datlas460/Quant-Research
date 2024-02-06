import configparser
import quandl as q
#nasdaq API key: zqJaFtiBCzzDx1i5C2sW


#config = configparser.ConfigParser()
#config.read()

#api_key = 'zqJaFtiBCzzDx1i5C2sW'

# data = q.get('BCHAIN/MKPRU', api_key='zqJaFtiBCzzDx1i5C2sW')
# data.info()
#
# print(data['Value'].resample('A').last())
#
# print('data2')
#
# data2 = q.get('FSE/SAP_X', start_date ='2018-1-1',end_date='2020-05-01',api_key='zqJaFtiBCzzDx1i5C2sW')
# data2.info()

#q.ApiConfig.api_key='zqJaFtiBCzzDx1i5C2sW'
print('1')
vol = q.get('CHRIS/ICE_G6.1', api_key='zqJaFtiBCzzDx1i5C2sW')
print('2')
vol.iloc[:,:10].info()

vol[['IvMean30', 'Ivmean60', 'Ivmean90']].tail()