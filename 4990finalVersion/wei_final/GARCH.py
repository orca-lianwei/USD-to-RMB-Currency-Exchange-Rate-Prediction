import numpy as np
import pyflux as pf
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
import time
import sys

def GARCH(days,cur):
    cny_usd = web.DataReader(cur,  'yahoo', datetime(2015,8,12), datetime.utcnow().date())
    log_diff = pd.DataFrame(np.diff(np.log(cny_usd['Adj Close'].values)))
    log_diff.index = cny_usd.index.values[1:cny_usd.index.values.shape[0]]
    log_diff.columns = ['Exponential Index']

    garch_model = pf.GARCH(log_diff,p=1,q=1)
    garch_model.fit()
    ##dynamic_Sample_Prediction is error is future
    dynamic_Sample_Prediction = garch_model.predict(h=days)

    ## dynamic_Sample_Prediction_log is error data with future error
    dynamic_Sample_Prediction_log = log_diff.append(dynamic_Sample_Prediction)
    dynamic_Sample_Prediction_log.drop_duplicates(inplace=True)

    ## dynamic_Sample_Prediction_log_cumsum is cumulate sum of all errors
    dynamic_Sample_Prediction_log_cumsum = dynamic_Sample_Prediction_log.cumsum()

    ## add 1.81094 to all cumsum respectively (1.81094 is log(2015-8-12))
    dynamic_GARCH_log = dynamic_Sample_Prediction_log_cumsum.add(1.81094)

    ## back to original data
    dynamic_Sample_Prediction_GARCH = np.exp(dynamic_GARCH_log)

    #print dynamic_ARIMA_log
    return dynamic_Sample_Prediction_GARCH
    

days = int(sys.argv[1])
currency = sys.argv[2];
currency += "=X";
data = GARCH(days,currency);
data = data[-days:]
datelist = data.index.tolist();
datelist = [time.strftime("%Y-%m-%d",time.strptime(str(i),"%Y-%m-%d %H:%M:%S"))\
        for i in datelist]
ratelist = data.values.tolist();
ans = []
for i in range(0,len(datelist)):
    ans.append([datelist[i],ratelist[i][0]])
print(ans)


