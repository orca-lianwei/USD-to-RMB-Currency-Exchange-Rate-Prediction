import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
try:
    from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
except:
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ohlc
import time
import sys
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import *
import statsmodels.api as sm

def getdata():
#get data from 5000days ago which got 3569 entries
    date1 = time.strftime("%Y-%m-%d").split('-');
    date2 = datetime.datetime.now() - datetime.timedelta(days=5000)
    date2 = date2.strftime("%Y-%m-%d").split('-');
    date1 = [int(i) for i in date1]
    date2 = [int(i) for i in date2]
    date1 = (date1[0],date1[1],date1[2]);
    date2 = (date2[0],date2[1],date2[2]);
    start = date2;
    end = date1;
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12
    try:
        #quotes = quotes_historical_yahoo_ohlc('CNY=X', start, end)
        currency = sys.argv[2];
        currency = currency + "=X";
        quotes = quotes_historical_yahoo_ohlc(currency, start, end)
        if len(quotes) == 0:
            print(0);
            raise SystemExit
        data = list(quotes);
        for i in range(0,len(data)):
            data[i] = list(data[i]);
            data[i][0] = datetime.datetime.fromordinal(int(data[i][0])).strftime('%Y-%m-%d');
        return data
    except:
        print(0);
        return 0;

def predict (steps):
    #dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%y')
    #data = pd.read_csv('rateData.csv', parse_dates=True, index_col='Date',date_parser=dateparse)
    #ts = data['RMBperUSD']
    #new_period = ts['2015-08-11':]
    
    data = getdata();
    data = [[pd.datetime.strptime(x[0],'%Y-%m-%d')   ,x[4]]\
        for x in data]
    col = ['date','RMBperUSD']
    data = pd.DataFrame.from_records(data,columns=col,index = col[0]);
    new_period = data['RMBperUSD']
    
    new_period_log = np.log(new_period)
       
    model = ARIMA(new_period_log, order=(2, 1, 2),freq='D')
    results_ARIMA = model.fit(disp=-1)

    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    startDate = new_period.index[-1]
    endDate = new_period.index[-1] + Day(steps)
    
    ##dynamic_Sample_Prediction is error is future
    dynamic_Sample_Prediction = results_ARIMA.predict(startDate.isoformat(), endDate.isoformat(), exog = None, dynamic = False)

    ## dynamic_Sample_Prediction_log is error data with future error
    dynamic_Sample_Prediction_log = dynamic_Sample_Prediction.add(predictions_ARIMA_diff,fill_value=0)

    ## dynamic_Sample_Prediction_log_cumsum is cumulate sum of all errors
    dynamic_Sample_Prediction_log_cumsum = dynamic_Sample_Prediction_log.cumsum()

    ## expand 1.81094 to date + future date
    dynamic_ARIMA_log = pd.Series(new_period_log.ix[0], index=dynamic_Sample_Prediction_log.index)

    ## add 1.81094 to all cumsum respectively
    dynamic_ARIMA_log = dynamic_ARIMA_log.add(dynamic_Sample_Prediction_log_cumsum,fill_value=0)

    ## back to original data
    dynamic_Sample_Prediction_ARIMA = np.exp(dynamic_ARIMA_log)

    ##prediction data
    predictResult = dynamic_Sample_Prediction_ARIMA[new_period.index[-1]:]
    datelist = predictResult.index.tolist();
    datelist = [time.strftime("%Y-%m-%d",time.strptime(str(i),"%Y-%m-%d %H:%M:%S"))\
            for i in datelist]
    ratelist = predictResult.values.tolist();
    ans = []
    for i in range(0,len(datelist)):
        ans.append([datelist[i],ratelist[i]])
    return ans


ans = 0;
try:
    days = int(sys.argv[1])
    ans = predict(days)
except:
    ans = 0;
print(ans);
