import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
import time
import sys
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import *
import statsmodels.api as sm
import warnings

#provide different currency
def getcurrecy():
    currency = ['CNY=X', 'GBP=X', 'MXN=X', 'INR=X']

#grap currency exchange data from Yhaoo Finance
def getdata(x):
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
        quotes = quotes_historical_yahoo_ohlc(x, start, end)
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

def deep_predict (cur,steps):
    #modify dataset
    data = getdata(cur);
    data = [[pd.datetime.strptime(x[0],'%Y-%m-%d')   ,x[4]]\
        for x in data]
    col = ['date','perUSD']
    data = pd.DataFrame.from_records(data,columns=col,index = col[0]);
    new_period = data['perUSD']
    new_period_log = np.log(new_period)
    
    #optimal parameters
    f = rss(cur,steps)
    p = int(f['optimal'].AR)
    d = int(f['optimal'].Diff)
    q = int(f['optimal'].MA)

    #main model
    model = ARIMA(new_period_log, order=(p, d, q),freq='D')
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

    
    dynamic_ARIMA_log = pd.Series(new_period_log.ix[0], index=dynamic_Sample_Prediction_log.index)
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




#define function to calculate Residual Sum of Square of errors and gain the optimal ARIMA paramaters
def rss (cur,steps):
    t1 = time.time()
    warnings.filterwarnings('ignore')
    
    rss = pd.DataFrame(columns=('AR', 'Diff', 'MA', 'MSE'))
    n = 0
    
    #define train and test data
    data = getdata(cur);
    data = [[pd.datetime.strptime(x[0],'%Y-%m-%d')   ,x[4]]\
            for x in data]
    col = ['date','perUSD']
    data = pd.DataFrame.from_records(data,columns=col,index = col[0]);
    new_period = data['perUSD']
    new_period_log = np.log(new_period)
    
    new_period_train = new_period[:int(len(new_period_log)*0.8)] #train data
    train = new_period_log[0:int(len(new_period_log)*0.8)] #train data after removing noise
    test_data = data[int(len(data)*0.8):]
    
    #find the optimal ARIMA paramaters
    for p in range(1,4):
        for q in range(1,4):
            #for d in range(1,2):
            try:
                train_model = ARIMA(train, order=(p, 1, q),freq='D')
                #train_model = ARIMA(train, order=(p, 1, q),freq='D')
                train_results_ARIMA = train_model.fit(disp=-1)
                train_predictions_ARIMA_diff = pd.Series(train_results_ARIMA.fittedvalues, copy=True)
                startDate = new_period_train.index[-1]
                endDate = new_period_train.index[-1] + Day(steps)
                train_dynamic_Sample_Prediction = train_results_ARIMA.predict(startDate.isoformat(), endDate.isoformat(), exog = None, dynamic = False)
                train_dynamic_Sample_Prediction_log = train_dynamic_Sample_Prediction.add(train_predictions_ARIMA_diff,fill_value=0)
                train_dynamic_Sample_Prediction_log_cumsum = train_dynamic_Sample_Prediction_log.cumsum()
                train_dynamic_ARIMA_log = pd.Series(train.ix[0], index=train_dynamic_Sample_Prediction_log.index)
                train_dynamic_ARIMA_log = train_dynamic_ARIMA_log.add(train_dynamic_Sample_Prediction_log_cumsum,fill_value=0)
                train_dynamic_Sample_Prediction_ARIMA = np.exp(train_dynamic_ARIMA_log)
                    
                train_predictResult = train_dynamic_Sample_Prediction_ARIMA[new_period_train.index[-1]:]
                train_datelist = train_predictResult.index.tolist();
                    
                train_datelist = [time.strftime("%Y-%m-%d",time.strptime(str(i),"%Y-%m-%d %H:%M:%S"))\
                                      for i in train_datelist]
                train_ratelist = train_predictResult.values.tolist()
                    
                train_ans = []
                for i in range(0,len(train_datelist)):
                    train_ans.append([train_datelist[i],train_ratelist[i]])
                        
                    
                diff=[]
                for i in range(1,steps+1):
                    diff.append((train_ans[i][1] - test_data.iloc[i-1]) ** 2)
                    
                mse = np.mean(diff)
                rss.loc[n] = [p,1,q,mse]
                n += 1
                    
            except:
                continue
    t2 = time.time()
    t = t2 - t1
    optimal = rss[rss.MSE == rss.MSE.min()] #return optimal p,d,q
    
    return {'rss_table':rss, 'optimal':optimal, 'time':t}
    #return optimal


def predict (cur,steps):
    #global p
    #global q
    #modify dataset
    data = getdata(cur);
    data = [[pd.datetime.strptime(x[0],'%Y-%m-%d')   ,x[4]]\
        for x in data]
    col = ['date','perUSD']
    data = pd.DataFrame.from_records(data,columns=col,index = col[0]);
    new_period = data['perUSD']
    new_period_log = np.log(new_period)
    #optimal parameters
    if cur == 'CNY=X' or cur== 'CAD=X' or cur == 'AUD=X' or cur == 'JPY=X':
        p = 2
        q = 2
    elif cur == 'GBP=X' or cur== 'MXN=X':
        p = 2
        q = 1
    elif cur == 'INR=X':
        p = 1
        q = 1 
    elif cur == 'EUR=X':
        p = 2
        q = 3
    

    #main model
    model = ARIMA(new_period_log, order=(p, 1, q),freq='D')
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

    
    dynamic_ARIMA_log = pd.Series(new_period_log.ix[0], index=dynamic_Sample_Prediction_log.index)
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
    currency = sys.argv[2];
    currency += "=X";
    ans = predict(currency,days)
except:
    ans = 0;
print(ans);
