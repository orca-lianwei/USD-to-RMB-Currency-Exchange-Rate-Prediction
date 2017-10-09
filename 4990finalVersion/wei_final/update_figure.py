import sys;
import matplotlib.pyplot as plt
import datetime
import json
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
try:
    from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
except:
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ohlc
start = sys.argv[1];
end = sys.argv[2];
currency = sys.argv[3];
currency = currency + "=X";

start = start.split('-');
end = end.split('-');
start = [int(i) for i in start];
end = [int(i) for i in end];
start = (start[0],start[1],start[2])
end = (end[0],end[1],end[2]);
    
try:
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12

    #quotes = quotes_historical_yahoo_ohlc('CNY=X', start, end)
    quotes = quotes_historical_yahoo_ohlc(currency, start, end)
    if len(quotes) == 0:
        print(0)
        raise SystemExit
    data = list(quotes);
    for i in range(0,len(data)):
        data[i] = list(data[i]);
        data[i][0] = datetime.datetime.fromordinal(int(data[i][0])).strftime('%Y-%m-%d');
    print(data);
except:
    print(0);
