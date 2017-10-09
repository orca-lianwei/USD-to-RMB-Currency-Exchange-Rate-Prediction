#print data from now to 20 days before
import matplotlib.pyplot as plt
import datetime
import time
import json
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
try:
    from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
except:
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ohlc


# (Year, month, day) tuples suffice as args for quotes_historical_yahoo
date1 = time.strftime("%Y-%m-%d").split('-');
date2 = datetime.datetime.now() - datetime.timedelta(days=20)
date2 = date2.strftime("%Y-%m-%d").split('-');
date1 = [int(i) for i in date1]
date2 = [int(i) for i in date2]
date1 = (date1[0],date1[1],date1[2]);
date2 = (date2[0],date2[1],date2[2]);
#print(date1)
start = date2;
end = date1;
#date1 = (2017, 4, 1)
#date2 = (2017, 5, 1)


mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12
try:
    quotes = quotes_historical_yahoo_ohlc('CNY=X', start, end)
    if len(quotes) == 0:
        print(0);
        raise SystemExit
    data = list(quotes);
    for i in range(0,len(data)):
        data[i] = list(data[i]);
        data[i][0] = datetime.datetime.fromordinal(int(data[i][0])).strftime('%Y-%m-%d');
    print(data);
except:
    print(0);
# with open('fig_data.json','w') as outfile:
    # json.dump(data,outfile);
    # exit();

