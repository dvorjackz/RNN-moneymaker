import requests
import os
import json
import time

### Grab all NASDAQ tickers

tickers = []

with open('tickers.txt', 'r') as f:
    lines = f.readlines()
    for i in range(1, len(lines) - 1):
        tickers.append(lines[i].split('|')[0])
    
### Use (two?) api keys to get 20-year time series price data for each ticker

av_api_key1 = os.environ.get('av_api_key1')
# av_api_key2 = os.environ.get('av_api_key2')
base_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}"

for i, ticker in enumerate(tickers):

    print("Pulling {}, no. {}".format(ticker, i))

    # if i % 2 == 0:
    #     url = base_url.format(ticker, av_api_key1)
    # else:
    #     url = base_url.format(ticker, av_api_key2)

    url = base_url.format(ticker, av_api_key1)

    r = requests.get(url) 
    try:
        data = r.json()["Time Series (Daily)"]
    except KeyError as e:
        print('\n')
        print(r.json())
        print("Stopped at {}.".format(ticker))
        break

    with open('data/' + ticker + '.txt', 'w') as f:
        json.dump(data, f)
    
    print("Done.")

    time.sleep(12.2)