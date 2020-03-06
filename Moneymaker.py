# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # RNN Moneymaker

# +
import numpy as np
import torch

import requests
import os

# +
av_api_key = os.environ.get('av_api_key')
base_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}"
stock = "MSFT"
url = base_url.format(stock, av_api_key)

r = requests.get(url) 
data = r.json()["Time Series (Daily)"]

print(len(data))

# +
X = []

for k, v in data.items():
    row = [ float(i[1]) for i in v.items() ]
    X.append(row)

# X[i] = [open, high, low, close, volume]
    
X = np.array(X)
print(X.shape)

# +
X_len = X.shape[0]
window_size = 1000
X_set, y_set = [], []

for i in range(X_len - window_size - 1):
    X_set.append(X[i:i+window_size])
    y_set.append(X[i+window_size][3]) # predict the closing price

X_set, y_set = np.array(X_set), np.array(y_set)
num_train = X_set.shape[0]

train_size = int(0.8 * num_train)
test_size = num_train - train_size
X_train, X_test = torch.utils.data.random_split(X_set, [train_size, test_size])
y_train, y_test = torch.utils.data.random_split(y_set, [train_size, test_size])

X_train, X_test = X_train.dataset, X_test.dataset
y_train, y_test = y_train.dataset, y_test.dataset

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
# -


