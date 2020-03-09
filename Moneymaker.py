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

import json
import os

# +
X = []
y = []
window_size = 122

# Only load a third of all stocks because of processing issues

for i, filename in enumerate(os.listdir('scrape/data')):
    
    if i % 3 == 0:
    
        len_stocks = len(os.listdir('scrape/data'))
        print("Loading stock {}/{}".format(i + 1, len_stocks))
        with open('scrape/final.txt', 'w') as f:
            with open('scrape/data/' + filename) as f:
                if not filename.startswith('.'):
                    print(filename)
                    prices = []
                    data = json.load(f)

                    for k, v in data.items():
                        row = [ float(i[1]) for i in v.items() ]
                        prices.append(row)

                    for i in range(len(prices) - window_size - 1):
                        X.append(prices[i:i+window_size])
                        y.append(prices[i+window_size][3]) # predict the closing price


# -


X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# +
num_train = X.shape[0]

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
