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
import csv
import re
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# -


root_path = 'scrape/data/'

# +
raw_data = {}
all_files = os.listdir(root_path)

for i, filename in enumerate(all_files):
    
    # Only load 1/N of all stocks
    if i % 3 == 0:
    
        len_stocks = len(all_files)
        print("Loading stock {}/{} ({})".format(i + 1, len_stocks, filename), end='\r')
        
        with open(root_path + filename) as f:
            if not filename.startswith('.'):
                data = json.load(f)
                                
                prices = []
                dates = []
                for k, v in data.items():
                    prices.append(np.array([ float(i[1]) for i in v.items() if i[0] != "5. volume" ]))
                    dates.append(k)
                
                # reverse so that data is increasing in time
                prices.reverse()
                dates.reverse()
                raw_data[filename.split('.')[0]] = (prices, dates)

print("")
print(len(raw_data))
print(y.shape)


# -


from datetime import date
print((datetime.strptime("2018-03-02", "%Y-%m-%d") - datetime.strptime("2018-03-01", "%Y-%m-%d")).days)

# +
X_train = []
y_train = []
X_val = []
y_val = []
window_size = 122

plen = 0
means_stds = {}
for i, items in enumerate(raw_data.items()):
    print("({}/{})".format(i, len(raw_data.items())), end="\r")
    k, v = items
    prices, _ = v
    if len(prices) < window_size:
        continue
    
    prices = torch.tensor(prices).float()
    mean = torch.mean(prices)
    std = torch.std(prices)
    prices = (prices - mean) / std
    means_stds[k] = (mean, std)
    
    plen += prices.shape[0] - window_size - 1
    if i % 5 != 0:
        for j in range(prices.shape[0] - window_size - 1):
            X_train.append(prices[j:j+window_size].unsqueeze(0))
            y_train.append(prices[j+window_size][3].item())
    else:
        for j in range(prices.shape[0] - window_size - 1):
            X_val.append(prices[j:j+window_size].unsqueeze(0))
            y_val.append(prices[j+window_size][3].item())

print("Converting to torch tensors...")
X_train = torch.cat(X_train)
y_train = torch.tensor(y_train).unsqueeze(1)
X_val = torch.cat(X_val)
y_val = torch.tensor(y_val).unsqueeze(1)
print("Done")
# -

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
print(plen == X_train.shape[0] + X_val.shape[0])


# +
# print("old:")
# print(X.shape)
# print(y.shape)

# a = torch.sum(torch.sum(X == X, dim=1), dim=1) != 0
# b = (y == y) != 0

# c = a if torch.sum(a) < torch.sum(b) else b

# X = X[c]
# y = y[c] # To make y (N, 1) from (N,)

# print("new:")
# print(X.shape)
# print(y.shape)
# -

# ### Training a RNN

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


batch_size = 32
dataset = Dataset(X_train, y_train)
loader = DataLoader(dataset, batch_size)


class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h=None):
        if type(h) == type(None):
            out, hn = self.rnn(x)
        else:
            out, hn = self.rnn(x, h.detach())
        out = self.drop(out)
        out = self.fc(out[:, -1, :])
        return out


input_dim = 4
hidden_dim = 20
output_dim = 1

model = RNNClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

train_accs = []
val_accs = []
train_losses = []
val_losses = []
epoch = 0

# +
model = RNNClassifier(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load('assets/partial_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.train()

# +
t0 = time.time()
num_epochs = 3
for ep in range(num_epochs):
    tstart = time.time()
    for i, data in enumerate(loader):
        print(i, end='\r')
        optimizer.zero_grad()
        outputs = model(data[0])
        loss = criterion(outputs, data[1])
        loss.backward()
        optimizer.step()
    
        if i % 1000==0:
            train_losses.append(loss.item())
            pXval = model(X_val)
            vloss = criterion(pXval, y_val)
            val_losses.append(vloss.item())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'assets/partial_model.pt')

            print("training loss: {:<3.3f} \t val loss: {:<3.3f}".format(loss, vloss))
    
#     ptrain = model(X_train)
#     tloss = criterion(ptrain, y_train)
#     train_losses.append(tloss.item())
    
    pXval = model(X_val)
    vloss = criterion(pXval, y_val)
    val_losses.append(vloss.item())
    epoch += 1    
    tend = time.time()
    print('epoch: {:<3d} \t time: {:<3.2f} \t val loss: {:<3.3f}'.format(epoch, 
            tend - tstart, vloss.item()))
time_total = time.time() - t0
print('Total time: {:4.3f}, average time per epoch: {:4.3f}'.format(time_total, time_total / num_epochs))
# -

torch.save(model, 'assets/model.pt')

# ### Model evaluation

model = torch.load('assets/model.pt')

t_losses = [i for i in train_losses if i < 4000]
plt.plot(t_losses)
plt.plot(val_losses)
plt.title('loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.legend(['train', 'val'])

model.eval()

pred = model(X_val)
print(criterion(pred, y_val))

# ### Model predictions compared to standard deviations

# +
pred = model(X_val)

plt.plot((pred - y_val).detach()[1000:1100])
plt.title('std difference')
# -

# ### Model predictions compared to actual price

# +
pred_abs = pred * X_std[val_selector][:,:,3] + X_mean[val_selector][:,:,3]
y_val_abs = y_val * X_std[val_selector][:,:,3] + X_mean[val_selector][:,:,3]

plt.plot((pred_abs - y_val_abs).detach()[1000:1100])
plt.title('absolute price difference')
# -

# ### Testing and visualizing random stock in test data

# +
val_names_and_dates = np.array(name_and_date)[val_selector]

stock = "AUTO";
stock_selector = torch.tensor(val_names_and_dates[:,0] == (stock + ".txt"))

s_pred_abs = pred_abs[stock_selector]
s_y_val_abs = y_val_abs[stock_selector]

stock_dates = val_names_and_dates[stock_selector][:,1]
# -

stock_dates.sort()
fig, ax = plt.subplots()
ax.plot(stock_dates[10:20], s_pred_abs.detach()[10:20])
ax.plot(stock_dates[10:20], s_y_val_abs.detach()[10:20])
fig.autofmt_xdate()
start, end = ax.get_xlim()
plt.show()

# ### Testing random unseen stock (skipped over in the data loading stage)

# +
seq = []
test_y = []
test_name_and_date = []
window_size = 122

with open(root_path + "AAPL.txt") as f:
    data = json.load(f)

    temp1 = [] # for prices
    temp2 = [] # for name and date

    for k, v in data.items():
        temp1.append(torch.tensor([ float(i[1]) for i in v.items() if i[0] != "5. volume" ]).unsqueeze(0))
        temp2.append(k)

    # reverse so that data is increasing in time
    temp1.reverse()
    temp2.reverse()

    prices = torch.cat(temp1, 0)

    for i in range(len(prices) - window_size - 1):
        seq.append(prices[i:i+window_size].unsqueeze(0))
        test_y.append(prices[i+window_size][3].item()) # predict the closing price
        test_name_and_date.append(['AAPL', temp2[i + window_size + 1]])
        
test_X = torch.cat(seq, 0)
test_y = torch.tensor(test_y).unsqueeze(1) # from (N,) to (N,1)

test_X_mean = torch.mean(test_X, dim=1).unsqueeze(1)
test_X_std = torch.std(test_X, dim=1).unsqueeze(1)

test_X = (test_X - test_X_mean) / test_X_std
test_y = (test_y - test_X_mean[:,:,3]) / test_X_std[:,:,3]

test_pred = model(test_X)
print(criterion(test_pred, test_y))

test_pred = test_pred * test_X_std[:,:,3] + test_X_mean[:,:,3]
test_y = test_y * test_X_std[:,:,3] + test_X_mean[:,:,3]

plt.plot(test_pred.detach()[3000:3100])
plt.plot(test_y[3000:3100])
# -

dates = [i[1] for i in test_name_and_date]
print(dates)
temp_pred = [ i.data[0] for i in test_pred.detach().numpy() ]
temp_y = [ i.data[0] for i in test_y.detach().numpy() ]

# +
import matplotlib.ticker as mticker

fig, ax = plt.subplots()
ax.plot(dates[1000:1010], temp_pred[1000:1010])
fig.autofmt_xdate()
start, end = ax.get_xlim()
plt.show()
# -

fig, ax = plt.subplots()
ax.plot(dates[1000:1010], temp_y[1000:1010])
fig.autofmt_xdate()
start, end = ax.get_xlim()
plt.show()
