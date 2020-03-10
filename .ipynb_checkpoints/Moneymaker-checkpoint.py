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
seq = []
y = []
name_and_date = []
window_size = 122

all_files = os.listdir(root_path)

for i, filename in enumerate(all_files):
    
    # Only load a 1/N of all stocks
    if i % 50 == 0:
    
        len_stocks = len(all_files)
        print("Loading stock {}/{} ({})".format(i + 1, len_stocks, filename), end='\r')
        
        with open(root_path + filename) as f:
            if not filename.startswith('.'):
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
                    y.append(prices[i+window_size][3].item()) # predict the closing price
                    name_and_date.append([filename, temp2[i + window_size + 1]])
                    
X = torch.cat(seq, 0)
y = torch.tensor(y).unsqueeze(1) # from (N,) to (N,1)

print(X.shape)
print(y.shape)


# -


torch.save(X, 'assets/X.pt')
torch.save(y, 'assets/y.pt')

X = torch.load('assets/X.pt')
y = torch.load('assets/y.pt')

# ### Normalize the input across each window-sized sequence

# +
X_mean = torch.mean(X, dim=1).unsqueeze(1)
X_std = torch.std(X, dim=1).unsqueeze(1)

X = (X - X_mean) / X_std
y = (y - X_mean[:,:,3]) / X_std[:,:,3]

print(X.shape)
print(y.shape)
plt.plot(X[0])
# -

# ### Remove all inputs and corresponding outputs that containt NaN's

# +
print("old:")
print(X.shape)
print(y.shape)

a = torch.sum(torch.sum(X == X, dim=1), dim=1) != 0
b = (y == y) != 0

c = a if torch.sum(a) < torch.sum(b) else b

X = X[c]
y = y[c] # To make y (N, 1) from (N,)

print("new:")
print(X.shape)
print(y.shape)
# -

# ### Split data into training and validation sets

# +
N, S, D = X.shape
perm = np.random.permutation(N)
num_train = int(0.99*N)

train_selector = perm[0:num_train]
val_selector = perm[num_train:]

X_train = X[train_selector]
y_train = y[train_selector]
X_val = X[val_selector]
y_val = y[val_selector]

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)


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
    
        if i % 100==0:
            train_losses.append(loss.item())
            pXval = model(X_val)
            vloss = criterion(pXval, y_val)
            val_losses.append(vloss.item())
            print("training loss: {:<3.3f} \t val loss: {:<3.3f}".format(loss, vloss))
    
    ptrain = model(X_train)
    tloss = criterion(ptrain, y_train)
    train_losses.append(tloss.item())
    
    pXval = model(X_val)
    vloss = criterion(pXval, y_val)
    val_losses.append(vloss.item())
    epoch += 1    
    tend = time.time()
    print('epoch: {:<3d} \t time: {:<3.2f} \t training loss: {:<3.3f} \t val loss: {:<3.3f}'.format(epoch, 
            tend - tstart, tloss.item(), vloss.item()))
time_total = time.time() - t0
print('Total time: {:4.3f}, average time per epoch: {:4.3f}'.format(time_total, time_total / num_epochs))

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

# +
pred = model(X_val)

plt.plot((pred - y_val).detach())
plt.title('std difference')

# +
pred_abs = pred * X_std[val_selector][:,:,3] + X_mean[val_selector][:,:,3]
y_val_abs = y_val * X_std[val_selector][:,:,3] + X_mean[val_selector][:,:,3]

plt.plot((pred_abs - y_val_abs).detach())
plt.title('absolute price difference')

# +
val_names_and_dates = np.array(name_and_date)[val_selector]

stock = "PESI";
stock_selector = torch.tensor(val_names_and_dates[:,0] == (stock + ".txt"))

s_pred_abs = pred_abs[stock_selector]
s_y_val_abs = y_val_abs[stock_selector]

stock_dates = val_names_and_dates[stock_selector][:,1]
# -

from datetime import datetime

