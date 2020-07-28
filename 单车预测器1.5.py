import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

data_path = 'E:\hour.csv'
rides = pd.read_csv(data_path)
rides.head()
counts = rides['cnt'][:50]
x = np.arange(len(counts))
y = np.array(counts)
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


n_hidden = 10
lr = 0.0001
losses = []

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_out)

    def forward(self,x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return x

net = Net(len(counts),n_hidden,len(counts))
loss_func = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr)

plt.ion()

for epoch in range(10000):
    prediction = net(x)

    loss = loss_func(prediction,y)
    losses.append(loss.data.numpy())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:
        print('loss:',loss)
        plt.cla()
        xplot, = plt.plot(x.data.numpy(),y.data.numpy(),'o-')
        yplot, = plt.plot(x.data.numpy(),prediction.data.numpy())
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend([xplot,yplot],['Data','Prediction'])
        plt.pause(0.1)

plt.ioff()
plt.show()



