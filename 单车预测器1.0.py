import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable

data_path = 'E:\hour.csv'
rides = pd.read_csv(data_path)
rides.head()
counts = rides['cnt'][:50]
x = np.arange(len(counts))
y = np.array(counts)
plt.figure(figsize=(10,7))
#plt.plot(x,y,'o-')
#plt.xlabel('X')
#plt.ylabel('Y')

x = Variable(torch.FloatTensor(x) / len(counts), requires_grad=True)
y = Variable(torch.FloatTensor(y), requires_grad=True)

n_hidden = 10
weights1 = Variable(torch.randn(1,n_hidden), requires_grad=True)
biases = Variable(torch.randn(n_hidden), requires_grad=True)
weights2 = Variable(torch.randn(n_hidden,1), requires_grad=True)

learning_rate = 0.0001
losses = []

for i in range (1000000):
    hidden = x.expand(n_hidden, len(x)).t() * weights1.expand(len(x),n_hidden) + biases.expand(len(x),n_hidden)

    hidden = torch.sigmoid(hidden)
    predictions = hidden.mm(weights2)
    
    loss = torch.mean((predictions - y)**2)
    losses.append(loss.data.numpy())

    if i % 10000 == 0:
        print('losses', loss)

    loss.backward()
    weights1.data.add_( - learning_rate * weights1.grad.data)
    biases.data.add_( - learning_rate * biases.grad.data)
    weights2.data.add_( - learning_rate * weights2.grad.data)

    weights1.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()


#plt.plot(loss)
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.show()

x_data = x.data.numpy()
y_data = y.data.numpy()
pred_data = predictions.data.numpy()       
plt.figure(figsize = (10,7))
xplot, = plt.plot(x_data,y_data,'-o')
yplot, = plt.plot(x_data,pred_data)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend([xplot,yplot],['Data','Prediction under 2000 epoches'])
plt.show()








