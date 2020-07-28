import torch
import torch.nn.functional as F
import torch.utils.data as Data
from matplotlib import pyplot as plt

#hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

#Simulated data
x = torch.unsqueeze(torch.linspace(-1,1,1000),dim = 1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True)

#framework of nn
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        return x

#different nets
net_SGD = Net(1, 10, 1)
net_Momentum = Net(1, 10, 1)
net_RMSprop = Net(1, 10, 1)
net_Adam = Net(1, 10, 1)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

#optimizer to be compared
optimizer_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
optimizer_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
optimizer_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
optimizer_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9,0.99))
optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSprop, optimizer_Adam]

#loss function
loss_func = torch.nn.MSELoss()
losses = [[], [], [], []]

for epoch in range(EPOCH):
    print(epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        for net, optimizer, loss_history in zip(nets, optimizers, losses):
            output = net(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.data)#record loss

#print the final result
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, loss_history in enumerate(losses):
    plt.plot(loss_history, label = labels[i])

plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0, 0.2)
plt.show()


