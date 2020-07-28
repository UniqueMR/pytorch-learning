import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_features ,n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1,n_hidden2)
        self.out = torch.nn.Linear(n_hidden2,n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x;

net = Net(1,10,10,1)

'''
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1))'''

'''
print(net)
'''

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.05)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    '''
    if t % 2 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(),'r-',lw = 2)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size':20, 'color': 'red'})
        plt.pause(0.1)
    '''

torch.save(net,'net.pkl')
torch.save(net.state_dict(),'net_parameters.pkl')

def restore_net():
    new_net = torch.load('net.pkl')
    return new_net;

def restore_parameters():
    new_net_para = Net(1,10,10,1)
    new_net_para.load_state_dict(torch.load(net_parameters.pkl))
    return new_net_para;

new_net = restore_net();
print(new_net)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(),'r-',lw = 2)        
plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size':20, 'color': 'red'})        


plt.ioff()
plt.show()

