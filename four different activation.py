import torch
import torch.nn.functional as Fuc
from torch.autograd import Variable
from matplotlib import pyplot as plt

#fake data
x = torch.linspace(-5, 5,200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = Fuc.relu(x).data.numpy()
y_sigmoid = Fuc.sigmoid(x).data.numpy()
y_tanh = Fuc.tanh(x).data.numpy()
y_softplus = Fuc.softplus(x).data.numpy()

plt.figure(1, figsize = (8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, color='red', label = 'relu')
plt.ylim((-1, 5))
plt.legend(loc = 'best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, color='red', label = 'sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc = 'best')

plt.subplot(223)
plt.plot(x_np, y_tanh, color='red', label = 'tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc = 'best')

plt.subplot(224)
plt.plot(x_np, y_softplus, color='red', label = 'softplus')
plt.ylim((-0.2, 6))
plt.legend(loc = 'best')

plt.show()

















