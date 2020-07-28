import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from matplotlib import pyplot as plt

#Hyper parameters
EPOCH = 1 #train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50 
LR = 0.001 #Learing rate 
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./minst',
    train=True,
    transform=torchvision.transforms.ToTensor(), #(0,1)<-0~255
    download=DOWNLOAD_MNIST
)

##plot one example
#print(train_data.train_data.size()) #(60000, 28, 28)
##print(train_data.train_label.size()) #(60000)
#plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
#plt.title('%i'% train_data.train_labels[0])
#plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./minst/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(    #(dimension=1, length=28, width=28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2, #if stride = 1, padding = (kernel_size-1)/2
            ),    # -> (16, 28, 28)
            torch.nn.ReLU(),    # -> (16, 28, 28)
            torch.nn.MaxPool2d(kernel_size=2),    # -> (16, 14, 14)
        )
        self.conv2 = torch.nn.Sequential(    # -> (16, 14, 14)
            torch.nn.Conv2d(16,32,5,1,2),    # -> (32, 14, 14)
            torch.nn.ReLU(),    # -> (32, 14, 14)
            torch.nn.MaxPool2d(2)    # -> (32, 7, 7)
           )
        self.out = torch.nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)    #(batch, 32, 7, 7)
        x = x.view(x.size(0), -1)    #(batch, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for steps,(batch_x, batch_y) in enumerate(train_loader):
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)






    

