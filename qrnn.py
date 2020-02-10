# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:18:57 2020

@author: gustav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QRNN(nn.Module):
    '''
    QRNN implementation in pytorch. Based on the kears implementation found at 
    https://github.com/atmtools/typhon/tree/master/typhon/retrieval/qrnn.
    '''

    def __init__(self, input_dim):
        super(QRNN, self).__init__()
        
        
        # an affine operation: y = Wx + b
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 9)   

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
       size = x.size()[1:]  # all dimensions except the batch dimension
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import GOES_GPM_Dataset
net = QRNN(784)



# load the data
import numpy as np
from sklearn.preprocessing import StandardScaler
xData =np.load('trainingData/xDataS100000_R28_P1000_R0.5.npy')
yData = np.load('trainingData/yDataS100000_R28_P1000_R0.5.npy')
times = np.load('trainingData/timesS100000_R28_P1000_R0.5.npy')
distance = np.load('trainingData/distanceS100000_R28_P1000_R0.5.npy') 
xData = np.reshape(xData, (len(xData),784))
# preprocess
scaler1 = StandardScaler()
scaler1.fit(xData)
xData = scaler1.transform(xData)

    
yData = yData/yData.max()


split_index = int(len(xData)*0.8)

x_train = xData[:split_index,:]
x_test = xData[split_index:,:]

y_train = yData[:split_index]
y_test = yData[split_index:]

#%%
def skewed_absolute_error(output, target, tau):
    """
    The quantile loss function for a given quantile tau:
    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)
    Where I is the indicator function.
    """
    dy = output - target
    #print(dy.size())
    return torch.mean((1.0 - tau) * torch.relu(dy) + tau * torch.relu(-dy), axis=-1)


def quantile_loss(output, target, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(
        torch.flatten(target), torch.flatten(output[:, 0]), taus[0])
    #print(e)
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(torch.flatten(target),
                                   torch.flatten(output[:, i + 1]),
                                   tau)
    return e

# define loss and optimizer
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.6) 
batch_size = 128
tau = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    print("here")
    #for i in range(2):
    for i in range(int(len(x_train)/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        batch_indexes = list(range(i*batch_size,(i+1)*batch_size))
        #print(batch_indexes)
        
        inputs = torch.tensor(np.reshape(x_train[batch_indexes,:],(batch_size,1,28,28)),dtype=torch.float)
        labels =torch.tensor( np.reshape(y_train[batch_indexes],(batch_size,)),dtype=torch.float)   
        #print(y_train[batch_indexes],)
        #print(labels)
        #print(labels.size())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        #print(outputs.size())
        #print(labels)
        #print(outputs)
        #loss = criterion(outputs, labels)
        loss = quantile_loss(outputs, labels, tau)
        #print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss/100))
            running_loss = 0.0

print('Finished Training')

#%% predict 
inputs = torch.tensor(np.reshape(x_test, (len(x_test),784)),dtype=torch.float)
        
output = net(inputs)

print(output.size())

#%%

import matplotlib.pyplot as plt

plt.plot(output.detach().numpy()[:,0])
plt.plot(y_test, alpha = 0.5)
plt.show()


#%%
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=10)

#%%print(type(testset))

for i,data in enumerate(testloader,0):
    print(i)