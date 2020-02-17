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
        
        self.conv1 = nn.Conv2d(2, 20, 2, padding = 1)
        self.conv2 = nn.Conv2d(20, 50, 2, padding = 1)
        #self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(7*7*50, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)
        #self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(500, 9)   
        
        '''
        dim = 256
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)  
        self.fc4 = nn.Linear(dim, dim)
        self.fc5 = nn.Linear(dim, dim)
        self.fc6 = nn.Linear(dim, dim)  
        self.fc7 = nn.Linear(dim, dim)
        self.fc8 = nn.Linear(dim, dim)
        self.fc9 = nn.Linear(dim ,9) 
        ''' 
    def forward(self, x):
        
        
        x = F.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #x = self.dropout1(x)
        x = x.view(-1, 7*7*50)
        x = F.sigmoid(self.fc1(x))
        x = self.bn1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.sigmoid(self.fc3(x))
        x = self.bn3(x)
        #x = self.dropout3(x)
        x = self.fc4(x)
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
       
        x = self.fc9(x)
        '''
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




# load the data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
xData =np.load('trainingData/xDataC8C13S130000_R28_P200_R0.5.npy')
yData = np.load('trainingData/yDataC8C13S130000_R28_P200_R0.5.npy')
times = np.load('trainingData/timesC8C13S130000_R28_P200_R0.5.npy')
distance = np.load('trainingData/distanceC8C13S130000_R28_P200_R0.5.npy') 
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)

#xData = xData[:,:,10:18,10:18]
print(xData.shape)
xData = np.reshape(xData, (len(xData),28*28*2))

#scaler1 = StandardScaler()
#scaler1.fit(xData)
#xData = scaler1.transform(xData)
xData = preprocessing.normalize(xData, norm='l2')


    
#yData = yData/yData.max()


split_index = int(len(xData)*0.8)

x_train = xData[:split_index,:]
x_test = xData[split_index:,:]

y_train = yData[:split_index]
y_test = yData[split_index:]
#%%
import matplotlib.pyplot as plt
plt.plot(y_train)
plt.show()
plt.plot(y_test)
plt.show()
print(np.mean(y_test))
print(np.mean(y_train))

print(x_train.sum()/len(x_train))
print(x_test.sum()/len(x_test))

#%%
for i in range(20):
    plt.imshow(np.reshape(x_train[i*20,64:128], (8,8)))
    plt.title(y_train[i*20])
    plt.show()
    #print(y_train[i*20])
#%%
net = QRNN(28*28*3)
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
criterion = nn.MSELoss()
import random
optimizer = optim.SGD(net.parameters(), lr=0.1) 
batch_size = 128
tau = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
net.train(True)
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    
    suffle_indexes = random.sample(range(0, x_train.shape[0]), x_train.shape[0])
    x_train = x_train[suffle_indexes,:]
    y_train = y_train[suffle_indexes]
    #for i in range(2):
    for i in range(int(len(x_train)/batch_size)):
        # get the inputs; data is a list of [inputs, labels]
        batch_indexes = list(range(i*batch_size,(i+1)*batch_size))
        #print(batch_indexes)
        #print(x_train[batch_indexes,:,:,:].shape)
        inputs = torch.tensor(np.reshape(x_train[batch_indexes,:],(batch_size,2,28,28)),dtype=torch.float, requires_grad = True)
        labels =torch.tensor( np.reshape(y_train[batch_indexes],(batch_size,)),dtype=torch.float)   
        #print(y_train[batch_indexes],)
        #print(labels)
        #print(labels.size())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(inputs.size())
        #print(outputs.size())
        #print(labels.size())        
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
        
    '''
    with torch.no_grad():
        val_input = torch.tensor(np.reshape(x_test,(len(x_test),3,8,8)),dtype=torch.float,requires_grad = True) 
        val_labels = torch.tensor( np.reshape(y_test,(len(y_test),)),dtype=torch.float) 
        val_loss = quantile_loss(net(val_input),val_labels,tau)
        #val_loss = criterion(net(val_input),val_labels)
    
    
    '''
    val_loss =0
    print('[%d, %5d] loss: %.3f val_loss: %3f' %(epoch + 1, i + 1, running_loss/int(len(x_train)/batch_size), val_loss))    
    running_loss = 0.0
    
    

print('Finished Training')
#%% val loss
val_input = torch.tensor(np.reshape(x_test,(len(x_test),2,28,28)),dtype=torch.float,requires_grad = True) 
val_labels = torch.tensor( np.reshape(y_test,(len(y_test),)),dtype=torch.float) 
val_loss = quantile_loss(net(val_input),val_labels,tau)
print(val_loss)
#val_loss = criterion(net(val_input),val_labels)
#%% predict 
inputs = torch.tensor(np.reshape(x_test, (x_test.shape[0],2,28,28)),dtype=torch.float)
        
output = net(inputs)

print(output.size())

#%%
print(net.fc4.weight)

#%%

import matplotlib.pyplot as plt

plt.plot(output.detach().numpy()[:,2])
plt.plot(y_test, alpha = 0.5)
plt.show()

#%%
from visulize_results import plotIntervalPredictions
plotIntervalPredictions(y_test, np.reshape(output.detach().numpy()[:,5],(len(y_test),1)), 0.005)
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