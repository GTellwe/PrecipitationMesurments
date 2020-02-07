# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# get training data
#xData, yData , times, distance = getTrainingData(50000,1000, 0.5)   

#%% load data
import numpy as np
xData = np.load('trainingData/xData_50k_R28_D50_2017-08_02-2017_10_01.npy')
yData = np.load('trainingData/yData_50k_R28_D50_2017-08_02-2017_10_01.npy')
times = np.load('trainingData/times_50k_R28_D50_2017-08_02-2017_10_01.npy')
distance = np.load('trainingData/distance_50k_R28_D50_2017-08_02-2017_10_01.npy')  
#%% 
plotTrainingData(xData,yData, times, 20)
#%% plot time differences
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.abs(times[:,0]-times[:,1]))

#%% plot distances
import matplotlib.pyplot as plt
plt.plot(distance)
#%%

#%% select the closest data


indexes = np.where(np.abs(times[:,0]-times[:,1]) <200)
print(len(indexes[0]))
print(xData.shape)
print(yData.shape)
print(times.shape)

xData = xData[indexes[0],:,:]
yData = yData[indexes[0]]
times = times[indexes[0],:]
print(xData.shape)
print(yData.shape)
print(times.shape)
#%% print how many rain days there is
print(len(np.where(yData == 0)[0]))
import matplotlib.pyplot as plt
plt.plot(yData)

#%%
#plotTrainingData(xData,yData,times, 20)



#%%
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
scaler1 = StandardScaler()
'''
# reshape data for the QRNN
newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]))

newYData = np.reshape(yData,(len(yData),1))
'''
'''
indexes_of_zeros = np.where(newYData[:,0] == 0)
indexes_of_non_zeros = np.where(newYData[:,0] > 0)
print(len(indexes_of_zeros[0]))
indexes_zeros = random.sample(range(0, len(indexes_of_zeros[0])),len(indexes_of_non_zeros[0]))
tmpX = np.zeros((len(indexes_of_non_zeros[0])+len(indexes_of_non_zeros[0]),xData.shape[1]*xData.shape[2]))
tmpY = np.zeros((len(indexes_of_non_zeros[0])+len(indexes_of_non_zeros[0]),1))

tmpY[:len(indexes_of_non_zeros[0]),0] =  newYData[indexes_of_non_zeros,0]
tmpY[len(indexes_of_non_zeros[0]):,0] =  newYData[indexes_zeros,0]

tmpX[:len(indexes_of_non_zeros[0]),:] =  newXData[indexes_of_non_zeros,:]
tmpX[len(indexes_of_non_zeros[0]):,:] =  newXData[indexes_zeros,:]

newOrder = random.sample(range(0,tmpX.shape[0]),tmpX.shape[0])

newXData = tmpX[newOrder,:]
newYData = tmpY[newOrder,:]

print(newXData)
'''
#%%
'''
tmp = np.zeros((xData.shape[0],xData.shape[1]*xData.shape[2]+2))
tmp[:,:xData.shape[1]*xData.shape[2]] = newXData
tmp[:,-1] = times[:,0]-times[:,1]
tmp[:,-2] = distance[:,0]
newXData = tmp

scaler1.fit(newXData)
newXData = scaler1.transform(newXData)
newYData = newYData/newYData.max()
'''
#print(newXData)
#%%
print(newYData.shape)
#%%#%%
import tensorflow as tf
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
newXData = in_data
input_dim = newXData.shape[1]
model = qrnn.QRNN(input_dim,quantiles, depth = 10, activation = 'relu')

# preprocess the data
#newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)
newYData = yData/yData.max()
newYData = np.reshape(newYData,(50000,1))
# split into training and test sets

cut_index = 40000
xTrain = newXData[:cut_index,:]
xTest = newXData[cut_index:,:]

yTrain = newYData[:cut_index]
yTest = newYData[cut_index:]

from sklearn import preprocessing
import numpy as np
#%%
import matplotlib.pyplot as plt
plt.plot(yTrain)

#%% fit the model
model.fit(x_train = xTrain, y_train = yTrain,x_val = xTest, y_val = yTest,batch_size = 128,maximum_epochs = 25)

#%%save model
model.save('model.h5')

#%%
prediction = model.predict(xTest)

#%%
mean = np.zeros((xTest.shape[0],1))
for i in range(xTest.shape[0]):
    
    mean[i,0] = model.posterior_mean(xTest[i,:])
    
#print(mean)
#%%
import matplotlib.pyplot as plt
plt.plot(mean)

print(np.mean(prediction[:,5]))
print(np.mean(yTest))
plt.ylim(0,0.2)

#plt.plot(prediction[:,8])
plt.plot(yTest, alpha = 0.5)
#plt.plot(prediction[:,0])
#%% generate QQ plot


#generateQQPLot(quantiles, yTest, prediction)

#%%


#generateRainfallImage(model,'data/OR_ABI-L1b-RadF-M4C13_G16_s20172322120227_e20172322125040_c20172322125109.nc')    

#%%

    
import datetime as datetime
    
plotGPMData(datetime.datetime(2020,1,26),[-90, -30, -30, 20])
#%%
import datetime as datetime

plotGPMData(GPM_data)
#%%
getGEOData(GPM_data[0,1], GPM_data[0,2],convertTimeStampToDatetime(GPM_data[0,3]))


#%%


    
    
plotGOESData('data/OR_ABI-L1b-RadF-M4C13_G16_s20172322120227_e20172322125040_c20172322125109.nc',[-80, -40, -20, 8])
#%%

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder = Model(input_img, encoded)
#%%

import numpy as np
# reshape data for the QRNN
#%%
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)  # adapt this if using `channels_first` image data format
#%%
print(x_test.shape)
#%%
#print(xTrain)
from sklearn.preprocessing import MinMaxScaler
import numpy as np


x_train = xData/xData.max()
x_train = np.reshape(x_train, (len(x_train), 28*28))

print(x_train.shape)
#%%
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=128,
                shuffle=True)
#%%
in_data = encoder.predict(x_train)
#%%
print(in_data.shape)
#%%
decoded_imgs = autoencoder.predict(x_train)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
     
