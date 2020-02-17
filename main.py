# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# get training data
xData, yData , times, distance = getTrainingData(300000,350, 0.5)   

#%%
print(xData.shape)
import matplotlib.pyplot as plt
entry = 23569
plt.imshow(xData[entry,0,:,:])
plt.show()
plt.imshow(xData[entry,1,:,:])
plt.show()
plt.imshow(xData[entry,2,:,:])
plt.show()
print(times[entry,1]-times[entry,0])
print(times[entry,2]-times[entry,0])
print(times[entry,3]-times[entry,0])
#%% load data
import numpy as np
xData =np.load('trainingData/xDataC8C13S40000_R28_P500_R0.5.npy')
yData = np.load('trainingData/yDataC8C13S40000_R28_P500_R0.5.npy')
times = np.load('trainingData/timesC8C13S40000_R28_P500_R0.5.npy')
distance = np.load('trainingData/distanceC8C13S40000_R28_P500_R0.5.npy') 
#%% remove nan values

nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)
#%% narrow the field of vision

xData = xData[:,:,10:18,10:18]
print(xData.shape)
#%%
import matplotlib.pyplot as plt
plt.plot(np.abs(times[:,3]-times[:,1])[:25000])

#%%
#%% remove images close to each other in time
import datetime as datetime
cleanXData = xData
cleanYData = yData
cleanTimes = times
cleanDistance = distance
index =0
tmp_indexes = list(range(0,len(xData)))

index = 0
for i in range(len(xData)):
#for i in range(5):
    if i+1 > len(tmp_indexes):
        break
    #print(tmp_indexes[i])
    #print(times[tmp_indexes[i]:tmp_indexes[i]+200,0])
    #print(np.where(np.abs(times[tmp_indexes[i]:tmp_indexes[i]+200,0]-times[tmp_indexes[i],0]) <2)[0])
    #print(len(tmp_indexes))
    #print(tmp_indexes[i:i+200])
    indexes_to_remove =  np.where(np.abs(times[tmp_indexes[i:i+200],0]-times[tmp_indexes[i],0]) <2)[0][1:]+i
    #print(len(indexes_to_remove))
    #print(indexes_to_remove)
    #print(tmp_indexes[0:200])
    #for index in indexes_to_remove:
    tmp_indexes = np.delete(tmp_indexes,indexes_to_remove)
    #tmp_indexes = (tmp_indexes,np.where(np.abs(times[tmp_indexes[i:i+200],0]-times[tmp_indexes[i],0]) <2)[0][1:])
    #print(tmp_indexes[0:100])
    #print(tmp_indexes[0:200])
    print(len(tmp_indexes))
    print(i)
        

#%%
tmp_x = xData[tmp_indexes,:,:,:]
tmp_y = yData[tmp_indexes]
tmp_times = times[tmp_indexes]
tmp_distace = distance[tmp_indexes]


#%%
import matplotlib.pyplot as plt
plt.plot(tmp_times[:200,0])

for i in range(20):
    plt.imshow(tmp_x[i+200,0,:,:])
    plt.show()
#%%

nmbImages = 20
import matplotlib.pyplot as plt
import numpy as np
for i in range(5800,5800+nmbImages):
    
    fig = plt.figure()
    fig.suptitle('timediff %s, rainfall %s' % (np.abs(times[i,0]-times[i,1]), yData[i]), fontsize=20)
    plt.imshow(xData[i,1,:,:])
#%%
#non_zero_indexes = np.where(yData > 0)[0]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(yData, bins = 400)
ax.set_xlim([0,20])

#yData[np.where(yData > 15)[0]] = 15
#%%
import matplotlib.pyplot as plt
plt.plot(yData)
#%% select the data within time limit

indexes = np.where(np.abs(times[:,0]-times[:,1])<200)[0]
xData = xData[indexes,:,:,:]
yData = yData[indexes]
times = times[indexes]
distance = distance[indexes]

'''
indexes = np.where(distance<0.0050)[0]
xData = xData[indexes,:,:]
yData = yData[indexes]
times = times[indexes]
distance = distance[indexes]
'''
#yData[np.where(yData > 15)] = 15
print(xData.shape)
print(yData.shape)
print(times.shape)
print(distance.shape)
#%%
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import preprocessDataForTraining

# preprocess and combine the data
#newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)
newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
input_dim = newXData.shape[1]
model = qrnn.QRNN(input_dim,quantiles, depth = 24,width = 512, activation = 'relu')

import random
#random_shuffle = random.sample(range(0,len(newXData)),len(newXData))
#newXData = newXData[random_shuffle,:]
#newYData = newYData[random_shuffle]
# split into training and test sets
cut_index = 32000
xTrain = newXData[:cut_index,:]
xTest = newXData[cut_index:,:]

yTrain = newYData[:cut_index]
yTest = newYData[cut_index:]


#yTrain[np.where(yTrain > 15)] = 15
#yTest[np.where(yTest > 15)] = 15

#%% fit the model
model.fit(x_train = xTrain, y_train = yTrain,batch_size = 128,maximum_epochs = 100)
#%%

model.save('model.h5')
#%%
from keras.models import load_model
model = load_model('model.h5')

#%% predict
test_set_x = xTest
test_set_y = yTest
prediction = model.predict(test_set_x)
#print(prediction)
#%% calculate the mean value
mean = np.zeros((test_set_x.shape[0],1))
for i in range(test_set_x.shape[0]):
    
    mean[i,0] = model.posterior_mean(test_set_x[i,:])
    

#%% plot 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))
line = 1
ax.set_ylim([0,20])

ax.plot(mean,linewidth=line)

print(np.mean(mean))
print(np.mean(test_set_y))

#print(prediction[:,5])
#plt.plot(prediction[:,0])
ax.plot(test_set_y, alpha = 0.5,linewidth=line)
#plt.plot(prediction[:,0])
#%% generate QQ plot
from visulize_results import generateQQPLot
generateQQPLot(quantiles, yTest, prediction)

#%% generate qq plots for intervals of y data
from visulize_results import generate_qqplot_for_intervals
generate_qqplot_for_intervals(quantiles, test_set_y, prediction, 1)
#%% get the error
from visulize_results import getMeansSquareError
getMeansSquareError(test_set_y, mean, 1)
#%% generate confision matrix, rain no rain
from visulize_results import confusionMatrix
confusionMatrix(test_set_y,mean)
#%%
from visulize_results import plotIntervalPredictions
plotIntervalPredictions(test_set_y, mean, 1)
#%%
print(np.mean(mean[np.where(yTest == 0)[0]]))
plt.plot(mean[np.where(test_set_y == 0)[0]])
#%%
print(np.mean(mean[np.where(yTest > 0)[0]]))
plt.plot(mean[np.where(test_set_y > 0)[0]])
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

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

encoder = Model(input_img, encoded)
#%%

import numpy as np
# reshape data for the QRNN
#%%
import numpy as np
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format  # adapt this if using `channels_first` image data format
#%%
print(x_test.shape)
newXData = np.reshape(xData, (len(xData), 28, 28, 1))  
print(newXData.shape)
#%%
#print(xTrain)
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler1 = StandardScaler()

# reshape data for the QRNN
newXData = np.reshape(xData, (len(xData), 28, 28, 1))  
print(newXData.shape)



# scale the data with unit variance and and between 0 and 1 for the labels
#scaler1.fit(newXData)
newXData = newXData / newXData.max()


#%%
#autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(newXData, newXData,
                epochs=100,
                batch_size=256,
                shuffle=True)
#%%
in_data = encoder.predict(newXData)
#%%
print(in_data.shape)
#%%
decoded_imgs = autoencoder.predict(newXData)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(newXData[i].reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
     
#%%
cudaRuntimeGetVersion()