# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# create training data
xData, yData , times, distance = getTrainingData(200000,200)   


#%% load data from files
import numpy as np

xData =np.load('trainingData/xDataC8C13S200000_R28_P200.npy')
yData = np.load('trainingData/yDataC8C13S200000_R28_P200.npy')
times = np.load('trainingData/timesC8C13S200000_R28_P200.npy')
distance = np.load('trainingData/distanceC8C13S200000_R28_P200.npy')  



#%% remove nan values
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)


#%% narrow the field of vision
xData = xData[:,:,10:18,10:18]
print(xData.shape)
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
   
    indexes_to_remove =  np.where(np.abs(times[tmp_indexes[i:i+200],0]-times[tmp_indexes[i],0]) <2)[0][1:]+i
    tmp_indexes = np.delete(tmp_indexes,indexes_to_remove)
    print(len(tmp_indexes))
    print(i)
        

#%%
tmp_x = xData[tmp_indexes,:,:,:]
tmp_y = yData[tmp_indexes]
tmp_times = times[tmp_indexes]
tmp_distace = distance[tmp_indexes]

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
#%% code for the typhon qrnn
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import preprocessDataForTraining

# preprocess and combine the data
#newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)
mean1 = np.mean(xData[:,0,:,:])
mean2 = np.mean(xData[:,1,:,:])
std1 = np.std(xData[:,0,:,:])
std2 = np.std(xData[:,1,:,:])
xData[:,0,:,:] = (xData[:,0,:,:]-mean1)/std1
xData[:,1,:,:] = (xData[:,1,:,:]-mean2)/std2
newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)

#newYData = newYData[:250000,:]
#newXData = newXData[:250000,:]
quantiles = [0.1,0.3,0.5,0.7,0.9]
input_dim = newXData.shape[1]
model = qrnn.QRNN(input_dim,quantiles, depth = 8,width = 256, activation = 'relu')



cut_index = 150000
xTrain = newXData[:cut_index,:]
xTest = newXData[cut_index:,:]

yTrain = newYData[:cut_index]
yTest = newYData[cut_index:]


#yTrain[np.where(yTrain > 15)] = 15
#yTest[np.where(yTest > 15)] = 15
#%%

#%% fit the model
model.fit(x_train = xTrain, y_train = yTrain,batch_size = 128,maximum_epochs = 50)
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

#%%
import matplotlib.pyplot as plt
#plt.plot(prediction[:,4])
print(prediction)
print(test_set_x)
print(np.mean(test_set_x))
#%% calculate the mean value
mean = np.zeros((test_set_x.shape[0],1))
for i in range(test_set_x.shape[0]):
    
    mean[i,0] = model.posterior_mean(test_set_x[i,:])
    

#%% plot 
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
generateQQPLot(quantiles, test_set_y, prediction)

#%% generate qq plots for intervals of y data
from visulize_results import generate_qqplot_for_intervals
generate_qqplot_for_intervals(quantiles, test_set_y, prediction, 1)
#%% get the error
from visulize_results import getMeansSquareError
getMeansSquareError(test_set_y, mean, 1)
#%% generate confision matrix, rain no rain
from visulize_results import confusionMatrix
print(len(np.where(test_set_y > 0)[0]))
confusionMatrix(test_set_y,prediction[:,4])
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
from visulize_results import generateRainfallImage

#data_files = ['ABI-L1b-RadF_2020_014_12_OR_ABI-L1b-RadF-M6C13_G16_s20200141200200_e20200141209519_c20200141210001.nc']
data_files = ['ABI-L1b-RadF_2020_014_12_OR_ABI-L1b-RadF-M6C13_G16_s20200141200200_e20200141209519_c20200141210001.nc','ABI-L1b-RadF_2020_014_12_OR_ABI-L1b-RadF-M6C08_G16_s20200141200200_e20200141209508_c20200141209587.nc']
generateRainfallImage(model,data_files)    

#%%
#tmpx =np.load('trainingData/xDataC8C13S200000_R28_P200.npy')


import datetime
from visulize_results import plot_predictions_and_labels


plot_predictions_and_labels(model, datetime.datetime(2020,1,16), mean1, std1, mean2, std2)
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

input_img = Input(shape=(28, 28, 2))  # adapt this if using `channels_first` image data format

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
decoded = Conv2D(2, (3, 3), activation='relu', padding='same')(x)

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
tmp  = np.zeros((len(xData),28,28,2))
tmp[:,:,:,0] = xData[:,0,:,:]/xData[:,0,:,:].max()
tmp[:,:,:,1] = xData[:,1,:,:]/xData[:,1,:,:].max()
newXData = tmp
#newXData = np.reshape(xData, (len(xData), 28, 28, 2))
#newXData = tmp/tmp.max()



# scale the data with unit variance and and between 0 and 1 for the labels

'''
flatten = np.reshape(xData, (len(xData),28*28*2))
scaler1.fit(flatten)
newXData = scaler1.transform(flatten)
newXData = np.reshape(newXData, (len(newXData),2,28,28))
newXData = np.reshape(newXData, (len(newXData),28,28,2))
'''

print(newXData.shape)
#%%
print(xData[:,0,:,:].sum())
print(xData[:,1,:,:].sum())
#%%
import matplotlib.pyplot as plt
print(newXData.shape)
plt.imshow(newXData[0,:,:,0])
#plt.imshow(xData[0,0,:,:])
#%%
#autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(newXData, newXData,
                epochs=20,
                batch_size=256,
                shuffle=True)
#%%
in_data = encoder.predict(newXData)
#%%
print(in_data.shape)
#%%
decoded_imgs = autoencoder.predict(newXData)
#%%
import matplotlib.pyplot as plt
print(newXData.shape)
plt.imshow(newXData[0,:,:,0])
#%%
n = 10  # how many digits we will display
plt.figure(figsize=(10, 10))
i =0
# display original
ax = plt.subplot(2, 2, 1)
plt.imshow(newXData[i,:,:,0])
#plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(2, 2,2)
plt.imshow(newXData[i,:,:,1])
#plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# display reconstruction
ax = plt.subplot(2, 2,3)
plt.imshow(decoded_imgs[i,:,:,0].reshape(28, 28))
#plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(2, 2,4)
plt.imshow(decoded_imgs[i,:,:,1].reshape(28, 28))
#plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()
     



