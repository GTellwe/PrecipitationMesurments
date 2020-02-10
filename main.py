# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# get training data
xData, yData , times, distance = getTrainingData(100000,100, 0.5)   

#%% load data
import numpy as np
xData =np.load('trainingData/xDataS100000_R28_P1000_R0.5.npy')
yData = np.load('trainingData/yDataS100000_R28_P1000_R0.5.npy')
times = np.load('trainingData/timesS100000_R28_P1000_R0.5.npy')
distance = np.load('trainingData/distanceS100000_R28_P1000_R0.5.npy') 
#%%

#%%
import matplotlib.pyplot as plt
plt.plot(distance)
#%% select the data within time limit
indexes = np.where(np.abs(times[:,0]-times[:,1])<100)[0]
xData = xData[indexes,:,:]
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
newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
input_dim = newXData.shape[1]
model = qrnn.QRNN(input_dim,quantiles, depth = 10,width = 128, activation = 'relu')



# split into training and test sets
cut_index = 19000
xTrain = newXData[:cut_index,:]
xTest = newXData[cut_index:,:]

yTrain = newYData[:cut_index]
yTest = newYData[cut_index:]


#%% fit the model
model.fit(x_train = xTrain, y_train = yTrain,x_val = xTest, y_val = yTest,batch_size = 128,maximum_epochs = 100)
#%%

model.save('model.h5')
#%%
from keras.models import load_model
model = load_model('model.h5')

#%% predict
prediction = model.predict(xTest)

#%% calculate the mean value
mean = np.zeros((xTest.shape[0],1))
for i in range(xTest.shape[0]):
    
    mean[i,0] = model.posterior_mean(xTest[i,:])
    

#%% plot 
import matplotlib.pyplot as plt
plt.plot(mean)

print(np.mean(prediction[:,5]))
print(np.mean(yTest))
#plt.ylim(0,0.2)
#print(prediction[:,8])
#plt.plot(prediction[:,8])
plt.plot(yTest, alpha = 0.5)
#plt.plot(prediction[:,0])
#%% generate QQ plot
from visulize_results import generateQQPLot
generateQQPLot(quantiles, yTest, prediction)

#%% generate qq plots for intervals of y data
from visulize_results import generate_qqplot_for_intervals
generate_qqplot_for_intervals(quantiles, yTest, prediction)
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