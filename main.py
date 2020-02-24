# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# create training data
xData, yData , times, distance = getTrainingData(350000,200, GPM_resolution = 3)   





#%%
indexes = np.where(yData > 0)[0]
print(len(indexes))
import matplotlib.pyplot as plt
index = indexes[263]
plt.imshow(xData[index,0,5:23,5:23:])
plt.show()
plt.imshow(xData[index,1,:,:])
plt.show()
plt.imshow(yData[index,:,:])
print(times[index,0]-times[index,1])
#print(yData[index,:,:])

#%% code for the typhon qrnn
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import preprocessDataForTraining
from data_loader import load_data



newXData, newYData, mean1, mean2,std1,std2 = load_data()
quantiles = [0.1,0.3,0.5,0.7,0.9]
input_dim = newXData.shape[1]
model = qrnn.QRNN(input_dim,quantiles, depth = 8,width = 256, activation = 'relu')


# split into training and validation set
cut_index = 175000
xTrain = newXData[:cut_index,:]
xTest = newXData[cut_index:,:]

yTrain = newYData[:cut_index]
yTest = newYData[cut_index:]

#%% train the model
model.fit(x_train = xTrain, y_train = yTrain,batch_size = 128,maximum_epochs = 300)
#%% save model

model.save('model.h5')
#%% load model
from typhon.retrieval import qrnn 
#loaded_model = qrnn.QRNN()
loaded_model = qrnn.QRNN.load('results/8_256_onyear_oneyear_28_7/model.h5')

#%%  generate results
from visulize_results import generate_all_results
generate_all_results(model, xTest,yTest, quantiles)

#%% create an image of predictions fromo GEO data
from visulize_results import generateRainfallImage

#data_files = ['ABI-L1b-RadF_2020_014_12_OR_ABI-L1b-RadF-M6C13_G16_s20200141200200_e20200141209519_c20200141210001.nc']
data_files = ['ABI-L1b-RadF_2020_014_12_OR_ABI-L1b-RadF-M6C13_G16_s20200141200200_e20200141209519_c20200141210001.nc','ABI-L1b-RadF_2020_014_12_OR_ABI-L1b-RadF-M6C08_G16_s20200141200200_e20200141209508_c20200141209587.nc']
generateRainfallImage(model,data_files)    

#%% create image of predictions and labels



import datetime
from visulize_results import plot_predictions_and_labels


plot_predictions_and_labels(model, datetime.datetime(2020,1,15), mean1, std1, mean2, std2)
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
#%% autoencoder

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
     
#%%
from data_loader import getGPMData
import datetime
start_DATE = datetime.datetime(2017,10,10)
pos_time_data, prec_data = getGPMData(start_DATE = datetime.datetime(2017,10,10), maxDataSize = 400, data_per_GPM_pass = 200, resolution = 3)

#%%
