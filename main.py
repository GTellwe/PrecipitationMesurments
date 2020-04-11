# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# create training data
xData, yData , times, distance = getTrainingData(6200,1400, GPM_resolution = 3)   

#%%


#%% code for the typhon qrnn
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import preprocessDataForTraining
from data_loader import load_data
import random

# load test data
#xTrain, yTrain, max1, max2 = load_data_training_data()
newXData, newYData = load_data()

#newYData =np.sqrt(newYData[:,3,3])



quantiles = [0.1,0.3,0.5,0.7,0.9]
input_dim = newXData.shape[1]
#model = qrnn.QRNN(input_dim,quantiles, depth = 8,width = 256, activation = 'relu')


# split into training and validation set
cut_index_1 =175000


xTest = newXData[cut_index_1:,:]
yTest = newYData[cut_index_1:,:]


xTrain = newXData[:cut_index_1,:]
yTrain = newYData[:cut_index_1,:]
#xTrain = newXData[:cut_index_1,:]

#yTrain = newYData[:cut_index_1]


# load training data

input_dim = newXData.shape[1]

# split into training and validation set



# train the model
#%%
indexes = random.sample(range(0,len(xTest)),50)
x_valid = xTest[indexes,:]
y_valid = yTest[indexes,:]
#indexes = random.sample(range(0,len(xTrain)),40000)
#x_train = xTrain[indexes,:]
#y_train = yTrain[indexes]


#%%
start_index = 8
end_index = 120
tmp_x_train = xTrain[:,start_index:end_index,start_index:end_index,:]
tmp_x_test = xTest[:,start_index:end_index,start_index:end_index,:]
#%%
print(xTrain.shape)



#%%
import matplotlib.pyplot as plt
index = 10
plt.imshow(tmp_x_train[index,:,:,0])
plt.show()
plt.imshow(tmp_x_train[index,:,:,1])
plt.show()
plt.imshow(tmp_y_train[index,:,:])
plt.show()
#%%
import extendedQRNN
model = extendedQRNN.QRNN((28*28*2+4,),quantiles, depth = 8,width = 256, activation = 'relu', model_name ='MLP')

model.fit(x_train = xTrain,
          y_train = yTrain[:,3,3],
          x_val=xTest,
          y_val =yTest[:,3,3],
          batch_size = 256,
          maximum_epochs = 500)

#%%
model = qrnn.QRNN(28*28*2,quantiles, depth = 16,width = 512, activation = 'relu')
model.fit(x_train = xTrain, y_train = yTrain,x_val = x_valid, y_val = y_valid, batch_size = 512,maximum_epochs = 500)
#%% save model

model.save('model.h5')

#%% load model
from keras.models import load_model
model = load_model('results\sqrt\model.h5')
#%% load model
from typhon.retrieval import qrnn
# = qrnn.QRNN()
model = qrnn.QRNN.load('results\\model.h5')

#%%generate results
from visulize_results import generate_all_results
generate_all_results(model, xTest,yTest[:,3,3],yTrain[:,3,3], quantiles)
#%%
predictions = model.predict(xTest[:,8:120,8:120,:])
#%%
#%%
print(predictions.shape)
tmp_yTest = np.reshape(yTest, (yTest.shape[0]*yTest.shape[1]*yTest.shape[2],1))
tmp_predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1]*predictions.shape[2],predictions.shape[3]))
tmp_yTrain = np.reshape(yTrain, (yTrain.shape[0]*yTrain.shape[1]*yTrain.shape[2],1))
#%%    
print(tmp_yTest.shape)
#%%
from visulize_results import generate_all_results_CNN
generate_all_results_CNN(tmp_predictions,np.reshape(tmp_predictions[:,2],(len(tmp_predictions[:,2]),1)),None,tmp_yTest,tmp_yTrain, quantiles)
#%%

#%%
prediction = model.predict(xTest)


#%%
import matplotlib.pyplot as plt

from visulize_results import generateQQPLot
generateQQPLot(quantiles, np.reshape(np.sqrt(yTest[:,3,3]),(len(yTest),1)), prediction, title = 'Network 3')
#%%
print(prediction.shape)
#%%

from visulize_results import generate_all_results_CNN
generate_all_results_CNN(prediction,np.reshape(prediction[:,2],(len(prediction[:,2]),1)),xTest, yTest,yTrain, quantiles)
#%%
from visulize_results import generateQQPLot
generateQQPLot(quantiles, yTrain, prediction)


#%%
from data_loader import convertTimeStampToDatetime
print(convertTimeStampToDatetime(times[305000,0]))

#%%
import numpy as np
'''
xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
#xData = xData[:,:,4:25,4:25]

'''

folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R128_P1400GPM_res3timeSeries.npy') 


# remove nan values
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)

#np.mean(xTrain, axis=0, keepdims=True)
max1 = xData[:,0,:,:].max()
max2 = xData[:,1,:,:].max()
min1 = xData[:,0,:,:].min()
min2 = xData[:,1,:,:].min()

mean1 = np.mean(xData[:,0,:,:])
mean2 = np.mean(xData[:,1,:,:])
std1 = np.std(xData[:,0,:,:])
std2 = np.std(xData[:,1,:,:])
x_test = np.zeros((3100,112,112,2))
x_test[:,:,:,0] = (xData[:3100,0,8:120,8:120]-min1)/(max1-min1)
x_test[:,:,:,1] = (xData[:3100,1,8:120,8:120]-min2)/(max2-min2)

#%%
predictions = model.predict(x_test)

#%% load model
import matplotlib.pyplot as plt
font1 = 15
font2 = 20
fig, ax = plt.subplots(1,2, figsize= (20,6))
ax[0].hist(np.abs(times[:,0]-times[:,1]),50, color = 'black')
ax[0].set_title('Time difference',fontsize = font2)
ax[0].set_ylabel('Frequency' ,fontsize = font1)
ax[0].set_xlabel('Abolute time difference',fontsize = font1)
ax[0].tick_params(labelsize=15)

ax[1].hist(distance[:,1],50, color = 'black')
ax[1].set_title('Distance difference',fontsize = font2)
ax[1].set_ylabel('Frequency',fontsize = font1)
ax[1].set_xlabel('Distance',fontsize = font1)
ax[1].tick_params(labelsize=15)

#%%
from typhon.retrieval import qrnn
# = qrnn.QRNN()
model = qrnn.QRNN.load('results/sqrt/model.h5')


#%%
from visulize_results import plot_predictions_and_labels
import datetime as datetime
DATE = datetime.datetime(2019,9,5)
plot_predictions_and_labels(model,DATE, max1,max2,min1, min2)
#%%



#%%
import datetime
maxLongitude = -51
minLongitde = -70
maxLatitide = 2.5
minLatitude = -11
extent = [-45,-75,7.5,-16]
 
DATE = datetime.datetime(2019,9,5)
from visulize_results import plotGPMData
plotGPMData(DATE)


#%%
import numpy as np
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S6200_R128_P1400GPM_res3timeSeries.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S6200_R128_P1400GPM_res3timeSeries.npy') 

#%%
print(xData.shape)
#%%
import matplotlib.pyplot as plt
index = 1
tmp = np.argwhere(yData >30)
print(tmp.shape)
#%%
from matplotlib.colors import LogNorm
fig = plt.figure(figsize=(15, 15))

axes=[]
axes.append(fig.add_subplot(1, 3, 1))
axes.append(fig.add_subplot(1, 3, 2))
axes.append(fig.add_subplot(1, 3, 3))
index = tmp[100,0]
axes[0].imshow(xData[index,0,:,:])
axes[1].imshow(xData[index,1,:,:])
im1 = axes[2].imshow(yData[index,:,:] ,cmap='jet',
                       norm = LogNorm(vmin=0.1, vmax = 100))
cbar = fig.colorbar(im1,ax = axes, shrink = 0.30, pad = 0.05,
               ticks = [0.1,0.5,1,5,10,25, 100])
cbar.ax.set_yticklabels([str(0.1),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 10)
cbar.ax.tick_params(labelsize=10)
plt.show()