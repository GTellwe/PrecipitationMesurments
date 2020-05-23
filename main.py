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
cut_index_1 =160000
cut_index_2 = len(newXData)

print(newYData.shape)
print(newXData.shape)

xTest = newXData[cut_index_1:cut_index_2,:]
yTest = newYData[cut_index_1:cut_index_2]


xTrain = newXData[:cut_index_1,:]
yTrain = newYData[:cut_index_1]
#xTrain = newXData[:cut_index_1,:]

#yTrain = newYData[:cut_index_1]


# load training data

input_dim = newXData.shape[1]

#%%
print(newXData.shape)
#%%
# split into training and validation set
tmp_xData = np.zeros((len(newXData),6,28,28,1))
for i in range(6):
    tmp_xData[:,i,:,:,0] = newXData[:,:,:,i]


xTest = tmp_xData[cut_index_1:,:]
#yTest = newYData[cut_index_1:]


xTrain = tmp_xData[:cut_index_1,:]
#yTrain = newYData[:cut_index_1]
#%%
import extendedQRNN
model = extendedQRNN.QRNN((28*28*2+4,),quantiles, depth = 8,width = 256, activation = 'relu', model_name ='CNN')

model.fit(x_train = xTrain,
          y_train = yTrain[:,3,3],
          x_val=xTest,
          y_val =yTest[:,3,3],
          batch_size = 256,
          maximum_epochs = 500)

#%%
print('sad')
#%% save model

model.save('CNN_model.h5')

#%% load model
from keras.models import load_model
model = load_model('results\sqrt\model.h5')
#%% load model
import extendedQRNN 
# = qrnn.QRNN()
model = extendedQRNN.QRNN.load('model2.h5')
#%%
print(model.x_mean)
#%%
print(np.mean(xTrain, axis=0, keepdims=True))
#%%
y_pred = model.predict(xTest)
crps = model.crps(y_pred, yTest[:,3,3], np.array(quantiles))
print(crps)
#%%
from visulize_results import generateCRPSIntervalPlot
generateCRPSIntervalPlot(crps,quantiles, yTest[:,3,3], y_pred, 1)
#%%
print(np.mean(crps))
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(newYData[:,1],newYData[:,0]))
print(correlation_target_prediction(newYData[:,0], newYData[:,1]))
#%%generate results
from visulize_results import generate_all_results
folder_path = 'C:\\Users\\gustav\\Documents\\Sorted\\PrecipitationMesurments\\code\\results\\timeseries2\\'
print(yTest.shape)
generate_all_results(model, xTest,yTest,yTrain, quantiles, False, folder_path)
#%%
predictions = model.predict(newXData)

#%%
mean = np.zeros((30000,1))
for i in range(30000):
    
    mean[i] = model.posterior_mean(newXData[i,:])
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(mean,newYData[:,0]))
print(correlation_target_prediction(newYData[:,0], mean))
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

#%%
y_test = yData[:3100,:,:]
#%%
tmp = np.argwhere(yTest >30)
print(tmp.shape)
print(predictions.shape)
#%%
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
index = tmp[0,0]
fig, ax = plt.subplots(2,3, figsize= (20,6))
max_val = 50
min_val = 0.1
im1 =ax[0,0].imshow(yTest[index,:,:], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
im2 =ax[0,1].imshow(predictions[index,:,:,0],cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
im3 = ax[0,2].imshow(predictions[index,:,:,1], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
ax[1,0].imshow(predictions[index,:,:,2], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
ax[1,1].imshow(predictions[index,:,:,3], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
im6 = ax[1,2].imshow(predictions[index,:,:,4], cmap='jet',norm = LogNorm(vmin=0.1, vmax = max_val))
cbar = fig.colorbar(im6,ax = ax, shrink = 0.79, pad = 0.025,
                   ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(100)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=20)
plt.show()
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
tmp = np.argwhere(yTest >30)
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


#%%
import numpy as np
nlin = 1613
ncol = 1349
filepath = "S11636382_201903200930.bin"
array = np.fromfile(filepath, dtype=np.int16,count=nlin*ncol).reshape(nlin, ncol)/10

#%%
import matplotlib.pyplot as plt
plt.imshow(array)
print(array.shape)

#%%
from os import listdir
from os.path import isfile, join
files = listdir('C:\\Users\gustav\\Documents\\Sorted\\PrecipitationMesurments\\ReferensData\\binaries')

print(files)

#%%
from data_loader import getReferenceData
getReferenceData()

#%%
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13S30000_R28_P10000GPM_res1reference.npy')
yData = np.load(folder_path+'/trainingData/yDataC8C13S30000_R28_P10000GPM_res1reference.npy')
times = np.load(folder_path+'/trainingData/timesC8C13S30000_R28_P10000GPM_res1reference.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13S30000_R28_P10000GPM_res1reference.npy') 
GPM_data = np.load(folder_path+'/trainingData/positionC8C13S30000_R28_P10000GPM_res1reference.npy') 

import numpy as np
scalexData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
scaleyData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
scaletimes = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
scaledistance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
#xData = xData[:,:,6:22,6:22]
import extendedQRNN 
# = qrnn.QRNN()
model = extendedQRNN.QRNN.load('results/CNN2_28_1_350k/model.h5')

# remove nan values
scalenanValues =np.argwhere(np.isnan(scalexData)) 
scalexData = np.delete(scalexData,np.unique(scalenanValues[:,0]),0)

#min1 = scalexData[:,0,:,:].min()
#max1 = scalexData[:,0,:,:].max()
#min2 = scalexData[:,1,:,:].min()
#max2 = scalexData[:,1,:,:].max()

tmpXData = np.zeros((len(xData),xData.shape[2],xData.shape[3],xData.shape[1]))
for i in range(xData.shape[1]):
 
        tmpXData[:,:,:,i] = (xData[:,i,:,:]-scalexData[:,i,:,:].min())/(scalexData[:,i,:,:].max()-scalexData[:,i,:,:].min()) 

predictions = model.predict(tmpXData)

#%%
import matplotlib.pyplot as plt
plt.plot(times[:10000])

#%%
extent = [-70, -50, -10, 2]
fig = plt.figure(figsize=(30, 30))
axes = []
start_index = 20000
end_index = 30000
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = 100

im1 = axes[-1].scatter(GPM_data[start_index:end_index,0], GPM_data[start_index:end_index,1], c = yData[start_index:end_index,0], s = 1, cmap='jet',
               norm = LogNorm(vmin=min_val, vmax = max_val)) 
axes[-1].set_title('DPR', fontsize = 20)


axes.append(fig.add_subplot(1, 3, 2, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)

im1 = axes[-1].scatter(GPM_data[start_index:end_index,0], GPM_data[start_index:end_index,1], c = yData[start_index:end_index,1], s = 1, cmap='jet',
               norm = LogNorm(vmin=min_val, vmax = max_val)) 
axes[-1].set_title('HE', fontsize = 20)

axes.append(fig.add_subplot(1, 3, 3, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)

tmp_pred = predictions[start_index:end_index,2]
tmp_gpm = GPM_data[start_index:end_index,:]
inds = np.where(tmp_pred > min_val)[0]

im1 = axes[-1].scatter(tmp_gpm[inds,0], tmp_gpm[inds,1], c = tmp_pred[inds], s = 1, cmap='jet',
               norm = LogNorm(vmin=min_val, vmax = max_val)) 
axes[-1].set_title('QRNN', fontsize = 20)
#plt.colorbar(im, ax=ax)

   




#plt.colorbar(im, ax=axes[-1])

cbar = fig.colorbar(im1,ax = axes, shrink = 0.79, pad = 0.025,
               ticks = [min_val,0.5,1,5,10,25, max_val])
cbar.ax.set_yticklabels([str(min_val),'0.5','1','5','10','25',str(max_val)])
cbar.set_label("Rain rate (mm/h)", fontsize = 20)
cbar.ax.tick_params(labelsize=20)

plt.show()
#%%
plt.plot(predictions[:,2])
#%%
from data_loader import convertTimeStampToDatetime
print(convertTimeStampToDatetime(GPM_data[0,2]))
#print(np.nan_to_num(grid_z0).max())

#%%
extent = [-70, -50, -10, 2]
import numpy as np
from scipy.interpolate import griddata
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from data_loader import get_single_GPM_pass
from data_loader import convertTimeStampToDatetime
from data_loader import getGEOData
from matplotlib.colors import LogNorm
nlin = 1613
ncol = 1349
filepath = "S11636382_201903200930.bin"
array = np.fromfile(filepath, dtype=np.int16,count=nlin*ncol).reshape(nlin, ncol)/10
# Hidro GOES16 coordinates
nlin = 1613
ncol = 1349
DY   = -0.0359477
DX   = 0.0382513
lati = 13.01202615
loni = -81.98087435


# creating lat lon vectors
latf = lati + (nlin*DY)
lonf = loni + (ncol*DX)

print(lati,latf)
print(loni,lonf)

lats = np.arange(lati,latf,DY)
lons = np.arange(loni,lonf,DX)
print(lats.shape)
print(lons.shape)
min_lat_index = np.argmin(np.abs(lats-extent[2]))
max_lat_index = np.argmin(np.abs(lats-extent[3]))
min_lon_index = np.argmin(np.abs(lons-extent[0]))
max_lon_index = np.argmin(np.abs(lons-extent[1]))
lats = lats[max_lat_index:min_lat_index]
lons = lons[min_lon_index:max_lon_index]
print(min_lat_index)
print(max_lat_index)
print(min_lon_index)
print(max_lon_index)
print(array.shape)
lats = np.reshape(lats, (len(lats),1))
lons = np.reshape(lons, (len(lons),1))
lats_matrix = np.repeat(lats,len(lons), axis=1)
lons_matrix = np.repeat(lons,len(lats), axis = 1)
print(lats_matrix.shape)
print(lats_matrix.shape)
#plt.plot(lats)
#plt.show()
#plt.plot(lons)
tmp_array = array[max_lat_index:min_lat_index, min_lon_index:max_lon_index]

fig = plt.figure(figsize=(30, 30))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = 100
#plt.imshow(tmp_array)
im1 = axes[-1].scatter(np.transpose(lons_matrix).flatten(),lats_matrix.flatten(), c = tmp_array.flatten(), s = 1, cmap='jet',
               norm = LogNorm(vmin=0.1, vmax = 100)) 
#%%

extent = [-70, -50, -10, 2]
fig = plt.figure(figsize=(30, 30))
axes = []
# Generate an Cartopy projection
pc = ccrs.PlateCarree()
fig.tight_layout()
fig.subplots_adjust(bottom = 0.60, left = 0.0, right = 0.8)
axes.append(fig.add_subplot(1, 3, 1, projection=pc))
axes[-1].set_extent(extent, crs=ccrs.PlateCarree())



axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
axes[-1].add_feature(ccrs.cartopy.feature.OCEAN, zorder=0)
axes[-1].add_feature(ccrs.cartopy.feature.LAND, zorder=0)


min_val = 0.1
max_val = max(yData[:,0])

im1 = axes[-1].scatter(GPM_data[:,0], GPM_data[:,1], c = yData[:,0], s = 1, cmap='jet',
               norm = LogNorm(vmin=0.1, vmax = max_val)) 
axes[-1].set_title('DPR', fontsize = 20)

#%%


#%%
from data_loader import load_gauge_data
hydro,gauge_data, xData, times, distance = load_gauge_data()
#%%
import numpy as np
folder_path = 'E:/Precipitation_mesurments'
xData =np.load(folder_path+'/trainingData/xDataC8C13Sreference.npy')
times = np.load(folder_path+'/trainingData/timesC8C13Sreference.npy')
distance = np.load(folder_path+'/trainingData/distanceC8C13Sreference.npy')
hydro = np.load(folder_path+'/trainingData/hydropredC8C13Sreference.npy')
gauge_data = np.load(folder_path+'/trainingData/gaugeC8C13Sreference.npy')

#%%
import matplotlib.pyplot as plt
plt.plot(hydro)
print(hydro)
print(hydro.shape)

#%%
#%%
import numpy as np
scalexData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
scaleyData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
scaletimes = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
scaledistance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
#xData = xData[:,:,6:22,6:22]
import extendedQRNN 
# = qrnn.QRNN()
model = extendedQRNN.QRNN.load('results/CNN2_28_1_350k/model.h5')

# remove nan values
scalenanValues =np.argwhere(np.isnan(scalexData)) 
scalexData = np.delete(scalexData,np.unique(scalenanValues[:,0]),0)
predictions = np.zeros((len(gauge_data),1))

#%%
min1 = scalexData[:,0,:,:].min()
max1 = scalexData[:,0,:,:].max()
min2 = scalexData[:,1,:,:].min()
max2 = scalexData[:,1,:,:].max()
tmp_values = np.zeros((3,1))
for i in range(gauge_data.shape[0]):
    #print(gauge_data.shape)
    tmp_mean = 0
    for j in range(3):
        tmpXData = np.zeros((1,xData.shape[3],xData.shape[4],xData.shape[2]))
        tmpXData[:,:,:,0] = (xData[i,j,0,:,:]-min1)/(max1-min1)
        tmpXData[:,:,:,1] = (xData[i,j,1,:,:]-min2)/(max2-min2)
        tmp_values[j,0] = model.posterior_mean(tmpXData)
        #tmp_mean += model.posterior_mean(tmpXData)
    
    tmp_values = np.sort(tmp_values)
    predictions[i,0] = (tmp_values[0]+2*tmp_values[1]+tmp_values[2])/4
    #print(i)
#%%
print(predictions.shape)
print(gauge_data.shape)
print(hydro.shape)
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(predictions,gauge_data[:,-1]))
print(correlation_target_prediction(gauge_data[:,-1], predictions))
#%%
from visulize_results import calculate_tot_MSE,correlation_target_prediction
print(calculate_tot_MSE(hydro,gauge_data[:,-1]))
print(correlation_target_prediction(gauge_data[:,-1], hydro))

#%% scatter plot the values
import matplotlib.pyplot as plt
s=4
plt.scatter(predictions, gauge_data[:,-1], s=s, color = 'black')
plt.xlim([0,30])
plt.ylim([0,30])
plt.ylabel('Gauge value')
plt.xlabel('QRNN prediction')
plt.show()
plt.scatter(hydro, gauge_data[:,-1],s= s, color = 'black')
plt.xlim([0,30])
plt.ylim([0,30])
plt.ylabel('Gauge value')
plt.xlabel('Hydro prediction')