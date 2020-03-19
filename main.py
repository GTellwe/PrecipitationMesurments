# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np
from data_loader import getTrainingData
  



# create training data
xData, yData , times, distance = getTrainingData(1000000,1400, GPM_resolution = 3)   


#%%
from data_loader import load_data
newXData, newYData, mean1, mean2,std1,std2 = load_data()

#%%
import matplotlib.pyplot as plt
print(newYData.shape)
print(newXData.shape)
indexes = np.where(newYData >0)[0]
index = 6
plt.imshow(newXData[indexes[index],:,:,0])
plt.show()
plt.imshow(newXData[indexes[index],:,:,1])
print(newYData[indexes[0],0])

#%%
import matplotlib.pyplot as plt
print(GPM_pos_time_data.shape)
print(GPM_pos_time_data[0,0,0,:])
plt.scatter(GPM_pos_time_data[0,0,:,:].flatten(),GPM_pos_time_data[0,1,:,:].flatten())
#%%
print(xData.shape)
print(times.shape)
#%%
import matplotlib.pyplot as plt
i = 0
for j in range(7):
    plt.imshow(xData[100,0,j,i,:,:])
    print(distance[100,0,i,j])
    #print(times[1,0,i,j])
    plt.show()

plt.imshow(xData[100,0,3,3,:,:])
#%% code for the typhon qrnn
from typhon.retrieval import qrnn 
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import preprocessDataForTraining
from data_loader import load_data, load_data_training_data
import random

# load test data
newXData, newYData, mean1, mean2,std1,std2 = load_data()


#newXData = (newXData-x_mean)/(x_sigma)
quantiles = [0.1,0.3,0.5,0.7,0.9]
input_dim = newXData.shape[1]
#model = qrnn.QRNN(input_dim,quantiles, depth = 8,width = 256, activation = 'relu')


# split into training and validation set
cut_index_1 = 350000




xTest = newXData[cut_index_1:,:]
yTest = newYData[cut_index_1:]

indexes = random.sample(range(0,len(xTest)),5000)
x_valid = xTest[indexes,:]
y_valid = yTest[indexes,:]

xTrain = newXData[:cut_index_1]
yTrain = newYData[:cut_index_1]
#xTrain = newXData[:cut_index_1,:]

#yTrain = newYData[:cut_index_1]


# load training data

input_dim = newXData.shape[1]

# split into training and validation set



# train the model
#%%
model = qrnn.QRNN(input_dim,quantiles, depth = 8,width = 256, activation = 'relu')
model.fit(x_train = xTrain, y_train = yTrain,x_val = x_valid, y_val = y_valid, batch_size = 512,maximum_epochs = 300)
#%% save model

model.save('model.h5')
#%% load model
from typhon.retrieval import qrnn
# = qrnn.QRNN()
model = qrnn.QRNN.load('results/8_256_oneyar_28_1_700k/model.h5')

#%%  generate results
from visulize_results import generate_all_results
generate_all_results(model, xTest,yTest,yTrain, quantiles)

#%% Comparing the apriori mean vs the posterieori mean
from data_loader import load_data
newXData, newYData, mean1, mean2,std1,std2 = load_data()
import numpy as np
# load data
xData =np.load('trainingData/xDataC8C13S3200_R28_P4GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S3200_R28_P4GPM_res3.npy')
times = np.load('trainingData/timesC8C13S3200_R28_P4GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S3200_R28_P4GPM_res3.npy')  

# remove nan values
nanValues =np.argwhere(np.isnan(xData)) 
xData = np.delete(xData,np.unique(nanValues[:,0]),0)
yData = np.delete(yData,np.unique(nanValues[:,0]),0)
times = np.delete(times,np.unique(nanValues[:,0]),0)
distance = np.delete(distance,np.unique(nanValues[:,0]),0)


xData[:,0,:,:,:,:] = (xData[:,0,:,:,:,:]-mean1)/std1
xData[:,1,:,:,:,:] = (xData[:,1,:,:,:,:]-mean2)/std2


#%%
predictions = np.zeros((len(xData),5))
for k in range(len(xData)):
    prediction = np.zeros((5,1))
    i = 3
    j = 3
    tmp_test_data = np.zeros((28*28*2+4,1))
    tmp_test_data[:28*28*2,0] = np.reshape(xData[k,:,i,j,:,:], 28*28*2)
    tmp_test_data[-1,0] = (times[k,0,i,j]-times[k,1,i,j])/1000
    tmp_test_data[-2,0] = distance[k,0,i,j]
    tmp_test_data[-3,0] = distance[k,1,i,j]
    tmp_test_data[-4,0] = (times[k,0,i,j]-times[k,2,i,j])/1000
           
    prediction[:,0] = model.predict(tmp_test_data[:,0])[0,:]
            #prediction 
            #print(tmp.shape)
    predictions[k,:] = prediction[:,0]
#%%

predictions = np.zeros((len(xData),5))
for k in range(len(xData)):
    prediction = np.zeros((5,1))
    for i in range(7):
        for j in range(7):
            tmp_test_data = np.zeros((28*28*2+4,1))
            tmp_test_data[:28*28*2,0] = np.reshape(xData[k,:,i,j,:,:], 28*28*2)
            tmp_test_data[-1,0] = (times[k,0,i,j]-times[k,1,i,j])/1000
            tmp_test_data[-2,0] = distance[k,0,i,j]
            tmp_test_data[-3,0] = distance[k,1,i,j]
            tmp_test_data[-4,0] = (times[k,0,i,j]-times[k,2,i,j])/1000
           
            prediction[:,0] += model.predict(tmp_test_data[:,0])[0,:]
            #prediction 
            #print(tmp.shape)
    predictions[k,:] = prediction[:,0]/49
    
#%%
tmp_yTest = np.zeros((len(xData),1))
for i in range(len(xData)):
    tmp_yTest[i,0] = yData[i,:,:].mean()
#%%
print(predictions.shape)
#%%    
from visulize_results import generate_all_results_CNN
generate_all_results_CNN(predictions,np.reshape(predictions[:,2],(len(predictions[:,2]),1)),None,tmp_yTest,yTrain, quantiles)
#%%
import logging
import keras
from keras.datasets import mnist
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
#from keras import backend as K

import keras.backend as K


logger = logging.getLogger(__name__)
def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:
    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)
    Where I is the indicator function.
    """
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)


def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(
        K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true),
                                   K.flatten(y_pred[:, i + 1]),
                                   tau)
    return e

class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom `__repr` function, so that the
    loss function object can be easily loaded using `keras.model.load`.
    Attributes:
        quantiles: List of quantiles that should be estimated with
                   this loss function.
    """

    def __init__(self, quantiles):
        self.__name__ = "QuantileLoss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"

class TrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.
    Attributes:
        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """

    def __init__(self, x_train, x_mean, x_sigma, y_train, sigma_noise, batch_size):
        self.bs = batch_size

        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_train = y_train
        self.sigma_noise = sigma_noise

        self.indices = np.random.permutation(x_train.shape[0])
        self.i = 0

    def __iter__(self):
        logger.info("iter...")
        return self

    def __next__(self):
        inds = self.indices[np.arange(self.i * self.bs,
                                      (self.i + 1) * self.bs)
                            % self.indices.size]
        x_batch = np.copy(self.x_train[inds, :])
        if not self.sigma_noise is None:
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        x_batch = (x_batch - self.x_mean) / self.x_sigma
        y_batch = self.y_train[inds]

        self.i = self.i + 1

        # Shuffle training set after each epoch.
        if self.i % (self.x_train.shape[0] // self.bs) == 0:
            self.indices = np.random.permutation(self.x_train.shape[0])

        return (x_batch, y_batch)

#class ValidationGenerator(keras.utils.Sequence):
class ValidationGenerator():
 
    """
    This Keras sample generator is similar to the training generator
    only that it returns the whole validation set and doesn't perform
    any randomization.
    Attributes:
        x_val: The validation input, i.e. the brightness temperatures
                 measured by the satellite.
        y_val: The validation output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
    """
    def __init__(self, x_val, x_mean, x_sigma, y_val, sigma_noise):
        self.x_val = x_val
        self.x_mean = x_mean
        self.x_sigma = x_sigma

        self.y_val = y_val

        self.sigma_noise = sigma_noise

    def __iter__(self):
        return self

    def __next__(self):
        x_val = np.copy(self.x_val)
        if not self.sigma_noise is None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        x_val = (x_val - self.x_mean) / self.x_sigma
        return (x_val, self.y_val)
    
   

        
        
        

class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.
    Attributes:
        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.
    """

    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        self.losses += [logs.get('val_loss')]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(
                self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0
            logger.info("\n Reduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])
        
class early_stopping(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.
    Attributes:
        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.
    """

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.loss_gaps = []
        
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        self.loss_gaps = logs.get('val_loss')-logs.get('loss')
        
        if self.loss_gaps > self.threshold:
            self.model.stop_training = True

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(21,21,2)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu'))

#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())

#model.add(Dropout(0.25))
model.add(Dense(5))


x_mean = np.mean(xTrain, axis=0, keepdims=True)
x_sigma = np.std(xTrain, axis=0, keepdims=True)

'''
activation_fun = 'sigmoid'
model = Sequential()

model.add(Dense(256, activation=activation_fun, input_dim=28*28*2+4))
model.add(BatchNormalization())
model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256, activation=activation_fun))
model.add(BatchNormalization())

#model.add(Dropout(0.5))
model.add(Dense(5 ,activation = None))
'''
quantiles = [0.1,0.3,0.5,0.7,0.9]
loss = QuantileLoss(quantiles)
model.compile(loss=loss,
              optimizer=keras.optimizers.Adam())


lrd = 2.0
lrm = 1e-6
ce = 10
bs = 512
me = 1000

#lr_callback = LRDecay(model, lrd, lrm, ce)
#lr_callback = early_stopping(model,0.1)

training_generator = TrainingGenerator(xTrain, x_mean, x_sigma,
                                                       yTrain, None, bs)

validation_generator = ValidationGenerator(x_valid, x_mean, x_sigma,
                                                       y_valid, None)
model.fit_generator(training_generator, steps_per_epoch=xTrain.shape[0] // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1,validation_freq = 1)

'''

model.fit_generator(training_generator, steps_per_epoch=xTrain.shape[0] // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1, callbacks=[lr_callback])
'''


'''
model.fit(xTrain, yTrain,
          batch_size=512,
          epochs=100,
          verbose=1,
          validation_data=(xTest, yTest),
          callbacks=[lr_callback])
score = model.evaluate(xTest, yTest, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
#%% Code for classifications cnns
x_mean = np.mean(xTrain, axis=0, keepdims=True)
x_sigma = np.std(xTrain, axis=0, keepdims=True)


#activation_fun = 'sigmoid'
model_rain_no_rain = Sequential()

model_rain_no_rain.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,2)))
#model.add(BatchNormalization())
model_rain_no_rain.add(Dropout(0.25))
model_rain_no_rain.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu'))

#model.add(BatchNormalization())
model_rain_no_rain.add(MaxPooling2D(pool_size=(2, 2)))

model_rain_no_rain.add(Flatten())
model_rain_no_rain.add(Dropout(0.25))
model_rain_no_rain.add(Dense(128, activation='relu'))
model_rain_no_rain.add(Dropout(0.25))
model_rain_no_rain.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())

#model.add(Dropout(0.25))
model_rain_no_rain.add(Dense(2, activation = 'softmax'))

quantiles = [0.1,0.3,0.5,0.7,0.9]
loss = QuantileLoss(quantiles)
model_rain_no_rain.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


lrd = 2.0
lrm = 1e-6
ce = 10
bs = 512
me = 1000

#lr_callback = LRDecay(model, lrd, lrm, ce)
#lr_callback = early_stopping(model,0.1)
# create 0 and 1
tmp_y_train = np.zeros((len(yTrain),2))
tmp_y_train[np.where(yTrain > 0)[0],1] = 1
tmp_y_train[np.where(yTrain == 0)[0],0] = 1
training_generator = TrainingGenerator(xTrain, x_mean, x_sigma,
                                                       tmp_y_train, None, bs)

tmp_y_test = np.zeros((len(y_valid),2))
tmp_y_test[np.where(y_valid > 0)[0],1] = 1
tmp_y_test[np.where(y_valid == 0)[0],0] = 1
validation_generator = ValidationGenerator(x_valid, x_mean, x_sigma,
                                                       tmp_y_test, None)
model_rain_no_rain.fit_generator(training_generator, steps_per_epoch=xTrain.shape[0] // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1,validation_freq = 1)

'''

model.fit_generator(training_generator, steps_per_epoch=xTrain.shape[0] // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1, callbacks=[lr_callback])
'''

#%%
print(tmp_y_train[:,1].sum()/len(yTrain))
#%%
print(tmp_y_test[:,0])
#%% Code for classifications cnns
x_mean = np.mean(xTrain, axis=0, keepdims=True)
x_sigma = np.std(xTrain, axis=0, keepdims=True)


#activation_fun = 'sigmoid'
model_class_rain = Sequential()

model_class_rain.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,2)))
#model.add(BatchNormalization())
model_class_rain.add(Dropout(0.25))
model_class_rain.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu'))

#model.add(BatchNormalization())
model_class_rain.add(MaxPooling2D(pool_size=(2, 2)))

model_class_rain.add(Flatten())
model_class_rain.add(Dropout(0.25))
model_class_rain.add(Dense(128, activation='relu'))
model_class_rain.add(Dropout(0.25))
model_class_rain.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())

#model.add(Dropout(0.25))
model_class_rain.add(Dense(5))

quantiles = [0.1,0.3,0.5,0.7,0.9]
loss = QuantileLoss(quantiles)
model_class_rain.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


lrd = 2.0
lrm = 1e-6
ce = 10
bs = 512
me = 1000

#lr_callback = LRDecay(model, lrd, lrm, ce)
#lr_callback = early_stopping(model,0.1)
# create 0 and 1


rain_train_set = yTrain[np.where(yTrain > 0)[0],:]
tmp_y_train = np.zeros((len(rain_train_set),5))
tmp_y_train[np.where(rain_train_set > 0 and rain_train_set <= 1)[0],0] = 1
tmp_y_train[np.where(rain_train_set > 1 and rain_train_set <= 2)[0],1] = 1
tmp_y_train[np.where(rain_train_set > 2 and rain_train_set <= 3)[0],2] = 1
tmp_y_train[np.where(rain_train_set > 3 and rain_train_set <= 4)[0],3] = 1
tmp_y_train[np.where(rain_train_set > 4)[0],4] = 1

rain_train_set_x = xTrain[np.where(yTrain > 0)[0],:]
training_generator = TrainingGenerator(rain_train_set_x, x_mean, x_sigma,
                                                       tmp_y_train, None, bs)

tmp_y_test = yTest
tmp_y_test[np.where(yTest > 0)[0],:] = 1
validation_generator = ValidationGenerator(x_valid, x_mean, x_sigma,
                                                       tmp_y_test, None)
model_class_rain.fit_generator(training_generator, steps_per_epoch=xTrain.shape[0] // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1,validation_freq = 1)

'''

model.fit_generator(training_generator, steps_per_epoch=xTrain.shape[0] // bs,
                                epochs=me, validation_data=validation_generator,
                                validation_steps=1, callbacks=[lr_callback])
'''



#%%
print(x_sigma.shape)

#%%
prediction = model.predict((xTest-x_mean)/x_sigma)

#%%
import matplotlib.pyplot as plt
indexes =np.where(yTest != 0)[0]

from visulize_results import generateQQPLot
generateQQPLot(quantiles, yTest[indexes], prediction[indexes,:])
#%%
print(prediction.shape)
#%%

from visulize_results import generate_all_results_CNN
generate_all_results_CNN(prediction,np.reshape(prediction[:,2],(len(prediction[:,2]),1)),xTest, yTest,yTrain, quantiles)
#%%
from visulize_results import generateQQPLot
generateQQPLot(quantiles, yTrain, prediction)

#%%
from keras.layers import Input, Dense, UpSampling2D
from keras.models import Model
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
decoded = Conv2D(2, (3, 3), padding='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.fit(xTrain, xTrain,
                epochs=100,
                batch_size=512,
                shuffle=True,
                validation_data=(xTest, xTest))
                #callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

#%%
endcoder_outputs = encoder.predict(xTrain)
#%%
encoder_test = encoder.predict(xTest)
#%%
print(endcoder_outputs.shape)
#%%
model = qrnn.QRNN((4*4*8),quantiles, depth = 3,width = 256, activation = 'relu')
autoencoder_input_train = np.reshape(endcoder_outputs,(len(endcoder_outputs),4*4*8))
autoencoder_input_test = np.reshape(encoder_test,(len(encoder_test),4*4*8))

model.fit(x_train = autoencoder_input_train, y_train = yTrain,x_val = autoencoder_input_test, y_val = yTest, batch_size = 512,maximum_epochs = 300)

#%%
#%%  generate results
from visulize_results import generate_all_results
generate_all_results(model, autoencoder_input_test,yTest,yTrain, quantiles)

#%%
import matplotlib.pyplot as plt
plt.hist(yTest, bins=[-0.15,-0.135,-0.125,-0.1])
#%%

xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')

#%%
from data_loader import convertTimeStampToDatetime
print(convertTimeStampToDatetime(times[305000,0]))