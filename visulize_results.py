# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:10:20 2020

@author: gustav
"""
def generateQQPLot(quantiles, yTest, prediction):
    import numpy as np
    q = np.zeros((len(quantiles),1))
    import matplotlib.pyplot as plt  
    for i in range(len(quantiles)):
        nmb = 0
        for j in range(yTest.shape[0]):
            if prediction[j,i] > yTest[j,0]:
                nmb +=1
        
        
        q[i,0] = nmb / yTest.shape[0]
        
    
    x = np.linspace(0, 1, 100)
    plt.plot(quantiles,q[:,0])
    plt.plot(x,x)
def generate_qqplot_for_intervals(quantiles, yTest, prediction, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    fig, ax=plt.subplots(nrows=10, ncols=1,figsize=(5,50))
    for k in range(10):
       
        # select the indexes within the specific interval
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        tmp_y = yTest[indexes]
        tmp_pred = prediction[indexes,:]
        
        q = np.zeros((len(quantiles),1))
        for i in range(len(quantiles)):
            nmb = 0
            for j in range(tmp_y.shape[0]):
                if tmp_pred[j,i] > tmp_y[j]:
                    nmb +=1
            
            if tmp_y.shape[0] != 0:
                
                q[i,0] = nmb / tmp_y.shape[0]
            else:
                q[i,0] = 0
            
      
        x = np.linspace(0, 1, 100)
        ax[k].set_title('interval: ('+str(sigma*k)+','+str(sigma*(k+1))+')'+' amount of observations'+ str(len(indexes)))
        ax[k].plot(quantiles,q[:,0])
        ax[k].plot(x,x)
    
    plt.show()
    
def getMeansSquareError(yTest, predictions, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    fig, ax=plt.subplots()
    error = np.zeros((15))
    for k in range(15):
       
        # select the indexes within the specific interval
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        tmp_y = yTest[indexes]
        tmp_pred = predictions[indexes,:]
        
        error[k] = np.abs(tmp_y-tmp_pred).sum()/len(tmp_y)
      
     
    ax.plot(error)
    
def plotIntervalPredictions(yTest, prediction, sigma):
    import numpy as np
    import matplotlib.pyplot as plt  
    fig, ax=plt.subplots(nrows=10, ncols=1,figsize=(5,50))
    for k in range(10):
       
        # select the indexes within the specific interval
        indexes = np.where((yTest > sigma*k)&(yTest < (k+1)*sigma))[0]
        tmp_y = yTest[indexes]
        tmp_pred = prediction[indexes,:]
    
            
      
        ax[k].set_title('mean pred:'+ str(np.mean(tmp_pred))+'mean actual:'+str(np.mean(tmp_y)))
        ax[k].plot(tmp_y)
        ax[k].plot(tmp_pred,alpha = 0.5)
    
    plt.show()
def confusionMatrix(yTest, predictions):
    #conf_matrix = np.zeros(2)
    
    pred_rain_was_rain = 0
    pred_rain_wasnt_rain= 0
    pred_no_rain_was_rain = 0
    pred_no_rain_wasnt_rain = 0
    print(yTest.shape)
    print(predictions.shape)
    
    for i in range(len(yTest)):
        
        if yTest[i] <= 0 and predictions[i] <= 0:
            pred_no_rain_wasnt_rain +=1
        
        elif yTest[i] <= 0 and predictions[i] > 0:
            pred_rain_wasnt_rain +=1
        
        elif yTest[i] > 0 and predictions[i] <= 0:
            pred_no_rain_was_rain +=1
        
        elif yTest[i] > 0 and predictions[i] > 0:
            pred_rain_was_rain +=1 
    
    print('pred_rain_was_rain:'+str(pred_rain_was_rain)+'pred_rain_wasnt_rain:'+str(pred_rain_wasnt_rain)+'pred_no_rain_was_rain:'+str(pred_no_rain_was_rain)+'pred_no_rain_wasnt_rain:'+str(pred_no_rain_wasnt_rain))
def generateRainfallImage(model,data_file_paths):
    from data_loader import downloadFile
    from data_loader import extractGeoData
    import numpy as np
    import matplotlib.pyplot as plt
    
    im_width = 200
    field_of_vision = 8
    half_field_of_vision = 4
    test = np.zeros((im_width*im_width,len(data_file_paths),field_of_vision,field_of_vision))
    channel =0
    for file in data_file_paths:
        #downloadFile(file)
        lons,lats,C,rad, x_data, y_data = extractGeoData(file)
        rad = rad.data
        
        rad = rad[:im_width,:im_width]
       
       
        print(rad.shape)
        
        # generate rad images to be evaluated
        index = 0
        for i in range(rad.shape[0]):
            #print(i)
            for j in range(rad.shape[1]):
                #print(np.reshape(rad[i-3:i+3,j-3:j+3].data,(1,36))/max_val)
                if i<half_field_of_vision or i >rad.shape[0]-half_field_of_vision or j<half_field_of_vision or j >rad.shape[1]-half_field_of_vision :
                    test[index,channel,:,:] = rad[:field_of_vision,:field_of_vision]
                else:
                    test[index,channel,:,:] = rad[i-half_field_of_vision:i+half_field_of_vision,j-half_field_of_vision:j+half_field_of_vision]
                #predictions[i,j] = model.predict(np.reshape(rad[i-3:i+3,j-3:j+3],(1,36))/max_val)[0,4]
                index = index +1
        channel +=1
            
    test2 = np.reshape(test,(rad.shape[0]*rad.shape[1],len(data_file_paths)*field_of_vision*field_of_vision))
    test3 = np.zeros((rad.shape[0]*rad.shape[1],len(data_file_paths)*field_of_vision*field_of_vision+4))
    test3[:,:len(data_file_paths)*field_of_vision*field_of_vision] = test2
    pred = model.predict(test3)
    #fig, axes = plt.subplots(nrows=2, ncols=3)
    plt.figure(figsize=(20, 20))
    
    for i in range(pred.shape[1]):
        
        ax = plt.subplot(3, 3, i+1)
        
        #fig, ax = plt.subplots()
        img = pred[:,i].reshape(im_width,im_width)
        #img_in =  np.ma.masked_less(img, 0.001)
        #im = axes.flat[i].imshow(img,vmin=0, vmax=20)
        #im = axes.flat[i].imshow(img)
        
        #fig.colorbar(im)
        im =plt.imshow(img)
        #plt.title(i)
        #im = ax.imshow(img_in)
        plt.colorbar(im, ax=ax)
    
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)

    plt.show()

def plotGPMData(DATE, extent):
    import numpy as np
    from scipy.interpolate import griddata
    import xarray
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(15, 12))
    
    # Generate an Cartopy projection
    pc = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=pc)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    
    
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    
    
    dataSize = 10000
    GPM_data = getGPMData(DATE,dataSize)
    
    extent1  = [min(GPM_data[:,1]),max(GPM_data[:,1]),min(GPM_data[:,2]),max(GPM_data[:,2])]
    grid_x, grid_y = np.mgrid[extent1[0]:extent1[1]:200j, extent1[2]:extent1[3]:200j]
    points = GPM_data[:,1:3]
    values = GPM_data[:,0]
    print(values.max())
    upper_threshold = 10
    indexPosList = [ i for i in range(len(values)) if values[i] >upper_threshold]
    #print(indexPosList)
    values[indexPosList] = upper_threshold
    #print(values.max())
    grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear')
    ax.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
    #plt.colorbar()
    #print(np.nan_to_num(grid_z0).max())
    
def plotGOESData(FILE, extent):
    from datetime import datetime
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import metpy  # noqa: F401
    import numpy as np
    import xarray
    from pyproj import Proj
    '''
    maxLongitude = -51
    minLongitde = -70
    maxLatitide = 2.5
    minLatitude = -11
    '''
    maxLongitude = -40
    minLongitde = -80
    maxLatitide = 8
    minLatitude = -20

    # Open the file with xarray.
    # The opened file is assigned to "C" for the CONUS domain.
    
    #FILE = ('data/OR_ABI-L1b-RadF-M6C13_G16_s20200260150156_e20200260159476_c20200260159547.nc')
    C = xarray.open_dataset(FILE)
     # Satellite height
    sat_h = C['goes_imager_projection'].perspective_point_height
    
    # Satellite longitude
    sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
    
    # Satellite sweep
    sat_sweep = C['goes_imager_projection'].sweep_angle_axis
    
    # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
    # See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary
    x_new = C['x'][:] * sat_h
    y_new = C['y'][:] * sat_h
    #x = C['x'][:] 
    #y = C['y'][:] 
    rad = C['Rad']
    #Dataset('data/OR_ABI-L1b-RadC-M6C01_G16_s20200160001163_e20200160003536_c20200160004010.nc','r')['nominal_satellite_subpoint_lat'][:]
    # Create a pyproj geostationary map object
    
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
  

    xmin_proj, ymax_proj = p(minLongitde, maxLatitide)
    xmax_proj, ymin_proj = p(maxLongitude, minLatitude)
    
    print()
    xmin_proj =min([xmin_proj,xmax_proj])
    xmax_proj =max([xmin_proj,xmax_proj])
    ymin_proj =min([ymin_proj,ymax_proj])
    ymax_proj =max([ymin_proj,ymax_proj])
    
    print(xmin_proj)
    print(xmax_proj)
    print(ymin_proj)
    print(ymax_proj)
    
    
    xmin_index = (np.abs(x_new.data-xmin_proj)).argmin()
    xmax_index = (np.abs(x_new.data-xmax_proj)).argmin()
    ymin_index = (np.abs(y_new.data-ymin_proj)).argmin()
    ymax_index = (np.abs(y_new.data-ymax_proj)).argmin()
     
    print(xmin_index)
    print(xmax_index)
    print(ymin_index)
    print(ymax_index)

    
    x_new = x_new[xmin_index:xmax_index]
    y_new = y_new[ymax_index:ymin_index]
    print(x_new)
    print(y_new)
    print(x_new.coords['x'])
    x_new.coords['x'] = x_new.coords['x']* sat_h
    y_new.coords['y'] = y_new.coords['y']* sat_h
    print(x_new)
    rad = rad[ymax_index:ymin_index,xmin_index:xmax_index]
    rad.coords['x'] =rad.coords['x']*sat_h
    rad.coords['y'] =rad.coords['y']*sat_h
    print(rad)
    #rad = rad[xmin_index.data:xmax_index.data,ymax_index.data:ymin_index.data]
    # We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
    dat = C.metpy.parse_cf('Rad')
    
    geos = dat.metpy.cartopy_crs
    
    x = dat.metpy.x
    y = dat.metpy.y
    print(x)
    print(y)
    #long = dat.metpy.longitude
    #lat = dat.metpy.latitude
    #print(long)
    x = x_new
    y = y_new
    
    fig = plt.figure(figsize=(15, 12))
    # Generate an Cartopy projection
    pc = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=pc)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    
    ax.imshow(rad, origin='upper',
              extent=(x.min(), x.max(), y.min(), y.max()),
              transform=geos,
              interpolation='none')
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    
    #plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
    #plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')
    
    plt.show()
def plot_predictions_and_labels(model,DATE, mean1, std1, mean2, std2):
    import numpy as np
    from scipy.interpolate import griddata
    import xarray
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import datetime
    from data_loader import get_single_GPM_pass
    from data_loader import convertTimeStampToDatetime
    from data_loader import getGEOData
    extent = [-70, -50, -10, 2]
    fig = plt.figure(figsize=(30, 24))
    
    # Generate an Cartopy projection
    pc = ccrs.PlateCarree()
    ax = fig.add_subplot(6, 1, 1, projection=pc)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    
    
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    
    
    
    perc_tot_rate, long, lat, time = get_single_GPM_pass(DATE)
    GPM_data = np.zeros((len(perc_tot_rate),4))
    GPM_data[:,0] = perc_tot_rate
    GPM_data[:,1] = long
    GPM_data[:,2] = lat
    GPM_data[:,3] = time
    receptiveField = 28
    dataSize = len(perc_tot_rate)
    xData = np.zeros((dataSize,2,receptiveField,receptiveField))
    times = np.zeros((dataSize,3))
    yData = np.zeros((dataSize,1))
    distance = np.zeros((dataSize,2))
   
    times[:,0] = time
    xData[:,0,:,:], times[:,1], distance[:,0] = getGEOData(GPM_data,'C13')
    xData[:,0,:,:] = (xData[:,0,:,:]-mean1)/std1
    
    xData[:,1,:,:], times[:,2], distance[:,1] = getGEOData(GPM_data,'C08')
    
    xData[:,1,:,:] = (xData[:,1,:,:]-mean2)/std2
    
    xData = xData[:,:,10:18,10:18]
    print(xData)
    print(np.mean(xData))
    #print(convertTimeStampToDatetime(time[0]))
    extent1  = [min(long),max(long),min(lat),max(lat)]
    grid_x, grid_y = np.mgrid[extent1[0]:extent1[1]:200j, extent1[2]:extent1[3]:200j]
    points = np.zeros((len(lat),2))
    points[:,0] = long
    points[:,1] = lat
    values = perc_tot_rate
    #print(grid_x)
    print(values.max())
    
    upper_threshold = 20
    indexPosList = [ i for i in range(len(values)) if values[i] >upper_threshold]
    #print(indexPosList)
    values[indexPosList] = upper_threshold
    #print(values.max())
    grid_z0 = griddata(points,values, (grid_x, grid_y), method='linear')
    im = ax.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
    plt.colorbar(im, ax=ax)
    
    input_1 = np.zeros((len(xData),8*8*2+4))
    input_1[:,:8*8*2] = np.reshape(xData,(len(xData),8*8*2))
    input_1[:,-1] = (times[:,0]-times[:,1])
    input_1[:,-2] = distance[:,0]
    input_1[:,-3] = distance[:,1]
    input_1[:,-4] = (times[:,0]-times[:,2])
    
    pred = model.predict(input_1)
    print(pred[:,4])
    print(np.mean(pred))
    # plot the precction
    axes = []
    for i in range(5):
        axes.append(fig.add_subplot(6, 1, i+2, projection=pc))
        axes[-1].set_extent(extent, crs=ccrs.PlateCarree())
        
        
        
        axes[-1].coastlines(resolution='50m', color='black', linewidth=0.5)
        #ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
        axes[-1].add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
        
        tmp = griddata(points,pred[:,i], (grid_x, grid_y), method='linear')
        im =axes[-1].imshow(tmp.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')
        plt.colorbar(im, ax=axes[-1])
    #plt.colorbar()
    #print(np.nan_to_num(grid_z0).max())