# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: Gustav Tellwe
"""
import netCDF4
import numpy as np

'''
    functions:
        getFileForHour():
        getClosestFile():
        downloadFile():
    
'''

  



#%% Functions

# Constants
rad = []
receptiveField = 28
maxLongitude = -51
minLongitde = -70
maxLatitide = 2.5
minLatitude = -11
lons = []
lats = []
oldFile = ""
def getFilesForHour(DATE):
    
    '''
        Returns an iterator of file names for the specific hour defines by DATE
    '''
    from google.cloud import storage
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    #hour = int(DATE.strftime('%H'))
    PATH_Storage = 'ABI-L1b-RadF/%s' % (DATE.strftime('%Y/%j/%H')) 
    #print(DATE)
    #print(PATH_Storage)
    return bucket.list_blobs(prefix = PATH_Storage)
def getMiddleTime(FILE):
    import datetime
    middleTimeDifference =(datetime.datetime.strptime(FILE.split('_')[4], 'e%Y%j%H%M%S%f')-datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')).total_seconds()
    return (datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')+datetime.timedelta(0, int(middleTimeDifference/2)))

def getClosestFile(DATE, CHANNEL):
    from datetime import datetime
    import numpy as np
    '''
    Returns the filepath closest to the DATE object
    
    '''
    files_for_hour = list(map(lambda x : x.name, getFilesForHour(DATE)))
   
    files_for_hour = [file for file in files_for_hour if file.split('_')[1] == CHANNEL ]
    date_diff = np.zeros((len(files_for_hour),1))
    #print(files_for_hour)
    for i in range(len(files_for_hour)):
        
        middleTime = getMiddleTime(files_for_hour[i])
        
        date_diff[i] = np.abs((DATE-middleTime).total_seconds())
        #print(totalMiddleTime)
        #print(totalSecondsData)
    #print(date_diff[:,0]) 
    #print(np.argmin(date_diff[:,0]))
    #print(files_for_hour[1])
    #print('%s' % (files_for_hour[np.argmin(date_diff[:,0])]))
    return '%s' % (files_for_hour[np.argmin(date_diff[:,0])]) 

def downloadFile(FILE):
    '''
    Downloads FILE and saves it in the data folder
    '''
    from google.cloud import storage
    #check if file lready exists
    
    try:
        f = open('data/'+FILE.split('/')[-1])
        # Do something with the file
        return
    except IOError:
        print("File does not exist, downloading")
   
        
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    blob = bucket.blob(FILE)
    blob.download_to_filename('data/'+FILE.split('/')[-1])
    
def extractGeoData(filePATH, oldFilePath):
    import xarray
    from pyproj import Proj
    import numpy as np
    global lons,lats
    global rad
    global x_data
    global y_data
    global sat_h
    global sat_lon
    global sat_sweep
    
    maxLongitude = -40
    minLongitde = -80
    maxLatitide = 8
    minLatitude = -20
    if oldFilePath == filePATH:
        return 
    
    FILE = 'data/'+filePATH.split('/')[-1]
    
    C = xarray.open_dataset(FILE)
    
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
    
   
    xmin_proj =min([xmin_proj,xmax_proj])
    xmax_proj =max([xmin_proj,xmax_proj])
    ymin_proj =min([ymin_proj,ymax_proj])
    ymax_proj =max([ymin_proj,ymax_proj])
    
    
    
    
    xmin_index = (np.abs(x_new.data-xmin_proj)).argmin()
    xmax_index = (np.abs(x_new.data-xmax_proj)).argmin()
    ymin_index = (np.abs(y_new.data-ymin_proj)).argmin()
    ymax_index = (np.abs(y_new.data-ymax_proj)).argmin()
     
    
    x_new = x_new[xmin_index:xmax_index]
    y_new = y_new[ymax_index:ymin_index]
   
    x_new.coords['x'] = x_new.coords['x']* sat_h
    y_new.coords['y'] = y_new.coords['y']* sat_h
    
    rad = rad[ymax_index:ymin_index,xmin_index:xmax_index]
    rad.coords['x'] =rad.coords['x']*sat_h
    rad.coords['y'] =rad.coords['y']*sat_h
    
    x = x_new
    y = y_new
    
    x_data = x
    y_data = y
    # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
    # to latitude and longitude values.
    XX, YY = np.meshgrid(x, y)
    print("traonsfrming data")
    lons, lats = p(XX, YY, inverse=True)
    #print(lons)
    print("transformation done")
'''    
def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude):
    import numpy as np
      # calculate the minimal dinstance
    X = np.abs(lons[int(lons.shape[0]/2),:] - longitude)
    idxLong = np.where(X == X.min())[0][0]
    X = np.abs(lats[:,idxLong] - latitude)
    idxLats = np.where(X == X.min())[0][0]
    square = 2
    minDistance = np.sqrt((lons[idxLats,idxLong]-longitude)**2+(lats[idxLats,idxLong]-latitude)**2)
    threshold = 0.01
    for k in range(500):
        if minDistance < threshold:
            break
        for j in range(idxLong-square,idxLong+square):
            for i in range(idxLats-square,idxLats+square):
                if j < lons.shape[1] and i < lons.shape[0]:
                    distance = np.sqrt((lons[i,j]-longitude)**2+(lats[i,j]-latitude)**2)
                    if(distance < minDistance):
                        minDistance = distance
                        idxLong = j
                        idxLats = i
    print(minDistance)
    return idxLats, idxLong
'''    
def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude, sat_h, sat_lon, sat_sweep, x_data,y_data):
    # project the longitude and latitude to geostationary references
    from pyproj import Proj
    import numpy as np
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    x, y = p(longitude, latitude)
    
    
    # get the x and y indexes
    x_index = (np.abs(x_data.data-x)).argmin()
    y_index = (np.abs(y_data.data-y)).argmin()
    
  
    return y_index, x_index
    
def getGEOData(longitude, latitude, TIME):
    from netCDF4 import Dataset
    import numpy as np
    global oldFile
    global rad
    import time
    from datetime import datetime
    global x_data
    global y_data
    global sat_h
    global sat_lon
    global sat_sweep
    
    
    '''
    returns the receptiveField by receptiveField pixel 10.8 radiance map for the geostationary 
    satelite at the time closest to TIME
    '''

    
    if longitude < minLongitde or longitude > maxLongitude or latitude < minLatitude or latitude >maxLatitide:
        return np.zeros((receptiveField,receptiveField))
    
    # Download the data file
   
    filePATH  =getClosestFile(TIME, 'ABI-L1b-RadF-M6C13')
    FILE = 'data/'+filePATH.split('/')[-1]
    
   
    downloadFile(filePATH)
   
   
    extractGeoData(filePATH, oldFile)
   
   
    
    
    oldFile = filePATH
    
  
    xIndex, yIndex = getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude, sat_h, sat_lon, sat_sweep, x_data,y_data)
   

    
    #print(xIndex)
    #print(yIndex)
    #return Dataset(FILE,'r')['Rad'][xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
    return rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
        
def getGPMFilesForSpecificDay(DATE):
    '''
        returning a list of file name for that spcific date
    '''
    import http.client
    import re
    c = http.client.HTTPSConnection("gpm1.gesdisc.eosdis.nasa.gov")

    request_string = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2BCMB.06/%s/' % (DATE.strftime('%Y/%j'))
    c.request("GET", request_string)
    r = c.getresponse()
    r = str(r.read())
   
    files = list(set(re.compile('"[^"]*.HDF5"').findall(r)))
  
    files.sort()
    return [f[1:-1] for f in files]

def downloadGPMFile(FILENAME, DATE):
    
    '''
        downloading rain totals form filename
    '''
    # check if file lready exists
    try:
        f = open('data/'+FILENAME)
        # Do something with the file
        return
    except IOError:
        print("File does not exist, downloading")
        
    maxLongitude = -51
    minLongitude = -70
    maxLatitude = 2.5
    minLatitude = -11
    host_name = 'https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?'
    filename = '%2Fdata%2FGPM_L2%2FGPM_2BCMB.06%2F' + DATE.strftime('%Y') + '%2F' + DATE.strftime('%j') + '%2F' + FILENAME
    p_format =  'aDUv'
    bbox = str(minLatitude) + '%2C' + str(minLongitude) + '%2C' + str(maxLatitude) + '%2C' + str(maxLongitude)
    label = FILENAME[:-8]+'SUB.HDF5'
    flags = 'GRIDTYPE__SWATH'
    variables = '..2FNS..2FsurfPrecipTotRate%2C..2FNS..2Fnavigation..2FtimeMidScanOffset'
    #URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FGPM_L2%2FGPM_2BCMB.06%2F2019%2F007%2F2B.GPM.DPRGMI.CORRA2018.20190107-S194520-E211752.027616.V06A.HDF5&FORMAT=aDUv&BBOX=-70%2C-180%2C70%2C180&LABEL=2B.GPM.DPRGMI.CORRA2018.20190107-S194520-E211752.027616.V06A.SUB.HDF5&FLAGS=GRIDTYPE__SWATH&SHORTNAME=GPM_2BCMB&SERVICE=SUBSET_LEVEL2&VERSION=1.02&DATASET_VERSION=06&VARIABLES=..2FNS..2FsurfPrecipTotRate%2C..2FNS..2Fnavigation..2FtimeMidScanOffset'
    URL = host_name + 'FILENAME=' + filename + '&FORMAT=' + p_format + '&BBOX=' + bbox + '&LABEL=' + label + '&FLAGS=' + flags + '&SHORTNAME=GPM_2BCMB&SERVICE=SUBSET_LEVEL2&VERSION=1.02&DATASET_VERSION=06&VARIABLES=' + variables
    
    
    import requests
    result = requests.get(URL)
    try:
       result.raise_for_status()
       f = open('data/' + FILENAME,'wb')
       f.write(result.content)
       f.close()
       print('contents of URL written to '+FILENAME)
    except:
       print('requests.get() returned an error code '+str(result.status_code))
       
def getGPMData(DATE, maxDataSize):
    
    '''
        retruns GPM data for the day provided. The data is in form of an array
        with each entry having attributes:
            1: position
            2: time
            3: rain amount
    '''
    
    import h5py
    import numpy as np
    data = np.zeros((maxDataSize,4))
    
    # get the files for the specific day
    files = getGPMFilesForSpecificDay(DATE)
    
    index = 0
    
    for FILENAME in files:
        # download the gpm data
        downloadGPMFile(FILENAME,DATE)
        
        # read the data
        path =  'data/'+FILENAME
        f = h5py.File(path, 'r')
        precepTotRate = f['NS']['surfPrecipTotRate'][:].flatten()
        longitude = f['NS']['Longitude'][:].flatten()
        latitude = f['NS']['Latitude'][:].flatten()
        time = np.array([f['NS']['navigation']['timeMidScan'][:],]*f['nrayNS_idx'].shape[0]).transpose().flatten()
        print(FILENAME)
        print(convertTimeStampToDatetime(time[0]))
        print(convertTimeStampToDatetime(time[-1]))
        # remove all the missing values
        indexes = np.where(np.abs(precepTotRate) < 200)[0]
        precepTotRate = precepTotRate[indexes]
        longitude = longitude[indexes]
        latitude = latitude[indexes]
        time = time[indexes]
        
        index1 = min(maxDataSize,index+len(precepTotRate))
        
        data[index:index1,0] = precepTotRate[:index1-(index+len(precepTotRate))]
        data[index:index1,1] = longitude[:index1-(index+len(precepTotRate))]
        data[index:index1,2] = latitude[:index1-(index+len(precepTotRate))]
        data[index:index1,3] = time[:index1-(index+len(precepTotRate))]
        
        
        
        index = index + len(precepTotRate)
        
        if index > maxDataSize:
            break
        
      
    return data
   
def convertTimeStampToDatetime(timestamp):
    from datetime import datetime, timedelta
    return datetime(1980, 1, 6) + timedelta(seconds=timestamp - (35 - 19))
    

def getTrainingData(dataSize):
    
    import numpy as np
    import datetime
    import time
    receptiveField = 28

    '''
    returns a set that conisit of radiance data for an area around every pixel
    in the given area together with its label
    '''
    
    xData = np.zeros((dataSize,receptiveField,receptiveField))
    times = np.zeros((dataSize,2))
    yData = np.zeros((dataSize,1))
    '''
        First step is to get the label data. To do this, we look at a specifi
        passing of the satelite over the area. We then extract the points
        in wich it passes. the result is a list of all the pixel that it 
        passet. Each entry in the list has the following atributes
            
            1: position
            2: time
            3: rain amount
    '''
    start_time = time.time()
    GPM_data = getGPMData(datetime.datetime(2020,1,26),dataSize)
    times[:,0] = GPM_data[:,3]
    end_time = time.time()
    print("time for collecting GPM Data %s" % (end_time-start_time))
    '''
    next step is to pair the label with the geostattionary data.
    '''
    start_time = time.time()
    for i in range(dataSize):
        #print(i)
        xData[i,:,:], times[i,1] = getGEOData(GPM_data[i,1], GPM_data[i,2],convertTimeStampToDatetime(GPM_data[i,3]))
        yData[i] = GPM_data[i,0]

    end_time = time.time()
    print("time for collecting GEO Data %s" % (end_time-start_time))
    return xData, yData, times


def plotTrainingData(xData,yData, times, nmbImages):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(nmbImages):
        
        fig = plt.figure()
        fig.suptitle('timediff %s, rainfall %s' % (np.abs(times[i,0]-times[i,1]), yData[i]), fontsize=20)
        plt.imshow(xData[i,:,:])
        #print(yData[i])



xData, yData , times = getTrainingData(100)
#%%
plotTrainingData(xData,yData,times, 100)
#%%
print(xData.shape)
#%%
from typhon.retrieval import qrnn 
import numpy as np
# reshape data for the QRNN
newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]))
print(newXData.shape)
print(xData.shape)
print(yData.shape)
#%%
quantiles = [0.1,0,2.0,3,0.4,0.5,0.6,0.7,0.8,0.9,1]
input_dim = newXData.shape[1]
model = qrnn.QRNN(input_dim,quantiles)

xTrain = newXData
yTrain = yData
print(xTrain)
from sklearn import preprocessing
import numpy as np
xTrain = preprocessing.scale(xTrain)
#xVal = newXData[80:,:]
#yVal = np.reshape(yData[80:,0],(20,1))
 
print(xTrain)

#model.fit(x_train = xTrain, y_train = yTrain,batch_size = 40,maximum_epochs = 50)



#%%
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
    
    #plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
    #plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')
    
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
    
import datetime as datetime
    
plotGPMData(datetime.datetime(2020,1,26),[-90, -30, -30, 20])
#%%
import datetime as datetime

plotGPMData(GPM_data)
#%%
getGEOData(GPM_data[0,1], GPM_data[0,2],convertTimeStampToDatetime(GPM_data[0,3]))


#%%

def plotGOESData(FILE, extent):
    from datetime import datetime
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import metpy  # noqa: F401
    import numpy as np
    import xarray
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
    
plotGOESData('data/OR_ABI-L1b-RadF-M6C13_G16_s20200260150156_e20200260159476_c20200260159547.nc',[-90, -30, -30, 20])
#%%
import xarray
from pyproj import Proj
import numpy as np
global lons,lats

maxLongitude = -51
minLongitde = -70
maxLatitide = 2.5
minLatitude = -11

FILE = 'data/OR_ABI-L1b-RadF-M6C13_G16_s20200260150156_e20200260159476_c20200260159547.nc'

C = xarray.open_dataset(FILE)

# Satellite height
sat_h = C['goes_imager_projection'].perspective_point_height

# Satellite longitude
sat_lon = C['goes_imager_projection'].longitude_of_projection_origin

# Satellite sweep
sat_sweep = C['goes_imager_projection'].sweep_angle_axis

# The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
# See details here: https://proj4.org/operations/projections/geos.html?highlight=geostationary
x = C['x'][:] * sat_h
y = C['y'][:] * sat_h
rad = C['Rad'].data
print(rad.shape)
#Dataset('data/OR_ABI-L1b-RadC-M6C01_G16_s20200160001163_e20200160003536_c20200160004010.nc','r')['nominal_satellite_subpoint_lat'][:]
# Create a pyproj geostationary map object
p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

xmin_proj, ymax_proj = p(minLongitde, maxLatitide)
xmax_proj, ymin_proj = p(maxLongitude, minLatitude)

xmin_index = (np.abs(x-xmin_proj)).argmin()
xmax_index = (np.abs(x-xmax_proj)).argmin()
ymin_index = (np.abs(y-ymin_proj)).argmin()
ymax_index = (np.abs(y-ymax_proj)).argmin()
print(ymin_index.data)
print(ymax_index.data)
x = x[xmin_index.data:xmax_index.data]
y = y[ymax_index.data:ymin_index.data]
#print(xmin_proj)
#print(x[min_distance.data])
#print(y)
# Perform cartographic transformation. That is, convert image projection coordinates (x and y)
# to latitude and longitude values.
XX, YY = np.meshgrid(x, y)
print("traonsfrming data")
lons, lats = p(XX, YY, inverse=True)
print("transformation done")
#print(x_proj)