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
receptiveField = 6
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
    PATH_Storage = 'ABI-L1b-RadF/%s' % (DATE.strftime('%Y/%j/%H')) 
    
    return bucket.list_blobs(prefix = PATH_Storage)

def getClosestFile(DATE, CHANNEL):
    import datetime
    import numpy as np
    '''
    Returns the filepath closest to the DATE object
    
    '''
    files_for_hour = list(map(lambda x : x.name, getFilesForHour(DATE)))
   
    files_for_hour = [file for file in files_for_hour if file.split('_')[1] == CHANNEL ]
    date_diff = map(lambda x: abs(datetime.strptime(x.split('_')[3], 's%Y%j%H%M%S%f')-DATE), files_for_hour)
    return '%s' % (files_for_hour[np.argmin(date_diff)]) 

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
    
    if oldFilePath == filePATH:
        return
    
    FILE = 'data/'+filePATH.split('/')[-1]
    
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
    #Dataset('data/OR_ABI-L1b-RadC-M6C01_G16_s20200160001163_e20200160003536_c20200160004010.nc','r')['nominal_satellite_subpoint_lat'][:]
    # Create a pyproj geostationary map object
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    
    # Perform cartographic transformation. That is, convert image projection coordinates (x and y)
    # to latitude and longitude values.
    XX, YY = np.meshgrid(x, y)
    print("traonsfrming data")
    lons, lats = p(XX, YY, inverse=True)
    print("transformation done")

def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude):
      # calculate the minimal dinstance
    X = np.abs(lons[int(lons.shape[0]/2),:] - longitude)
    idxLong = np.where(X == X.min())[0][0]
    X = np.abs(lats[:,idxLong] - latitude)
    idxLats = np.where(X == X.min())[0][0]
    square = 2
    minDistance = np.sqrt((lons[idxLats,idxLong]-longitude)**2+(lats[idxLats,idxLong]-latitude)**2)
  
    for k in range(500):
        for j in range(idxLong-square,idxLong+square):
            for i in range(idxLats-square,idxLats+square):
                if j < lons.shape[1] and i < lons.shape[0]:
                    distance = np.sqrt((lons[i,j]-longitude)**2+(lats[i,j]-latitude)**2)
                    if(distance < minDistance):
                        minDistance = distance
                        idxLong = j
                        idxLats = i
    return idxLats, idxLong
    
def getGEOData(longitude, latitude, TIME):
    from netCDF4 import Dataset
    import numpy as np
    global oldFile
    
    
    import matplotlib.pyplot as plt
    '''
    returns the receptiveField by receptiveField pixel 10.8 radiance map for the geostationary 
    satelite at the time closest to TIME
    '''

    
    if longitude < minLongitde or longitude > maxLongitude or latitude < minLatitude or latitude >maxLatitide:
        return np.zeros((receptiveField,receptiveField))
    
    # Download the data file
   
    filePATH  =getClosestFile(TIME, 'ABI-L1b-RadF-M6C15')
    FILE = 'data/'+filePATH.split('/')[-1]
  
    downloadFile(filePATH)
    extractGeoData(filePATH, oldFile)
    oldFile = filePATH
  
    xIndex, yIndex = getIndexOfGeoDataMatricFromLongitudeLatitude(longitude,latitude)
   # print(minDistance)
    return Dataset(FILE,'r')['Rad'][xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)]
        
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
    import datetime
    UNIXTime = datetime.datetime(1970,1,1)
    GPSTime = datetime.datetime(1980,1,6)
    delta = (GPSTime-UNIXTime).total_seconds()
    return datetime.datetime.fromtimestamp(timestamp+delta)

def getTrainingData(dataSize):
    
    import numpy as np
    import datetime
    receptiveField = 6

    '''
    returns a set that conisit of radiance data for an area around every pixel
    in the given area together with its label
    '''
    xData = np.zeros((dataSize,receptiveField,receptiveField))
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
    GPM_data = getGPMData(datetime.datetime(2020,1,26),dataSize)
    '''
    next step is to pair the label with the geostattionary data.
    '''
    
    for i in range(dataSize):
        print(i)
        xData[i,:,:] = getGEOData(GPM_data[i,1], GPM_data[i,2],convertTimeStampToDatetime(GPM_data[i,3]))
        yData[i] = GPM_data[i,0]

    
    return xData, yData



xData, yData = getTrainingData(100)


#%%
def plotGPMData(GPM_data):
    import numpy as np
    from scipy.interpolate import griddata
   
    extent1  = [min(GPM_data[:,1]),max(GPM_data[:,1]),min(GPM_data[:,2]),max(GPM_data[:,2])]
    grid_x, grid_y = np.mgrid[extent1[0]:extent1[1]:200j, extent1[2]:extent1[3]:200j]
    points = GPM_data[:,1:3]
    values = GPM_data[:,0]*1000
    grid_z0 = griddata(points,values, (grid_x, grid_y), method='cubic')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,10))
    plt.imshow(grid_z0.T,extent=(extent1[0],extent1[1],extent1[2],extent1[3]), origin='lower')

def plotGEOData(FILE, extent):
    import numpy as np
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    xmin, ymin = getIndexOfGeoDataMatricFromLongitudeLatitude(extent[0],extent[3])
    xmax, ymax = getIndexOfGeoDataMatricFromLongitudeLatitude(extent[1],extent[2])
    print(xmin)
    print(xmax)
    print(ymin)
    print(ymax)
    
    plt.figure(figsize=(20,10))
    plt.imshow(Dataset(FILE,'r')['Rad'][xmin:xmax,ymin:ymax])


FILE = 'data/'+'OR_ABI-L1b-RadF-M6C15_G16_s20200260200156_e20200260209470_c20200260209555.nc'
plotGEOData(FILE,[-70,-51,-11,2.5])

#%%
import datetime as datetime
dataSize = 10000
GPM_data = getGPMData(datetime.datetime(2020,1,26),dataSize)
plotGPMData(GPM_data)
#%%
getGEOData(GPM_data[0,1], GPM_data[0,2],convertTimeStampToDatetime(GPM_data[0,3]))

#%%
import matplotlib.pyplot as plt

for i in range(10):
    plt.figure()
    plt.imshow(xData[i,:,:])
#%%
plt.plot(yData)    

#%%
    
import xarray
from pyproj import Proj
import numpy as np
global lons,lats


FILE = 'data/'+'OR_ABI-L1b-RadF-M6C15_G16_s20200260200156_e20200260209470_c20200260209555.nc'

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
#Dataset('data/OR_ABI-L1b-RadC-M6C01_G16_s20200160001163_e20200160003536_c20200160004010.nc','r')['nominal_satellite_subpoint_lat'][:]
# Create a pyproj geostationary map object
p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

# Perform cartographic transformation. That is, convert image projection coordinates (x and y)
# to latitude and longitude values.
XX, YY = np.meshgrid(x, y)
print("traonsfrming data")
lons, lats = p(XX, YY, inverse=True)
print("transformation done")

#%%


#%%
plt.plot(GPM_data[:,0])