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
def getFilesForHour(DATE):
    
    '''
        Returns an iterator of file names for the specific hour defines by DATE
    '''
    from google.cloud import storage
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    PATH_Storage = 'ABI-L1b-RadC/%s' % (DATE.strftime('%Y/%j/%H')) 
    
    return bucket.list_blobs(prefix = PATH_Storage)

def getClosestFile(DATE):
    import datetime
    import numpy as np
    '''
    Returns the filepath closest to the DATE object
    
    '''
    files_for_hour = list(map(lambda x : x.name, getFilesForHour(DATE)))
    date_diff = map(lambda x: abs(datetime.strptime(x.split('_')[3], 's%Y%j%H%M%S%f')-DATE), files_for_hour)
    return '%s' % (files_for_hour[np.argmin(date_diff)]) 

def downloadFile(FILE):
    '''
    Downloads FILE and saves it in the data folder
    '''
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name='gcp-public-data-goes-16', user_project=None)
    blob = bucket.blob(FILE)
    blob.download_to_filename('data/'+FILE.split('/')[-1])

def getGEOData(longitude, latitude, TIME):
    from netCDF4 import Dataset
    '''
    returns the receptiveField by receptiveField pixel 10.8 radiance map for the geostationary 
    satelite at the time closest to TIME
    '''
    # Download the data file
    filePATH  =getClosestFile(TIME)
    downloadFile(filePATH)
    
    fileName = 'data/'+filePATH.split('/')[-1]
    #TODO: convert the longitude and latitude to indexes of the matrix
    
    xIndex = 1000
    yIndex = 1000
    return Dataset(fileName,'r')['Rad'][xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)]
        
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
    import numpy as np
    data = np.zeros((maxDataSize,4))
    
    # get the files for the specific day
    files = getGPMFilesForSpecificDay(DATE)
    
    index = 0
    
    for FILENAME in files:
        # download the gpm data
        downloadGPMFile(FILENAME,DATE)
        
        # read the data
        path =  filename
        f = h5py.File(path, 'r')
        precepTotRate = f['NS']['surfPrecipTotRate'][:].flatten()
        longitude = f['NS']['Longitude'][:].flatten()
        latitude = f['NS']['Latitude'][:].flatten()
        time = np.array([f['NS']['navigation']['timeMidScan'][:],]*f['nrayNS_idx'].shape[0]).transpose().flatten()
        
        index1 = min(maxDataSize,index+len(precepTotRate))
        
        data[index:index1,0] = precepTotRate[:index1-(index+len(precepTotRate))]
        data[index:index1,1] = longitude[:index1-(index+len(precepTotRate))]
        data[index:index1,2] = latitude[:index1-(index+len(precepTotRate))]
        data[index:index1,3] = time[:index1-(index+len(precepTotRate))]
        
        index = index + len(precepTotRate)
        
        if index > maxDataSize:
            break
        
        
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
        xData[i,:,:] = getGEOData(GPM_data[i,1], GPM_data[i,2],convertTimeStampToDatetime(GPM_data[i,3]))
        yData[i] = GPM_data[i,0]

    
#%%

   
#%%
   
import h5py
filename = 'data/2B.GPM.DPRGMI.CORRA2018.20200126-S004153-E021426.033577.V06A.hdf5'
f = h5py.File(filename, 'r')
print(f.keys())
print(f['NS'].keys())
print(f['nrayNS_idx'].shape[0])
#print(f['NS'][].shape)
print(f['NS']['navigation']['timeMidScan'][0])
seconds = f['NS']['navigation']['timeMidScan'][0]
UNIXTime = datetime.datetime(1970,1,1)
GPSTime = datetime.datetime(1980,1,6)
delta = (GPSTime-UNIXTime).total_seconds()

print(datetime.datetime.fromtimestamp(seconds+delta))

#%%

#%%
from netCDF4 import Dataset
Dataset('data/OR_ABI-L1b-RadC-M6C01_G16_s20200160001163_e20200160003536_c20200160004010.nc','r')['y'][1230]