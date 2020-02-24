# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:52:46 2020

@author: gustav
"""

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
    PATH_Storage = 'ABI-L1b-RadF/%s' % (DATE.strftime('%Y/%j/%H')) 
    
    return bucket.list_blobs(prefix = PATH_Storage)
def getMiddleTime(FILE):
    import datetime
    middleTimeDifference =(datetime.datetime.strptime(FILE.split('_')[4], 'e%Y%j%H%M%S%f')-datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')).total_seconds()
    return (datetime.datetime.strptime(FILE.split('_')[3], 's%Y%j%H%M%S%f')+datetime.timedelta(0, int(middleTimeDifference/2)))

def getClosestFile(DATE, CHANNEL):
    from datetime import datetime, timedelta
    import numpy as np
    '''
    Returns the filepath closest to the DATE object
    
    '''
    files_for_hour = list(map(lambda x : x.name, getFilesForHour(DATE)))
   
    files_for_hour = [file for file in files_for_hour if file.split('_')[1][-3:] == CHANNEL ]
    if len(files_for_hour) == 0:
        print("could npt get closest file"+str(DATE))
        
     
    
    date_diff = np.zeros((len(files_for_hour),1))
    
    for i in range(len(files_for_hour)):
        
        middleTime = getMiddleTime(files_for_hour[i])
        
        date_diff[i] = np.abs((DATE-middleTime).total_seconds())
    
    # get the file before and after as well
    index_closest_file = np.argmin(date_diff[:,0])
    #print("the file")
    #print(files_for_hour[index_closest_file])
    '''
    if index_closest_file == 0 and DATE.hour == 0:
        day_of_year = DATE.timetuple().tm_yday
        day_before = datetime(DATE.year, 1, 1,23) + timedelta(day_of_year - 2)
        #print("here")
        #print(day_before)
        #datetime(DATE.year,DATE.day-1,23)
        files_for_hour_before = list(map(lambda x : x.name, getFilesForHour(day_before)))
        files_for_hour_before = [file for file in files_for_hour_before if file.split('_')[1][-3:] == CHANNEL ]
        if len(files_for_hour_before) == 0:
            print(day_before)
        #print(files_for_hour_before)
        file_before = files_for_hour_before[-1]
    elif index_closest_file == 0:
       
        hour_before = datetime(DATE.year,DATE.month,DATE.day, DATE.hour -1)
        files_for_hour_before = list(map(lambda x : x.name, getFilesForHour(hour_before)))
        files_for_hour_before = [file for file in files_for_hour_before if file.split('_')[1][-3:] == CHANNEL ]
        if len(files_for_hour_before) == 0:
            print(hour_before)
        file_before = files_for_hour_before[-1]
        #print("here1")
        #print(hour_before)
        #print(files_for_hour_before)
    else:
        
        file_before = files_for_hour[index_closest_file-1]
        #print("here2")
        #print(file_before)
    
    if index_closest_file == len(files_for_hour)-1 and DATE.hour == 23:
        day_of_year = DATE.timetuple().tm_yday
        day_after = datetime(DATE.year, 1, 1,23) + timedelta(day_of_year)
        #day_after = datetime(DATE.year,DATE.day+1,0)
        files_for_hour_after = list(map(lambda x : x.name, getFilesForHour(day_after)))
        files_for_hour_after = [file for file in files_for_hour_after if file.split('_')[1][-3:] == CHANNEL ]
        if len(files_for_hour_after) == 0:
            print(day_after)
        file_after = files_for_hour_after[0]
        #print("here10")
        #print( day_after)
    elif index_closest_file == len(files_for_hour)-1:
        hour_after = datetime(DATE.year,DATE.month,DATE.day,DATE.hour +1)
        files_for_hour_after = list(map(lambda x : x.name, getFilesForHour(hour_after)))
        files_for_hour_after = [file for file in files_for_hour_after if file.split('_')[1][-3:] == CHANNEL ]
        if len(files_for_hour_after) == 0:
            print(hour_after)
            
        file_after = files_for_hour_after[0]
        #print("here3")
        #print(hour_after)
    else:
        file_after = files_for_hour[index_closest_file+1]
        #print("here4")
        #print(file_after)
        
    '''
    return '%s' % (files_for_hour[index_closest_file]), getMiddleTime(files_for_hour[np.argmin(date_diff[:,0])]) 

    #return '%s' % (file_before),'%s' % (files_for_hour[index_closest_file]),'%s' % (file_after), getMiddleTime(files_for_hour[np.argmin(date_diff[:,0])]) 

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
    
def extractGeoData(filePATH, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []):
    import xarray
    from pyproj import Proj
    import numpy as np
    import time
    
    maxLongitude = -40
    minLongitde = -80
    maxLatitide = 8
    minLatitude = -20
    
    #start_time = time.time()
   
    
    
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
    #end_time = time.time()
    
    #print("time for first section %s" % (end_time-start_time))
    
    #start_time = time.time()
    if prev_sat_h == sat_h and prev_sat_lon == sat_lon and prev_sat_sweep == sat_sweep and (prev_x.data == x.data).all() and (prev_y.data == y.data).all():
        lons = prev_lons
        lats = prev_lats
    else:
      
        XX, YY = np.meshgrid(x, y)
        print("traonsfrming data")
        lons, lats = p(XX, YY, inverse=True)
        print("transformation done")
    #end_time = time.time()
    
    #print("time for checking %s" % (end_time-start_time))
    
    return lons,lats,C,rad, x_data, y_data

#def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude, sat_h, sat_lon, sat_sweep, x_data,y_data):
def getIndexOfGeoDataMatricFromLongitudeLatitude(longitude, latitude,proj, x_data,y_data):
    # project the longitude and latitude to geostationary references
    from pyproj import Proj
    import numpy as np
    import time
    
    #start_time = time.time()
    
    #p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    x, y = proj(longitude, latitude)
    #x = 0
    #y = 0
    #end_time = time.time()
    
    #print("time_for_projection %s" % (end_time-start_time))
    
    #start_time = time.time()
    # get the x and y indexes
    x_index = (np.abs(x_data.data-x)).argmin()
    y_index = (np.abs(y_data.data-y)).argmin()
    #end_time = time.time()
    
    #print("time for finding the index : %s" % (end_time-start_time))
    
    lons, lats = proj(x_data[x_index], y_data[y_index], inverse=True)
  
    return y_index, x_index, np.sqrt((longitude-lons)*(longitude-lons)+(latitude-lats)*(latitude-lats))
    #return 50, 50, 0
    
    
def getGEOData(GPM_data, channel):
    from netCDF4 import Dataset
    import numpy as np
    import time
    from datetime import datetime
    from pyproj import Proj
    
    #if longitude < minLongitde or longitude > maxLongitude or latitude < minLatitude or latitude >maxLatitide:
    #    return np.zeros((receptiveField,receptiveField))
    
    # Download the data file
    #file_paths_before = []
    #file_paths_after = []
    filePaths = []
    newFileIndexes = []
    previousFileName = ""
    start_time = time.time()
 
   
    middleTime = datetime(1980,1,1)
    
    for i in range(len(GPM_data[:,2])):
        currentTime = convertTimeStampToDatetime(GPM_data[i,2])
        if(np.abs((currentTime-middleTime).total_seconds()) > 600):
            
            #file_path_before, filePath, file_path_after, middleTime = getClosestFile(currentTime, channel)
            filePath, middleTime = getClosestFile(currentTime, channel)
            #print(file_path_before)
            #print(filePath)
            #print(file_path_after)
            
           
        if filePath != previousFileName:
            newFileIndexes.append(i)
            filePaths.append(filePath)
            #file_paths_before.append(file_path_before)
            #file_paths_after.append(file_path_after)
            previousFileName = filePath
        
    end_time = time.time()
    print("time for getting file paths %s" % (end_time-start_time))
    # iterate through all unique file names
    nmb_files = 1
    xData = np.zeros((len(GPM_data),nmb_files,receptiveField,receptiveField))
    times = np.zeros((len(GPM_data),nmb_files))
    distance = np.zeros((len(GPM_data),nmb_files))
    
    lons = []
    lats = []
    sat_h = 0
    sat_lon = 0
    sat_sweep = 0
    x_data = []
    y_data =  []
    
    for i in range(len(newFileIndexes)):
        '''
        filePATH = file_paths_before[i]
        FILE = 'data/'+file_paths_before[i].split('/')[-1]
        print(FILE)
       
        downloadFile(filePATH)
        
        
        # file before
        lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
      
        sat_h = C['goes_imager_projection'].perspective_point_height
        
        # Satellite longitude
        sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
        
        # Satellite sweep
        sat_sweep = C['goes_imager_projection'].sweep_angle_axis
       
        if i == len(newFileIndexes)-1:
            endIndex = len(GPM_data)
        else:
            endIndex = newFileIndexes[i+1]
        
        
        
        for j in range(newFileIndexes[i],endIndex):
            xIndex, yIndex , distance[j,0]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,1], GPM_data[j,2], sat_h, sat_lon, sat_sweep, x_data,y_data)
            xData[j,0,:,:], times[j,0] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
        '''
        # closest file
        filePATH = filePaths[i]
        FILE = 'data/'+filePaths[i].split('/')[-1]
        if i % 100 == 0:
            print(i/len(newFileIndexes))
       
        downloadFile(filePATH)
        #filePATH, prev_sat_h = 0, prev_sat_lon = 0, prev_sat_sweep = 0, prev_x = [], prev_y = [], prev_lons =[], prev_lats = []
        #start_time = time.time()
        lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH,sat_h,sat_lon,sat_sweep, x_data, y_data,lons,lats)
        #end_time = time.time()
        #print("time for extractng geo data %s" % (end_time-start_time))
        sat_h = C['goes_imager_projection'].perspective_point_height
        
        # Satellite longitude
        sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
        
        # Satellite sweep
        sat_sweep = C['goes_imager_projection'].sweep_angle_axis
       
        if i == len(newFileIndexes)-1:
            endIndex = len(GPM_data)
        else:
            endIndex = newFileIndexes[i+1]
        
        
        #start_time = time.time()
        proj = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
        for j in range(newFileIndexes[i],endIndex):
            #start_time = time.time()
            xIndex, yIndex , distance[j,0]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,0], GPM_data[j,1], proj, x_data,y_data)
            #end_time = time.time()
            #print(" time for getting index %s" %(end_time -start_time))
            
            #start_time = time.time()
            xData[j,0,:,:], times[j,0] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
            #end_time = time.time()
            #print("time for extracting data: %s" %(end_time-start_time))
        #end_time = time.time()
        #print("time for for lop %s" %(end_time-start_time))
        '''
        # file after
        filePATH = file_paths_after[i]
        FILE = 'data/'+file_paths_after[i].split('/')[-1]
        print(FILE)
       
        downloadFile(filePATH)
        
        lons,lats,C,rad, x_data, y_data = extractGeoData(filePATH)
      
        sat_h = C['goes_imager_projection'].perspective_point_height
        
        # Satellite longitude
        sat_lon = C['goes_imager_projection'].longitude_of_projection_origin
        
        # Satellite sweep
        sat_sweep = C['goes_imager_projection'].sweep_angle_axis
       
        if i == len(newFileIndexes)-1:
            endIndex = dataSize
        else:
            endIndex = newFileIndexes[i+1]
        
        
        
        for j in range(newFileIndexes[i],endIndex):
            xIndex, yIndex , distance[j,2]= getIndexOfGeoDataMatricFromLongitudeLatitude(GPM_data[j,1], GPM_data[j,2], sat_h, sat_lon, sat_sweep, x_data,y_data)
            xData[j,2,:,:], times[j,2] = rad.data[xIndex-int(receptiveField/2):xIndex+int(receptiveField/2),yIndex-int(receptiveField/2):yIndex+int(receptiveField/2)], (getMiddleTime(FILE)-datetime(1980,1,6)).total_seconds()
        
        '''
        
    
    return xData[:,0,:,:], times[:,0], distance[:,0]   

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
       
def getGPMData(start_DATE, maxDataSize, data_per_GPM_pass, resolution):
    
    '''
        retruns GPM data for the day provided. The data is in form of an array
        with each entry having attributes:
            1: position
            2: time
            3: rain amount
    '''
    
    import h5py
    import numpy as np
    from datetime import datetime, timedelta
    import random
    days_missing_in_GEO = [str(datetime(2017,8,3)),
                           str(datetime(2017,9,26)),
                           str(datetime(2017,9,27)),
                           str(datetime(2017,9,28)),
                           str(datetime(2017,9,29)),
                           str(datetime(2017,11,30)),
                           str(datetime(2017,12,1)),
                           str(datetime(2017,12,2)),
                           str(datetime(2017,12,3)),
                           str(datetime(2017,12,4)),
                           str(datetime(2017,12,5)),
                           str(datetime(2017,12,6)),
                           str(datetime(2017,12,7)),
                           str(datetime(2017,12,8)),
                           str(datetime(2017,12,9)),
                           str(datetime(2017,12,10)),
                           str(datetime(2017,12,11)),
                           str(datetime(2017,12,12)),
                           str(datetime(2017,12,13)),
                           str(datetime(2017,12,14)),
                           str(datetime(2018,1,28)),
                           str(datetime(2018,2,21)),
                           str(datetime(2018,2,22)),
                           str(datetime(2018,5,6)),
                           str(datetime(2019,2,25)),
                           str(datetime(2019,4,15)),
                           str(datetime(2019,5,30)),
                           str(datetime(2019,6,27)),
                           str(datetime(2019,6,28)),
                           str(datetime(2019,11,6))
                           ]
    pos_time_data = np.zeros((maxDataSize,3))
    prec_data = np.zeros((maxDataSize,2*resolution+1,2*resolution+1))
    
    # get the files for the specific day
    
    DATE = start_DATE
    index = 0
    while index < maxDataSize:
        
        
        if str(DATE) in days_missing_in_GEO:
            DATE += timedelta(days=1)
            continue
        
        files = getGPMFilesForSpecificDay(DATE)
        
        for FILENAME in files:
            # download the gpm data
            downloadGPMFile(FILENAME,DATE)
            
            # read the data
            try:
                path =  'data/'+FILENAME
                f = h5py.File(path, 'r')
                dim_1 = f['NS']['surfPrecipTotRate'][:].shape[0]
                dim_2 = f['NS']['surfPrecipTotRate'][:].shape[1]
                precepTotRate = f['NS']['surfPrecipTotRate'][:].flatten()
                longitude = f['NS']['Longitude'][:].flatten()
                latitude = f['NS']['Latitude'][:].flatten()
                time = np.array([f['NS']['navigation']['timeMidScan'][:],]*f['nrayNS_idx'].shape[0]).transpose().flatten()
            except:
                continue
            # remove null data 
            ''''
            indexes = np.where(np.abs(precepTotRate) < 200)[0]
            precepTotRate = precepTotRate[indexes]
            longitude = longitude[indexes]
            latitude = latitude[indexes]
            time = time[indexes]
            '''
            # get indexes of rainy data
            '''
            rain_indexes = np.where(np.abs(precepTotRate) > 0)[0]
            norain_indexes = np.where(np.abs(precepTotRate) == 0)[0]
            '''
            '''
            print(len(precepTotRate))
            print(len(rain_indexes))
            print(len(norain_indexes))
            print(int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division))
            print(len( norain_indexes[:int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division)]))
            '''
            
            
            #tot_indexes = np.concatenate((rain_indexes, norain_indexes[:int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division)]))
            #tot_indexes = rain_indexes + norain_indexes[:int((len(rain_indexes) - rain_norain_division*len(rain_indexes))/rain_norain_division)]
            #print(len(tot_indexes))
            '''
            precepTotRate = precepTotRate[tot_indexes]
            longitude = longitude[tot_indexes]
            latitude = latitude[tot_indexes]
            time = time[tot_indexes]
            '''
            
            
            index1 = min(maxDataSize,index+len(precepTotRate),index+data_per_GPM_pass)
            #print('index')
            #print(index1)
            # select random data
            
            #indexes = random.sample(range(0, len(precepTotRate)), index1-index)
            res = resolution
            random_sample = random.sample(range(0, len(precepTotRate)), len(precepTotRate))
            
            prec_matrix = np.reshape(precepTotRate,(dim_1,dim_2))
            
            
            chosen_indexes = []
            #chosen_prec = []
           # print(index)
            #print(index1)
            tmp_prec_data = np.zeros((index1-index,2*resolution+1,2*resolution+1))
            
            for tmp_index in random_sample:
                # check if index is to close to any of the sides
                #print(tmp_index)
                column = tmp_index % dim_2
                row = int(tmp_index / dim_2)
                
                tmp_prec_matrix = prec_matrix[row-res:row+res+1,column-res:column+res+1]
                if tmp_prec_matrix.shape[0] != 2*resolution+1 or tmp_prec_matrix.shape[1] != 2*resolution+1:
                    continue
                #if tmp_index % dim_2 < res or tmp_index-dim_2*res <0 or tmp_index > len(precepTotRate)-res*dim_2:
                #    continue
                
                # get the neigbour values
                
                #print(dim_2)
                #print(dim_1)
                #print(row)
                #print(column)
                #print(prec_matrix[row-res:row+res+1,column-res:column+res+1])
                # check if any values are nan
                if(len(np.where(np.abs(tmp_prec_matrix)>300)[0])>0):
                   continue
               
                # calculate the mean rainfall
                
                #print(men_prec)
                chosen_indexes.append(tmp_index)
                #print(tmp_prec_data.shape)
                tmp_prec_data[len(chosen_indexes)-1,:,:] =tmp_prec_matrix
                #chosen_indexes.append(tmp_index)
                #chosen_prec.append(men_prec)
                
                # break if maximum number of elements foun
                #print(len(chosen_indexes))
            
                if len(chosen_indexes) >= index1-index:
                    break
            
            if index1 > index + len(chosen_indexes):
                #print(index1)
                #print(index)
                #print(len(chosen_indexes))
                #print(prec_matrix)
                index1 = index+len(chosen_indexes)
                tmp_prec_data = tmp_prec_data[:len(chosen_indexes),:,:]
                #print('yess')
            
            #print(chosen_prec)
            #data[index:index1,0] = precepTotRate[chosen_indexes]
            #data[index:index1,0] = chosen_prec
            pos_time_data[index:index1,0] = longitude[chosen_indexes]
            pos_time_data[index:index1,1] = latitude[chosen_indexes]
            pos_time_data[index:index1,2] = time[chosen_indexes]
            
            prec_data[index:index1,:,:] = tmp_prec_data
            
            index = index1
            print(index)
            if index >= maxDataSize:
                break
            
           
                
        
        DATE += timedelta(days=1)
        print(DATE)
      
    return pos_time_data, prec_data
   
def convertTimeStampToDatetime(timestamp):
    from datetime import datetime, timedelta
    return datetime(1980, 1, 6) + timedelta(seconds=timestamp - (35 - 19))
    

def getTrainingData(dataSize, nmb_GPM_pass, GPM_resolution):
    
    import numpy as np
    import datetime
    import time
    
    #receptiveField = 28

    '''
    returns a set that conisit of radiance data for an area around every pixel
    in the given area together with its label
    '''
    
    xData = np.zeros((dataSize,2,receptiveField,receptiveField))
    times = np.zeros((dataSize,3))
    yData = np.zeros((dataSize,1))
    distance = np.zeros((dataSize,2))
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
    GPM_pos_time_data, GPM_prec_data = getGPMData(datetime.datetime(2017,8,2),dataSize,nmb_GPM_pass,GPM_resolution)
    times[:,0] = GPM_pos_time_data[:,2]
    end_time = time.time()
    
    print("time for collecting GPM Data %s" % (end_time-start_time))
    '''
    next step is to pair the label with the geostattionary data.
    '''
    start_time = time.time()
    
    xData[:,0,:,:], times[:,1], distance[:,0] = getGEOData(GPM_pos_time_data,'C13')
    
    xData[:,1,:,:], times[:,2], distance[:,1] = getGEOData(GPM_pos_time_data,'C08')
    
    yData = GPM_prec_data
    end_time = time.time()
    
    print("time for collecting GEO Data %s" % (end_time-start_time))
    import numpy as np
    
    np.save('trainingData/xDataC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'.npy', xData)   
    np.save('trainingData/yDataC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'.npy', yData)   
    np.save('trainingData/timesC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'.npy', times)
    np.save('trainingData/distanceC8C13S'+str(dataSize)+'_R'+str(receptiveField)+'_P'+str(nmb_GPM_pass)+'GPM_res'+str(GPM_resolution)+'.npy', distance)  
    
    
    return xData, yData, times, distance


def plotTrainingData(xData,yData, times, nmbImages):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(5800,5800+nmbImages):
        
        fig = plt.figure()
        fig.suptitle('timediff %s, rainfall %s' % (np.abs(times[i,0]-times[i,1]), yData[i]), fontsize=20)
        plt.imshow(xData[i,:,:])
        
def extract_data_within_timewindow(xData, yData, times, distance):

    indexes = np.where(np.abs(times[:,0]-times[:,1]) <200)
    return xData[indexes[0],:,:] , yData[indexes[0]] , times[indexes[0],:], distance[indexes[0],:]

def preprocessDataForTraining(xData, yData, times, distance):
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    scaler1 = StandardScaler()

    # reshape data for the QRNN
    newXData = np.reshape(xData,(xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]))
    newYData = np.reshape(yData,(len(yData),1))
    
    #scaler1.fit(newXData)
    #newXData = scaler1.transform(newXData)
    #newYData = newYData/newYData.max()
    
    # comine the IR images and the distance and time difference
    tmp = np.zeros((xData.shape[0],xData.shape[1]*xData.shape[2]*xData.shape[3]+4))
    tmp[:,:xData.shape[1]*xData.shape[2]*xData.shape[3]] = newXData
    
    '''
    tmp[:,-1] = (times[:,0]-times[:,1])/(times[:,0]-times[:,1]).max()
    tmp[:,-2] = distance[:,0]/distance.max()
    tmp[:,-3] = (times[:,0]-times[:,2])/(times[:,0]-times[:,2]).max()
    tmp[:,-4] = distance[:,1]/distance.max()
    '''
    
    tmp[:,-1] = (times[:,0]-times[:,1])/1000
    tmp[:,-2] = distance[:,0]
    tmp[:,-3] = distance[:,1]
    tmp[:,-4] = (times[:,0]-times[:,2])/1000
    
  
    
    newXData = tmp
    
    # scale the data with unit variance and and between 0 and 1 for the labels
    
    return newXData, newYData
def get_single_GPM_pass(DATE): 
    import numpy as np
    import h5py
    
    files = getGPMFilesForSpecificDay(DATE)
   # FILENAME = files[0]
    
    # download the gpm data
   # downloadGPMFile(files[0],DATE)
    
    # read the data
    
    for FILENAME in files:
            # download the gpm data
            downloadGPMFile(FILENAME,DATE)
            
            # read the data
            try:
                path =  'data/'+FILENAME
                f = h5py.File(path, 'r')
                precepTotRate = f['NS']['surfPrecipTotRate'][:].flatten()
                longitude = f['NS']['Longitude'][:].flatten()
                latitude = f['NS']['Latitude'][:].flatten()
                time = np.array([f['NS']['navigation']['timeMidScan'][:],]*f['nrayNS_idx'].shape[0]).transpose().flatten()
                indexes = np.where(np.abs(precepTotRate) < 200)[0]
                precepTotRate = precepTotRate[indexes]
                longitude = longitude[indexes]
                latitude = latitude[indexes]
                time = time[indexes]
                return precepTotRate, longitude, latitude, time
            except:
                continue
            # remove null data 
def load_data():
    import numpy as np

    xData =np.load('trainingData/xDataC8C13S350000_R28_P200GPM_res3.npy')
    yData = np.load('trainingData/yDataC8C13S350000_R28_P200GPM_res3.npy')
    times = np.load('trainingData/timesC8C13S350000_R28_P200GPM_res3.npy')
    distance = np.load('trainingData/distanceC8C13S350000_R28_P200GPM_res3.npy')  
    
    # remove nan values
    nanValues =np.argwhere(np.isnan(xData)) 
    xData = np.delete(xData,np.unique(nanValues[:,0]),0)
    yData = np.delete(yData,np.unique(nanValues[:,0]),0)
    times = np.delete(times,np.unique(nanValues[:,0]),0)
    distance = np.delete(distance,np.unique(nanValues[:,0]),0)

    # get the mean of the yData
    tmp_yData = np.zeros((len(yData),1))
    for i in range(len(yData)):
        tmp_yData[i,0] = yData[i,3,3]
    
    yData = tmp_yData    
    # narrow the field of vision
    #xData = xData[:,:,10:18,10:18]
    
    # select the data within time limit
    '''
    indexes = np.where(np.abs(times[:,0]-times[:,1])<200)[0]
    xData = xData[indexes,:,:,:]
    yData = yData[indexes]
    times = times[indexes]
    distance = distance[indexes]
    '''
    mean1 = np.mean(xData[:,0,:,:])
    mean2 = np.mean(xData[:,1,:,:])
    std1 = np.std(xData[:,0,:,:])
    std2 = np.std(xData[:,1,:,:])
    xData[:,0,:,:] = (xData[:,0,:,:]-mean1)/std1
    xData[:,1,:,:] = (xData[:,1,:,:]-mean2)/std2
    newXData, newYData = preprocessDataForTraining(xData, yData, times, distance)

    return newXData, newYData, mean1, mean2,std1,std2