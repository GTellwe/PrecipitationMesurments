# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:19:27 2020

@author: gustav
"""

import h5py
filename = '2B.GPM.DPRGMI.CORRA2018.20200119-S234145-E011417.033483.V06A.SUB.hdf5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    
    
#%%
    # Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

import h5py
filename = '2B.GPM.DPRGMI.CORRA2018.20200119-S234145-E011417.033483.V06A.SUB.hdf5'
f = h5py.File(filename, 'r')
print(f.keys())
print(f['MS'].keys())
print(f['nrayMS_idx'][:])
data1 = f['MS']['surfPrecipTotRate']
print(data1.shape)
plt.plot(data1[100:1200,15])

#%%
import netCDF4
import numpy as np
f = netCDF4.Dataset('M6C01_G16_s20200240000163_e20200240009471_c20200240009528.nc')