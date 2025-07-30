# general
import numpy as np
from numpy.random import PCG64, SeedSequence
import pandas as pd
import verde as vd
import xarray as xr
import skgstat as skg
from skgstat import models
import gstatsim as gsm
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm

# plotting
import matplotlib.pyplot as plt

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import time
import numbers
import warnings

import gstatsim_custom as gsim

if __name__=='__main__':

    ds = xr.load_dataset('test_data.nc')
    bed_cond = ds.surface_topography.values - ds.thick_cond.values
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    cond_msk = ~np.isnan(bed_cond)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_cond = bed_cond[cond_msk]
    
    smoothing = 1e11
    interp = RBFInterpolator(np.array([x_cond, y_cond]).T, data_cond, smoothing=smoothing)
    trend = interp(np.array([xx.flatten(), yy.flatten()]).T).reshape(xx.shape)
    res_cond = bed_cond - trend
    
    res_norm, nst_trans = gsim.utilities.gaussian_transformation(res_cond, cond_msk)
    vgrams, experimental, bins = gsim.utilities.variograms(xx, yy, res_cond, maxlag=30e3, n_lags=20, downsample=10)
    
    parameters = vgrams['matern']
    
    # set variogram parameters
    nugget = parameters[-1]
    major_range = parameters[0]
    minor_range = parameters[0]
    sill = parameters[1]
    smoothness = parameters[2]
    nugget = parameters[-1]
    azimuth = 0
    
    k = 20
    rad = 50e3
    
    vario = {
        'azimuth' : azimuth,
        'nugget' : nugget,
        'major_range' : major_range,
        'minor_range' : minor_range,
        'sill' : sill,
        's' : smoothness,
        'vtype' : 'matern',
    }
    
    # tic = time.time()
    # res_newsim = interpolate.sgs(xx, yy, res_cond, vario, rad, k, seed=0)
    # toc = time.time()
    # print(f'{toc-tic:.2f} seconds normal simulation')
    
    # plt.pcolormesh(ds.x/1000, ds.y/1000, res_newsim + trend)
    # plt.axis('scaled')
    # plt.xlabel('X [km]')
    # plt.ylabel('Y [km]')
    # plt.title('New simulation')
    # plt.colorbar()
    # plt.show()
    
    tic = time.time()
    res_parsim = gsim.parallel.parallel_sgs(xx, yy, res_cond, vario, rad, k, seed=0, chunk_size=20e3, n_workers=7)
    toc = time.time()
    print(f'{toc-tic:.2f} seconds')
    
    plt.pcolormesh(ds.x/1000, ds.y/1000, res_parsim + trend)
    plt.axis('scaled')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Parallelsimulation')
    plt.colorbar()
    plt.show()