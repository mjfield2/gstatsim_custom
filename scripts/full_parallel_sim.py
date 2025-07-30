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
from cmcrameri import cm

# plotting
import matplotlib.pyplot as plt

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import time
import numbers
import warnings

import sys
sys.path.append('..')

import gstatsim_custom as gsim

if __name__=='__main__':

    ds = xr.open_dataset(Path('../../bedmap/bedmap3_mod_1km.nc'))

    exposed_rock_cond = np.full(ds.thick_cond.shape, np.nan)
    exposed_rock_cond[ds.mask == 4] = True
    thick_cond = np.where(exposed_rock_cond==True, 0, ds.thick_cond.values)
    ground_ice_msk = ds.mask==1
    
    bed_cond = ds.surface_topography.values - thick_cond
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    cond_msk = ~np.isnan(bed_cond)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_cond = bed_cond[cond_msk]
    trend = ds.trend.values

    res_cond = bed_cond - trend

    print('calculating variogram')
    res_norm, nst_trans = gsim.utilities.gaussian_transformation(res_cond, cond_msk)
    vgrams, experimental, bins = gsim.utilities.variograms(xx, yy, res_cond, maxlag=30e3, n_lags=20, downsample=100)
    
    parameters = vgrams['exponential']
    vario = {
        'azimuth' : 0,
        'nugget' : parameters[-1],
        'major_range' : parameters[0],
        'minor_range' : parameters[0],
        'sill' : parameters[1],
        'vtype' : 'exponential',
    }
    rng = np.random.default_rng(0)
    k = 20
    rad = 50e3
    sim_mask = (ds.mask==1) | (ds.mask==4)

    print('starting simulation')
    tic = time.time()
    sim = gsim.parallel.parallel_sgs(xx, yy, res_cond, vario, rad, k, seed=rng, sim_mask=sim_mask, chunk_size=50e3, n_workers=7)
    toc = time.time()
    print(f'{toc-tic} seconds')

    ilow = 1000
    ihigh = 5800
    jlow = 700
    jhigh = 6200
    
    xx_trim = xx[ilow:ihigh,jlow:jhigh]
    yy_trim = yy[ilow:ihigh,jlow:jhigh]
    sim_trim = sim[ilow:ihigh,jlow:jhigh]
    trend_trim = trend[ilow:ihigh,jlow:jhigh]

    plt.figure(figsize=(13,10))
    im = plt.pcolormesh(xx_trim/1000, yy_trim/1000, sim_trim+trend_trim, cmap=cm.batlowW)
    plt.axis('scaled')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('SGS parallel simulation+trend, exponential covariance')
    plt.colorbar(im, pad=0.03, aspect=40, shrink=0.7, label='bed elevation [meters]')
    plt.savefig('full_parallel_simulation_2.png', dpi=300, bbox_inches='tight')
    plt.show()