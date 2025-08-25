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
import multiprocessing as mp

# plotting
import matplotlib.pyplot as plt

# io
from tqdm.auto import tqdm
import os
from pathlib import Path
import time
import numbers
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('..')

import gstatsim_custom as gsim

def local_variogram(xi, yi, coordinates, values, rng, kn, mindist=50e3, dist_change=10e3, mindata=500, directional=False):
    # get cond data within distance of point
    dist = mindist
    n_cond = 0
    while n_cond < mindata:
        dist_msk = vd.distance_mask((xi, yi), dist, coordinates)
        norm_near = values[dist_msk]
        n_cond = norm_near.size
        dist += 10e3
    
    coords_near = np.array(coordinates).T[dist_msk,:]
    
    inds = rng.integers(0, norm_near.shape[0]-1, mindata)
    norm_near = norm_near[inds]
    coords_near = coords_near[inds]
        
    var = np.var(norm_near)
    
    # compute experimental (isotropic) variogram
    V = skg.Variogram(coords_near, norm_near, bin_func='even', n_lags=20, 
                   maxlag=30e3, normalize=False)
    V.model = 'matern'
    
    rang = V.parameters[0]
    sill = V.parameters[1]
    smooth = V.parameters[2]
    
    # find nearest azimuth from velocity
    azim = kn.predict((xi, yi))
    
    if azim < 90:
        second_azim = azim + 90
    else:
        second_azim = azim - 90
    
    V_primary = skg.DirectionalVariogram(coords_near, norm_near, bin_fun="even", n_lags=20, 
                                      maxlag=30e3, normalize=False, azimuth=azim, tolerance=15)
    V_secondary = skg.DirectionalVariogram(coords_near, norm_near, bin_func="even", n_lags=20, 
                                  maxlag=30e3, normalize=False, azimuth=second_azim, tolerance=15)
    V_primary.model = 'matern'
    V_secondary.model = 'matern'
    
    major_range = V_primary.parameters[0]
    minor_range = V_secondary.parameters[0]
    
    return var, rang, sill, smooth, major_range, minor_range

if __name__ == '__main__':

    tic = time.time()

    rng = np.random.default_rng()

    ds = xr.load_dataset(Path('../test_data.nc'))
    
    bed_cond = ds.surface_topography.values - ds.thick_cond.values
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    cond_msk = ~np.isnan(bed_cond)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    coordinates = (x_cond, y_cond)
    data_cond = bed_cond[cond_msk]
    
    smoothing = 1e11
    interp = RBFInterpolator(np.array([x_cond, y_cond]).T, data_cond, smoothing=smoothing)
    trend = interp(np.array([xx.flatten(), yy.flatten()]).T).reshape(xx.shape)
    res_cond = bed_cond - trend

    res_norm, nst_trans = gsim.utilities.gaussian_transformation(res_cond, cond_msk)
    norm_values = res_norm[cond_msk]

    n_points = 50
    
    # randomly sample with latin hypercube sampling
    bounds = [-1.25e6, -0.9e6, -1.15e6, -0.75e6]
    l_bounds = bounds[:2]
    u_bounds = bounds[2:]
    
    sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    sample = sampler.random(n=n_points)
    points = qmc.scale(sample, l_bounds, u_bounds)
    
    ds_coarse = ds.coarsen(x=20, y=20, boundary='trim').mean()
    
    vel_mag = np.sqrt(ds.vx.values**2+ds.vy.values**2)
    vel_angle = np.arctan2(ds.vy.values, ds.vx.values)*180/np.pi
    
    kn = vd.KNeighbors(k=1)
    kn.fit((xx, yy), vel_angle)

    mindist = 50e3
    dist_change = 10e3
    mindata = 1000
    directional=True

    params = []
    for p in points:
        params.append([p[0], p[1], coordinates, norm_values, rng, kn, mindist, dist_change, mindata, directional])
    
    with mp.Pool(7) as p:
        result = p.starmap(local_variogram, params)
    
    result = np.array(result)
    
    df = pd.DataFrame(
        {
            'x' : points[:,0],
            'y' : points[:,1],
            'vars' : result[:,0],
            'range' : result[:,1],
            'sill' : result[:,2],
            'smooth' : result[:,3],
            'major_range' : result[:,4],
            'minor_range' : result[:,5]
        }
    )
    
    df.to_csv(Path('../example_variogram_params.csv'), index=False)

    toc = time.time()
    print(f'{toc-tic:.3f} seconds')