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
import psutil
from netCDF4 import Dataset

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
from utilities import spline_interp_msk

def local_variogram(xi, yi, coordinates, values, rng, azim, maxlag=100e3, dist_change=50e3, n_samples=500, n_lags=50, directional=False):
    # get cond data within distance of point
    dist = maxlag
    n_cond = 0
    while n_cond < n_samples:
        dist_msk = vd.distance_mask((xi, yi), dist, coordinates)
        norm_near = values[dist_msk]
        n_cond = norm_near.size
        dist += dist_change

    # dist_msk = vd.distance_mask((xi, yi), maxlag, coordinates)
    # norm_near = values[dist_msk]
    
    coords_near = np.array(coordinates).T[dist_msk,:]
    
    inds = rng.integers(0, norm_near.shape[0]-1, n_samples)
    norm_near = norm_near[inds]
    coords_near = coords_near[inds]

    coords_near = coords_near[~np.isnan(norm_near)]
    norm_near = norm_near[~np.isnan(norm_near)]
        
    var = np.var(norm_near)

    try:
        # compute experimental (isotropic) variogram
        V = skg.Variogram(coords_near, norm_near, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False)
        V.model = 'matern'
        
        rang = V.parameters[0]
        sill = V.parameters[1]
        smooth = V.parameters[2]
    except:
        rang = np.nan
        sill = np.nan
        smooth = np.nan
    
    if azim < 90:
        second_azim = azim + 90
    else:
        second_azim = azim - 90

    try:
        V_primary = skg.DirectionalVariogram(coords_near, norm_near, bin_fun="even", n_lags=n_lags, 
                                          maxlag=maxlag, normalize=False, azimuth=azim, tolerance=15)
        V_primary.model = 'matern'
        major_range = V_primary.parameters[0]
    except:
        major_range = np.nan

    try:
        V_secondary = skg.DirectionalVariogram(coords_near, norm_near, bin_func="even", n_lags=n_lags, 
                                      maxlag=maxlag, normalize=False, azimuth=second_azim, tolerance=15)
        V_secondary.model = 'matern'
        minor_range = V_secondary.parameters[0]
    except:
        minor_range = np.nan
    
    return var, rang, sill, smooth, major_range, minor_range

if __name__ == '__main__':

    csv_path = Path('../processed_data/continent_variogram_params_1000.csv')
    nc_path = Path('../processed_data/continental_variogram_1000.nc')

    tic = time.time()

    rng = np.random.default_rng(0)

    ds = xr.open_dataset(Path('../processed_data/bedmap3_mod_1000.nc'))

    # add exposed bedrock to conditioning data
    thick_cond = np.where(ds.mask == 4, 0, ds.thick_cond.values)
    
    bed_cond = ds.surface_topography.values - thick_cond
    ice_rock_msk = (ds.mask == 1) | (ds.mask == 4) | (ds.mask == 2)
    bed_cond = np.where(ice_rock_msk, bed_cond, np.nan)
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    cond_msk = ~np.isnan(bed_cond)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    coordinates = (x_cond, y_cond)
    data_cond = bed_cond[cond_msk]
    trend = ds.trend.values

    res_cond = bed_cond - trend

    res_norm, nst_trans = gsim.utilities.gaussian_transformation(res_cond, cond_msk)
    norm_values = res_norm[cond_msk]

    ################################################

    n_points = 2000

    ################################################
    
    # randomly sample with latin hypercube sampling
    bounds = [ds.x.values.min(), ds.y.values.min(), ds.x.values.max(), ds.y.values.max()]
    l_bounds = bounds[:2]
    u_bounds = bounds[2:]
    
    sampler = qmc.LatinHypercube(d=2, optimization="lloyd", scramble=True, rng=rng)
    sample = sampler.random(n=n_points)
    points = qmc.scale(sample, l_bounds, u_bounds)

    ground_ice_rock_msk = (ds.mask==1) | (ds.mask==4) | (ds.mask==2)

    good_points = np.full(points.shape[0], True)
    
    for k, p in enumerate(points):
        i = np.argmin(np.abs(ds.y.values - p[1]))
        j = np.argmin(np.abs(ds.x.values - p[0]))
        if ground_ice_rock_msk[i, j] == False:
            good_points[k] = False

    points = points[good_points,:]
    
    vel = xr.open_dataset(Path('D:/phase_vel/antarctic_ice_vel_phase_map_v01.nc'))
    vel = vel.coarsen(x=10, y=10, boundary='trim').mean()
    vel_angle = np.arctan2(vel.VY.values, vel.VX.values)*180/np.pi

    xx_vel, yy_vel = np.meshgrid(vel.x, vel.y)

    kn = vd.KNeighbors(k=1)
    kn.fit((xx_vel, yy_vel), vel_angle)
    azims = kn.predict((points[:,0], points[:,1]))

    ##############################################################

    maxlag = 250e3
    dist_change = 50e3
    n_samples = 1000
    directional=True
    n_lags = 70

    ###############################################################

    # make list of parameters for parallel
    params = []
    for p, azim in zip(points, azims):
        params.append([p[0], p[1], coordinates, norm_values, rng, azim, maxlag, dist_change, n_samples, n_lags, directional])

    # run in parallel
    n_cores = psutil.cpu_count(logical=False)-1
    with mp.Pool(n_cores) as p:
        result = p.starmap(local_variogram, params)

    # # run serially
    # result = []
    # for p, azim in tqdm(zip(points, azims), total=points.shape[0]):
    #     result.append(local_variogram(p[0], p[1], coordinates, norm_values, rng, azim, maxlag, dist_change, mindata, n_lags, directional))
    
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
            'minor_range' : result[:,5],
            'azimuth' : azims
        }
    )
    df = df.dropna()

    smooth_max = 1

    # limit smoothness to range
    df.loc[df.smooth < 0.5, 'smooth'] = 0.5
    df.loc[df.smooth > smooth_max, 'smooth'] = smooth_max
    
    df.to_csv(csv_path, index=False)

    print('interpolating variogram parameters')
    points = np.array([df.x.values, df.y.values]).T

    damping = 1e-5
    
    interp_vars = spline_interp_msk(points, df.vars.values, xx, yy, ice_rock_msk, damping)
    
    interp_azim = spline_interp_msk(points, df.azimuth.values, xx, yy, ice_rock_msk, damping)
    interp_azim = np.where(interp_azim < -180, -180, interp_azim)
    interp_azim = np.where(interp_azim > 180, 180, interp_azim)
    
    interp_ranges = spline_interp_msk(points, df.range.values, xx, yy, ice_rock_msk, damping)
    interp_ranges = np.where(interp_ranges < 5e3, 5e3, interp_ranges)
    interp_ranges = np.where(interp_ranges > maxlag, maxlag, interp_ranges)
    
    interp_sills = spline_interp_msk(points, df.sill.values, xx, yy, ice_rock_msk, damping)
    
    interp_smooths = spline_interp_msk(points, df.smooth.values, xx, yy, ice_rock_msk, damping)
    interp_smooths = np.where(interp_smooths < 0.5, 0.5, interp_smooths)
    interp_smooths = np.where(interp_smooths > smooth_max, smooth_max, interp_smooths)

    interp_major_ranges = spline_interp_msk(points, df.major_range.values, xx, yy, ice_rock_msk, damping)
    interp_major_ranges = np.where(interp_major_ranges < 5e3, 5e3, interp_major_ranges)
    interp_major_ranges = np.where(interp_major_ranges > maxlag, maxlag, interp_major_ranges)
    
    interp_minor_ranges = spline_interp_msk(points, df.minor_range.values, xx, yy, ice_rock_msk, damping)
    interp_minor_ranges = np.where(interp_minor_ranges < 5e3, 5e3, interp_minor_ranges)
    interp_minor_ranges = np.where(interp_minor_ranges > maxlag, maxlag, interp_minor_ranges)

    dsv = xr.Dataset(
        data_vars=dict(
            varr=(('y', 'x'), interp_vars),
            ranges=(('y', 'x'), interp_ranges),
            sill=(('y', 'x'), interp_sills),
            smooth=(('y', 'x'), interp_smooths),
            major_range=(('y', 'x'), interp_major_ranges),
            minor_range=(('y', 'x'), interp_minor_ranges),
            azimuth=(('y', 'x'), interp_azim)
        ),
        coords=dict(
            y=('y', ds.y.values),
            x=('x', ds.x.values)
        )
    )

    dsv.to_netcdf(nc_path)

    toc = time.time()
    print(f'{toc-tic:.3f} seconds')
    
    dsv_trim = dsv.coarsen(x=10, y=10, boundary='trim').mean()
    
    plots = [dsv_trim.azimuth.values, dsv_trim.ranges.values, dsv_trim.major_range.values, dsv_trim.minor_range.values, dsv_trim.sill.values, dsv_trim.smooth.values]
    titles = ['azimuth', 'range', 'primary range', 'secondary range', 'sill', 'smoothness']
    
    fig, axs = plt.subplots(2, 3, figsize=(16,8.5), sharey=True)
    
    for p, ax, title in zip(plots, axs.flatten(), titles):
        im = ax.pcolormesh(dsv_trim.x/1000, dsv_trim.y/1000, p)
        ax.axis('scaled')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    plt.savefig(Path('../figures/continent_variogram_1000.png'), dpi=300, bbox_inches='tight')
    plt.show()

    
    
