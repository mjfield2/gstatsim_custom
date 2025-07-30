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

import sys
sys.path.append('..')

import gstatsim_custom as gsim

ds = xr.load_dataset(Path('../test_data.nc'))

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
df_grid = pd.DataFrame({'X' : x_cond, 'Y' : y_cond, 'residual' : res_cond[cond_msk], 'NormZ' : res_norm[cond_msk]})

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

vario = {
    'azimuth' : azimuth,
    'nugget' : nugget,
    'major_range' : major_range,
    'minor_range' : minor_range,
    'sill' : sill,
    's' : smoothness,
    'vtype' : 'matern',
}

rng = np.random.default_rng(0)

k = 64
rad = 50e3

ok_mean, ok_std = gsim.interpolate.krige(xx, yy, res_cond, vario, rad, k)
ok_std = ok_std - np.min(ok_std)

fig, axs = plt.subplots(1, 2, figsize=(10,4), sharey=True)
ax = axs[0]
im = ax.pcolormesh(ds.x, ds.y, ok_mean)
ax.axis('scaled')
plt.colorbar(im, ax=ax)

ax = axs[1]
im = ax.pcolormesh(ds.x, ds.y, ok_std, cmap='magma')
ax.axis('scaled')
plt.colorbar(im, ax=ax)
plt.show()