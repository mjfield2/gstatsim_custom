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
from cmcrameri import cm

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

ds = xr.open_dataset(Path('../../bedmap/bedmap3_mod_1km.nc'))

# exposed_rock_cond = np.full(ds.thick_cond.shape, np.nan)
# exposed_rock_cond[ds.mask == 4] = True
thick_cond = np.where(ds.mask == 4, 0, ds.thick_cond.values)
#ground_ice_msk = ds.mask==1

bed_cond = ds.surface_topography.values - thick_cond
ice_rock_msk = (ds.mask == 1) | (ds.mask == 4) | (ds.mask == 2)
bed_cond = np.where(ice_rock_msk, bed_cond, np.nan)
xx, yy = np.meshgrid(ds.x, ds.y)

cond_msk = ~np.isnan(bed_cond)
#cond_msk = np.where(
x_cond = xx[cond_msk]
y_cond = yy[cond_msk]
data_cond = bed_cond[cond_msk]
trend = ds.trend.values

res_cond = bed_cond - trend

sim = np.load(Path('../../gstatsim_custom/results/nonstationary_sim.npy'))

ilow = 1000
ihigh = 5800
jlow = 700
jhigh = 6200

xx_trim = xx[ilow:ihigh,jlow:jhigh]
yy_trim = yy[ilow:ihigh,jlow:jhigh]
sim_trim = sim[ilow:ihigh,jlow:jhigh]
trend_trim = trend[ilow:ihigh,jlow:jhigh]

plt.figure(figsize=(13,10))
im = plt.pcolormesh(xx_trim/1000, yy_trim/1000, sim_trim+trend_trim, cmap=cm.batlowW, vmin=-3000, vmax=4500)
plt.axis('scaled')
plt.xlabel('X [km]')
plt.ylabel('Y [km]')
plt.title('SGS simulation+trend, matern covariance, nonstationary, anisotropic 1 km')
plt.colorbar(im, pad=0.03, aspect=40, shrink=0.7, label='bed elevation [meters]')
plt.savefig('figures/matern_nonstationary_1km.png', dpi=300, bbox_inches='tight')
plt.show()