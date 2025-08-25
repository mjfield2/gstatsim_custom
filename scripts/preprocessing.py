import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import verde as vd
from pathlib import Path
import os
import time
from tqdm.auto import tqdm
import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")

from utilities import *

def plot_bad():

    ilow = 2000
    ihigh = 11600
    jlow = 1400
    jhigh = 12400
    
    xx, yy = np.meshgrid(ds.x, ds.y)
    
    xx_trim = xx[ilow:ihigh,jlow:jhigh]
    yy_trim = yy[ilow:ihigh,jlow:jhigh]

    ice_rock_msk = (ds.mask == 1) | (ds.mask == 4) | (ds.mask == 2)
    
    fig = plt.figure(figsize=(15,12), constrained_layout=True)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.02], figure=fig)
    
    ax = fig.add_subplot(gs[0,0])
    ax.scatter(df.x[good_surface][::100], df.y[good_surface][::100], c='tab:blue', s=0.1)
    ax.scatter(df.x[~good_surface][::100], df.y[~good_surface][::100], c='tab:red', s=0.1)
    ax.contour(xx_trim, yy_trim, ice_rock_msk[ilow:ihigh,jlow:jhigh], levels=[0.5], colors='k')
    ax.axis('scaled')
    ax.set_title('bad surface', fontsize=14)
    ax.set_xlim(xx_trim.min(), xx_trim.max())
    ax.set_ylim(yy_trim.min(), yy_trim.max())
    
    ax = fig.add_subplot(gs[0,1])
    im = ax.scatter(df.x[::100], df.y[::100], c=df.surface[::100], s=0.1, cmap='plasma')
    ax.contour(xx_trim, yy_trim, ice_rock_msk[ilow:ihigh,jlow:jhigh], levels=[0.5], colors='k')
    ax.axis('scaled')
    ax.set_title('surface', fontsize=14)
    cax = fig.add_subplot(gs[0,2])
    plt.colorbar(im, cax=cax, label='meters')
    ax.set_xlim(xx_trim.min(), xx_trim.max())
    ax.set_ylim(yy_trim.min(), yy_trim.max())
    
    ax = fig.add_subplot(gs[1,0])
    ax.scatter(df.x[good_thick][::100], df.y[good_thick][::100], c='tab:blue', s=0.1)
    ax.scatter(df.x[~good_thick][::100], df.y[~good_thick][::100], c='tab:red', s=0.1)
    ax.contour(xx_trim, yy_trim, ice_rock_msk[ilow:ihigh,jlow:jhigh], levels=[0.5], colors='k')
    ax.axis('scaled')
    ax.set_title('bad thickness', fontsize=14)
    ax.set_xlim(xx_trim.min(), xx_trim.max())
    ax.set_ylim(yy_trim.min(), yy_trim.max())
    
    ax = fig.add_subplot(gs[1,1])
    im = ax.scatter(df.x[::100], df.y[::100], c=df.thickness[::100], s=0.1)
    ax.contour(xx_trim, yy_trim, ice_rock_msk[ilow:ihigh,jlow:jhigh], levels=[0.5], colors='k')
    ax.axis('scaled')
    ax.set_title('thickness', fontsize=14)
    cax = fig.add_subplot(gs[1,2])
    plt.colorbar(im, cax=cax, label='meters')
    ax.set_xlim(xx_trim.min(), xx_trim.max())
    ax.set_ylim(yy_trim.min(), yy_trim.max())
    plt.savefig(Path('../figures/bad_surface_thickness.png'), dpi=300, bbox_inches='tight')
    plt.show(block=False)

rename_dict = {
    'surface_altitude (m)' : 'surface',
    'land_ice_thickness (m)' : 'thickness',
    'bedrock_altitude (m)' : 'bed',
    'longitude (degree_east)' : 'lon',
    'latitude (degree_north)' : 'lat'
}

def collect_files(path, verbose=False):
    total = 0
    for item in os.scandir(path):
        if item.name.endswith('.csv'):
            total += 1
    
    dfs = []
    
    for i, item in tqdm(enumerate(os.scandir(path)), total=total):
        if item.name.endswith('.csv'):
            if verbose==True:
                print(item.path)
            dfs.append(pd.read_csv(item.path, header=18))

    if len(dfs)==1:
        df = dfs[0]
    else:
        df = pd.concat(dfs)
    df = df.rename(columns=rename_dict)
    df = df[['surface', 'thickness', 'bed', 'lat', 'lon']]
    df = df.replace(-9999, np.nan)
    
    return df

tic = time.time()

bm1path = Path('D:/bedmap/BEDMAP1')
bm2path = Path('D:/bedmap/BEDMAP2')
bm3path = Path('D:/bedmap/BEDMAP3')
bmgrid_path = Path('D:/bedmap/bedmap3.nc')
bmach_path = Path('D:/bedmachine/BedMachineAntarctica-v3.nc')
stream_path = Path('D:/bedmap/bm3_streamlines_pt/bm3_streamlines_pt.shp')

print('collecting files')
bm1 = collect_files(bm1path)
bm2 = collect_files(bm2path)
bm3 = collect_files(bm3path)

df = pd.concat([bm1, bm2, bm3])

del(bm1)
del(bm2)
del(bm3)

x_coords, y_coords = geo2ant(df['lat'], df['lon'])
df['x'] = x_coords
df['y'] = y_coords

msk = (df['thickness'].isna()==True) & (df['surface'].isna()==False) & (df['bed'].isna()==False)
df.loc[msk, 'thickness'] = df.loc[msk, 'surface'] - df.loc[msk, 'bed']

df = df.loc[df['thickness'].isna()==False]

ds = xr.open_dataset(bmgrid_path)
xx, yy = np.meshgrid(ds.x, ds.y)

print('finding bad surface and thickness')

# find bad surface
kn = vd.KNeighbors(k=1)
kn.fit((xx.flatten(), yy.flatten()), ds.surface_topography.values.flatten())
preds = kn.predict((df.x, df.y))

good_surface = (np.abs(df['surface'].values-preds)<200) | (df['surface'].isna()==True)

# find bad ice thickness
kn = vd.KNeighbors(k=1)
kn.fit((xx.flatten(), yy.flatten()), ds.ice_thickness.values.flatten())
preds_thick = kn.predict((df.x, df.y))

good_thick = (df['thickness'].values-preds_thick)<200

plot_bad()

prev_shape = df.shape[0]
df = df.loc[good_surface, :]
print(f'{prev_shape-df.shape[0]:,} bad surface points removed')

good_thick = (df['thickness'].values-preds_thick[good_surface])<200

prev_shape = df.shape[0]
df = df.loc[good_thick, :]
print(f'{prev_shape-df.shape[0]:,} bad thickness points removed')

# add new COLDEX data
coldex = pd.read_csv(Path('D:/bedmap/2023_Antarctica_BaslerMKB.csv'))

x_coords, y_coords = geo2ant(coldex['LAT'], coldex['LON'])
coldex['x'] = x_coords
coldex['y'] = y_coords

coldex = coldex.rename(columns={'LAT' : 'lat', 'LON' : 'lon', 'THICK' : 'thickness', 'SURFACE' : 'surface', 'BOTTOM' : 'bed'})
coldex = coldex[['surface', 'thickness', 'bed', 'lat', 'lon', 'x', 'y']]

df = pd.concat([df, coldex])

print('doing block reduction')

# Block reduction onto Bedmap3 grid
xmin = ds.x.min().values - 250
xmax = ds.x.max().values + 250
ymin = ds.y.min().values - 250
ymax = ds.y.max().values + 250
region = [xmin, xmax, ymin, ymax]

block = vd.BlockReduce(region=region, spacing=500, adjust='region', center_coordinates=True, reduction=np.median)
block_coords, block_data = block.filter((df.x.values, df.y.values), df['thickness'].values)

block_coords = np.rint(block_coords)

thick_grid = xy_into_grid(ds.x.values, ds.y.values, block_coords, block_data)

# Interpolate BedMachine geoid onto Bedmap3
bmach = xr.open_dataset(bmach_path)

xx_bmach, yy_bmach = np.meshgrid(bmach.x, bmach.y)

linear = vd.KNeighbors(k=1)
linear.fit((xx_bmach, yy_bmach), bmach.geoid.values)
preds = linear.predict((xx, yy))

ds['geoid'] = (('y', 'x'), preds)
ds['thick_cond'] = (('y', 'x'), thick_grid)
# ds['bed_ell'] = (('y', 'x'), ds.bed_topography.values + preds)
# ds['surface_ell'] = (('y', 'x'), ds.surface_topography.values + preds)

ice_rock_msk = (ds.mask == 1) | (ds.mask == 4) | (ds.mask == 2)
print(f'{np.count_nonzero(np.isnan(ds.thick_cond.values) & ice_rock_msk):,} grid cells to simulate at 500 m resolution')

# Bedmap3 streamline ice thickness
print('getting streamlines')
pts = gpd.read_file(stream_path)

coords = pts.get_coordinates()
coords = (np.rint(coords['x'].values), np.rint(coords['y'].values))
thick = pts.thick.values

stream_thick = xy_into_grid(ds.x.values, ds.y.values, coords, thick)

ds['stream_thick'] = (('y', 'x'), stream_thick)

# create trend
print('creating trend')

bed_cond = ds.surface_topography.values - ds.thick_cond.values
xx, yy = np.meshgrid(ds.x, ds.y)

cond_msk = ~np.isnan(bed_cond)
x_cond = xx[cond_msk]
y_cond = yy[cond_msk]
data_cond = bed_cond[cond_msk]

cond_coords = np.array([x_cond[::1000], y_cond[::1000]]).T
trend = spline_interp_msk(cond_coords, data_cond[::1000], xx, yy, ice_rock_msk, damping=1e-5)

res_cond = bed_cond - trend

ds['trend'] = (('y', 'x'), trend)

# save to netCDF
ds.to_netcdf(Path('../processed_data/bedmap3_mod_500.nc'))

# coarsen to 1 km resolution
print('coarsening to 1 km')
ds = ds.coarsen(x=2, y=2, boundary='trim').median()
ds['mask'] = (('y', 'x'), np.where(np.isnan(ds.mask.values), np.nan, np.rint(ds.mask.values).astype(int)))

ds['bed_topography'] = (('y', 'x'), np.where(ds.mask.values==4, ds.surface_topography.values, ds.bed_topography.values))

ice_rock_msk = (ds.mask == 1) | (ds.mask == 4) | (ds.mask == 2)
print(f'{np.count_nonzero(np.isnan(ds.thick_cond.values) & ice_rock_msk):,} grid cells to simulate at 1 km resolution')

ds.to_netcdf(Path('../processed_data/bedmap3_mod_1000.nc'))

toc = time.time()

print(f'time elapsed: {toc-tic}')

fig, axs = plt.subplots(1, 2, figsize=(14,5), sharey=True)
ax = axs[0]
im = ax.imshow(trend, extent=(ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()))
ax.axis('scaled')
plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_title('Trend')

ax = axs[1]
im = ax.imshow(res_cond, extent=(ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()))
ax.axis('scaled')
plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_title('Residual')
plt.savefig(Path('../figures/trend_residual.png'), dpi=300, bbox_inches='tight')
plt.show()