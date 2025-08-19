from pathlib import Path
import os
from netCDF4 import Dataset

csv_path = Path('../continent_variogram_params.csv')
nc_path = Path('../continental_variogram.nc')

if (os.path.exists(csv_path)==True) & (os.access(csv_path, os.W_OK)==False):
    raise PermissionError(f'The file {csv_path} is open, please close it')
else:
    print('can write')

if (os.path.exists(nc_path)==True) & (os.access(nc_path, os.W_OK)==False):
    raise PermissionError(f'The file {nc_path} is open, please close it')
else:
    print('can write')
    
try:
    with Dataset(nc_path, 'r+') as nc_file:
        print('nc writable')
except Exception as e:
    print(e)
    print('nc not writable')

from netCDF4 import Dataset
import os

filename = 'continental_variogram.nc'

try:
    # Attempt to open in 'r+' (read and write) mode, which often requires exclusive access
    with Dataset(filename, 'r+') as nc_file:
        print(f"NetCDF file '{filename}' is accessible and not exclusively locked by another process.")
        # Perform operations with nc_file
except OSError as e:
    if "Permission denied" in str(e) or "Access denied" in str(e):
        print(f"NetCDF file '{filename}' is likely open and locked by another process.")
    else:
        print(f"An unexpected OSError occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")