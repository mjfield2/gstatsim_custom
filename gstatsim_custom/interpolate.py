import numpy as np
from copy import deepcopy
import numbers
from tqdm import tqdm
import multiprocessing as mp

from ._krige import *
from .utilities import *
from .neighbors import *

def krige(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None):
    """
    Ordinary or simple kriging interpolation using nearest neighbors found in an octant search.
    Note: Unsure why the output differs so much from GStatSim.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        num_points (int): Number of nearest neighbors to find. Default is 20.
        ktype (string): 'ok' for ordinary kriging or 'sk' for simple kriging. Default is 'ok'.
        sim_mask (numpy.ndarray or None): Mask True where to do kriging. Default None will do whole grid.
        quiet (book): Turn off progress bar when True. Default False.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.

    Returns:
        (numpy.ndarray, numpy.ndarray): 2D arrays of kriging mean and standard deviation
    """
    
    # check arguments
    _sanity_checks(xx, yy, grid, variogram, radius, num_points, ktype)

    # preprocess some grids and variogram parameters
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)
    cond_grid = deepcopy(out_grid)
    var_grid = np.zeros(grid.shape)

    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')

    # iterate over indicies
    for k in tqdm(range(inds.shape[0]), disable=quiet):
        
        i, j = inds[k]

        nearest = np.array([])
        rad = radius
        stenc = stencil

        # check if grid cell needs to be simulated
        if cond_msk[i, j] == False:
            # make local variogram
            local_vario = {}
            for key in vario.keys():
                if key=='vtype':
                    local_vario[key] = vario[key]
                else:
                    local_vario[key] = vario[key][i,j]

            # find nearest neighbors, increasing search distance if none are found
            while nearest.shape[0] == 0:
                nearest = neighbors(i, j, ii, jj, xx, yy, out_grid, cond_msk, rad, num_points, stencil=stenc)
                if nearest.shape[0] > 0:
                    break
                else:
                    rad += 100e3
                    stenc, _, _ = make_circle_stencil(xx[0,:], rad)

            # solve kriging equations
            if ktype=='ok':
                est, var = ok_solve((xx[i,j], yy[i,j]), nearest, local_vario)
            elif ktype=='sk':
                est, var = sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean)

            # put value in grid
            out_grid[i,j] = est
            var_grid[i,j] = var

    var_grid = np.where(var_grid < 0, 0, var_grid)

    std_grid = np.sqrt(var_grid)
    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)
    std_trans = nst_trans.inverse_transform(std_grid.reshape(-1,1)).squeeze().reshape(xx.shape)
    
    return sim_trans, std_trans


def sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, seed=None):
    """
    Sequential Gaussian Simulation with ordinary or simple kriging using nearest neighbors found in an octant search.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        num_points (int): Number of nearest neighbors to find. Default is 20.
        ktype (string): 'ok' for ordinary kriging or 'sk' for simple kriging. Default is 'ok'.
        sim_mask (numpy.ndarray or None): Mask True where to do simulation. Default None will do whole grid.
        quiet (book): Turn off progress bar when True. Default False.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.
        seed (int, None, or numpy.random.Generator): If None, a fresh random number generator (RNG)
            will be created. If int, a RNG will be instantiated with that seed. If an instance of
            RNG, that will be used.

    Returns:
        (numpy.ndarray): 2D simulation
    """
    
    # check arguments
    _sanity_checks(xx, yy, grid, variogram, radius, num_points, ktype)

    # preprocess some grids and variogram parameters
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)

    # make random number generator if not provided
    rng = get_random_generator(seed)

    # shuffle indices
    rng.shuffle(inds)

    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')

    # iterate over indicies
    for k in tqdm(range(inds.shape[0]), disable=quiet):
        
        i, j = inds[k]

        nearest = np.array([])
        rad = radius
        stenc = stencil

        # check if grid cell needs to be simulated
        if cond_msk[i, j] == False:
            # make local variogram
            local_vario = {}
            for key in vario.keys():
                if key=='vtype':
                    local_vario[key] = vario[key]
                else:
                    local_vario[key] = vario[key][i,j]

            # find nearest neighbors, increasing search distance if none are found
            while nearest.shape[0] == 0:
                nearest = neighbors(i, j, ii, jj, xx, yy, out_grid, cond_msk, rad, num_points, stencil=stenc)
                if nearest.shape[0] > 0:
                    break
                else:
                    rad += 100e3
                    stenc, _, _ = make_circle_stencil(xx[0,:], rad)

            # solve kriging equations
            if ktype=='ok':
                est, var = ok_solve((xx[i,j], yy[i,j]), nearest, local_vario)
            elif ktype=='sk':
                est, var = sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean)

            var = np.abs(var)

            # put value in grid
            out_grid[i,j] = rng.normal(est, np.sqrt(var), 1)
            cond_msk[i,j] = True

    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)

    return sim_trans

def _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil):
    """
    Sequential Gaussian Simulation with ordinary or simple kriging using nearest neighbors found in an octant search.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        sim_mask (numpy.ndarray or None): Mask True where to do simulation. Default None will do whole grid.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.

    Returns:
        (out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil)
    """
    
    # get masks and gaussian transform data
    cond_msk = ~np.isnan(grid)
    out_grid, nst_trans = gaussian_transformation(grid, cond_msk)

    if sim_mask is None:
        sim_mask = np.full(xx.shape, True)

    # get index coordinates and filter with sim_mask
    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')
    inds = np.array([ii[sim_mask].flatten(), jj[sim_mask].flatten()]).T

    vario = deepcopy(variogram)

    # turn scalar variogram parameters into grid
    for key in vario:
        if isinstance(vario[key], numbers.Number):
            vario[key] = np.full(grid.shape, vario[key])

    # mean of conditioning data for simple kriging
    global_mean = np.mean(out_grid[cond_msk])

    # make stencil for faster nearest neighbor search
    if stencil is None:
        stencil, _, _ = make_circle_stencil(xx[0,:], radius)

    return out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil

def _sanity_checks(xx, yy, grid, vario, radius, num_points, ktype):
    """
    Do sanity checks and throw errors.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        num_points (int): Number of nearest neighbors to find. Default is 20.
        ktype (string): 'ok' for ordinary kriging or 'sk' for simple kriging. Default is 'ok'.

    Returns:
        Nothing
    """
    
    if (isinstance(xx, np.ndarray) == False) | (len(xx.shape) != 2):
        raise ValueError('xx must be a 2D NumPy array')
    if (isinstance(yy, np.ndarray) == False) | (len(yy.shape) != 2):
        raise ValueError('yy must be a 2D array')
    if (isinstance(grid, np.ndarray) == False) | (len(grid.shape) != 2):
        raise ValueError('grid must be a 2D array')

    expected_keys = [
        'major_range',
        'minor_range',
        'azimuth',
        'sill',
        'nugget',
        'vtype'
    ]
    missing_vario = []
    for k in expected_keys:
        if k not in vario.keys():
            missing_vario.append(k)

    if len(missing_vario) > 0:
        raise ValueError(f"Variogram missing {', '.join(missing_vario)}")

    if vario['vtype'].lower() not in ['gaussian', 'exponential', 'spherical', 'matern']:
        raise ValueError(f"vtype must be exponential, gaussian, spherical, or matern")

    if vario['vtype'].lower() == 'matern':
        if 's' not in vario.keys():
            raise ValueError(f"Matern covariance requires the s parameter in the variogram")

    if isinstance(radius, numbers.Number) == False:
        raise ValueError('radius must be a number')
    if isinstance(num_points, numbers.Number) == False:
        raise ValueError('num_points must be a number')

    if ktype not in ['ok', 'sk']:
        raise ValueError("ktype must be 'ok' or 'sk'")