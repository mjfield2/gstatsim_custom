import numpy as np
from sklearn.preprocessing import QuantileTransformer
from copy import deepcopy
import skgstat as skg
from tqdm.auto import tqdm

def gaussian_transformation(grid, cond_msk, n_quantiles=500):
    """
    Gaussian quantile transformation.

    Args:
        grid (numpy.ndarray): Gridded data, NaN where there is not conditioning data.
        cond_msk (numpy.ndarray): Mask where conditioning data is.
        n_quantiles (int): Number of quantiles. Default 500.
        
    Returns:
        (numpy.ndarray) Gaussian transformed grid, fitted transformer
    """
    
    data_cond = grid[cond_msk].reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal").fit(data_cond)
    norm_data = nst_trans.transform(data_cond).squeeze()
    grid_norm = np.full(grid.shape, np.nan)
    np.place(grid_norm, cond_msk, norm_data)

    return grid_norm, nst_trans

def dists_to_cond(xx, yy, grid):
    """
    Find minimum distance to conditioning data.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): Gridded data, NaN where there is not conditioning data.
        
    Returns:
        (numpy.ndarray) 2D array of minimum distances to conditioning data.
    """
    
    cond_msk = ~np.isnan(grid)
    min_dists = np.zeros(grid.shape)
    for i in tqdm(range(grid.shape[0])):
        for j in range(grid.shape[1]):
            dist = np.sqrt((xx[i,j]-xx[cond_msk])**2 + (yy[i,j]-yy[cond_msk])**2)
            min_dists[i,j] = np.min(dist)

    return min_dists

def get_random_generator(seed):
    """
    Conveniance function to get numpy random number generator for SGS. If seed is None, a random
    seed is used. If seed is an integer, that integer is used to seed the RNG. If seed is
    already an instance of a numpy RNG that is returned.

    Args:
        seed (int, None, or numpy.random.Generator): how to initialize generator

    Returns:
        numpy.random.Generator
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed=seed)
    elif isinstance(seed, np.random._generator.Generator):
        rng = seed
    else:
        raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
    return rng

def variograms(xx, yy, grid, bin_func='even', maxlag=100e3, n_lags=70, covmodels=['gaussian', 'spherical', 'exponential', 'matern'], downsample=None):
    """
    Make experimental variogram and fit covariance models.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): Gridded data, NaN where there is not conditioning data.
        bin_func (str, or sequence): binning function or array of bin edges.
        maxlag (int, float): Maximum lag for experimental variogram.
        n_lags (int): Number of lag bins for variogram.
        covmodels (list): Covariance models to fit to variogram.
        
    Returns:
        Dictionary of variograms, experimental variogram values, and bins.
    """
    cond_msk = ~np.isnan(grid)
    grid_norm, nst_trans = gaussian_transformation(grid, cond_msk)
    x_cond = xx[cond_msk]
    y_cond = yy[cond_msk]
    data_norm = grid_norm[cond_msk]
    coords_cond = np.array([x_cond, y_cond]).T

    if isinstance(downsample, int):
        data_norm = data_norm[::downsample]
        coords_cond = coords_cond[::downsample]

    vgrams = {}

    # compute experimental (isotropic) variogram
    V = skg.Variogram(coords_cond, data_norm, bin_func=bin_func, n_lags=n_lags, 
                   maxlag=maxlag, normalize=False)
    
    V.model = covmodels[0]
    vgrams[covmodels[0]] = V.parameters

    if len(covmodels) > 1:
        for i, cov in enumerate(covmodels[1:]):
            V_i = deepcopy(V)
            V_i.model = cov
            vgrams[cov] = V_i.parameters

    return vgrams, V.experimental, V.bins