import numpy as np
import cupy as cp
from copy import deepcopy
import numbers
from tqdm import tqdm
import multiprocessing as mp
from scipy.stats import truncnorm
# from cupyx.scipy.spatial.distance import cdist, pdist, squareform

# from ._krige import *
from .utilities import *
# from .neighbors import *
from .covariance import *

def sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, rcond=None, bounds=None, seed=None):
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

    xp = cp.get_array_module(grid)  # 'xp' is a standard usage in the community

    print("Using:", xp.__name__)
    
    # check arguments
    _sanity_checks(xx, yy, grid, variogram, radius, num_points, ktype, sim_mask)

    # preprocess some grids and variogram parameters
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil, bounds = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil, bounds)

    xx = cp.asarray(xx)
    yy = cp.asarray(yy)
    out_grid = cp.asarray(out_grid)
    cond_msk = cp.asarray(cond_msk)
    # sim_mask = cp.asarray(sim_mask)

    # make random number generator if not provided
    rng = get_random_generator(seed)

    # shuffle indices
    rng.shuffle(inds)

    ii, jj = cp.meshgrid(cp.arange(xx.shape[0]), cp.arange(xx.shape[1]), indexing='ij')

    # iterate over indicies
    for k in tqdm(range(inds.shape[0]), disable=quiet):
        
        i, j = inds[k]

        nearest = cp.array([])
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
                est, var = ok_solve((xx[i,j], yy[i,j]), nearest, local_vario, rcond)
            elif ktype=='sk':
                est, var = sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean, rcond)

            var = cp.abs(var)

            # put value in grid
            # out_grid[i,j] = rng.normal(est, cp.sqrt(var), 1)
            if bounds is None:
                out_grid[i,j] = rng.normal(est, cp.sqrt(var), 1)
            else:
                scale = cp.sqrt(var)
                a_transformed, b_transformed = (bounds[0][i,j] - est) / scale, (bounds[1][i,j] - est) / scale
                out_grid[i,j] = truncnorm.rvs(a_transformed, b_transformed, loc=est, scale=scale, size=1, random_state=rng)
            cond_msk[i,j] = True

    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)

    return sim_trans

def ok_solve(sim_xy, nearest, vario, rcond=None, precompute=False):
    """
    Solve ordinary kriging system given neighboring points
    
    Args:
        sim_xy (list): x- and y-coordinate of grid cell being simulated
        nearest (numpy.ndarray): coordinates and values of neighboring points with shape (N,3)
        vario (dictionary): variogram parameters including azimuth, nugget, major_range, 
            minor_range, sill, vtype, and s
    
    Returns:
        numpy.ndarray: 2x2 rotation matrix used to perform coordinate transformations
    """
    
    rotation_matrix = make_rotation_matrix(vario['azimuth'], vario['major_range'], vario['minor_range'])

    xy_val = nearest[:,:2]
    local_mean = cp.mean(nearest[:,2])
    n = nearest.shape[0]
    
    # covariance between data
    Sigma = cp.zeros((n+1, n+1))
    Sigma[0:n,0:n] = make_sigma(xy_val, rotation_matrix, vario)
    Sigma[n,0:n] = 1
    Sigma[0:n,n] = 1

    # Set up Right Hand Side (covariance between data and unknown)
    rho = cp.zeros((n+1))
    rho[0:n] = make_rho(xy_val, sim_xy, rotation_matrix, vario)
    rho[n] = 1

    # solve for kriging weights
    k_weights, res, rank, s = cp.linalg.lstsq(Sigma, rho, rcond=rcond) 
    var = vario['sill'] - cp.sum(k_weights[0:n]*rho[0:n])

    if precompute == True:
        return k_weights, var
    else:
        est = local_mean + cp.sum(k_weights[0:n]*(nearest[:,2] - local_mean))
        return est, var

def sk_solve(sim_xy, nearest, vario, global_mean, rcond=None, precompute=False):
    """
    Solve simple kriging system given neighboring points
    
    Args:
        sim_xy (list): x- and y-coordinate of grid cell being simulated
        nearest (numpy.ndarray): coordinates and values of neighboring points with shape (N,3)
        vario (dictionary): variogram parameters including azimuth, nugget, major_range, 
            minor_range, sill, vtype, and s
        global_mean (float): mean of all conditioning points
    
    Returns:
        numpy.ndarray: 2x2 rotation matrix used to perform coordinate transformations
    """

    rotation_matrix = make_rotation_matrix(vario['azimuth'], vario['major_range'], vario['minor_range'])
    
    xy_val = nearest[:,:2]
    local_mean = cp.mean(nearest[:,2])
    n = nearest.shape[0]
    
    # covariance between data
    Sigma = make_sigma(xy_val, rotation_matrix, vario)

    # covariance between data and unknown
    rho = make_rho(xy_val, sim_xy, rotation_matrix, vario)

    # solve for kriging weights
    k_weights, res, rank, s = cp.linalg.lstsq(Sigma, rho, rcond=rcond) 
    var = vario['sill'] - cp.sum(k_weights*rho)

    if precompute == True:
        return k_weights, var
    else:
        est = global_mean + (cp.sum(k_weights*(nearest[:,2] - global_mean))) 
        return est, var

def make_rotation_matrix(azimuth, major_range, minor_range):
    """
    Make rotation matrix for accommodating anisotropy
    
    Args:
        azimuth (int, float): angle (in degrees from horizontal) of axis of orientation
        major_range (int, float): range parameter of variogram in major direction, or azimuth
        minor_range (int, float): range parameter of variogram in minor direction, or orthogonal to azimuth
    
    Returns:
        numpy.ndarray: 2x2 rotation matrix used to perform coordinate transformations
    """
    
    theta = (azimuth / 180.0) * cp.pi 
    
    rotation_matrix = cp.dot(
        cp.array([[cp.cos(theta), -cp.sin(theta)],
                [cp.sin(theta), cp.cos(theta)],]),
        cp.array([[1 / major_range, 0], [0, 1 / minor_range]]))
    
    return rotation_matrix

def make_sigma(coord, rotation_matrix, vario):
    """
    Make covariance matrix showing covariances between each pair of icput coordinates
    
    Args:
        coord (numpy.ndarray): coordinates of data points
        rotation_matrix (numpy.ndarray): rotation matrix used to perform coordinate transformations
        vario (dictionary): variogram parameters including azimuth, nugget, major_range, 
            minor_range, sill, vtype, and s
    
    Returns:
        numpy.ndarray: nxn matrix of covariance between n points
    """
    
    norm_range = squareform(pdist(coord @ rotation_matrix))
    Sigma = covmodels[vario['vtype'].lower()](norm_range, **vario)

    return Sigma

def make_rho(coord1, coord2, rotation_matrix, vario):
    """
    Make covariance array showing covariances between each data points and grid cell of interest
    
    Args:
        coord1 (numpy.ndarray): coordinates of n data points
        coord2 (numpy.ndarray): coordinate of grid cell being simulated repeated n times
        rotation_matrix (numpy.ndarray): rotation matrix used to perform coordinate transformations
        vario (dictionary): variogram parameters including azimuth, nugget, major_range, 
            minor_range, sill, vtype, and s
    
    Returns:
        numpy.ndarray: nx1 array of covariance between n points and grid cell of interest
    """
    
    mat1 = coord1 @ rotation_matrix
    mat2 = coord2 @ rotation_matrix
    norm_range = cp.sqrt(cp.square(mat1 - mat2).sum(axis=1))
    rho = covmodels[vario['vtype'].lower()](norm_range, **vario)

    return rho

def neighbors(i, j, ii, jj, xx, yy, grid, cond_msk, radius, num_points, stencil=None):
    """
    Find nearest neighbors using octant search

    Args:
        i (int): i index of simulation grid cell
        j (int): j index of simulation grid cell
        ii (numpy.ndarray): 2D array of i indices
        jj (numpy.ndarray): 2D array of j indicies
        xx (numpy.ndarray): 2D array of x-coordinates
        yy (numpy.ndarray): 2D array of y-coordinates
        grid (numpy.ndarray): grid with NaN where there is not conditioning data
        cond_msk (numpy.ndarray): boolean mask True where there is conditioning data
        radius (float): distance to search for neighbors
        num_points: Total number of points to search for
        stencil (numpy.ndarray): Cookie cutter distance mask

    Returns:
        numpy.ndarray: A 2D array nearest neighbor entries in rows. Columns are
        x-coordinates, y-coordinates, values, i-indices, j-indices.
    """

    if stencil is not None:
        ni, nj = grid.shape
        hw = math.floor(stencil.shape[0]//2)
    
        # make sure block extent inside domain
        ilow = max(0, i-hw)
        ihigh = min(ni, i+hw+1)
        jlow = max(0, j-hw)
        jhigh = min(nj, j+hw+1)

        # trim arrays to smaller extent
        grid = grid[ilow:ihigh,jlow:jhigh]
        xx = xx[ilow:ihigh,jlow:jhigh]
        yy = yy[ilow:ihigh,jlow:jhigh]
        cond_msk = cond_msk[ilow:ihigh,jlow:jhigh]
        ii = ii[ilow:ihigh,jlow:jhigh]
        jj = jj[ilow:ihigh,jlow:jhigh]

        # adjust indices for trimmed extent
        i = i-ilow
        j = j-jlow
    
    # calculate distances and angles for filtering
    distances = cp.sqrt((xx[i,j] - xx)**2 + (yy[i,j] - yy)**2)
    angles = cp.arctan2(yy[i,j] - yy, xx[i,j] - xx)

    points = []
    # uses range because cp.arange causes issues with equality
    for b in range(-4, 4, 1):
        msk = (distances < radius) & (angles > b/4*cp.pi) & (angles <= (b+1)/4*cp.pi) & cond_msk
        sort_inds = cp.argsort(distances[msk])
        p = cp.array([xx[msk], yy[msk], grid[msk], ii[msk], jj[msk]]).T
        p = p[sort_inds,:]
        p = p[:num_points//8,:]
        points.append(p)
    points = cp.concatenate(points)
    points = points[~cp.isnan(points[:,2]),:]
    
    return points

def make_circle_stencil(x, rad):
    """
    Creates a circle mask on a grid.

    Args:
        x (numpy.ndarray): x-values of grid
        rad (int, float): Radius of the circle

    Returns:
        numpy.ndarray: A 2D array with 1s inside the circle and 0s elsewhere.
    """
    dx = cp.abs(x[1]-x[0])
    ncells = math.ceil(rad/dx)
    x_stencil = cp.linspace(-rad, rad, 2*ncells+1)
    xx_st, yy_st = cp.meshgrid(x_stencil, x_stencil)
    stencil = cp.sqrt(xx_st**2 + yy_st**2) < rad

    return stencil, xx_st, yy_st

def sample_bounded_value(bounds, loc, scale, rng):
    a_transformed, b_transformed = (bounds[0][i,j] - loc) / scale, (bounds[1][i,j] - loc) / scale
    return truncnorm.rvs(a_transformed, b_transformed, loc=loc, scale=scale, size=1, random_state=rng)

def _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil, bounds):
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

    # put bounds into grid if int
    new_bounds = []
    if bounds is not None:
        try:
            if len(bounds) != 2:
                raise ValueError('bounds be an iterable of length 2 with lower and upper bounds')
            else:
                for i, bound in enumerate(bounds):
                    if isinstance(bound, numbers.Number):
                        trans_bound = nst_trans.transform(np.array([bound]).reshape(-1,1)).squeeze()
                        new_bounds.append(np.full(xx.shape, trans_bound))
                    elif isinstance(bound, np.ndarray):
                        if bound.shape != xx.shape:
                            raise ValueError('bounds must have same shape as grid')
                        else:
                            new_bounds.append(nst_trans.transform(bound.reshape(-1,1)).reshape(xx.shape))
                    else:
                        raise ValueError('bounds must be None or a 2D numpy array')
        except:
            raise ValueError('bounds must be None or a 2D numpy array')
    else:
        new_bounds = None

    return out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil, new_bounds

def _sanity_checks(xx, yy, grid, vario, radius, num_points, ktype, sim_mask):
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

    if isinstance(xx, np.ndarray):
        if (len(xx.shape) != 2):
            raise ValueError('xx must be a 2D NumPy array')
    else:
        raise ValueError('xx must be a 2D NumPy array')
        
    if isinstance(yy, np.ndarray):
        if (len(yy.shape) != 2):
            raise ValueError('yy must be a 2D NumPy array')
    else:
        raise ValueError('yy must be a 2D NumPy array')

    if isinstance(grid, np.ndarray):
        if (len(grid.shape) != 2):
            raise ValueError('grid must be a 2D NumPy array')
    else:
        raise ValueError('grid must be a 2D NumPy array')

    if (xx.shape != yy.shape) | (xx.shape != grid.shape):
        raise ValueError('xx, yy, and grid must have same shape')

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

    if vario['vtype'].lower() not in covmodels.keys():
        raise ValueError(f"vtype must be exponential, gaussian, spherical, or matern")

    if vario['vtype'].lower() == 'matern':
        if 's' not in vario.keys():
            raise ValueError(f"Matern covariance requires the s parameter in the variogram")

    if sim_mask is not None:
        if isinstance(sim_mask, np.ndarray):
            if sim_mask.shape != grid.shape:
                raise ValueError('sim_mask shape must be same as grid if provided')
        else:
            raise ValueError('sim_mask must be None or a 2D array')

    for k in vario.keys():
        if k == 'vtype':
            continue
        else:
            if isinstance(vario[k], numbers.Number):
                if np.isnan(vario[k]) == True:
                    raise ValueError(f'variogram parameter {k} is NaN')
            elif isinstance(vario[k], np.ndarray):
                if sim_mask is None:
                    if np.count_nonzero(np.isnan(vario[k])) > 0:
                        raise ValueError(f'Variogram parameter {k} contains NaN')
                else:
                    if np.count_nonzero((sim_mask==True) & (np.isnan(vario[k]))) > 0:
                        raise ValueError(f'Variogram parameter {k} contains NaN in sim_mask')
                

    if isinstance(radius, numbers.Number) == False:
        raise ValueError('radius must be a number')
    if isinstance(num_points, numbers.Number) == False:
        raise ValueError('num_points must be a number')

    if ktype not in ['ok', 'sk']:
        raise ValueError("ktype must be 'ok' or 'sk'")