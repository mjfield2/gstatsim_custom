import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from .covariance import *

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
    local_mean = np.mean(nearest[:,2])
    n = nearest.shape[0]
    
    # covariance between data
    Sigma = np.zeros((n+1, n+1))
    Sigma[0:n,0:n] = make_sigma(xy_val, rotation_matrix, vario)
    Sigma[n,0:n] = 1
    Sigma[0:n,n] = 1

    # Set up Right Hand Side (covariance between data and unknown)
    rho = np.zeros((n+1))
    rho[0:n] = make_rho(xy_val, sim_xy, rotation_matrix, vario)
    rho[n] = 1

    # solve for kriging weights
    k_weights, res, rank, s = np.linalg.lstsq(Sigma, rho, rcond=rcond) 
    var = vario['sill'] - np.sum(k_weights[0:n]*rho[0:n])

    if precompute == True:
        return k_weights, var
    else:
        est = local_mean + np.sum(k_weights[0:n]*(nearest[:,2] - local_mean))
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
    local_mean = np.mean(nearest[:,2])
    n = nearest.shape[0]
    
    # covariance between data
    Sigma = make_sigma(xy_val, rotation_matrix, vario)

    # covariance between data and unknown
    rho = make_rho(xy_val, sim_xy, rotation_matrix, vario)

    # solve for kriging weights
    k_weights, res, rank, s = np.linalg.lstsq(Sigma, rho, rcond=rcond) 
    var = vario['sill'] - np.sum(k_weights*rho)

    if precompute == True:
        return k_weights, var
    else:
        est = global_mean + (np.sum(k_weights*(nearest[:,2] - global_mean))) 
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
    
    theta = (azimuth / 180.0) * np.pi 
    
    rotation_matrix = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / major_range, 0], [0, 1 / minor_range]]))
    
    return rotation_matrix

def make_sigma(coord, rotation_matrix, vario):
    """
    Make covariance matrix showing covariances between each pair of input coordinates
    
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
    norm_range = np.sqrt(np.square(mat1 - mat2).sum(axis=1))
    rho = covmodels[vario['vtype'].lower()](norm_range, **vario)

    return rho