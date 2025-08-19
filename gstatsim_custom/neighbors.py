import numpy as np
import math

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
    distances = np.sqrt((xx[i,j] - xx)**2 + (yy[i,j] - yy)**2)
    angles = np.arctan2(yy[i,j] - yy, xx[i,j] - xx)

    points = []
    # uses range because np.arange causes issues with equality
    for b in range(-4, 4, 1):
        msk = (distances < radius) & (angles > b/4*np.pi) & (angles <= (b+1)/4*np.pi) & cond_msk
        sort_inds = np.argsort(distances[msk])
        p = np.array([xx[msk], yy[msk], grid[msk], ii[msk], jj[msk]]).T
        p = p[sort_inds,:]
        p = p[:num_points//8,:]
        points.append(p)
    points = np.concatenate(points)
    points = points[~np.isnan(points[:,2]),:]
    
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
    dx = np.abs(x[1]-x[0])
    ncells = math.ceil(rad/dx)
    x_stencil = np.linspace(-rad, rad, 2*ncells+1)
    xx_st, yy_st = np.meshgrid(x_stencil, x_stencil)
    stencil = np.sqrt(xx_st**2 + yy_st**2) < rad

    return stencil, xx_st, yy_st

def make_ellipse_stencil(x, major_axis, minor_axis, angle_degrees):
    """
    Creates a 2D NumPy array representing an ellipse.

    Args:
        x (numpy.ndarray): x-values of grid
        major_axis (float): The length of the semi-major axis.
        minor_axis (float): The length of the semi-minor axis.
        angle_rad (float): The rotation angle of the ellipse in radians.

    Returns:
        numpy.ndarray: A 2D array with 1s inside the ellipse and 0s elsewhere.
    """
    angle_rad = (180-angle_degrees)*np.pi/180
    dx = np.abs(x[1]-x[0])
    ncells = math.ceil(major_axis/dx)
    x_stencil = np.linspace(-major_axis, major_axis, 2*ncells+1)
    xx, yy = np.meshgrid(x_stencil, x_stencil)
    
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Rotate the translated coordinates
    x_rotated = xx * cos_angle + yy * sin_angle
    y_rotated = -xx * sin_angle + yy * cos_angle
    
    # ellipsoid equation
    ell_eq = (x_rotated / major_axis)**2 + (y_rotated / minor_axis)**2
    
    # Check if the point is inside the ellipse
    ellipse_array = np.where(ell_eq <= 1, 1, 0)

    return ellipse_array, xx, yy