import numpy as np
from scipy.special import kv, gamma

def exponential_cov_norm(norm_range, sill, nugget, **kwargs):
    c = (sill - nugget)*np.exp(-3*norm_range)
    return c

def gaussian_cov_norm(norm_range, sill, nugget, **kwargs):
    c = (sill - nugget)*np.exp(-3 * np.square(norm_range))
    return c

def spherical_cov_norm(norm_range, sill, nugget, **kwargs):
    c = sill - nugget - 1.5 * norm_range + 0.5 * np.power(norm_range, 3)
    c[norm_range > 1] = sill - 1
    return c

def matern_cov_norm(norm_range, sill, nugget, s, **kwargs):
    scale = 0.45246434*np.exp(-0.70449189*s)+1.7863836
    norm_range[norm_range==0.0] = 1e-8
    c = (sill-nugget)*2/gamma(s)*np.power(scale*norm_range*np.sqrt(s), s)*kv(s, 2*scale*norm_range*np.sqrt(s))
    c[np.isnan(c)] = sill-nugget
    return c

covmodels = {
    'matern' : matern_cov_norm,
    'exponential' : exponential_cov_norm,
    'gaussian' : gaussian_cov_norm,
    'spherical' : spherical_cov_norm
}