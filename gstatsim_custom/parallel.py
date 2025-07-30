import numpy as np
from copy import deepcopy
import numbers
from tqdm import tqdm
import multiprocessing as mp
from numba import njit, prange

from ._krige import *
from .utilities import *
from .neighbors import *
from .interpolate import _sanity_checks, _preprocess


def parallel_sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, seed=None, chunk_size=20e3, n_workers=8):

    # check arguments
    _sanity_checks(xx, yy, grid, variogram, radius, num_points, ktype)

    # preprocess some grids and variogram parameters
    out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)
    
    # make random number generator if not provided
    rng = get_random_generator(seed)

    # shuffle indices
    rng.shuffle(inds)

    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')

    print('precomputing kriging weights')
    # precompute kriging weights
    kweights_grid, nearest_i_grid, nearest_j_grid, var_grid = get_weights(xx, yy, ii, jj, out_grid, cond_msk, vario, inds, radius, 
                                                                          num_points, ktype, stencil, global_mean, quiet, chunk_size, n_workers)

    print('serial simulation')
    # do serial calculation
    for k in tqdm(range(inds.shape[0]), disable=quiet):

        i, j = inds[k]

        # check if grid cell needs to be simulated
        if cond_msk[i, j] == False:

            # extract indices of nearest neighbors
            nearest_i = nearest_i_grid[i,j,:]
            nearest_i = nearest_i[~np.isnan(nearest_i)]
            nearest_j = nearest_j_grid[i,j,:]
            nearest_j = nearest_j[~np.isnan(nearest_j)]

            # extract nearest neighbor values
            nearest_v = out_grid[nearest_i.astype(int), nearest_j.astype(int)]
            local_mean = np.mean(nearest_v)
            n = nearest_v.size
    
            k_weights = kweights_grid[i,j,:]
            k_weights = k_weights[~np.isnan(k_weights)]

            # compute kriging estimate
            if ktype=='ok':
                est = local_mean + np.sum(k_weights[0:n]*(nearest_v - local_mean))
            elif ktype=='sk':
                est = global_mean + (np.sum(k_weights*(nearest_v - global_mean))) 
    
            var = np.abs(var_grid[i,j])
    
            # put value in grid
            out_grid[i,j] = rng.normal(est, np.sqrt(var), 1)

    sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)

    return sim_trans

#############################################################

# Parallel kriging weights with worker taking many grid cells

#############################################################

def get_weights(xx, yy, ii, jj, out_grid, cond_msk, vario, inds, radius, num_points, ktype, stencil, global_mean, quiet, chunk_size, n_workers):

    # calculate weights in parallel
    kweights_grid = np.full((*out_grid.shape, num_points), np.nan)
    nearest_i_grid = np.full((*out_grid.shape, num_points), np.nan)
    nearest_j_grid = np.full((*out_grid.shape, num_points), np.nan)
    var_grid = np.full(out_grid.shape, np.nan)

    # progress bar
    pbar = tqdm(total=inds.shape[0], disable=quiet)

    for chunk_start in range(0, inds.shape[0], int(chunk_size)):
        if chunk_start + chunk_size > inds.shape[0]:
            chunk_end = int(inds.shape[0])
        else:
            chunk_end = int(chunk_start + chunk_size)

        params = []
        inds_i = inds[chunk_start:chunk_end,:]
        params.append([inds_i, inds, xx, yy, ii, jj, out_grid, cond_msk, vario, radius, num_points, ktype, stencil, global_mean, quiet])
    
        with mp.Pool(n_workers) as pool:
            result = pool.starmap(get_weights_worker, params)

        total = 0
        for chunk in result:
            iis = chunk[0]
            jjs = chunk[1]
            kws = chunk[2]
            nears = chunk[3]
            varss = chunk[4]
            
            for it in range(len(iis)):
                i = iis[it]
                j = jjs[it]
                kw = kws[it]
                nearest = nears[it]
                var = varss[it]
                # i, j, kw, nearest, var = r
                
                kweights_grid[i,j,:kw.size] = kw
                nearest_i_grid[i,j,:nearest.shape[0]] = nearest[:,3]
                nearest_j_grid[i,j,:nearest.shape[0]] = nearest[:,4]
                var_grid[i,j] = var
            total += len(chunk[0])

        pbar.update(total)
            
    return kweights_grid, nearest_i_grid, nearest_j_grid, var_grid

def get_weights_worker(inds, full_inds, xx, yy, ii, jj, out_grid, cond_msk, vario, radius, num_points, ktype, stencil, global_mean, quiet):

    iis = []
    jjs = []
    kws = []
    nears = []
    varss = []
    
    for k in range(inds.shape[0]):
        
        i, j = inds[k]

        if cond_msk[i,j]==False:
            # put all previous inds into tmp cond_msk
            cond_msk_tmp = cond_msk.copy()
            cond_msk_tmp[full_inds[:k,0], full_inds[:k,1]] = True
        
            nearest = np.array([])
            rad = radius
            stenc = stencil
        
            # make local variogram
            local_vario = {}
            for key in vario.keys():
                if key=='vtype':
                    local_vario[key] = vario[key]
                else:
                    local_vario[key] = vario[key][i,j]
        
            # find nearest neighbors, increasing search distance if none are found
            while nearest.shape[0] == 0:
                nearest = neighbors(i, j, ii, jj, xx, yy, out_grid, cond_msk_tmp, rad, num_points, stencil=stenc)
                if nearest.shape[0] > 0:
                    break
                else:
                    rad += 100e3
                    stenc, _, _ = make_circle_stencil(xx[0,:], rad)
        
            # solve kriging equations
            if ktype=='ok':
                kw, var = ok_solve((xx[i,j], yy[i,j]), nearest, local_vario, precompute=True)
            elif ktype=='sk':
                kw, var = sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean, precompute=True)

            iis.append(i)
            jjs.append(j)
            kws.append(kw)
            nears.append(nearest)
            varss.append(var)

    return iis, jjs, kws, nears, varss

###########################################################

# Parallel kriging weights with worker as single grid cell

###########################################################

# def get_weights(xx, yy, ii, jj, out_grid, cond_msk, vario, inds, radius, num_points, ktype, stencil, global_mean, quiet, chunk_size, n_workers):

#     # calculate weights in parallel
#     kweights_grid = np.full((*out_grid.shape, num_points), np.nan)
#     nearest_i_grid = np.full((*out_grid.shape, num_points), np.nan)
#     nearest_j_grid = np.full((*out_grid.shape, num_points), np.nan)
#     var_grid = np.full(out_grid.shape, np.nan)

#     # progress bar
#     pbar = tqdm(total=inds.shape[0], disable=quiet)

#     for chunk_start in range(0, inds.shape[0], int(chunk_size)):
#         if chunk_start + chunk_size > inds.shape[0]:
#             chunk_end = int(inds.shape[0])
#         else:
#             chunk_end = int(chunk_start + chunk_size)

#         params = []
#         for k in range(chunk_start, chunk_end):
#             i, j = inds[k]
#             if cond_msk[i,j] == False:
#                 params.append([k, xx, yy, ii, jj, out_grid, cond_msk, vario, inds, radius, num_points, ktype, stencil, global_mean, quiet])
    
#         with mp.Pool(n_workers) as pool:
#             result = pool.starmap(get_weights_worker, params)
    
#         for it, r in enumerate(result):
            
#             i, j, kw, nearest, var = r
            
#             kweights_grid[i,j,:kw.size] = kw
#             nearest_i_grid[i,j,:nearest.shape[0]] = nearest[:,3]
#             nearest_j_grid[i,j,:nearest.shape[0]] = nearest[:,4]
#             var_grid[i,j] = var

#         pbar.update(len(result))
            
#     return kweights_grid, nearest_i_grid, nearest_j_grid, var_grid

# def get_weights_worker(k, xx, yy, ii, jj, out_grid, cond_msk, vario, inds, radius, num_points, ktype, stencil, global_mean, quiet):
    
#     # put all previous inds into tmp cond_msk
#     cond_msk_tmp = cond_msk.copy()
#     cond_msk_tmp[inds[:k,0], inds[:k,1]] = True

#     i, j = inds[k]

#     nearest = np.array([])
#     rad = radius
#     stenc = stencil

#     # make local variogram
#     local_vario = {}
#     for key in vario.keys():
#         if key=='vtype':
#             local_vario[key] = vario[key]
#         else:
#             local_vario[key] = vario[key][i,j]

#     # find nearest neighbors, increasing search distance if none are found
#     while nearest.shape[0] == 0:
#         nearest = neighbors(i, j, ii, jj, xx, yy, out_grid, cond_msk_tmp, rad, num_points, stencil=stenc)
#         if nearest.shape[0] > 0:
#             break
#         else:
#             rad += 100e3
#             stenc, _, _ = make_circle_stencil(xx[0,:], rad)

#     # solve kriging equations
#     if ktype=='ok':
#         kw, var = ok_solve((xx[i,j], yy[i,j]), nearest, local_vario, precompute=True)
#     elif ktype=='sk':
#         kw, var = sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean, precompute=True)

#     return i, j, kw, nearest, var

###########################################################################

# Parallelize whole simulation algorithm. Violates sequential assumptions.

###########################################################################

# def sgs_parallel(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, seed=None, chunk_size=20e3, n_workers=8):
#     # check arguments
#     _sanity_checks(xx, yy, grid, variogram, radius, num_points, ktype)

#     # preprocess some grids and variogram parameters
#     out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)

#     # make random number generator if not provided
#     rng = get_random_generator(seed)

#     # shuffle indices
#     rng.shuffle(inds)

#     n_chunks = math.ceil(inds.shape[0] / chunk_size)
#     if n_workers > n_chunks:
#         n_workers = n_chunks

#     # progress bar
#     pbar = tqdm(total=inds.shape[0])

#     # split simulation path into chunks
#     for k in range(n_chunks):
#         if k == n_chunks-1:
#             inds_k = inds[int(k*chunk_size):,:]
#         else:
#             inds_k = inds[int(k*chunk_size):int((k+1)*chunk_size),:]
            
#         worker_chunk_size = math.ceil(inds_k.shape[0] / n_workers)

#         # split chunk into smaller chunks to do in parallel
#         total = 0
#         params = []
        
#         for w in range(n_workers):
#             if w == n_workers-1:
#                 inds_w = inds_k[int(w*worker_chunk_size):,:]
#             else:
#                 inds_w = inds_k[int(w*worker_chunk_size):int((w+1)*worker_chunk_size),:]
    
#             params.append([inds_w, xx, yy, out_grid, vario, radius, num_points, ktype, sim_mask, True, stencil, rng])

#             total += inds_w.shape[0]
    
#         with mp.Pool(n_workers) as p:
#             result = p.starmap(sgs_worker, params)
    
#         for r in result:
#             out_grid[r[0],r[1]] = r[2]

#         pbar.update(total)

#     sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)
    
#     return sim_trans
    

# def sgs_worker(inds, xx, yy, grid, vario, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=True, stencil=None, seed=None):

#     rng = seed

#     cond_msk = ~np.isnan(grid)

#     out_arr = np.full(inds.shape[0], np.nan)
#     ii = np.zeros(inds.shape[0]).astype(int)
#     jj = np.zeros(inds.shape[0]).astype(int)
    
#     # mean of conditioning data for simple kriging
#     global_mean = np.mean(grid[cond_msk])
    
#     # iterate over indicies
#     for k in tqdm(range(inds.shape[0]), disable=quiet):
        
#         i, j = inds[k]

#         nearest = np.array([])
#         rad = radius
#         stenc = stencil

#         # check if grid cell needs to be simulated
#         if cond_msk[i, j] == False:
#             # make local variogram
#             local_vario = {}
#             for key in vario.keys():
#                 if key=='vtype':
#                     local_vario[key] = vario[key]
#                 else:
#                     local_vario[key] = vario[key][i,j]

#             # find nearest neighbors, increasing search distance if none are found
#             while nearest.shape[0] == 0:
#                 nearest = neighbors(i, j, xx, yy, grid, cond_msk, rad, num_points, stencil=stenc)
#                 if nearest.shape[0] > 0:
#                     break
#                 else:
#                     rad += 100e3
#                     stenc, _, _ = make_circle_stencil(xx[0,:], rad)

#             # solve kriging equations
#             if ktype=='ok':
#                 est, var = ok_solve((xx[i,j], yy[i,j]), nearest, local_vario)
#             elif ktype=='sk':
#                 est, var = sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean)

#             var = np.abs(var)

#             # put value in grid
#             out_arr[k] = rng.normal(est, np.sqrt(var), 1)
#         else:
#             out_arr[k] = grid[i,j]
            
#         ii[k] = i
#         jj[k] = j
            
#     return ii, jj, out_arr
