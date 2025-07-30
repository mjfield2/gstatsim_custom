# Variation of GStatSim

This is designed for geostatistical interpolation

### Key features

* Faster nearest neighbor search
* Nearest neighbor uses "cookie cutter" mask for fast distance filtering
* interpolation algorithms operate on grid, no extracting conditioning data and prediction locations
* Variogram is passed as a dictionary and there is not ambiguity about position in a list
* Variogram parameters can be nonstationary, represented by 2D arrays
* Use a `sim_msk` to limit the extend of the interpolation
* Covariance functions are independent functions
* Kriging system code is reaused in kriging and SGS functions
* One function for kriging, and one for SGS, with argument for Ordinary or Simple
