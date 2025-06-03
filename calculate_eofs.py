import sys
import os 
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _eof_svd(data: xr.DataArray, n_modes, weight_latitude=True):
    """
    Compute EOFs using SVD.

    Parameters:
        data (xr.DataArray): Anomaly field with dims (time, lat, lon)
        n_modes (int): Number of EOF modes to return
        weight_latitude (bool): If True, apply sqrt(cos(lat)) weighting

    Returns:
        EOFs (xr.DataArray): EOF patterns with dims (mode, lat, lon)
        PCs (xr.DataArray): Principal component time series (time, mode)
        variance_explained (xr.DataArray): Variance fraction per mode (mode,)
    """
    # Check dimensions
    assert all(dim in data.dims for dim in ['time', 'lat', 'lon']), "Expected dimensions: time, lat, lon"

    # Latitude weights following Ding et al. 2012
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])))
    data = data * weights

    # Reshape data: (time, lat*lon)
    nt, nlat, nlon = data.shape
    data_2d = data.values.reshape(nt, nlat * nlon)
    # Remove spatial mean if desired
    data_2d = data_2d - np.nanmean(data_2d, axis=0, keepdims=True)

    logger.info(f'Calculating EOFs and saving {n_modes} modes...')
    # SVD
    U, s, VT = np.linalg.svd(data_2d, full_matrices=False)

    # Truncate to n_modes
    PCs = U[:, :n_modes] * s[:n_modes]
    EOFs = VT[:n_modes, :].reshape(n_modes, nlat, nlon)
    variance_explained = (s[:n_modes] ** 2) / np.sum(s ** 2)

    # Wrap in DataArrays
    EOFs_da = xr.DataArray(EOFs, dims=('mode', 'lat', 'lon'),
                           coords={'mode': np.arange(n_modes), 'lat': data['lat'], 'lon': data['lon']})
    PCs_da = xr.DataArray(PCs, dims=('time', 'mode'),
                          coords={'time': data['time'], 'mode': np.arange(n_modes)})
    var_exp_da = xr.DataArray(variance_explained, dims='mode', coords={'mode': np.arange(n_modes)})

    # we're done!
    return EOFs_da, PCs_da, var_exp_da

def eofs_for_sam(dataset:str, modes=5, ll_cache:str=None, eof_cache:str=None, overwrite=False):
    """
    Calculate EOFs for the Southern Annular Mode (SAM) from z250 anomalies.
    Remaps DLESyM HPX mesh to lat-lon, selects relevant monthly/seasonal data,
    and computes EOFs via `_eof_svd`.

    Parameters:
        dataset (str): Path to input z250 anomaly dataset
        modes (int, optional): Number of EOF modes to compute (default: 5)
        ll_cache (str, optional): Path to cache lat-lon remap (default: None)
        eof_cache (str, optional): Path to cache EOFs (default: None)
        overwrite (bool, optional): Overwrite existing cache files (default: False)

    Returns:
        None
    """
    # mount module DLESyM for remapping utility 
    try:
        from data_processing.remap import healpix as hpx
    except ImportError as e:
        logger.info(f"Error importing DLESyM: {e}")
        return
    logger.info(f"Successfully import DLESyM module. Make sure PYTHONPATH is set to include DLESyM.")

    # We use this to get from HPX mesh to Lat-Lon. 1 degree resolution
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )

    # open dataset, select z250 
    ds = xr.open_dataset(dataset).z250

    # initilaize lat-lon dataarray, remap data
    logger.info("Remapping data to Lat-Lon grid...")
    # check for cacheing 
    if ll_cache is not None and os.path.exists(ll_cache) and not overwrite:
        logger.info(f"Loading cached Lat-Lon data from {ll_cache}")
        ds_ll = xr.open_dataarray(ll_cache)
    else:
        ds_ll = xr.DataArray(
                np.array([mapper.hpx2ll(ds.sel(time=t).values) for t in tqdm(ds.time.values)]),
                dims=['time', 'lat', 'lon'],
                coords={'time': ds.time.values, 'lat': np.arange(90, -90.1, -1),'lon': np.arange(0, 360, 1)}
            )
        if ll_cache is not None:
            logger.info(f"Caching Lat-Lon data to {ll_cache}")
            ds_ll.to_netcdf(ll_cache, mode='w', engine='netcdf4')

    # again, check for cahed values 
    if eof_cache is not None and os.path.exists(eof_cache) and not overwrite:
        logger.info(f"Loading cached EOFs from {eof_cache}")
        ds_eofs = xr.open_dataset(eof_cache)
    else:
        # calculate EOFs for southern annular mode (SAM) by selecting data south of 20 degrees
        ds_ll = ds_ll.sel(lat=slice(-20, -90))
        eofs, pcs, var_exp = _eof_svd(ds_ll, n_modes=modes, weight_latitude=True)
        # combine into a dataset
        ds_eofs = xr.Dataset({
            'eofs': eofs,
            'pcs': pcs,
            'variance_explained': var_exp
        })
        if eof_cache is not None:
            logger.info(f"Caching EOFs to {eof_cache}")
            ds_eofs.to_netcdf(eof_cache, mode='w', engine='netcdf4')
    
    logger.info("EOFs calculated successfully.")
    return



if __name__ == "__main__":
    
    # EOFs for DLESyM monthly z250 data
    eofs_for_sam(dataset='data/monthly_z250_dlesym.nc', 
                ll_cache='data/monthly_z250_dlesym_ll.nc', 
                eof_cache='data/monthly_z250_dlesym_eofs.nc')
    # EOFs for observational monthly z250 data
    eofs_for_sam(dataset='data/monthly_z250_obs.nc',
                ll_cache='data/monthly_z250_obs_ll.nc', 
                eof_cache='data/monthly_z250_obs_eofs.nc')





