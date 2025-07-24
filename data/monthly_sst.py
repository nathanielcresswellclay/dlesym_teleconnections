import xarray as xr 
import numpy as np
from dask.diagnostics import ProgressBar
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple routines to extract monthly data and remove the annual cycle.
def get_monthly_sst_obs(obs_file:str, fcst_file:str, output_file:str) -> None:
    """
    Extracts monthly sst data from an observational dataset.
    Args:
        obs_file (str): Path to the nc dataset containing SST data.
        fcst_file (str): Path to the nc dataset containing forecast data for time reference.
        output_file (str): Path to save the monthly SST data.
    Returns:
        None
    """
    
    # open_dataset, select SST variable
    ds = xr.open_dataset(obs_file, engine='netcdf4', chunks={'face':1}).sst

    # select the exact same time used to calculate monthly DLESyM data. 
    # probably not necessary for results but good for consistency
    forecasted_time = xr.open_dataset(fcst_file, engine='netcdf4').step.values
    ds = ds.rename({'sample': 'time'})
    ds = ds.sel(time=forecasted_time)
    # let's rechunk here for good measure
    ds = ds.chunk({'face': 1})
    # monthly mean
    ds_monthly = ds.resample(time='1M').mean()

    # Remove the monthly climatology
    climatology = ds_monthly.groupby('time.month').mean('time')
    ds_monthly_anom = ds_monthly - climatology.sel(month=ds_monthly['time.month'])
    # in chunks, use progress bar
    with ProgressBar():
        ds_monthly_anom.to_netcdf(output_file, mode='w', compute=True, engine='netcdf4')
    logger.info(f"Monthly SST data saved to {output_file}")
    logger.info(ds_monthly_anom)

    return

if __name__ == "__main__":


    get_monthly_sst_obs('/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_sst.nc',
                        '/home/disk/rhodium/nacc/forecasts/testing_dlesym/forced_atmos_dlesym_1983-2017.nc',
                        'data/monthly_sst_obs.nc')