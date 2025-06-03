import xarray as xr 
import numpy as np
from dask.diagnostics import ProgressBar
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple routines to extract monthly data and remove the annual cycle.


def get_monthly_z250_dlesym(forecast_file: str, output_file: str) -> None:
    """
    Extracts monthly z250 data from a dlesym forecast.
    Args:
        forecast_file (str): Path to the nc dataset containing z250 data.
        output_file (str): Path to save the monthly z250 data.
    Returns:
        None
    """
    
    # open_dataset, select z250 variable
    ds = xr.open_dataset(forecast_file, engine='netcdf4', chunks={'face':1}).z250

    # clean up time representation
    ds = ds.squeeze().reset_coords('time', drop=True).rename({'step': 'time'})
    # monthly mean
    ds_monthly = ds.resample(time='1M').mean()
    # Remove the monthly climatology
    climatology = ds_monthly.groupby('time.month').mean('time')
    ds_monthly_anom = ds_monthly - climatology.sel(month=ds_monthly['time.month'])
    # write in chunks, use progress bar
    with ProgressBar():
        ds_monthly_anom.to_netcdf(output_file, mode='w', compute=True, engine='netcdf4')
    logger.info(f"Monthly z250 data saved to {output_file}")
    logger.info(ds_monthly_anom)

    return

def get_monthly_z250_obs(obs_file:str, fcst_file:str, output_file:str) -> None:
    """
    Extracts monthly z250 data from an observational dataset.
    Args:
        obs_file (str): Path to the nc dataset containing z250 data.
        fcst_file (str): Path to the nc dataset containing forecast data for time reference.
        output_file (str): Path to save the monthly z250 data.
    Returns:
        None
    """
    
    # open_dataset, select z250 variable
    ds = xr.open_dataset(obs_file, engine='netcdf4', chunks={'face':1}).z250

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
    logger.info(f"Monthly z250 data saved to {output_file}")
    logger.info(ds_monthly_anom)

    return

if __name__ == "__main__":

    get_monthly_z250_dlesym('/home/disk/rhodium/nacc/forecasts/testing_dlesym/forced_atmos_dlesym_1983-2017.nc',
                            'data/monthly_z250_dlesym.nc')
    get_monthly_z250_obs('/home/disk/rhodium/dlwp/data/HPX64/era5_0.25deg_3h_HPX64_1979-2021_z250.nc',
                         '/home/disk/rhodium/nacc/forecasts/testing_dlesym/forced_atmos_dlesym_1983-2017.nc',
                         'data/monthly_z250_obs.nc')
