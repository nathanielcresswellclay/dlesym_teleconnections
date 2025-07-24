import os
import logging
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
from scipy.stats import linregress


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seasonal_sst_sam_regression(
    monthly_sam_file: str, 
    monthly_sst_file: str, 
    regression_cache: str = None,
    overwrite: bool = False): 
    """
    Calculate seasonal SST regressions against SAM indices.
    Parameters:
        monthly_sam_file (str): Path to the monthly SAM indices file.
        monthly_sst_file (str): Path to the monthly SST data file.
        output_prefix (str): Prefix for the output files.
        regression_cache (str): Path to cache regression results. If None, no caching is done.
        overwrite (bool): If True, overwrite existing cached results.
    Returns:
        None
    """


    def _extract_seasonal_means(data):

        # Determine season and adjusted year
        seasons = data['time'].dt.season
        year = data['time'].dt.year
        month = data['time'].dt.month
        season_year = year.where(month != 12, year + 1)
        # Add coordinates
        data = data.assign_coords(season=seasons, season_year=season_year)
        # Function to compute seasonal means
        def seasonal_average(season_name):
            season_data = data.where(data.season == season_name)
            grouped = season_data.groupby('season_year').mean('time', skipna=True)
            grouped = grouped.rename({'season_year': 'time'})
            grouped['time'] = pd.to_datetime(grouped['time'].values, format='%Y')
            return grouped
        # Compute seasonal means
        seasonal_ds = xr.Dataset({
            'DJF': seasonal_average('DJF'),
            'MAM': seasonal_average('MAM'),
            'JJA': seasonal_average('JJA'),
            'SON': seasonal_average('SON')
        })
        return seasonal_ds
        
    if os.path.isfile(regression_cache) and not overwrite:
        logger.info(f"Regressions already calculated and cached in {regression_cache}")
        return
    else:
        logger.info("No cached regression results found or overwrite is set to True. Calculating regressions...")   
        # open sam indices, leading PC of the EOF calculation 
        ds_sam = xr.open_dataset(monthly_sam_file).pcs.isel(mode=0)
        # open sst data
        ds_sst = xr.open_dataset(monthly_sst_file).sst

        # calculate seasonal means for both datasets
        ds_sam_seasonal = _extract_seasonal_means(ds_sam)
        ds_sst_seasonal = _extract_seasonal_means(ds_sst)

        # now we want to regress the SST data against the SAM indices
        # for each season. 
        def _intialize_dataarray():
            return xr.DataArray(
                np.zeros([4, 12, 64, 64]),
                dims=['season', 'face', 'height', 'width'],
                coords={
                    'season': ['DJF', 'MAM', 'JJA', 'SON'],
                    'face': ds_sst_seasonal.MAM.face,
                    'height': ds_sst_seasonal.MAM.height,
                    'width': ds_sst_seasonal.MAM.width
                }
            )
        # initialize data array to hold regression results
        regressions = _intialize_dataarray()
        # intialize data array to hold p-values
        p_values = _intialize_dataarray()
        # initialize data array to hold r_values
        r_values = _intialize_dataarray()
        
        # loop through seasons and calculate regression
        logger.info("Starting regression calculations...")
        for season in tqdm(['DJF', 'MAM', 'JJA', 'SON']):

            # seasonal data. 
            sam_season = ds_sam_seasonal[season]
            sst_season = ds_sst_seasonal[season]

            for face in sst_season.face.values:
                for height in sst_season.height.values:
                    for width in sst_season.width.values:
                        sst_data = sst_season.sel(face=face, height=height, width=width).values
                        sam_values = sam_season.values

                        # here we're checking for nans and masking them.  
                        mask = np.isfinite(sst_data) & np.isfinite(sam_values)
                        if np.count_nonzero(mask) >= 3:
                            slope, intercept, r_val, p_val, std_err = linregress(
                                sam_values[mask], sst_data[mask]
                            )
                            regressions.loc[season, face, height, width] = slope
                            p_values.loc[season, face, height, width] = p_val
                            r_values.loc[season, face, height, width] = r_val
                        else:
                            regressions.loc[season, face, height, width] = np.nan
                            p_values.loc[season, face, height, width] = np.nan
                            r_values.loc[season, face, height, width] = np.nan

        logger.info("Regression calculations completed.")
        if regression_cache is not None:
            logger.info(f"Caching regression results to {regression_cache}")
            # combine into a dataset
            ds_regressions = xr.Dataset({
                'regressions': regressions,
                'p_values': p_values,
                'r_values': r_values
            })
            # save to netcdf
            ds_regressions.to_netcdf(regression_cache, mode='w', engine='netcdf4')
    
        return

def plot_regressions(
    regression_cache: str,
    output_dir: str,
    output_prefix: str,
    title_prefix: str,
):

    # open regression results from cache
    logger.info(f"Loading regression results from {regression_cache}")
    ds_regressions = xr.open_dataset(regression_cache)
    regressions = ds_regressions.regressions
    p_values = ds_regressions.p_values
    r_values = ds_regressions.r_values

    # now we want to map this data onto lat-lon grid
    # and plot the results.
    # mount module DLESyM for remapping utility 
    try:
        from data_processing.remap import healpix as hpx
    except ImportError as e:
        logger.info(f"Error importing DLESyM, make sure PYTHONPATH is set to include DLESyM: {e}")
        return
    logger.info(f"Successfully import DLESyM module.")

    # We use this to get from HPX mesh to Lat-Lon. 1 degree resolution
    mapper = hpx.HEALPixRemap(
        latitudes=181,
        longitudes=360,
        nside=64,
    )

    # function for quickly remapping data and returning a DataArray
    # uses the above initialized mapper.
    def _remap_data(data):
        return xr.DataArray(
            mapper.hpx2ll(data.values),
            dims=['lat', 'lon'],
            coords={'lat': np.arange(90, -90.1, -1), 'lon': np.arange(0, 360, 1)}
        )
    
    def _get_custom_cmap(cmap_base='PiYG'):
        # Create a colormap that goes from the base colormap to 'whitesmoke'
        cmap_base = plt.get_cmap(cmap_base)
        # Split the base colormap into two parts
        colors_lower = cmap_base(np.linspace(0, 0.45, 120))
        colors_upper = cmap_base(np.linspace(0.55, 1, 120))
        colors_middle = mcolors.LinearSegmentedColormap.from_list("mycmap", ['whitesmoke', 'whitesmoke'])
        # Create a new colormap with 'whitesmoke' in the middle
        colors = np.vstack((colors_lower, colors_middle(np.linspace(.4,.6,48)), colors_upper))
        cmap_combined = mcolors.LinearSegmentedColormap.from_list('custom_divergent_colormap', colors)
        return cmap_combined
    
    # helper function to initilaize axis 
    def _initialize_axis():
        fig, ax = plt.subplots(
            subplot_kw={'projection': ccrs.Robinson(central_longitude=180)},
            figsize=(8, 4)
        )
        # Add filled land feature (acts as a patch)
        land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='black', facecolor='lightgray', linewidth=0.3)
        ax.add_feature(land, zorder=10)
        return fig, ax
    
    # make output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # now we plot a map of the regressions for each season
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    logger.info("Plotting regression results...")
    for season in tqdm(seasons):
        # remap the data to lat-lon grid
        reg_data = _remap_data(regressions.sel(season=season))
        p_data = _remap_data(p_values.sel(season=season))
        r_data = _remap_data(r_values.sel(season=season))

        # plot the regression and p-values 
        fig, ax = _initialize_axis()
        # plot regression (also add cyclic point for better visualization)
        reg_data_cyc, lon_cyc = add_cyclic_point(reg_data, coord=reg_data.lon)
        im = ax.contourf(lon_cyc, reg_data.lat, reg_data_cyc, 
            transform=ccrs.PlateCarree(), cmap=_get_custom_cmap(), extend='both',
            levels=np.arange(-12, 13, 2))
        # plot pvalue contour at 0.05 significance level (again add cyclic point)
        p_data_cyc, lon_p_cyc = add_cyclic_point(p_data, coord=p_data.lon)
        ax.contour(lon_p_cyc, p_data.lat, p_data_cyc,
            transform=ccrs.PlateCarree(), levels=[0.05], colors='black', linewidths=0.5, linestyles='dashed')
        
        ax.set_title(f"{title_prefix} SAM vs. SST: {season}")
        # add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6, aspect=40, ticks=np.arange(-12, 13, 4),)
        cbar.set_label(label='$^{\circ}$C per unit SAM', fontsize=12)
        # tighten the layout and save
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{output_prefix}_{season}_regression.png"), dpi=500, bbox_inches='tight')
        plt.close(fig)

        # plot r value and p-values
        fig, ax = _initialize_axis()
        r_data_cyc, lon_cyc = add_cyclic_point(r_data, coord=r_data.lon)
        im = ax.contourf(lon_cyc, r_data.lat, r_data_cyc,
            transform=ccrs.PlateCarree(), cmap=_get_custom_cmap(), levels=np.arange(-.8, .81, 0.2), extend='both')
        ax.contour(lon_p_cyc, p_data.lat, p_data_cyc,
            transform=ccrs.PlateCarree(), levels=[0.05], colors='black', linewidths=0.5, linestyles='dashed')
        ax.set_title(f"{title_prefix} SAM-SST Correlation: {season}")
        # add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6, aspect=40, ticks=np.arange(-.8, .81, 0.4))
        cbar.set_label(label='$r$-value', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{output_prefix}_{season}_r_values.png"), dpi=500, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    
    # Run for DLESyM forced output
    seasonal_sst_sam_regression(
        monthly_sam_file='data/monthly_z250_dlesym_eofs.nc',
        monthly_sst_file='data/monthly_sst_obs.nc',
        regression_cache='data/dlesym_sam_sst_regression.nc',
    )
    # plot the results
    plot_regressions(
        regression_cache='data/dlesym_sam_sst_regression.nc',
        output_dir='plots/sst_regressions/',
        output_prefix='dlesym_sam_sst_regression',
        title_prefix='DLESyM'
    )

    # Run regression for observational data
    seasonal_sst_sam_regression(
        monthly_sam_file='data/monthly_z250_obs_eofs.nc',
        monthly_sst_file='data/monthly_sst_obs.nc',
        regression_cache='data/obs_sam_sst_regression.nc',
    )
    # plot the results
    plot_regressions(
        regression_cache='data/obs_sam_sst_regression.nc',
        output_dir='plots/sst_regressions/',
        output_prefix='obs_sam_sst_regression',
        title_prefix='ERA5'
    )
