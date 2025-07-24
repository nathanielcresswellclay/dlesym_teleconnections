import sys
import os 
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath

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
    EOFs = (s[:n_modes, None] * VT[:n_modes, :]).reshape(n_modes, nlat, nlon)
    PCs = U[:, :n_modes]
    variance_explained = (s[:n_modes] ** 2) / np.sum(s ** 2)

    # Wrap in DataArrays
    EOFs_da = xr.DataArray(EOFs, dims=('mode', 'lat', 'lon'),
                           coords={'mode': np.arange(n_modes), 'lat': data['lat'], 'lon': data['lon']})
    PCs_da = xr.DataArray(PCs, dims=('time', 'mode'),
                          coords={'time': data['time'], 'mode': np.arange(n_modes)})
    var_exp_da = xr.DataArray(variance_explained, dims='mode', coords={'mode': np.arange(n_modes)})

    # we're done!
    return EOFs_da, PCs_da, var_exp_da

def eofs_for_sam(dataset:str, modes=20, ll_cache:str=None, eof_cache:str=None, overwrite=False):
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

def plot_eofs(eof_cache, output_dir, ouput_prefix, title_prefix):
    """
    Plot EOFs and their principal components from a cached EOF dataset.
    Parameters:
        eof_cache (str): Path to cached EOF dataset, should contain 'eofs', 'pcs', and 'variance_explained'
        output_dir (str): Directory to save plots
        ouput_prefix (str): Prefix for output filenames
        title_prefix (str): Prefix for plot titles
    Returns:
        None
    """

    # load EOFs from cache
    eofs = xr.open_dataset(eof_cache).eofs
    pcs = xr.open_dataset(eof_cache).pcs
    var_exp = xr.open_dataset(eof_cache).variance_explained

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # function for drawing boundary
    def _draw_circle(ax, theta, center_x=0.5, center_y=0.5):
        center, radius = [center_x, center_y], 0.5
        # Adjust theta to span only half the circle
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        return ax
    
    def _get_custom_cmap(cmap_base='coolwarm'):

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
    
    # plot each EOF
    logger.info("Plotting EOFs...")
    for i in tqdm(range(len(eofs))):

        # format axis
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([0, 360, -25, -90], ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.6, zorder=20)
        ax = _draw_circle(ax, np.linspace(0, 2*np.pi, 100),.5, .5)

        # set title
        ax.set_title(f"{title_prefix} EOF{i+1} ({var_exp[i].values:.2%})", fontsize=12)

        # plot EOF
        eof_cyclic, lon_cyclic = add_cyclic_point(
            eofs[i, :, :].values,
            coord=eofs['lon'].values,
        )
        im = ax.contourf(
            lon_cyclic, eofs['lat'].values,
            eof_cyclic / 9.81,  # convert to m from m^2/s^2
            transform=ccrs.PlateCarree(),
            cmap=_get_custom_cmap(),
            levels=np.linspace(-750, 750, 14),
            extend='both',
            zorder=10
        )
        # add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', 
                            pad=0.05, aspect=40, shrink=0.8,
                            ticks=np.linspace(-750, 750, 7))

        # save plot using output prefix
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{ouput_prefix}_mode{i+1}.png"), dpi=300, bbox_inches='tight')
        # close figure to free memory
        plt.close(fig)

        # now plot PC time series
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2)
        ax.plot(pcs['time'], pcs[:, i], linewidth=1.5)
        ax.set_xlim(pcs['time'].min(), pcs['time'].max())
        ax.set_ylim(-.2, .2)
        ax.set_title(f"{title_prefix} EOF{i+1}", fontsize=12)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('PC Index', fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{ouput_prefix}_pc{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info("EOFs plotted successfully. Now plotting the variance explained...")
    # plot variance explained
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(var_exp)) + 1, var_exp.values * 100, color='steelblue', edgecolor='black')
    # add uncertaintly using Northh's rule of thumb
    uncert = np.sqrt(var_exp.values * (1 - var_exp.values) / (len(pcs['time']) - 1)) * 100
    ax.errorbar(np.arange(len(var_exp)) + 1, var_exp.values * 100, yerr=uncert, fmt='none', color='black', capsize=3)
    ax.set_xticks(np.arange(len(var_exp)) + 1)
    ax.set_xticklabels([f'{i+1}' for i in range(len(var_exp))])
    ax.set_xlabel('EOF Mode', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title(f"{title_prefix} Variance Explained by EOFs", fontsize=14)
    ax.set_ylim(0, 30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{ouput_prefix}_explained_var.png"), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    
    # EOFs for DLESyM monthly z250 data
    eofs_for_sam(dataset ='data/monthly_z250_dlesym.nc', 
                ll_cache ='data/monthly_z250_dlesym_ll.nc', 
                eof_cache ='data/monthly_z250_dlesym_eofs.nc',
                overwrite=False)   
    # plot EOFs 
    plot_eofs(eof_cache = 'data/monthly_z250_dlesym_eofs.nc', 
              output_dir ='plots/eofs', 
              ouput_prefix ='monthly_z250_dlesym_eofs',
              title_prefix='DLESyM ')
    
    # EOFs for observational monthly z250 data
    eofs_for_sam(dataset='data/monthly_z250_obs.nc',
                ll_cache='data/monthly_z250_obs_ll.nc', 
                eof_cache='data/monthly_z250_obs_eofs.nc',
                overwrite=False)
    # plot EOFs
    plot_eofs(eof_cache = 'data/monthly_z250_obs_eofs.nc', 
              output_dir ='plots/eofs', 
              ouput_prefix ='monthly_z250_obs_eofs',
              title_prefix='ERA5 ')
