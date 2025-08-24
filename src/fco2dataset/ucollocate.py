import pandas as pd
import pathlib
import requests
import os
import requests
import xarray as xr
from functools import lru_cache

def is_reachable_url(url: str) -> bool:
    try:
        response = requests.head(url, timeout=5)
        return response.status_code >= 200 and response.status_code < 400
    except requests.Timeout:
        print(f"Timeout occurred while checking URL: {url}. Proceed anyway maybe dataset already downloaded.")
        return True
    except requests.RequestException:
        return False
    
def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    r = requests.get(url, stream=True)
    # raise an error if the download fails
    r.raise_for_status()
    # open local file for writing
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            f.write(chunk)
    return local_filename

def rename_coords(ds, df, abbrev):
    coords = list(ds.coords)
    rename = {k: f"{k}_{abbrev}" for k in coords}

    return df.rename(columns=rename)

@lru_cache(10)  # Cache the function results for efficiency
def _load_netcdf(path, vars_tuple: tuple):
    """
    Loads a NetCDF file and renames its variables.
    
    Args:
        path (str): Path to the NetCDF file.
        vars_tuple (tuple): Tuple of (original_var_name, new_var_name) pairs.

    Returns:
        xarray.Dataset: Processed dataset.
    """
    vars = dict(vars_tuple)  # Convert the tuple to a dictionary for easy renaming
    ds = xr.open_dataset(path, chunks={}, decode_timedelta=True)  # Load the NetCDF file with no chunking

    # Rename selected variables
    ds = ds[list(vars)].rename(**vars)
    
    # Standardize time and longitude
    ds['time'] = ds.time.dt.round('1D')  # Round time to the nearest day
    ds['lon'] = ds.lon % 360  # Ensure longitude is in the [0, 360] range
    ds = ds.sortby('lon')  # Sort dataset by longitude
    
    # If depth dimension exists, select the surface level (depth=0)
    if 'depth' in ds.dims:
        ds = ds.sel(depth=0, method='nearest')
    
    return ds

BASE_URL = "https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/"
VARIABLES = {
    'soda': {
        #'temp': 'temp_soda',
        'salt': 'salt_soda',
        'mlp': 'mld_dens_soda',
        #'xt_ocean': 'lon',
        #'yt_ocean': 'lat',
        #'st_ocean': 'depth'
           },
    'chl': {
        'CHL': 'chl_globcolour',
#        'CHL_uncertainty': 'chl_globcolour_uncert',
#        'flags': 'chl_globcolour_flags'
          },
    'cmems': {
        'adt': 'ssh_adt',
        'sla': 'ssh_sla',
        #'latitude': 'lat',
        #'longitude': 'lon'
        },
    'sst_cci': {
        'analysed_sst': 'sst_cci',
        #'analysed_sst_uncertainty': 'sst_cci_uncertainty',
        #'sea_ice_fraction': 'ice_cci'
        },
    'sss_cci': {
        'sss': 'sss_cci', 
        #'sss_random_error': 
        #'sss_cci_random_error'
        },
    'sss_multiobs': {
        'sos': 'sss_multiobs',
        'sos_error': 'sss_multiobs_error',
        'latitude': 'lat',
        'longitude': 'lon'}
        }

def get_zarr_data(year='2022'):
    dict_catalog = {
        'chl': 'chl_globcolour.zarr',
        'soda': 'soda.zarr',
        'cmems': 'ssh_duacs_cmems.zarr',
        'sss_cci': 'sss_cci.zarr',
        #'sss_multiobs': 'sss_multiobs.zarr',
        'sst_cci': 'sst_cci_cdr.zarr'
    }
    dss = {}
    for key, path in dict_catalog.items():
        #print(key)
        varnames_map = VARIABLES[key]
        full_path = os.path.join(BASE_URL, path)
        ds = xr.open_zarr(full_path, group=year)
        if key == "chl":
            ds["lon"] = ds["lon"] % 360
            ds = ds.sortby("lon")
        ds = ds[list(varnames_map)].rename(**varnames_map)
        dss[key] = ds
    
    return dss

#def collocate(df, date, save_path):
def get_day_data(date, save_path):

    dict_catalog = lambda date: {
        'chl': f'chl_globcolour/{date.strftime("%Y")}/{date.strftime("%Y%m%d")}_cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D.nc',
        'soda': f'soda/soda3.15.2_5dy_ocean_reg_{date.strftime("%Y_%m_%d")}.nc',
        'cmems': f'ssh_duacs_cmems/{date.strftime("%Y")}/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_{date.strftime("%Y%m%d")}.nc',
        'sss_cci': f'sss_cci/{date.strftime("%Y")}/ESACCI-SEASURFACESALINITY-L4-SSS-GLOBAL-MERGED_OI_7DAY_RUNNINGMEAN_DAILY_0.25deg-{date.strftime("%Y%m%d")}-fv4.41.nc',
        'sss_multiobs': f'sss_multiobs/{date.strftime("%Y")}/dataset-sss-ssd-nrt-daily_{date.strftime("%Y%m%d")}T1200Z.nc',
        'sst_cci': f'sst_cci_cdr/{date.strftime("%Y")}/{date.strftime("%Y%m%d")}120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR3.0-v02.0-fv01.0.nc'
    }

    # find the closest later date with all files reachable
    all_reachable = False
    while not all_reachable and date < pd.to_datetime("2025-01-01"):
        all_reachable = True
        for key, path in dict_catalog(date).items():
            if not is_reachable_url(BASE_URL + path):
                all_reachable = False
                date += pd.Timedelta(days=1)
                break

    if not all_reachable:
        print(f"Could not find a date with all files reachable after {date}.")
        return None
    
    print(f"All files reachable for date {date}")

    # create a new directory for the gridded data
    pathlib.Path(f"{save_path}_{date:%Y-%m-%d}").mkdir(parents=True, exist_ok=True)
    save_dir = f"{save_path}_{date:%Y-%m-%d}/"
    local_paths = []
    for key, url in dict_catalog(date).items():
        local_filename = os.path.join(save_dir, key + ".nc")
        local_paths.append((key, local_filename))
        # check if the file already exists
        if not os.path.exists(local_filename):
            print(f"Downloading {key} data from {url} to {local_filename}")
            download_file(BASE_URL + url, local_filename)
        else:
            print(f"{key} data already exists at {local_filename}")
    local_paths = dict(local_paths)
    
    # df['time'] = date
    # df.reset_index(inplace=True)
    # coords = ['lat', 'lon', 'time']
    # selection = df[coords].to_xarray()   
    # df_list = [df]
    dss = dict({})
    for key, local_path in local_paths.items():
        print(f"Reading {key} data from {local_path}")
        ds = _load_netcdf(local_path, tuple(VARIABLES[key].items()))
        dss[key] = ds
        # df_matched = ds.sel(selection, method='nearest').to_dataframe()
        # df_matched = rename_coords(ds, df_matched, key)
        # df_list.append(df_matched)
    
    
    # df_collocated = pd.concat(df_list, axis=1)

    # return df_collocated
    return dss

def collocate(df, date, save_path, dss=None, verbose=True):
    """Collocate the data from the dataframe with the data from the NetCDF files."""
    if dss is None:
        # If dss is not provided, load the data from the NetCDF files
        dss = get_day_data(date, save_path)
    df['time'] = date
    df.reset_index(inplace=True)
    coords = ['lat', 'lon', 'time']
    selection = df[coords].to_xarray()   
    df_list = [df]
    for key, ds in dss.items():
        if verbose:
            print(f"Reading {key} data from {ds}")
        df_matched = ds.sel(selection, method='nearest').to_dataframe()
        df_matched = rename_coords(ds, df_matched, key)
        df_list.append(df_matched)
    
    df_collocated = pd.concat(df_list, axis=1)

    return df_collocated