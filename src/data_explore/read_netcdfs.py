from functools import lru_cache
import xarray as xr
import pandas as pd
import pathlib
from loguru import logger


def read_data_sources(fname_yaml:str):
    """Reads a yaml file and returns a munch data type"""
    import yaml
    import munch

    with open(fname_yaml, 'r') as f:
        sources = munch.munchify(yaml.safe_load(f))
    return sources  # type: ignore
    

def open_and_rename(entry, t):
    """
    Opens and processes a dataset if it contains the requested time range.
    
    Args:
        entry (dict): Entry from the dataset catalog.
        t (pd.Timestamp): Time for which data is requested.
    
    Returns:
        xarray.Dataset or None: Processed dataset or None if file is unavailable.
    """
    if not in_time_range(entry, t):  # Check if the requested time is within the dataset's range
        return None
    
    try:
        path = get_entry_local_path(entry, t)  # Get local path for the requested time
    except FileNotFoundError as e:
        logger.warning(str(e))  # Log a warning if the file is missing
        return None
    
    vars_tuple = tuple(entry.variables.items())  # Convert variable dictionary to tuple
    ds = _load_netcdf(path, vars_tuple)  # Load and rename dataset
    
    return ds


def in_time_range(entry, t):
    """
    Checks if a given time falls within the dataset's available time range.
    
    Args:
        entry (dict): Dataset entry from the catalog.
        t (pd.Timestamp): Requested time.
    
    Returns:
        bool: True if the time is within range, False otherwise.
    """
    t0 = pd.Timestamp(entry['time']['start'])  # Start time of dataset
    t1 = pd.Timestamp(entry['time']['end'])  # End time of dataset
    
    return (t > t0) & (t < t1)  # Return True if time is within range

    
def get_entry_local_path(entry, t, max_dt=5):
    """
    Retrieves the local file path for a dataset, checking for small time offsets if needed.
    
    Args:
        entry (dict): Dataset entry from the catalog.
        t (pd.Timestamp): Requested time.
        max_dt (int): Maximum number of days to search for a nearby file if exact match is unavailable.
    
    Returns:
        str: File path of the dataset.
    
    Raises:
        FileNotFoundError: If no suitable file is found.
    """
    local = entry['storage_options']['cache_storage']  # Retrieve base storage path
    fpath_test = pathlib.Path(local.format(t=t))  # Format path with requested time
    
    # If file exists or is explicitly a NetCDF file, return the formatted path
    if fpath_test.is_file() or fpath_test.name.endswith('nc'):
        return str(fpath_test)
    
    # Otherwise, construct the full path using the dataset URL
    fname = str(pathlib.Path(local) / entry['url'].split('/')[-1])
    fname_out = pathlib.Path(fname.format(t=t))
    
    if fname_out.is_file():
        return str(fname_out)  # Return file path if it exists
    
    # If the exact file isn't found, try adjusting the date by up to max_dt days
    for dt in range(1, max_dt + 1):
        dt = pd.Timedelta(days=dt)
        fname_up = pathlib.Path(fname.format(t=t + dt))
        fname_dn = pathlib.Path(fname.format(t=t - dt))
        
        if fname_dn.is_file():
            return str(fname_dn)  # Return file if found with earlier date
        elif fname_up.is_file():
            return str(fname_up)  # Return file if found with later date
    
    # If no file is found, raise an error
    raise FileNotFoundError(f'Cannot find a file for {entry["abbrev"]} on {t:%Y-%m-%d}')




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
    ds = xr.open_dataset(path, chunks={})  # Load the NetCDF file with no chunking

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