import pandas as pd
import xarray as xr
from functools import lru_cache
from .read_netcdfs import (
    open_and_rename as open_and_rename_netcdf, 
    read_data_sources as read_data_sources_yaml,
)


def collocate(df, dict_catalog, time_name='time', lat_name='lat', lon_name='lon'):
    import numpy as np
    from loguru import logger

    coords = {'time_name': time_name, 'lat_name': lat_name, 'lon_name': lon_name}
    
    time = df[time_name]
    time_daily = np.unique(time.values.astype('datetime64[D]'))

    df_list = []
    try:
        for t in time_daily:
            df_list += collocate_timestep(df, dict_catalog, t, **coords),
    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt passed, returning and concatenating current data')
        pass
        
    df_collocated = pd.concat(df_list, axis=0)
    
    return df_collocated


def collocate_timestep(df, dict_catalog, time, time_name='time', lat_name='lat', lon_name='lon'):
    print(time, end=': ')
    coords = [time_name, lat_name, lon_name]
    
    t0 = t = pd.Timestamp(time)
    t1 = t0 + pd.DateOffset(days=1, seconds=-1)

    # data for current timestep
    df_t = df.loc[ 
        (df[time_name] >= t0)
        & 
        (df[time_name] <= t1)
    ]

    if len(df_t) == 0:
        raise ValueError("Something wrong with your time-based selection, no data returned")
    
    # selection of coordinates to retrieve
    selection = (
        df_t[coords]
        .to_xarray()
        .rename(**{
            lat_name:'lat', 
            lon_name:'lon', 
            time_name:'time'}))

    df_list = [df_t]
    for key in dict_catalog:
        entry = dict_catalog[key]

        try:  # dodgey try since all exceptions are passed
            ds = open_and_rename_netcdf(entry, t)
            if ds is None:
                continue
                print(f"No data found for {key} on {t:%Y-%m-%d}")
        except:  # dodgey exception 
            ds = None
            print(f"No data found for {key} on {t:%Y-%m-%d}")
            continue
        
        print(key, end=', ')
        
        df_matched = ds.sel(selection, method='nearest').to_dataframe()
        df_list += rename_coords(ds, df_matched, entry.abbrev),
    df_out = pd.concat(df_list, axis=1)
    print()

    return df_out


def rename_coords(ds, df, abbrev):
    coords = list(ds.coords)
    rename = {k: f"{k}_{abbrev}" for k in coords}

    return df.rename(columns=rename)


def extract_tar(fname_tar, dest_dir):
    import pathlib
    import tarfile
    
    dest_dir = pathlib.Path(dest_dir)
    
    tar_obj = tarfile.open(fname_tar)
    tar_obj.extractall(path=dest_dir)
    
    member0 = tar_obj.getnames()[0]
    out_paths = [str(p) for p in (dest_dir / member0).glob('*')]

    return out_paths
    