from datetime import datetime
import pandas as pd
import numpy as np
from fco2dataset.ucruise import divide_cruise, interpolate_ship_positions
from haversine import haversine_vector

VERSION = datetime.now().strftime("%Y%m%d")

def segment_cruise_data():
    NUM_BINS = 64
    WINDOW_LEN = 5
    
    total_cruises = 0
    dfs = []
    for year in range(1982, 2022):
        # Load interpolated data
        fname_interp_socat = f'../data/SOCATv2024_interpolated2/.gridded_2d_ocean_data_for_ML/SOCATv2024-cruise_tracks_interp-collocated/socat2024-interp_tracks-collocated-{year}.pq'
        df_interp_socat = pd.read_parquet(fname_interp_socat)
        df_interp_socat.set_index(['expocode', 'time_1d', 'lat', 'lon'], inplace=True)
        df_interp_socat = df_interp_socat.astype(np.float32, errors='ignore')
        #df_interp_socat = df_interp_socat.groupby(level=[0,1,2,3]).mean()
    
        cruises = len(df_interp_socat.index.unique(level=0))
        total_cruises += cruises
        print(f"number of cruises in {year}: {cruises}")
    
        # read raw socat data
        fname_socat = f'../data/SOCATv2024_raw-collocated-1982_2021/SOCATv2024_raw_r20250307-1982_2021/SOCATv2024v_collocated-{year}.pq'
        df_socat = pd.read_parquet(fname_socat)
        df_socat['time_1d'] = df_socat['time'].dt.round('D')
        df_socat.set_index(['expocode', 'time_1d','lat', 'lon'], inplace=True)
        df_socat = df_socat.astype(np.float32, errors='ignore')
        df_socat['interpolated'] = False
    
        df = pd.concat([df_socat, df_interp_socat], axis=0)
        print(df.shape)
        df2 = df.groupby(level='expocode', group_keys=False).apply(
            lambda cruise: divide_cruise(
                cruise,
                num_windows=NUM_BINS,
                len_window=WINDOW_LEN,
                max_time_delta=pd.Timedelta(days=2),
            )
        )
    
        dfs.append(df2)
    
    df = pd.concat(dfs, axis=0)
    df.interpolated = df.interpolated.astype(bool)
    
    # save df to parquet
    df.to_parquet(
        f'../data/training_data/SOCAT_1982_2021_collocated_augm_binned-v{VERSION}.pq',
        index=True,
    )
    
    # save grouped by expocode and window_id
    df.reset_index(inplace=True)
    df_grouped = df.groupby(['expocode', 'window_id']).mean()
    df_grouped.reset_index(inplace=True)
    
    df_grouped.to_parquet(
        f'../data/training_data/SOCAT_1982_2021_grouped_colloc_augm_binned-v{VERSION}.pq',
        index=False,
    )

# mostly chatgpt
def interpolate_ship_positions(df: pd.DataFrame, interval_km=5, max_dist=100):
    TIME_COL = 'time'
    LAT_COL = 'lat'
    LON_COL = 'lon'
    df.sort_values(by=TIME_COL, inplace=True)
    # Convert timestamp to numeric (Unix time) for interpolation
    df[TIME_COL] = pd.to_datetime(df[TIME_COL]).astype(int) / 10**9
    
    # Extract latitude, longitude, and timestamps as NumPy arrays
    # lat1, lon1 = df[LAT_COL].values[:-1], df[LON_COL].values[:-1]
    # lat2, lon2 = df[LAT_COL].values[1:], df[LON_COL].values[1:]
    coord = df[[LAT_COL, LON_COL]].values.reshape((-1, 2))
    coord1 = coord[:-1]
    coord2 = coord[1:]
    
    # Compute distances using fast haversine_vector (in km)
    # distances = haversine_vector(list(zip(lat1, lon1)), list(zip(lat2, lon2)), unit='km')
    distances = haversine_vector(coord1, coord2, normalize=True)
    
    # Identify where interpolation is needed
    num_insertions = np.zeros_like(distances, dtype=int)
    mask = (distances > interval_km) * (distances < max_dist)
    num_insertions[mask] = (distances[mask] // interval_km + 1).astype(int)  # Points to add per gap (add one just to be sure)

    if not mask.any(): # if no points must be added just return
        df['interpolated'] = False
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit='s')
        return df
    
    # Vectorized interpolation setup
    insert_points = np.repeat(np.arange(len(df) - 1), num_insertions)  # Indices to insert new points
    step_counts = np.hstack([np.arange(1, n + 1) for n in num_insertions])  # Steps for interpolation

    # Extract corresponding values for interpolation
    lat_start, lon_start, time_start = df.iloc[insert_points][[LAT_COL, LON_COL, TIME_COL]].values.T
    lat_end, lon_end, time_end = df.iloc[insert_points + 1][[LAT_COL, LON_COL, TIME_COL]].values.T

    # Compute interpolation factors
    factors = step_counts[:, None] / (num_insertions.repeat(num_insertions)[:, None] + 1)
    # print(insert_points.shape)
    # print(factors.shape)
    
    # Interpolate latitude, longitude, and timestamp
    new_lats = lat_start + factors.flatten() * (lat_end - lat_start)
    lon_diff = (lon_end - lon_start + 180) % 360 - 180  # Ensure shortest path interpolation
    new_lons = ((lon_start + factors.flatten() * lon_diff) % 360).astype(np.float32)  # Keep within [0, 360]
    # new_lons = lon_start + factors.flatten() * (lon_end - lon_start)
    new_times = time_start + factors.flatten() * (time_end - time_start)

    # Create DataFrame for interpolated points
    df_new = pd.DataFrame({LAT_COL: new_lats, LON_COL: new_lons, TIME_COL: new_times})

    # Convert timestamp back to datetime and format as dd-mm-yyyy hh:mm:ss
    df_new[TIME_COL] = pd.to_datetime(df_new[TIME_COL], unit='s')
    df_new['interpolated'] = True

    # Convert original timestamps to correct format before merging
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit='s')
    df['interpolated'] = False

    # Merge original and interpolated points, sorting by timestamp
    df_final = pd.concat([df, df_new]).sort_values(by=TIME_COL)

    cruise_id = df.index.unique(level=0)[0]
    df_final['expocode'] = cruise_id
    df_final.set_index('expocode', inplace=True)

    # df_final[LON_COL] = df_final[LON_COL].map(lambda x: f"{x:.6f}")
    
    return df_final

def interpolate_all_cruises():
    # interpolate points for all years
    for year in range(1982, 2022):
        fname = f'../data/SOCATv2024_raw-collocated-1982_2021/SOCATv2024_raw_r20250307-1982_2021/SOCATv2024v_collocated-{year}.pq'
        df_raw = pd.read_parquet(fname)
        df_raw.set_index(['expocode'], inplace=True)
        df2 = (df_raw.groupby(level=0, group_keys=False)
               .apply(
                   lambda cruise: interpolate_ship_positions(cruise)
                   ))
        df2['time_1d'] = df2['time'].dt.round('D')
        df2.reset_index(inplace=True)
        df2_new = df2.set_index(['expocode', 'time_1d', 'lat', 'lon'])
        df_interpolated = df2_new.loc[df2_new.interpolated, ['interpolated', 'time']]
        df_interpolated.to_parquet(f'../data/interpolated_points/interpolated_{year}')


if __name__ == "__main__":
    # interpolate_all_cruises()
    segment_cruise_data()

