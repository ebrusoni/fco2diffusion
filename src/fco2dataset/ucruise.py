from itertools import accumulate, chain
import numpy as np
from haversine import haversine_vector
import pandas as pd

TIME_COL = 'time'
LAT_COL = 'lat'
LON_COL = 'lon'
# TIME_COL = 'time_avg'
# LAT_COL = 'lat_005'
# LON_COL = 'lon_005'

def get_dtoprev(srt_cruise):
    # calculate distance in kms to previous locations
    coords = np.column_stack([
        srt_cruise.index.get_level_values(LAT_COL).to_numpy(),
        srt_cruise.index.get_level_values(LON_COL).to_numpy()
    ])
    
    d = np.full(coords.shape[0], np.nan, dtype=np.float32)
    if coords.shape[0] > 1:
        d[1:] = haversine_vector(coords[1:], coords[:-1], normalize=True)
    
    return d

def ifprint(verbose, data):
    if verbose:
        print(data)

def divide_cruise(cruise, num_windows=64, len_window=5, max_time_delta=pd.Timedelta(days=1000), max_d_delta=np.inf, verbose=False):

    srt_cruise = cruise.sort_values(by=TIME_COL)
    d_diff = get_dtoprev(srt_cruise)
    ifprint(verbose, d_diff[30:40])
    time_diff = srt_cruise[TIME_COL].diff()
    # ifprint(verbose, time_diff)
    track_len = num_windows * len_window

    cs = 0
    segs = []
    cur_seg = [0]
    large_time_diffs = []
    for (i, dprev) in enumerate(d_diff[1:]):
        cs += dprev
        if dprev > 20:
            large_time_diffs.append(time_diff.iloc[i + 1])
        # if the segment length exceeds 64*5 kms or the time jumps than max_time_delta or jumps more than max_d_delta kilometers
        # end the segment
        if cs >= track_len or time_diff.iloc[i + 1] >= max_time_delta or dprev >= max_d_delta:
            segs.append(cur_seg)
            cur_seg = [0]
            cs = 0
        else:
            cur_seg.append(dprev.astype(np.float32))
    if cur_seg:
        segs.append(cur_seg)
    # ifprint(verbose, segs[:4])
    # ifprint(verbose, large_time_diffs)

    ix_segs = chain(*[[i]*len(seg) for (i, seg) in enumerate(segs)])
    cum_segs = chain(*[list(accumulate(seg)) for seg in segs])
    
    sdf = pd.DataFrame({'segment_id': list(ix_segs), 'track_length':list(cum_segs)}, index=srt_cruise.index)
    # ifprint(verbose, sdf.iloc[:4].track_length)
    bins = np.arange(-len_window / 2., track_len + len_window, len_window)
    cut_sdf = pd.cut(sdf.track_length, bins=bins, labels=False)
    
    srt_cruise.set_index(cruise.index)
    srt_cruise['bin_id'] = cut_sdf.values
    srt_cruise['segment_id'] = sdf.segment_id
    return srt_cruise


import numpy as np
import pandas as pd
from haversine import haversine_vector
TIME_COL = 'time'
LAT_COL = 'lat'
LON_COL = 'lon'

# mostly chatgpt
def interpolate_ship_positions(df: pd.DataFrame, interval_km=5, max_dist=100):
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
    new_lons = ((lon_start + factors.flatten() * lon_diff) % 360)  # Keep within [0, 360]
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
    
    return df_final


def df_to_numpy(df2, num_bins, predictors):
    """
    Function to create the actual numpy dataset from the binned dataframe

    df2: pandas dataframe of collocated data plus segment_id and bin_id columns
    num_bins: number of bins in the dataset
    predictors: list of predictor column names
    
    returns: numpy array of shape (len(predictors) + 1, num_segs_tot, num_bins + 1) and expocode map for visualization
    """

    binned = df2.groupby([pd.Grouper(level=0), 'segment_id', 'bin_id']).mean() # first bin all values in the 5km buckets found above
    print("time_feat binned: ", binned['time_feat'].head())
    # index level of <binned> : [expocode, segment_id, bin_id]
    bins_per_seg = binned.groupby([pd.Grouper(level=0), pd.Grouper(level=1)]).size() # number of non-empty buckets per segment
    num_segs_tot = bins_per_seg.size # total number of segments in dataset
    segs_per_expocode = bins_per_seg.groupby(level=0).size() # number of segments per expocode
    
    offsets = segs_per_expocode.values.cumsum() # offsets to index segments in expocode table
    offsets_expocode = np.zeros_like(offsets)
    offsets_expocode[1:] = offsets[:-1] # offsets at expocode level
    offsets_seg = np.repeat(offsets_expocode, segs_per_expocode.values)
    offsets_seg = offsets_seg + bins_per_seg.index.get_level_values(level=1).to_numpy() # offset for each segment
    offsets_seg_per_bin = np.repeat(offsets_seg, bins_per_seg) # offset for each bin (row index in dataset does not change)

    dataset = np.full((len(predictors) + 1, num_segs_tot, num_bins + 1), 
                      np.nan, 
                      dtype=np.float64)
    bin_ids = binned.index.get_level_values(level=2).to_numpy()

    # I only add the fco2 values for the moment
    y = binned.fco2rec_uatm.values
    X = binned[predictors].values.T
    dataset[0, offsets_seg_per_bin, bin_ids] = y
    dataset[1:, offsets_seg_per_bin, bin_ids] = X

    expomap = pd.DataFrame(offsets_seg, index=bins_per_seg.index)

    return dataset, expomap

import matplotlib.pyplot as plt
def ds_nanstats(ds):
    n_obs = ds.size
    n_segs, n_bins = ds.shape
    print("dataset shape: ", ds.shape)
    print("total number of entries: ", n_obs)
    notnans = (np.where(~np.isnan(ds))[0]).shape[0]
    print("fraction of valid observations: ", notnans / n_obs)
    nonans = np.apply_along_axis(lambda track: ~np.isnan(track).any(), 1, ds)
    full_tracks = np.where(nonans)[0]
    print("fraction of segments without any nans: ", full_tracks.size / n_segs)
    bins_nan_stat = []
    for i in range(n_bins + 1):
        badsegs = np.apply_along_axis(lambda seg: np.sum(np.isnan(seg)) == i, 1, ds)
        bins_nan_stat.append(badsegs.sum() / n_segs)
    plt.figure(figsize=(20, 5))
    plt.title("Fraction of segments with i nans")
    plt.xticks(np.arange(0, n_bins + 1, 1))
    plt.bar(range(n_bins + 1), bins_nan_stat)
    plt.show()

def filter_nans(ds, y, predictors, col_map):
    """
    Filter out samples where any of the predictors contains nans
    """
    X = ds[[col_map[p] for p in predictors]]
    # keep samples where all predictors are not nan
    not_nan_mask = np.isnan(X).any(axis=2)
    not_nan_mask = np.sum(not_nan_mask, axis=0) == 0
    print("Number of samples after filtering: ", np.sum(not_nan_mask))
    return X[:, not_nan_mask, :], y[not_nan_mask, :]

def plot_segment(X, y, titles, seg):
    num_bins = X.shape[2]
    fig, axs = plt.subplots(X.shape[0] + 1, 1, figsize=(5*(X.shape[0] + 1), 10), sharex=True)
    plt.xlim((0, num_bins))
    axs[0].plot(y[seg])
    print(y[seg])
    axs[0].set_title(titles[0])
    for i in range(X.shape[0]):
        axs[i + 1].plot(X[i, seg])
        axs[i + 1].set_title(titles[i + 1])
    plt.show()


def divide_cruise_random(cruise, num_windows=64, len_window=5, max_time_delta=pd.Timedelta(days=1000), max_d_delta=np.inf, verbose=False):
    """
    Divide the cruise into segments of length num_windows * len_window, by randomly selecting a starting point

    """
    
    srt_cruise = cruise.sort_values(by=TIME_COL)
    d_diff = get_dtoprev(srt_cruise)
    time_diff = srt_cruise[TIME_COL].diff()
    track_len = (num_windows - 1) * len_window
    
    total_distance = np.sum(d_diff[1:]).astype(int)
    random_starts = np.random.randint(0, len(srt_cruise), size=(total_distance // track_len + 1) * 2)

    segs = []
    for location in random_starts:
        dprev = 0
        cur_seg = [0]
        cs = 0
        while cs < track_len and location < len(srt_cruise):
            location += 1
            dprev += d_diff[location]
            cs += dprev
            # if the segment length exceeds 64*5 kms or the time jumps than max_time_delta or jumps more than max_d_delta kilometers
            # end the segment
            if cs >= track_len or time_diff.iloc[location] >= max_time_delta or dprev >= max_d_delta:
                segs.append(cur_seg)
                cur_seg = []
                break
            else:
                cur_seg.append(dprev.astype(np.float32))
        if cur_seg:
            segs.append(cur_seg)

    # print("segs shape: ", len(segs))
    # print("random_starts shape: ", random_starts.shape)
    
    # index of the segment for each entry in the cruise
    ix_segs = list(chain(*[[i]*len(seg) for (i, seg) in enumerate(segs)]))
    # print("ix_segs shape: ", len(ix_segs))
    print(ix_segs[:20])
    # index of the entry in originial cruise df
    sequences = [np.arange(start, start + len(seg), 1, dtype=np.int32) for (start, seg) in zip(random_starts, segs)]
    sequences = np.concatenate(sequences)
    # print("sequences shape: ", sequences.shape)
    print(sequences[:20])
    # total length of each segment
    cum_segs = chain(*[list(accumulate(seg)) for seg in segs])
    
    bins = np.arange(-len_window / 2., track_len, len_window)
    cut_sdf = pd.cut(pd.Series(cum_segs), bins=bins, labels=False)
    binned = np.zeros((sequences.shape[0], 3), dtype=np.float32)
    binned[:, 0] = sequences
    binned[:, 1] = ix_segs
    binned[:, 2] = cut_sdf.values
    df_binned = pd.DataFrame(binned, columns=['index', 'segment_id', 'bin_id'] , index=srt_cruise.index[sequences])
    df_binned[cruise.columns] = srt_cruise[cruise.columns].iloc[sequences].values
    return df_binned
    
