import torch
import numpy as np
import pandas as pd
from diffusers import DDIMScheduler
from utils import add_src_and_logger
save_dir = None
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

def normalize(df, stats, mode):
    for i in range(len(stats['means']) - 1): # first column is the target
        col = df.columns[i]
        # print(f"Normalizing {col} with {mode}")
        if mode == 'min_max':
            # print(f"Min: {stats['mins'][i]}, Max: {stats['maxs'][i]}")
            df[col] = 2 * (df[col] - stats['mins'][i + 1]) / (stats['maxs'][i + 1] - stats['mins'][i + 1]) - 1
        elif mode == 'mean_std':
            df[col] = (df[col] - stats['means'][i + 1]) / stats['stds'][i + 1]
        else:
            raise ValueError(f"Unknown mode {mode}")
    return df

def diffusion_step(model, noise_scheduler, x, t, jump):
    # Get model pred
    sample = x[:, 0:1, :]  # Assuming the first channel is the sample
    with torch.no_grad():
        residual = model(x, t, return_dict=False)[0]
    output_scheduler = noise_scheduler.step(residual, t, sample, eta=0)
    if jump is not None:
        x_0 = output_scheduler.pred_original_sample
        if t < jump:
            sample = x_0
        else:
            sample = noise_scheduler.add_noise(x_0, torch.randn_like(sample), t - jump)
    else:
        # Update sample with step
        sample = output_scheduler.prev_sample # this is used with the DDIM sampler
    return sample

def step_and_collocate(model, t, dataloader, noise_scheduler, n_recs, expo_ids, windows, recs):
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in dataloader:
        batch = batch.to(device)
        sample = diffusion_step(model, noise_scheduler, batch, t, jump=None)
        samples.append(sample.cpu().numpy())

    samples = np.concatenate(samples, axis=0).squeeze(axis=1)
    rec_dfs = []
    for i in range(n_recs):
        rec_df = pd.DataFrame({
            'expocode': expo_ids[recs == i].flatten(),
            'window_id': windows[recs == i].flatten(),
            f'rec_{i}': samples[recs == i].flatten()
        })
        rec_df.set_index(['expocode', 'window_id'], inplace=True)
        rec_dfs.append(rec_df)
    
    sample_df = pd.concat(rec_dfs, axis=1)
    return sample_df

from fco2models.ueval import get_expocode_map
def init_denoise_df(df, n_recs=10, n=64):
    """
    Initialize a DataFrame for denoising with the required columns.
    """
    rec_cols = [f'rec_{i}' for i in range(n_recs)]
    df = reflect_dataframe_edges(df, n=n) # pad edges of each cruise
    df.loc[:, rec_cols] = np.random.randn(len(df), n_recs)  # Random values for recs
    df.loc[:, 'expocode_id'] = df.loc[:, 'expocode'].map(get_expocode_map(df.loc[:, :])) # map expocode to int id for ndarray bookkeeping
    return df

def reflect_dataframe_edges(df: pd.DataFrame, group_col: str = "expocode", n: int = 64) -> pd.DataFrame:
    def reflect_group(group):
        g_n = len(group)
        if g_n < n :
            return group
        
        top_reflection = group.iloc[:n][::-1]
        top_reflection.window_id = np.arange(-1, -n-1, -1)
        bottom_reflection = group.iloc[-n:][::-1]
        bottom_reflection.window_id = np.arange(g_n, g_n + n, 1)
        return pd.concat([top_reflection, group, bottom_reflection], ignore_index=True)
    
    def repeat_group(group):
        g_n = len(group)
        
        # Repeat top n rows m times
        top_repeat = pd.concat([group.iloc[[0]]] * n, axis=0)
        top_repeat.window_id = np.arange(-n, 0, 1)  # Assign window IDs (adjust logic if needed)

        # Repeat bottom n rows m times
        rest = n - g_n % n if g_n >= n else n - g_n # add also a small rest to make track sizes divisible by 64
        bottom_repeat = pd.concat([group.iloc[[-1]]] * (n + rest), axis=0) # extend to n exactly so that get_segments_fast works
        bottom_repeat.window_id = np.arange(g_n, g_n + n + rest, 1)

        #Concatenate top repeats, original group, bottom repeats
        return pd.concat([top_repeat, group, bottom_repeat], ignore_index=True)
    
    return df.groupby(group_col, group_keys=False)[df.columns].apply(repeat_group).reset_index(drop=True)

from numpy.lib.stride_tricks import sliding_window_view
def get_segments_fast(df, cols, num_windows=64, step=64, offset=0):
    """
    Return an array of shape (n_segments, len(cols), num_windows)
    without any Python‑level looping.
    """
    arr = df[cols].to_numpy(dtype=np.float32, copy=False)     # (N, C)
    # windows: (N - num_windows + 1, C, num_windows)
    windows = sliding_window_view(arr, num_windows, axis=0)
    windows = windows[offset::step]                           # stride in one go
    # Note that windows of neighbouring cruise tracks (which are completely unrelated) can have overlapping paddings.
    # This is not optimal, but it is much easier and faster to implement.
    return windows 

def step_get_samples(model, t, dataloader, noise_scheduler):
    outs = []
    with torch.inference_mode():
        for batch in dataloader:                    # batch is CPU
            batch = batch.to(model.device, non_blocking=True)
            sample = diffusion_step(model, noise_scheduler, batch, t, jump=None)
            outs.append(sample.cpu().numpy())
    return np.concatenate(outs, axis=0).squeeze(axis=1)  # (N, L)


def segment_rec_df(df, predictors, n_recs, segment_len=64):
    """
    Segment the DataFrame into smaller chunks for processing.
    Include data as well as indexing information (expocode_id, window_id, rec_id)
    """
    segments = []
    cols =  predictors + ['expocode_id', 'window_id']
    random_offset = np.random.randint(0, 64, size=n_recs)
    for i in range(n_recs):
        rec = f'rec_{i}'
        segments_rec = get_segments_fast(df, [rec] + cols, offset=random_offset[i]) # get segments for every sample with a different offset
        rec_col = np.full_like(segments_rec[:, 0:1, :], i) # column storing sample id for assigning to correct column after denoising step
        segments_rec = np.concatenate([segments_rec, rec_col], axis=1) # concat data and sample id column (each datapoint is now indexed by: expocode_id, window_id, rec_id)
        segments.append(segments_rec)

    segments = np.concatenate(segments, axis=0)
    return segments

from tqdm import tqdm
from torch.utils.data import DataLoader
import time
def denoise_df(df, model_info, n_recs, jump=20):
    params = model_info['params']
    predictors = params['predictors']
    model = model_info['model']
    noise_scheduler = model_info['noise_scheduler']
    # Use ddim for inference
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
        beta_schedule=noise_scheduler.config.beta_schedule,
        clip_sample_range=noise_scheduler.config.clip_sample_range,
       )
    #timesteps = np.concatenate((np.arange(0, 40, 2), np.arange(40, 1000, 20)))[::-1]
    #print(timesteps)
    # /opt/conda/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddim.py, line 335 for passing inference steps as list
    ddim_scheduler.set_timesteps(50)
   
    stats = {
        'means': params['train_means'],
        'stds': params['train_stds'],
        'mins': params['train_mins'],
        'maxs': params['train_maxs'],
    }
    
    df.loc[:, predictors] = normalize(df.loc[:, predictors], stats, params['mode'])
    denoise_df = init_denoise_df(df, n_recs=n_recs)
    rec_cols = [f'rec_{i}' for i in range(n_recs)]

    t_loop = tqdm(ddim_scheduler.timesteps[::jump], desc="Denoising steps")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for t in t_loop:
        start = time.time()
        ds = segment_rec_df(denoise_df, predictors, n_recs)
        
        expo_ids = ds[:, -3, :].astype(int)
        windows = ds[:, -2, :].astype(int)
        #print(windows)
        recs = ds[:, -1, :].astype(int)
        end = time.time()
        model_input = np.concatenate([ds[:, :-3, :], np.ones_like(ds[:, 0:1, :])], axis=1)
        model_input = torch.from_numpy(model_input).float().to(model.device)
        dataloader = DataLoader(model_input, batch_size=2048, shuffle=False)
        sample_df = step_and_collocate(model, t, dataloader, ddim_scheduler, n_recs, expo_ids, windows, recs)

        denoise_df.set_index(['expocode_id', 'window_id'], inplace=True)
        denoise_df.loc[sample_df.index, rec_cols] = denoise_df.loc[sample_df.index, rec_cols].where(sample_df[rec_cols].isna(), sample_df[rec_cols])
        denoise_df.reset_index(inplace=True)
    
    return denoise_df

def denoise_df_2(df, model_info, n_recs, jump=None, n=64):
    """
    Generate smooth track samples using shifting procedure outlines in Section 3.3 of the writeup
    This version was made mostly with chatGPT to move all data processing to numpy arrays.
    Denoise_df was made by me and works with dataframes (easier to index, but much slower)
    """
    params = model_info['params']
    predictors = params['predictors']
    model = model_info['model']
    noise_scheduler = model_info['noise_scheduler']

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
        beta_schedule=noise_scheduler.config.beta_schedule,
        clip_sample_range=noise_scheduler.config.clip_sample_range,
    )
    ddim_scheduler.set_timesteps(50) # 50 equally spaced time steps 

    stats = {
        'means': params['train_means'],
        'stds': params['train_stds'],
        'mins': params['train_mins'],
        'maxs': params['train_maxs'],
    }

    df.loc[:, predictors] = normalize(df.loc[:, predictors], stats, params['mode'])
    denoise_df = init_denoise_df(df, n_recs=n_recs, n=n) # initialize dataframe for procedure
    rec_cols = [f'rec_{i}' for i in range(n_recs)]

    # Ensure integer dtypes for keys (helps get_indexer)
    denoise_df['expocode_id'] = denoise_df['expocode_id'].astype(np.int64)
    denoise_df['window_id']   = denoise_df['window_id'].astype(np.int64)

    # --- build fixed indexer and a direct view of the rec block
    row_mi = pd.MultiIndex.from_frame(denoise_df[['expocode_id', 'window_id']])
    rec_block = denoise_df[rec_cols].to_numpy(copy=False)  # (n_rows, n_recs)

    t_loop = tqdm(ddim_scheduler.timesteps[::jump], desc="Denoising steps")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for t in t_loop:
        ds = segment_rec_df(denoise_df, predictors, n_recs) # split cruise in segments and include indexing information

        # indexing info
        expo_ids = ds[:, -3, :].astype(np.int64)  # (N, L)
        windows  = ds[:, -2, :].astype(np.int64)  # (N, L)
        recs     = ds[:, -1, :].astype(np.int64)  # (N, L)

        # input to the model data (remoce indexing columns) and add all-one mask
        model_input = np.concatenate([ds[:, :-3, :], np.ones_like(ds[:, 0:1, :])], axis=1)
        model_input = torch.from_numpy(model_input).float()       # stay on CPU here
        dataloader = DataLoader(model_input, batch_size=2048, shuffle=False, pin_memory=True)

        # get denoised samples using the model
        samples = step_get_samples(model, t, dataloader, ddim_scheduler)  # (N, L)

        # Vectorised row lookup: -1 where (expocode_id, window_id) is missing
        idx = pd.MultiIndex.from_arrays([expo_ids.ravel(), windows.ravel()])
        rows = row_mi.get_indexer(idx)                      # ndarray of int64 (−1 for miss)
        mask = rows != -1
        if mask.any():
            # use indexing information
            rows_i = rows[mask]
            cols_i = recs.ravel()[mask]
            vals   = samples.ravel()[mask]
            rec_block[rows_i, cols_i] = vals
        
        denoise_df[rec_cols] = rec_block # save reconstruction in df for the next iteration
    return denoise_df



import pandas as pd
import numpy as np
from fco2models.utraining import prep_df, make_monthly_split, impute_df
from fco2models.ueval import rescale
from fco2models.models import UNet2DModelWrapper, MLPNaiveEnsemble, UNet1DModelWrapper, Unet1DClassifierFreeModel
from fco2models.ueval import load_models
from fco2models.umeanest import MLPModel

save_path = '../models/anoms_sea_1d/'
model_dict = {
    'dm': [save_path, 'e_200.pt', UNet1DModelWrapper],
}
models = load_models(model_dict)
df = pd.read_parquet(DATA_PATH + "SOCAT_1982_2021_grouped_colloc_augm_bin.pq", engine='pyarrow')
df = prep_df(df, bound=True)[0]
df['sst_clim'] += 273.15
df['sst_anom'] = df['sst_cci'] - df['sst_clim']
df['sss_anom'] = df['sss_cci'] - df['sss_clim']
df['chl_anom'] = df['chl_globcolour'] - df['chl_clim']
df['ssh_anom'] = df['ssh_sla'] - df['ssh_clim']
df['mld_anom'] = np.log10(df['mld_dens_soda'] + 1e-5) - df['mld_clim']
mask_train, mask_val, mask_test = make_monthly_split(df)
df_train = df.loc[df.expocode.map(mask_train), :]
df_val = df.loc[df.expocode.map(mask_val), :]
df_test = df.loc[df.expocode.map(mask_test), :]

expocodes = ["AG5W20141113", "33RO20180307", "49P120060717", "49P120100104", "AG5W20141113", "AG5W20151120", "AG5W20170115"]
#expocodes = df_val.expocode.unique()
n_recs = 50
rec_cols = [f'rec_{i}' for i in range(n_recs)]
model_info = models['dm']
#model_info['model'].set_w(0.5) # set classifier-free guidance weight
params = model_info['params']
#df_val = impute_df(df_val, params['predictors'])
df_test = impute_df(df_test, params['predictors']) # impute nans so that Nans do not propagate, when shifting with offsets.
df_in = pd.concat((df_test, ), axis=0)
#df_in = df_in[df_in.expocode.isin(expocodes)] # select a subset of expocodes for faster processing
pad = 64 # pad size
ds = denoise_df_2(df_in, model_info, n_recs=n_recs, jump=None, n=pad)
ds.set_index('index', inplace=True)

# Finally rescale and add atmospheric co2 trend
xco2_ix = params['predictors'].index('xco2') + 1
ds.loc[:, rec_cols] = rescale(ds.loc[:, rec_cols].values.reshape(-1, 1), params, params['mode']).reshape(-1, n_recs)
ds.loc[:, rec_cols] += (ds['xco2'].values[:, None] * params['train_stds'][xco2_ix] + params['train_means'][xco2_ix])

df_in = df_in.reset_index()
print(f"shape df_in: {df_in.shape}")
def trim(df):
    len_expocode = len(df_in[df_in.expocode==df.expocode.iloc[0]])
    return df.sort_values(by='window_id').iloc[pad:pad+len_expocode] # remove padding
    
ds = ds.groupby("expocode").apply(trim)
print(f"shape ds: {ds.shape}")
ds.to_parquet(f'{save_path}selection_imputed.pq')