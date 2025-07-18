import torch
import numpy as np
import pandas as pd
from utils import add_src_and_logger
save_dir = None
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

def normalize(df, stats, mode):
    for i in range(1, len(df.columns)): # first column is the target
        col = df.columns[i - 1]
        # print(f"Normalizing {col} with {mode}")
        if mode == 'min_max':
            # print(f"Min: {stats['mins'][i]}, Max: {stats['maxs'][i]}")
            df[col] = 2 * (df[col] - stats['mins'][i]) / (stats['maxs'][i] - stats['mins'][i]) - 1
        elif mode == 'mean_std':
            df[col] = (df[col] - stats['means'][i]) / stats['stds'][i]
        else:
            raise ValueError(f"Unknown mode {mode}")
    return df

def diffusion_step(model, noise_scheduler, x, t, jump):
    # Get model pred
    sample = x[:, 0:1, :]  # Assuming the first channel is the sample
    with torch.no_grad():
        residual = model(x, t, return_dict=False)[0]
    output_scheduler = noise_scheduler.step(residual, t, sample)
    if jump is not None:
        x_0 = output_scheduler.pred_original_sample
        if t < jump:
            sample = x_0
        else:
            sample = noise_scheduler.add_noise(x_0, torch.randn_like(sample), t - jump)
    else:
        # Update sample with step
        sample = output_scheduler.prev_sample
    return sample

def step_and_collocate(model, t, dataloader, noise_scheduler, n_recs, expo_ids, windows, recs):
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in dataloader:
        batch = batch.to(device)
        sample = diffusion_step(model, noise_scheduler, batch, t, jump=20)
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
    # print(f"Sample DataFrame shape: {sample_df.shape}")
    # print(f"Sample DataFrame Nans: {sample_df.isna().sum()}")
    return sample_df

from fco2models.ueval import get_expocode_map
def init_denoise_df(df, n_recs=10):
    """
    Initialize a DataFrame for denoising with the required columns.
    """
    rec_cols = [f'rec_{i}' for i in range(n_recs)]
    df.loc[:, rec_cols] = np.random.randn(len(df), n_recs)  # Random values for recs
    #df.loc[:, 'window'] = df.groupby('expocode').cumcount()  # Create a window column
    df.loc[:, 'expocode_id'] = df['expocode'].map(get_expocode_map(df))
    df = reflect_dataframe_edges(df)
    return df

def reflect_dataframe_edges(df: pd.DataFrame, group_col: str = "expocode", n: int = 128) -> pd.DataFrame:
    def reflect_group(group):
        g_n = len(group)
        if g_n < n :
            return group
        
        top_reflection = group.iloc[:n][::-1]
        top_reflection.window_id = np.arange(-1, -n-1, -1)
        bottom_reflection = group.iloc[-n:][::-1]
        bottom_reflection.window_id = np.arange(g_n, g_n + n, 1)
        return pd.concat([top_reflection, group, bottom_reflection], ignore_index=True)

    return df.groupby(group_col, group_keys=False).apply(reflect_group).reset_index(drop=True)

from fco2models.utraining import get_segments
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
    return windows

def segment_rec_df(df, predictors, n_recs, segment_len=64):
    """
    Segment the DataFrame into smaller chunks for processing.
    """
    segments = []
    cols =  predictors + ['expocode_id', 'window_id']
    random_offset = np.random.randint(0, 64, size=n_recs)
    for i in range(n_recs):
        rec = f'rec_{i}'
        segments_rec = get_segments_fast(df, [rec] + cols, offset=random_offset[i])
        rec_col = np.full_like(segments_rec[:, 0:1, :], i)
        segments_rec = np.concatenate([segments_rec, rec_col], axis=1)
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
   
    stats = {
        'means': params['train_means'],
        'stds': params['train_stds'],
        'mins': params['train_mins'],
        'maxs': params['train_maxs'],
    }
    
    df.loc[:, predictors] = normalize(df.loc[:, predictors], stats, params['mode'])
    denoise_df = init_denoise_df(df, n_recs=n_recs)
    rec_cols = [f'rec_{i}' for i in range(n_recs)]

    t_loop = tqdm(noise_scheduler.timesteps[::jump], desc="Denoising steps")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for t in t_loop:
        start = time.time()
        ds = segment_rec_df(denoise_df, predictors, n_recs)
        
        expo_ids = ds[:, -3, :].astype(int)
        windows = ds[:, -2, :].astype(int)
        recs = ds[:, -1, :].astype(int)
        end = time.time()
        #print(f"time for creating ds: {end-start}")
        model_input = np.concatenate([ds[:, :-3, :], np.ones_like(ds[:, 0:1, :])], axis=1)
        model_input = torch.from_numpy(model_input).float().to(model.device)
        dataloader = DataLoader(model_input, batch_size=1024, shuffle=False)
        sample_df = step_and_collocate(model, t, dataloader, noise_scheduler, n_recs, expo_ids, windows, recs)

        denoise_df.set_index(['expocode_id', 'window_id'], inplace=True)
        #nan_mask = sample_df[rec_cols].isna()
        #sample_df = sample_df.loc[~nan_mask, :]
        #denoise_df.loc[:, rec_cols] = np.nan
        denoise_df.loc[sample_df.index, rec_cols] = denoise_df.loc[sample_df.index, rec_cols].where(sample_df[rec_cols].isna(), sample_df[rec_cols])
        #denoise_df.loc[sample_df.index, rec_cols] = sample_df[rec_cols]
        denoise_df.reset_index(inplace=True)
    
    return denoise_df

import pandas as pd
import numpy as np
from fco2models.utraining import prep_df, make_monthly_split
from fco2models.ueval import rescale
from fco2models.models import UNet2DModelWrapper, MLPNaiveEnsemble
from fco2models.ueval import load_models
from fco2models.umeanest import MLPModel

save_path = '../models/anoms_sea/'
model_dict = {
    'xco2_100': [save_path, 'e_200.pt', UNet2DModelWrapper],
    #'sota_ensemble': ['../models/sota_ensemble/', 'e_30.pt', MLPNaiveEnsemble],
}
models = load_models(model_dict)
#DATA_PATH = "../data/training_data/"
df = pd.read_parquet(DATA_PATH + "SOCAT_1982_2021_grouped_colloc_augm_bin.pq", engine='pyarrow')
df = prep_df(df, bound=True)[0]
df['sst_clim'] += 273.15
df['sst_anom'] = df['sst_cci'] - df['sst_clim']
df['sss_anom'] = df['sss_cci'] - df['sss_clim']
df['chl_anom'] = df['chl_globcolour'] - df['chl_clim']
df['ssh_anom'] = df['ssh_sla'] - df['ssh_clim']
df['mld_anom'] = np.log10(df['mld_dens_soda'] + 1e-5) - df['mld_clim']
_, mask_val, mask_test = make_monthly_split(df)
df_val = df.loc[df.expocode.map(mask_val), :]
df_test = df.loc[df.expocode.map(mask_test), :]

#expocodes = ["AG5W20141113"]
expocodes = df_val.expocode.unique()
n_recs = 10
rec_cols = [f'rec_{i}' for i in range(n_recs)]
model_info = models['xco2_100']
params = model_info['params']
ds = denoise_df(df_val[df_val.expocode.isin(expocodes)].copy(), model_info, n_recs=n_recs)
#ds = denoise_df(df_val, model_info, n_recs=n_recs)
ds.set_index('index', inplace=True)
ds.loc[:, rec_cols] = rescale(ds.loc[:, rec_cols].values.reshape(-1, 1), params, params['mode']).reshape(-1, n_recs)

ds = ds.groupby("expocode").apply(lambda df: df.sort_values(by='window_id').iloc[128:-128])
ds.to_parquet(f'{save_path}cruise.pq')