from utils import add_src_and_logger
save_dir = f'../models/cfree_mini/'
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
from fco2models.models import Unet2DClassifierFreeModel, UNet2DModelWrapper
from fco2models.ueval import load_model
from fco2models.utraining import prep_df, make_monthly_split, get_segments, get_context_mask, normalize_dss, get_stats_df, full_denoise

# load model
model_path = 'e_200.pt'
model_class = Unet2DClassifierFreeModel
model, noise_scheduler, params, losses = load_model(save_dir, model_path, model_class, training_complete=True)
print("Model loaded")

# load data
df = pd.read_parquet(DATA_PATH + "SOCAT_1982_2021_grouped_colloc_augm_bin.pq")
df = prep_df(df, bound=True)[0]
df['sst_clim'] += 273.15
df['sst_anom'] = df['sst_cci'] - df['sst_clim']
df['sss_anom'] = df['sss_cci'] - df['sss_clim']
df['chl_anom'] = df['chl_globcolour'] - df['chl_clim']
df['ssh_anom'] = df['ssh_sla'] - df['ssh_clim']
df['mld_anom'] = df['mld_dens_soda'] - df['mld_clim']
# map expocode column to int
expocode_map = df['expocode'].unique()
expocode_map = {expocode: i for i, expocode in enumerate(expocode_map)}
df['expocode_id'] = df['expocode'].map(expocode_map ) 
print(df.columns)

# print(df.fco2rec_uatm.max(), df.fco2rec_uatm.min())

mask_train, mask_val, mask_test = make_monthly_split(df)
df_train = df[df.expocode.map(mask_train)]
df_val = df[df.expocode.map(mask_val)]
df_test = df[df.expocode.map(mask_test)]
print(df_train.fco2rec_uatm.max(), df_train.fco2rec_uatm.min())


target = "fco2rec_uatm"
predictors = params["predictors"]
#predictors = predictors[1:] # REMOVE AFTERWRADS
coords = ['expocode_id', 'window_id']
all_cols = predictors + coords
cols = [target] + all_cols

train_stats = get_stats_df(df_train, [target] + predictors)


def prep_for_eval(df, predictors, coords):
    target = "fco2rec_uatm"
    all_cols = [target] + predictors + coords
    n_coords = len(coords)
    segment_df = df.groupby("expocode").apply(
        lambda x: get_segments(x, all_cols),
        include_groups=False,
    )
    ds = np.concatenate(segment_df.values, axis=0)
    ds_input = ds[:, :-n_coords, :]  # remove expocode and window_id
    ds_index = ds[:, -n_coords:, :]
    return ds_input, ds_index


train_ds, train_index = prep_for_eval(df_train, predictors, coords)
val_ds, val_index = prep_for_eval(df_val, predictors, coords)
test_ds, test_index = prep_for_eval(df_test, predictors, coords)

print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
train_context_mask, val_context_mask, test_context_mask = get_context_mask([train_ds, val_ds, test_ds])
train_ds, train_index = train_ds[train_context_mask], train_index[train_context_mask]
val_ds, val_index = val_ds[val_context_mask], val_index[val_context_mask]
test_ds, test_index = test_ds[test_context_mask], test_index[test_context_mask]
print(f"removing context nans")
print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")

mode = params["mode"]
train_ds_norm, val_ds_norm, test_ds_norm = normalize_dss([train_ds.copy(), val_ds.copy(), test_ds.copy()], train_stats, mode, ignore=[])

# count nans in first column of the dataset
print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")

# CHECK THAT STATS ARE THE SAME AS IN TRAINING
#assert np.allclose(train_stats['maxs'], params['train_maxs'], atol=1e0)
#assert np.allclose(train_stats['mins'], params['train_mins'], atol=1e0)
#assert np.allclose(train_stats['means'], params['train_means'], atol=1e0)
#assert np.allclose(train_stats['stds'], params['train_stds'], atol=1e0)
print(train_stats['maxs'])
print(params['train_maxs'])

# Use ddim for inference
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
    beta_schedule=noise_scheduler.config.beta_schedule,
    clip_sample_range=noise_scheduler.config.clip_sample_range,
    )
ddim_scheduler.set_timesteps(50)


n_rec=20 # number of samples to generate

def denoise_samples(ds_norm, model, scheduler, n_rec):
    context = ds_norm[:, 1:, :] # remove target column
    context_ds = torch.from_numpy(np.repeat(context, n_rec, axis=0)).float()
    print("context_ds shape: ", context_ds.shape)
    context_loader = DataLoader(context_ds, batch_size=512, shuffle=False)
    with torch.no_grad():
        samples_norm = full_denoise(model, scheduler, context_loader, jump=None, eta=0)
    return samples_norm

def rescale_samples(samples_norm, params):
    if params["mode"] == "mean_std":
        samples = samples_norm * params["train_stds"][0] + params["train_means"][0]
    elif params["mode"] == "min_max":
        samples = (samples_norm + 1) * (params["train_maxs"][0] - params["train_mins"][0]) / 2 + params["train_mins"][0]
    else:
        raise ValueError(f"Unknown mode: {params['mode']}")
    return samples

w=0.8
model.set_w(w)
print("Denoise validation set")
val_samples_norm = denoise_samples(val_ds_norm, model, ddim_scheduler, n_rec)
val_samples = rescale_samples(val_samples_norm, params).reshape(-1, n_rec, 64)

print("Denoise training set")
train_samples_norm = denoise_samples(train_ds_norm, model, ddim_scheduler, n_rec)
train_samples = rescale_samples(train_samples_norm, params).reshape(-1, n_rec, 64)

print("Denoise test set")
test_samples_norm = denoise_samples(test_ds_norm, model, ddim_scheduler, n_rec)
test_samples = rescale_samples(test_samples_norm, params).reshape(-1, n_rec, 64)


sample_cols = [f"sample_{i}" for i in range(n_rec)]
def samples_to_df(samples, index):
    samples_index = np.concatenate((samples, index), axis=1)
    n_samples, n_cols, n_bins = samples_index.shape
    samples_index_flat = np.full((n_samples*n_bins, n_cols), np.nan)
    for i in range(samples_index.shape[1]):
        samples_index_flat[:, i] = samples_index[:, i, :].flatten()
    df = pd.DataFrame(samples_index_flat, columns=sample_cols + ['expocode_id', 'window_id'])
    df['expocode_id'] = df['expocode_id'].astype(int)
    df['window_id'] = df['window_id'].astype(int)
    df.set_index(['expocode_id', 'window_id'], inplace=True)
    return df

def concat_to_dataframe(df, samples_df):
    df.set_index(['expocode_id', 'window_id'], inplace=True)
    df_pred = pd.concat([df, samples_df], axis=1)
    df_pred = df_pred.reset_index()
    return df_pred

val_samples_df = samples_to_df(val_samples, val_index)
train_samples_df = samples_to_df(train_samples, train_index)
test_samples_df = samples_to_df(test_samples, test_index)

df_val = concat_to_dataframe(df_val, val_samples_df)
df_train = concat_to_dataframe(df_train, train_samples_df)
df_test = concat_to_dataframe(df_test, test_samples_df)

def get_df_err_stats(df):
    seamask = df.seamask.astype(bool)
    fco2_nans = ~df.fco2rec_uatm.isna()
    pred_nans = ~df.sample_0.isna()
    mask = seamask & fco2_nans & pred_nans
    
    truth = df.loc[mask, "fco2rec_uatm"].values
    mean = df.loc[mask, sample_cols].mean(axis=1).values
    
    low = df.loc[mask, sample_cols].min(axis=1).values
    high = df.loc[mask, sample_cols].max(axis=1).values
    coverage = (truth >= low) & (truth <= high)
    
    rmse = root_mean_squared_error(truth, mean)
    r2 = r2_score(truth, mean)
    mae = mean_absolute_error(truth, mean)
    bias = (truth - mean).mean()

    return dict({
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'coverage': coverage.mean(),
        'avg_interval_width': (high-low).mean(),
        'avg_interval_width_std': (high-low).std(),
        'max_interval_width': (high-low).max(),
        'r2_interval_width_error': r2_score((truth-mean)**2, df.loc[mask, sample_cols].var(axis=1)),
        'corr_variance_error': pearsonr((truth-mean)**2, df.loc[mask, sample_cols].var(axis=1))[0],
        'ratio_error_interval_width': (mae/(high-low)).mean(),
    })

val_err_stats = get_df_err_stats(df_val)
train_err_stats = get_df_err_stats(df_train)
test_err_stats = get_df_err_stats(df_test)

print("Validation error statistics:")
for key, value in val_err_stats.items():
    print(f"{key}: {value:.4f}")
print("\nTraining error statistics:")
for key, value in train_err_stats.items():
    print(f"{key}: {value:.4f}")
print("\nTest error statistics:")
for key, value in test_err_stats.items():
    print(f"{key}: {value:.4f}")

# Save the results
results = {
    'val_err_stats': val_err_stats,
    'train_err_stats': train_err_stats,
    'test_err_stats': test_err_stats,
    'w': w,
    'n_rec':n_rec
}

path_info = f"w{w}_"
path = save_dir + path_info
with open(path+'error_stats.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save the dataframes
val_samples_df.to_parquet(path + 'val_samples.parquet')
train_samples_df.to_parquet(path + 'train_samples.parquet')
test_samples_df.to_parquet(path + 'test_samples.parquet')