from utils import add_src_and_logger
save_dir = f'../models/anoms_sea_1d/'
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
#from diffusers import DDIMScheduler
from mydiffusers.scheduling_ddim import DDIMScheduler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
from fco2models.models import Unet2DClassifierFreeModel, UNet2DModelWrapper, TSEncoderWrapper, UNet2DShipMix, UNet1DModelWrapper
from fco2models.ueval import load_model
from fco2models.utraining import prep_df, make_monthly_split, get_segments, get_context_mask, normalize_dss, get_stats_df, full_denoise

# load model
model_path = 'e_200.pt'
model_class = UNet1DModelWrapper
model, noise_scheduler, params, losses = load_model(save_dir, model_path, model_class, training_complete=False)
print("Model loaded")

# load data
df = pd.read_parquet(DATA_PATH + "SOCAT_1982_2021_grouped_colloc_augm_bin.pq")
df = prep_df(df, bound=True)[0]
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
assert df_val.expocode.isin(df_test.expocode).sum() == 0, "expocode ids overlap between validation and test sets"
assert df_train.expocode.isin(df_val.expocode).sum() == 0, "expocode ids overlap between train and validation sets"
assert df_train.expocode.isin(df_test.expocode).sum() == 0, "expocode ids overlap between train and test sets"
print(df_train.fco2rec_uatm.max(), df_train.fco2rec_uatm.min())

print(f"train df shape: {df_train.shape}, val df shape: {df_val.shape}, test df shape: {df_test.shape}")

target = "fco2rec_uatm"
predictors = params["predictors"]
coords = ['expocode_id', 'window_id']
all_cols = predictors + coords
cols = [target] + all_cols

train_stats = get_stats_df(df_train, [target] + predictors) # get training set stats (mean, std, min, max)

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
    return ds_input, ds_index # index is used when rearranging the samples back in the dataframe

train_ds, train_index = prep_for_eval(df_train, predictors, coords)
val_ds, val_index = prep_for_eval(df_val, predictors, coords)
test_ds, test_index = prep_for_eval(df_test, predictors, coords)

print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
# remove invalid entries for faster processing
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

# CHECK THAT STATS ARE THE SAME AS IN TRAINING (to be sure that preprocessing was equal)
assert np.allclose(train_stats['maxs'], params['train_maxs'])
assert np.allclose(train_stats['mins'], params['train_mins'])
assert np.allclose(train_stats['means'], params['train_means'])
assert np.allclose(train_stats['stds'], params['train_stds'])
print(train_stats['means'])
print(params['train_means'])
print(train_stats['stds'])
print(params['train_stds'])

# Use ddim for inference
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
    beta_schedule=noise_scheduler.config.beta_schedule,
    clip_sample_range=noise_scheduler.config.clip_sample_range,
    )
ddim_scheduler.set_timesteps(50) # 50 equally-spaced inference steps

n_rec=50 # number of samples to generate
eta=0
def denoise_samples(ds_norm, model, scheduler, n_rec):
    context = ds_norm[:, 1:, :] # remove target fco2 column
    context_ds = torch.from_numpy(np.repeat(context, n_rec, axis=0)).float()
    print("context_ds shape: ", context_ds.shape)
    context_loader = DataLoader(context_ds, batch_size=1028, shuffle=False)
    with torch.no_grad():
        samples_norm = full_denoise(model, scheduler, context_loader, jump=None, eta=eta)
    return samples_norm

def rescale_samples(samples_norm, params):
    if params["mode"] == "mean_std":
        samples = samples_norm * params["train_stds"][0] + params["train_means"][0]
    elif params["mode"] == "min_max":
        samples = (samples_norm + 1) * (params["train_maxs"][0] - params["train_mins"][0]) / 2 + params["train_mins"][0]
    else:
        raise ValueError(f"Unknown mode: {params['mode']}")
    return samples

w=None
#model.set_w(w)
#print("Denoise validation set")
#val_samples_norm = denoise_samples(val_ds_norm, model, ddim_scheduler, n_rec)
#val_samples = rescale_samples(val_samples_norm, params).reshape(-1, n_rec, 64)

#print("Denoise training set")
#train_samples_norm = denoise_samples(train_ds_norm, model, ddim_scheduler, n_rec)
#train_samples = rescale_samples(train_samples_norm, params).reshape(-1, n_rec, 64)

print("Denoise test set")
test_samples_norm = denoise_samples(test_ds_norm, model, ddim_scheduler, n_rec)
test_samples = rescale_samples(test_samples_norm, params).reshape(-1, n_rec, 64)


sample_cols = [f"sample_{i}" for i in range(n_rec)]
def samples_to_df(samples, index):
    """
    function to rearrange samples in dataframe indexed by window_id and expocode_id
    """
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
    """
    Concatenates the original DataFrame with the samples DataFrame.
    """
    df.set_index(['expocode_id', 'window_id'], inplace=True)
    df_pred = pd.concat([df, samples_df], axis=1)
    df_pred = df_pred.reset_index()
    return df_pred

#val_samples_df = samples_to_df(val_samples, val_index)
#train_samples_df = samples_to_df(train_samples, train_index)
test_samples_df = samples_to_df(test_samples, test_index)

#df_val = concat_to_dataframe(df_val, val_samples_df)
#df_train = concat_to_dataframe(df_train, train_samples_df)
df_test = concat_to_dataframe(df_test, test_samples_df)
#df_test[sample_cols] += df_test.xco2.values[:, None]
#df_test['fco2rec_uatm'] += df_test.xco2

def get_df_err_stats(df):
    seamask = df.seamask.astype(bool)
    fco2_nans = ~df.fco2rec_uatm.isna()
    pred_nans = ~df.sample_0.isna()
    mask = seamask & fco2_nans & pred_nans
    
    print(f"save pred_nans to pq") # for matching MLP and diffusion test sets during MLP validation
    pd.DataFrame(mask).to_parquet(f"{save_dir}pred_nans.pq")
    
    print(f"no. test samples {mask.sum()}") # number of test samples after filtering
    
    truth = df.loc[mask, "fco2rec_uatm"].values
    mean = df.loc[mask, sample_cols].mean(axis=1).values
    print(f"fco2 climatology RMSE: {root_mean_squared_error(truth, df.loc[mask, 'co2_clim8d'])}") # climatology error (sanity check)
    
    #coverage
    low = df.loc[mask, sample_cols].min(axis=1).values
    high = df.loc[mask, sample_cols].max(axis=1).values
    coverage = (truth >= low) & (truth <= high)
    
    # RMSE
    rmse = root_mean_squared_error(truth, mean)
    # R2
    r2 = r2_score(truth, mean)
    # MAE
    mae = mean_absolute_error(truth, mean)
    # BIAS
    bias = (truth - mean).mean()
    
    # Central Coverage Calibration
    # S: (n_rows, n_samp) samples from your model. Replace df.values with your samples.
    S = df.loc[mask, sample_cols].values            # shape (n_rows, 50) in your real case
    y = truth                # shape (n_rows,)

    levels = np.arange(0.1, 1.0, 0.1)  # 10%,...,90% central coverage

    q_lo = np.quantile(S, (1 - levels)/2, axis=1).T   # shape (n_rows, len(levels))
    q_hi = np.quantile(S, 1 - (1 - levels)/2, axis=1).T

    covered = ((y[:, None] >= q_lo) & (y[:, None] <= q_hi)).mean(axis=0)  # empirical coverage
    cal_dict = {f"{int(100*l)}%": c for l, c in zip(levels, covered)}
    
    # CRPS
    # E|S - y|
    term1 = np.mean(np.abs(S - y[:, None]), axis=1)
    # 0.5 * E|S - S'|
    term2 = 0.5 * np.mean(np.abs(S[:, None, :] - S[:, :, None]), axis=(1,2))
    crps = term1 - term2                      # one score per row (lower is better)
    crps_mean = crps.mean()

    return dict({
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'coverage': coverage.mean(),
        'samples_std': df.loc[mask, sample_cols].std(axis=1).mean(),
        'avg_interval_width': (high-low).mean(),
        'avg_interval_width_std': (high-low).std(),
        'max_interval_width': (high-low).max(),
        'r2_interval_width_error': r2_score((truth-mean)**2, df.loc[mask, sample_cols].var(axis=1)),
        'corr_variance_error': pearsonr((truth-mean)**2, df.loc[mask, sample_cols].var(axis=1))[0],
        'ratio_error_interval_width': (mae/(high-low)).mean(),
        'calibration': cal_dict,
        'crps': crps_mean
    })

#val_err_stats = get_df_err_stats(df_val)
#train_err_stats = get_df_err_stats(df_train)
test_err_stats = get_df_err_stats(df_test)

print("Validation error statistics:")
#for key, value in val_err_stats.items():
#    print(f"{key}: {value:.4f}")
print("\nTraining error statistics:")
#for key, value in train_err_stats.items():
#    print(f"{key}: {value:.4f}")
print("\nTest error statistics:")
for key, value in test_err_stats.items():
    print(f"{key}: {value}")

# Save the results
results = {
    #'val_err_stats': val_err_stats,
    #'train_err_stats': train_err_stats,
    'test_err_stats': test_err_stats,
    'w': w,
    'n_rec':n_rec,
    'eta':eta
}

path_info = f"eta_{eta}_"
path = save_dir + path_info
with open(path+f'error_stats_{n_rec}recs.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save the dataframes
#val_samples_df.to_parquet(path + 'val_samples.pq')
#train_samples_df.to_parquet(path + 'train_samples.pq')
df_test.to_parquet(path + 'test_samples.pq')