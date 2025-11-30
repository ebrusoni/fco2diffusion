from utils import add_src_and_logger
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from fco2models.models import MLPNaiveEnsemble
from fco2models.ueval import load_models
from fco2models.utraining import prep_df, make_monthly_split

save_dir = f'../models/sota_anoms64_big/'
model_info = {
    'sota_ensemble': [save_dir, 'e_20.pt', MLPNaiveEnsemble],
}
models = load_models(model_info)


df = pd.read_parquet(DATA_PATH + "SOCAT_1982_2021_grouped_colloc_augm_bin.pq", engine='pyarrow')
df = prep_df(df, bound=True)[0]
_, mask_val, mask_test = make_monthly_split(df)
df_val = df.loc[df.expocode.map(mask_val), :]
df_test = df.loc[df.expocode.map(mask_test), :]

def normalize(df, stats, mode):
    for i in range(len(stats['means']) - 1): # first column is the target
        col = df.columns[i]
        if mode == 'min_max':
            df[col] = 2 * (df[col] - stats['mins'][i + 1]) / (stats['maxs'][i + 1] - stats['mins'][i + 1]) - 1
        elif mode == 'mean_std':
            df[col] = (df[col] - stats['means'][i + 1]) / stats['stds'][i + 1]
        else:
            raise ValueError(f"Unknown mode {mode}")
    return df

def get_samples_ensemble(df, model_info):
    params = model_info['params']
    predictors = params['predictors']
    stats = {
        'means': params['train_means'],
        'stds': params['train_stds'],
        'mins': params['train_mins'],
        'maxs': params['train_maxs'],
    }
    df.loc[:, predictors] = normalize(df.loc[:, predictors], stats, params['mode'])

    model = model_info['model']
    ds = torch.from_numpy(df.loc[:, predictors].values).float()
    print(ds.shape)
    ds = TensorDataset(ds.unsqueeze(-1))
    dataloader = DataLoader(ds, batch_size=2048, shuffle=False)
    samples = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # _, samples = predict_mean_eval(model, dataloader)
    for batch in tqdm(dataloader, desc="Ensemble sampling"):
        batch = batch[0].to(device)
        with torch.no_grad():
            sample = model(batch, 0, return_dict=False)[0]
        samples.append(sample.cpu().numpy())
    samples = np.concatenate(samples, axis=1)

    print(f"Samples shape: {samples.shape}")
    return samples 

from fco2models.ueval import rescale
samples = get_samples_ensemble(df_test.copy(), models['sota_ensemble'])

params = models['sota_ensemble']['params']
ensemble_size = models['sota_ensemble']['model'].E
samples = rescale(samples.reshape(-1, 1), params, params['mode']).reshape(ensemble_size, -1).T
samples += df_test['xco2'].values[:, None]
df_test.loc[:, 'fco2rec_uatm'] += df_test.loc[:, 'xco2'] # remove xco2 offset

mlp_pred_cols = [f'mlp_{i}' for i in range(samples.shape[1])]
df_test.loc[:, mlp_pred_cols] = samples

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
import numpy as np
def get_df_err_stats(df, sample_cols):
    seamask = df.seamask.astype(bool)
    fco2_nans = ~df.fco2rec_uatm.isna()
    pred_nans = ~df.mlp_0.isna()
    mask = seamask & fco2_nans & pred_nans
    print(f"number of samples: {mask.sum()}")
    
    truth = df.loc[mask, "fco2rec_uatm"].values
    mean = df.loc[mask, sample_cols].mean(axis=1).values
    
    low = df.loc[mask, sample_cols].min(axis=1).values
    high = df.loc[mask, sample_cols].max(axis=1).values
    coverage = (truth >= low) & (truth <= high)
    
    rmse = root_mean_squared_error(truth, mean)
    r2 = r2_score(truth, mean)
    mae = mean_absolute_error(truth, mean)
    bias = (truth - mean).mean()
    
    # S: (n_rows, n_samp) samples from your model. Replace df.values with your samples.
    S = df.loc[mask, sample_cols].values            # shape (n_rows, 50) in your real case
    y = truth                # shape (n_rows,)

    levels = np.arange(0.1, 1.0, 0.1)  # 10%,...,90% central coverage

    q_lo = np.quantile(S, (1 - levels)/2, axis=1).T   # shape (n_rows, len(levels))
    q_hi = np.quantile(S, 1 - (1 - levels)/2, axis=1).T

    covered = ((y[:, None] >= q_lo) & (y[:, None] <= q_hi)).mean(axis=0)  # empirical coverage
    cal_dict = {f"{int(100*l)}%": c for l, c in zip(levels, covered)}
    
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

dm_preds = pd.read_parquet("../models/anoms_sea_1d/eta_0_test_samples.pq")
# load mask selecting predictions made also by the diffusion model to ensure fairness
pred_mask = pd.read_parquet("../models/anoms_sea_1d/pred_nans.pq")

# match variances
dm_samples = [f"sample_{i}" for i in range(50)]
# check same ordering for both dataframes
assert np.allclose(dm_preds.fco2rec_uatm.values, df_test.fco2rec_uatm.values, equal_nan=True)
dm_std = dm_preds.loc[pred_mask.values, dm_samples].std(axis=1).mean()
mlp_std = df_test.loc[pred_mask.values, mlp_pred_cols].std(axis=1).mean()
print(f"diffusion model mean std: {dm_std}")
print(f"MLP model mean std: {mlp_std}")
scale_const = dm_std/mlp_std # scaling constant to match stds

mlp_mean = (scale_const - 1) * df_test.loc[pred_mask.values, mlp_pred_cols].mean(axis=1) # needed so that samples have the same mean estimates as before
mlp_pred_cols_scaled = ["scaled_"+col for col in mlp_pred_cols] 
df_test[mlp_pred_cols_scaled] = (scale_const * df_test.loc[pred_mask.values, mlp_pred_cols]) - mlp_mean.values[:, None]


err = get_df_err_stats(df_test.loc[pred_mask.values], mlp_pred_cols_scaled) # get error_statistics
for k in err: 
    print(f"{k}: {err[k]}")

df_test.to_parquet(f'{save_dir}test_predictions.pq')