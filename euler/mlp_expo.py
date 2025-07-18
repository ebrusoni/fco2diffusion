from utils import add_src_and_logger
save_dir = f'../models/sota_anoms/'
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

model_info = {
    'sota_ensemble': [save_dir, 'e_30.pt', MLPNaiveEnsemble],
}
models = load_models(model_info)


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
    dataloader = DataLoader(ds, batch_size=128, shuffle=False)
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
samples = get_samples_ensemble(df_val.copy(), models['sota_ensemble'])
params = models['sota_ensemble']['params']
samples = rescale(samples.reshape(-1, 1), params, params['mode']).reshape(20, -1)

mlp_pred_cols = [f'mlp_{i}' for i in range(samples.shape[0])]
df_val.loc[:, mlp_pred_cols] = samples.T

df_val.to_parquet(f'{save_dir}val_predictions.pq')