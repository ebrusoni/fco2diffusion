from utils import add_src_and_logger
save_dir = None
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)
import numpy as np
import pandas as pd
import healpy as hp
from fco2dataset.ucollocate import get_day_data, collocate, get_zarr_data
from fco2models.utraining import prep_df

def get_day_dataset(date):
    # get global satellite data for a given date
    dss = get_zarr_data(str(date.year))
    return dss

def collocate_coords(df, dss, date):
    save_path = '../data/inference/collocated'
    df_collocated = collocate(df, date, save_path=save_path, dss=dss, verbose=False)
    return df_collocated


# lat_grid = np.linspace(-90, 90, int((90 - -90) / 0.25) + 1)
# lon_grid = np.linspace(0, 360, int((360 - 0) / 0.25) + 1)
# print(len(lat_grid), len(lon_grid))
# grid_df = pd.DataFrame({
#     'lat': np.repeat(lat_grid, len(lon_grid)),
#     'lon': np.tile(lon_grid, len(lat_grid))
# })

nside = 2**10
npix = hp.nside2npix(nside)

m = np.arange(npix, dtype=np.int32)
lon, lat = hp.pix2ang(nside, m, lonlat=True, nest=True)

grid_df = pd.DataFrame({
    'lat': lat,
    'lon': lon
})

print(grid_df.shape)
print(grid_df.duplicated(subset=['lat', 'lon']).sum())
date = pd.Timestamp('2022-10-04')


dss = get_day_dataset(date)
df_collocated = collocate_coords(grid_df, dss, date)
print(df_collocated.duplicated(subset=['lat', 'lon']).sum())
df_collocated['time_1d'] = date
df_collocated = prep_df(df_collocated, with_target=False)[0]
df_collocated['sst_clim'] += 273.15
df_collocated['sst_anom'] = df_collocated['sst_cci'] - df_collocated['sst_clim']
df_collocated['sss_anom'] = df_collocated['sss_cci'] - df_collocated['sss_clim']
df_collocated['chl_anom'] = df_collocated['chl_globcolour'] - df_collocated['chl_clim']
df_collocated['ssh_anom'] = df_collocated['ssh_sla'] - df_collocated['ssh_clim']
df_collocated['mld_anom'] = np.log10(df_collocated['mld_dens_soda'] + 1e-5) - df_collocated['mld_clim']

df_collocated = df_collocated.fillna(df_collocated.mean())  # fill NaNs with mean of each column

print(df_collocated.duplicated(subset=['lat', 'lon']).sum())

from fco2models.ueval import load_models, print_loss_info
from fco2models.umeanest import MLPModel
from fco2models.models import MLPNaiveEnsemble

save_dir = '../models/sota_anoms64/'
model_path = 'e_20.pt'
model_name = 'sota_mlp'
model_info = {
    model_name: [save_dir, model_path, MLPNaiveEnsemble]
    }

models = load_models(model_info)
train_loss = models['sota_mlp']['losses']['train_losses']
val_loss = models['sota_mlp']['losses']['val_losses']
model_info = models['sota_mlp']

import numpy as np
from fco2models.umeanest import predict_mean_eval, get_loader_point_ds

params = model_info['params']
predictors = params['predictors']

train_stats = dict([
    ('means', params['train_means']),
    ('stds', params['train_stds']),
    ('mins', params['train_mins']),
    ('maxs', params['train_maxs'])
              ])

mode = params['mode']
model = model_info['model']
target = 'fco2rec_uatm'

cols = [target] + predictors
df_collocated[target] = np.nan  # Initialize target column
dataloader = get_loader_point_ds(df_collocated[cols], train_stats, mode, dropna=False)
_, preds = predict_mean_eval(model, dataloader, is_sequence=False)
print(preds.shape)

# remove last two axes and transpose
preds = preds[:, :, 0, 0].T
#rescale 
preds = (preds * train_stats["stds"][0]) + train_stats["means"][0]
# add back xco2
preds += df_collocated["xco2"].values[:, None]

sample_cols = [f"mlp_{i}" for i in range(20)]
df_collocated[sample_cols] = preds

df_collocated.to_parquet(f"{save_dir}global.pq")