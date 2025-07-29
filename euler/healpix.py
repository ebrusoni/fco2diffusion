from utils import add_src_and_logger
save_dir = f'../models/mean_std/'
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

import torch, platform, os
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version   :", torch.version.cuda)
print("GPU seen       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

import numpy as np
def segment_sample(data, orientation, segment_len=64, random_offset=0):
    padded_side, _, num_cols = data.shape
    side = padded_side - 2 * segment_len
    if orientation == 'vertical':
        data = np.transpose(data, (1, 0, 2))

    #padded_width = width + 2 * segment_len
    segments = np.zeros(( (side**2) // segment_len, num_cols, segment_len), dtype=np.float32)

    offset = segment_len + random_offset
    for i in range(num_cols):
        segments[:, i, :] = data[segment_len:-segment_len, offset:offset + side, i].reshape((side ** 2) // segment_len, segment_len)
    return segments

from fco2dataset.ucollocate import get_day_data, collocate
def get_day_dataset(date):
    # get global satellite data for a given date
    dss = get_day_data(date, save_path='../data/inference/gridded')
    return dss

def collocate_coords(df, dss, date):
    save_path = '../data/inference/collocated'
    df_collocated = collocate(df, date, save_path=save_path, dss=dss, verbose=False)
    return df_collocated

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

import torch
from tqdm import tqdm
import healpy as hp
import numpy as np
from fco2models.utraining import prep_df
import time


import torch
from tqdm import tqdm
import healpy as hp
import numpy as np
from fco2models.utraining import prep_df
import time


def segment_ds(data, orientation, segment_len=64):
    side, _, num_cols = data.shape

    if orientation == 'vertical':
        data = np.transpose(data, (1, 0, 2))
    elif not (orientation == 'horizontal'):
        raise ValueError(f"Unknown orientation {orientation}")
    segments = np.zeros(( (side**2) // segment_len, num_cols, segment_len), dtype=np.float32)
    for i in range(num_cols):
        segments[:, i, :] = data[:, :, i].reshape((side ** 2) // segment_len, segment_len)
    return segments


def fdf_to_numpy(df, cols, nside):
    fnpix = df.shape[0]
    fpix = np.arange(fnpix)
    xyf = hp.pix2xyf(nside, fpix, nest=True)
    x,y = xyf[0], xyf[1]
    fds = np.full((nside, nside, len(cols)), np.nan, dtype=np.float32)
    fds[x, y, :] = df.values
    assert np.isnan(df.values[:, 0]).sum() == 0, "NaN values found in dataframe" # there should be no NaN values in the sample column (every coordinate should have a sample)
    assert np.isnan(fds[:, :, 0]).sum() == 0, "NaN values found in fds" # there should be no NaN values in the sample column (every coordinate should have a sample)
    return fds, x, y

def do_step_loader(model, noise_scheduler, dataloader, t, device, jump):
    samples = []
    for (ix, batch) in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            sample_prev = batch[:, 0:1, :].to(device)
            #sample_prev = noise_scheduler.add_noise(sample_prev, torch.randn_like(sample_prev), t)
            noise_pred = model(batch, t, return_dict=False)[0]
            x_0 = noise_scheduler.step(noise_pred, t, sample_prev).pred_original_sample
            if jump is None:
                sample = noise_scheduler.step(noise_pred, t, sample_prev).prev_sample
            elif t - jump > 0:
                #sample = x_0 
                sample = noise_scheduler.add_noise(x_0, torch.randn_like(sample_prev), t - jump)
            else:
                sample = x_0
                #sample = noise_scheduler.step(noise_pred, t, sample_prev).prev_sample
            sample[torch.isnan(sample)] = sample_prev[torch.isnan(sample)]
            samples.append(sample.cpu().numpy())
    return np.concatenate(samples, axis=0)

def do_random_rot(lon, lat):
        random_zrotation = np.random.random() * 360
        #random_yrotation = np.random.random() * 20
        #random_xrotation = np.random.random() * 20
        rot = hp.Rotator(rot=(random_zrotation, 0, 0), eulertype="ZYX") # rotate only around the z-axis, so we do not mix equatirial and polar coordinates
        lon_rot, lat_rot = rot(lon, lat, lonlat=True)
        m_rot = hp.ang2pix(nside, lon_rot, lat_rot, nest=True, lonlat=True)
        return m_rot


def infer_patch_with_rotations(model, noise_scheduler, params, date, nside=1024, jump=None, n_samples=1):

    npix = hp.nside2npix(nside)
    print(f"Number of pixels: {npix}")
    m = np.arange(npix)
    lon, lat = hp.pix2ang(nside, m, nest=True, lonlat=True)
    sample = np.random.randn(npix, 1).astype(np.float32)
    dss = get_day_dataset(date)
    context_df = collocate_coords(pd.DataFrame({
        'lon': lon.flatten(),
        'lat': lat.flatten(),
        'time_1d': date,
        }), dss, date)
    print(f"Collocated data shape: {context_df.shape}")
    
    context_df['lon'] = (context_df['lon'] + 180) % 360 - 180
    df = prep_df(context_df, with_target=False, with_log=True)[0]
    df['sst_clim'] += 273.15
    df['sst_anom'] = df['sst_cci'] - df['sst_clim']
    df['sss_anom'] = df['sss_cci'] - df['sss_clim']
    df['chl_anom'] = df['chl_globcolour'] - df['chl_clim']
    df['ssh_anom'] = df['ssh_sla'] - df['ssh_clim']
    df['mld_anom'] = np.log10(df['mld_dens_soda'] + 1e-5) - df['mld_clim']
    context_df = df.loc[:, params['predictors']]
    stats = {
        'means': params['train_means'],
        'stds': params['train_stds'],
        'mins': params['train_mins'],
        'maxs': params['train_maxs']
    }
    context_df = normalize(context_df,stats, params['mode'])
    context_df = context_df.fillna(context_df.mean())
    print(f"Preprocessed data shape: {context_df.shape}")
    
    # add sample and metadata
    context_df['healpix_id'] = m
    context_df['ring_id'] = hp.nest2ring(nside, m)
    context_df['m_rotated'] = m  # placeholder for rotated healpix id
    context_df.set_index('healpix_id', inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #t_loop_jump = noise_scheduler.timesteps[::jump][:-1]
    #t_loop_full = torch.arange(jump-1, 0, -1)
    #print(f"t_loop_jump: {t_loop_jump}, t_loop_full: {t_loop_full}")
    #t_loop = tqdm(torch.cat((t_loop_jump, t_loop_full)), desc="Denoising steps")
    for i in range(n_samples):
        #cols = [f'sample_{i}'] + params['predictors']
        context_df[f'sample_{i}'] = np.random.randn(npix).astype(np.float32)
    
    for i in range(n_samples):
        cols = [f'sample_{i}'] + params['predictors']
        t_loop = tqdm(noise_scheduler.timesteps[::jump], desc="Denoising steps")
        step = 0
        for t in t_loop:
            context_df['m_rotated'] = do_random_rot(lon, lat)
            #context_df[f'sample_{i}'] = np.random.randn(npix).astype(np.float32)
            if step % 3 == 0:
                #print("ring")
                ring = context_df.loc[:, 'ring_id']
                context_df.reset_index(inplace=True)
                context_df.set_index('ring_id', inplace=True)
                context_df.sort_index(inplace=True)
                df = context_df.loc[:, cols]
                segments = df.values.reshape(npix//64, 64, len(cols))
                segments = np.transpose(segments, (0, 2, 1))
                
                ds = torch.from_numpy(segments).float()
                ds = torch.cat((ds, torch.ones_like(ds[:, 0:1, :])), axis=1)
                dataloader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=False, num_workers=2)

                samples = do_step_loader(model, noise_scheduler, dataloader, t, device, jump)
                context_df.loc[:, f'sample_{i}'] = samples.flatten()
                context_df.reset_index(inplace=True)
                context_df.set_index('healpix_id', inplace=True)
                step+=1
                continue
                
            #print("nested")
            for face in range(12):
                fnpix = npix // 12 
                fpixs = np.arange(face * fnpix, (face + 1) * fnpix)
                fpixs_rotated = context_df.loc[fpixs, 'm_rotated']
                
                df = context_df.loc[fpixs_rotated, cols]
                fds, x, y = fdf_to_numpy(df, cols, nside)

                if step % 2 == 0:
                    segments = segment_ds(fds, 'horizontal', segment_len=64)
                else:
                    segments = segment_ds(fds, 'vertical', segment_len=64)
                
                #segments = np.concatenate((hsegments, vsegments), axis=0)
                ds = torch.from_numpy(segments).float()
                ds = torch.cat((ds, torch.ones_like(ds[:, 0:1, :])), axis=1)
                dataloader = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=False, num_workers=2)

                samples = do_step_loader(model, noise_scheduler, dataloader, t, device, jump)
                
                #hsamples = samples[:samples.shape[0] // 2] 
                #vsamples = samples[samples.shape[0] // 2:] 
                
                samples = samples.reshape(nside, nside)
                #hsamples = hsamples.reshape(nside, nside)
                #vsamples = vsamples.reshape(nside, nside)
                context_df.loc[fpixs_rotated, f'sample_{i}'] = samples[x, y] if step % 2 == 0 else samples[y, x]
                #context_df.loc[fpixs_rotated, f'sample_{i}'] = (hsamples[x, y] + vsamples[y, x]) / 2.0
                t_loop.set_postfix({"face": {face}, "sample": {i}})
            step += 1
    return context_df


import pandas as pd
from fco2models.models import UNet2DModelWrapper, Unet2DClassifierFreeModel, UNet2DShipMix
from fco2models.ueval import load_model
# load baseline model
save_path = '../models/mean_std/'
model_path = 'e_200.pt'
model_class = UNet2DModelWrapper
model, noise_scheduler, params, losses = load_model(save_path, model_path, model_class,training_complete=True)
print("model loaded")
print("predictors:", params['predictors'])
date = pd.Timestamp('2022-10-04')
nside = 2**10

from diffusers import DDIMScheduler
#model.set_w(1)
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
    beta_schedule=noise_scheduler.config.beta_schedule,
    clip_sample_range=noise_scheduler.config.clip_sample_range,
    #timestep_spacing="trailing"
    )
ddim_scheduler.set_timesteps(50)
n=1
df = infer_patch_with_rotations(model, ddim_scheduler, params, date, nside=nside, jump=None, n_samples=n)
sample_cols = [f"sample_{i}" for i in range(n)]
df['xco2'] = df['xco2'] * params['train_stds'][6] + params['train_means'][6]
df[sample_cols] =df[sample_cols] * params['train_stds'][0] + params['train_means'][0] + df['xco2'].values[:, np.newaxis]
df.to_parquet(f'{save_path}ring.pq')