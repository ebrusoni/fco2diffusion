from utils import add_src_and_logger
#save_dir = f'../models/mean_std/'
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

import torch
import numpy as np
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version   :", torch.version.cuda)
print("GPU seen       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

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

import torch
from tqdm import tqdm
import healpy as hp
import numpy as np
from fco2models.utraining import prep_df
def segment_ds(data, orientation, segment_len=64):
    _, side, num_cols = data.shape

    if orientation == 'vertical':
        data = np.transpose(data, (1, 0, 2))
    elif not (orientation == 'horizontal'):
        raise ValueError(f"Unknown orientation {orientation}")
    #segments = np.zeros(((side**2) // segment_len, num_cols, segment_len), dtype=np.float32)
    #for i in range(num_cols):
        #segments[:, i, :] = data[:, :, i].reshape((side ** 2) // segment_len, segment_len)
    data = data.reshape((side ** 2) // segment_len, segment_len, -1)
    return np.swapaxes(data, 1, 2)


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

def do_random_rot(lon, lat, nest=True):
        random_zrotation = np.random.random() * 360
        #random_yrotation = np.random.random() * 20
        #random_xrotation = np.random.random() * 20
        rot = hp.Rotator(rot=(random_zrotation, 0, 0), eulertype="ZYX") # rotate only around the z-axis, so we do not mix equatirial and polar coordinates
        lon_rot, lat_rot = rot(lon, lat, lonlat=True)
        m_rot = hp.ang2pix(nside, lon_rot, lat_rot, nest=nest, lonlat=True)
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
    ring = hp.nest2ring(nside, m)
    context_df['m_rotated'] = m  # placeholder for rotated healpix id
    context_df.set_index('healpix_id', inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for i in range(n_samples):
        context_df[f'sample_{i}'] = np.random.randn(npix).astype(np.float32)
    
    sample_cols = [f'sample_{i}' for i in range(n_samples)]
    cds = context_df[sample_cols + params['predictors']].values
    for i in range(n_samples):
        cols = [f'sample_{i}'] + params['predictors']
        colixs = np.array([i] + list(range(n_samples, n_samples + len(params['predictors']))))
        t_loop = tqdm(noise_scheduler.timesteps[::jump], desc="Denoising steps")
        step = 0
        for t in t_loop:
            #m_rot = do_random_rot(lon, lat)
            #context_df[f'sample_{i}'] = np.random.randn(npix).astype(np.float32)
            if True:#step % 3 == 0:
                order = np.argsort(ring)
                ring_rot = do_random_rot(lon[order], lat[order], nest=False)
                #cds_ring = cds[order] #WHAT THE HELL
                #print(cds.shape)
                #print(cds_ring.shape)
                df = cds[order, colixs[:, np.newaxis]].T
                df = df[ring_rot]
                segments = df.reshape(npix//64, 64, len(cols))
                segments = np.transpose(segments, (0, 2, 1))
                
                ds = torch.from_numpy(segments).float()
                ds = torch.cat((ds, torch.ones_like(ds[:, 0:1, :])), axis=1)
                dataloader = torch.utils.data.DataLoader(ds, batch_size=8096, shuffle=False, num_workers=2)

                samples = do_step_loader(model, noise_scheduler, dataloader, t, device, jump)
                #cds[:, i] = samples.flatten()[np.argsort(order)]
                tmp = np.empty(npix)
                tmp[ring_rot] = samples.flatten()
                cds[:, i] = tmp[np.argsort(order)]
                step+=1
                continue
            m_rot = do_random_rot(lon, lat)  # get rotated healpix indices
            # --- pre-computed constants -------------------------------------------------
            fnpix  = npix // 12           # pixels per face     = nside ** 2
            faces  = np.arange(12)        # [0 … 11]
            x, y   = hp.pix2xyf(nside, np.arange(fnpix), nest=True)[:2]   # shape (fnpix,)
            cols_n = len(cols)            # number of feature columns
            
            # --- 1. gather the data for *all* faces at once -----------------------------
            # pixels of each face before and after rotation
            fpix_faces      = faces[:, None] * fnpix + np.arange(fnpix)          # (12, fnpix)
            fpix_rot_faces  = m_rot[fpix_faces]                                  # (12, fnpix)
            
            # df_all: (12, fnpix, cols_n)
            df_all = cds[fpix_rot_faces][:, :, colixs]
            
            # --- 2. put every face into its own (nside × nside × cols) cube -------------
            # create empty array (12, nside, nside, cols_n)
            fds_all = np.full((12, nside, nside, cols_n), np.nan, dtype=np.float32)
            
            # broadcasted assignment
            fds_all[
                np.repeat(faces, fnpix),          # 12 × fnpix
                np.tile(x,    12),                # 12 × fnpix
                np.tile(y,    12)                 # 12 × fnpix
            ] = df_all.reshape(-1, cols_n)        # flatten face & pixel axes
            
            # --- 3. segment all faces in one go -----------------------------------------
            if step % 2:                          # vertical
                fds_all = fds_all.swapaxes(1, 2)  # swap x ↔ y for every face
            
            side            = nside
            n_total_pixels  = 12 * side * side
            n_segments      = n_total_pixels // 64
            
            segments = (fds_all
                        .reshape(n_total_pixels, cols_n)        # flatten to 2-D
                        .reshape(n_segments, 64, cols_n)        # cut into 64-px strips
                        .transpose(0, 2, 1)                     # (seg, col, 64)
                        .astype(np.float32))                    # final dtype
            
            # --- 4. push through the model once -----------------------------------------
            ds = torch.from_numpy(segments)
            ds = torch.cat((ds, torch.ones_like(ds[:, :1, :])), dim=1)   # add the "ones" channel
            loader = torch.utils.data.DataLoader(ds, batch_size=8096, shuffle=False, num_workers=2)
            
            samples = do_step_loader(model, noise_scheduler, loader, t, device, jump)
            # samples now has length = n_total_pixels
            samples = samples.reshape(12, side, side)          # (face, y, x)
            
            # undo the earlier swap if necessary
            #if step % 2:
                #samples = samples.transpose(1, 2)              # back to original orientation
            
            # --- 5. write every prediction back into cds in one vectorised hit ----------
            cds[fpix_rot_faces.reshape(-1), i] = samples[
                np.repeat(faces, fnpix),                       # face index per pixel
                np.tile(x if step % 2 == 0 else y, 12),        # y-coord chosen as in original
                np.tile(y if step % 2 == 0 else x, 12)         # x-coord chosen as in original
            ]
            step+=1
    context_df[sample_cols] = cds[:, :n_samples]
    return context_df


import pandas as pd
from fco2models.models import UNet2DModelWrapper, Unet2DClassifierFreeModel, UNet2DShipMix, UNet1DModelWrapper
from fco2models.ueval import load_model
# load baseline model
save_path = '../models/anoms_sea_1d/'
model_path = 'e_200.pt'
model_class = UNet1DModelWrapper
model, noise_scheduler, params, losses = load_model(save_path, model_path, model_class,training_complete=True)
print("model loaded")
print("predictors:", params['predictors'])
date = pd.Timestamp('2022-10-04')
nside = 2**7

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
df['xco2'] = df['xco2'] * params['train_stds'][11] + params['train_means'][11]
df[sample_cols] =df[sample_cols] * params['train_stds'][0] + params['train_means'][0] + df['xco2'].values[:, np.newaxis]
df.to_parquet(f'{save_path}test.pq')