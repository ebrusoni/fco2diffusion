from utils import add_src_and_logger
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)

import torch
import numpy as np
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version   :", torch.version.cuda)
print("GPU seen       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

from fco2dataset.ucollocate import get_day_data, collocate, get_zarr_data
def get_day_dataset(date):
    dss = get_zarr_data(str(date.year))
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
        data = np.transpose(data, (1, 0, 2)) # transpose for vertical segmentation
    elif not (orientation == 'horizontal'):
        raise ValueError(f"Unknown orientation {orientation}")
    data = data.reshape((side ** 2) // segment_len, segment_len, -1) # divide dataset into segments (n_side is a power of 2, so it is always divisible by segment_len)
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

def do_step_loader(model, noise_scheduler, dataloader, t, device, jump, eta):
    samples = []
    for (ix, batch) in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            sample_prev = batch[:, 0:1, :].to(device)
            noise_pred = model(batch, t, return_dict=False)[0]
            x_0 = noise_scheduler.step(noise_pred, t, sample_prev).pred_original_sample
            if jump is None:
                sample = noise_scheduler.step(noise_pred, t, sample_prev, eta=eta).prev_sample
            elif t - jump > 0:
                #sample = x_0 
                sample = noise_scheduler.add_noise(x_0, torch.randn_like(sample_prev), t - jump)
            else:
                sample = x_0
            sample[torch.isnan(sample)] = sample_prev[torch.isnan(sample)] # fill possible Nans with values from previous iteration to avoid propagation
            samples.append(sample)
    return torch.cat(samples, axis=0)

def do_random_rot(lon, lat, nest=True):
        random_zrotation = np.random.random() * 360
        #random_yrotation = np.random.random() * 5
        #random_xrotation = np.random.random() * 5
        # rotate only around the z-axis, so we do not mix equatirial and polar coordinates
        rot = hp.Rotator(rot=(random_zrotation, 0, 0), eulertype="ZYX")
        # rotate canonical coordinates (coordinates of unrotated pixels)
        lon_rot, lat_rot = rot(lon, lat, lonlat=True)
        # calculate canonical pixel indices corresponding to rotated coordinates (if we rotate around z-axis mapping is onr-to-one)
        m_rot = hp.ang2pix(nside, lon_rot, lat_rot, nest=nest, lonlat=True) 
        return m_rot


import time 
def infer_patch_with_rotations(model, noise_scheduler, params, date, nside=1024, jump=None, n_samples=1):

    npix = hp.nside2npix(nside)
    print(f"Number of pixels: {npix}")
    m = np.arange(npix)
    lon, lat = hp.pix2ang(nside, m, nest=True, lonlat=True)
    sample = np.random.randn(npix, 1).astype(np.float32)
    start = time.time()
    dss = get_day_dataset(date)
    end = time.time()
    print(f"download time: {end-start}")
    start = time.time()
    context_df = collocate_coords(pd.DataFrame({
        'lon': lon.flatten(),
        'lat': lat.flatten(),
        'time_1d': date,
        }), dss, date)
    end = time.time()
    print(f"collocation time: {end-start}")
    print(f"Collocated data shape: {context_df.shape}")
    
    context_df['lon'] = (context_df['lon'] + 180) % 360 - 180
    df = prep_df(context_df, with_target=False, with_log=True)[0]
    df['sst_clim'] += 273.15
    df['sst_anom'] = df['sst_cci'] - df['sst_clim']
    df['sss_anom'] = df['sss_cci'] - df['sss_clim']
    df['chl_anom'] = df['chl_globcolour'] - df['chl_clim']
    df['ssh_anom'] = df['ssh_sla'] - df['ssh_clim']
    df['mld_anom'] = np.log10(df['mld_dens_soda'] + 1e-5) - df['mld_clim']
    context_df = df.loc[:, params['predictors']+['seamask']]
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
    cds_all = context_df[sample_cols + params['predictors']].values
    for i in range(n_samples):
        cols = [f'sample_{i}'] + params['predictors'] + ['seamask']
        colixs = np.array([i] + list(range(n_samples, n_samples + len(params['predictors']))))
        cds = cds_all[:, colixs]
        print(f"cds shape: {cds.shape}")
        t_loop = tqdm(noise_scheduler.timesteps[::jump], desc="Denoising steps")
        step = 0
        for t in t_loop:
            #m_rot = do_random_rot(lon, lat)
            #context_df[f'sample_{i}'] = np.random.randn(npix).astype(np.float32)
            if step % 3 == 0:
                order = np.argsort(ring)
                ring_rot = do_random_rot(lon[order], lat[order], nest=False)
                #cds_ring = cds[order] #WHAT THE HELL
                #print(cds.shape)
                #print(cds_ring.shape)
                df = cds[order, :]
                #print(f"df shape{df.shape}")
                df = df[ring_rot]
                segments = df.reshape(npix//64, 64, len(cols))
                segments = np.transpose(segments, (0, 2, 1))
                
                ds = torch.from_numpy(segments).float()
                ds = torch.cat((ds, torch.ones_like(ds[:, 0:1, :])), axis=1)
                dataloader = torch.utils.data.DataLoader(ds, batch_size=8096, shuffle=False, num_workers=2)

                samples = do_step_loader(model, noise_scheduler, dataloader, t, device, jump, eta=1.0)
                #cds[:, i] = samples.flatten()[np.argsort(order)]
                tmp = np.empty(npix)
                tmp[ring_rot] = samples.flatten()
                cds[:, 0] = tmp[np.argsort(order)]
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
            df_all = cds[fpix_rot_faces][:, :, :]
            
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
            
            samples = do_step_loader(model, noise_scheduler, loader, t, device, jump, eta=1.0)
            # samples now has length = n_total_pixels
            samples = samples.reshape(12, side, side)          # (face, y, x)
            
            # undo the earlier swap if necessary
            #if step % 2:
                #samples = samples.transpose(1, 2)              # back to original orientation
            
            # --- 5. write every prediction back into cds in one vectorised hit ----------
            cds[fpix_rot_faces.reshape(-1), 0] = samples[
                np.repeat(faces, fnpix),                       # face index per pixel
                np.tile(x if step % 2 == 0 else y, 12),        # y-coord chosen as in original
                np.tile(y if step % 2 == 0 else x, 12)         # x-coord chosen as in original
            ]
            step+=1
        cds_all[:, i] = cds[:, 0]
    context_df[sample_cols] = cds_all[:, :n_samples]
    return context_df

import time 
import time
import numpy as np
import pandas as pd
import torch
import healpy as hp
from tqdm import tqdm


def infer_patch_with_rotations_gpu(
    model,
    ddim_scheduler,
    params,
    date,
    *,
    nside: int = 1024,
    jump: int | None = None,
    n_samples: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,   # change to float32 for reproducibility
):
    """GPU-resident rewrite of the original function (logic unchanged). Written mostly with chatGPT"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ────────────────────────── 0. metadata & DataFrame build ─────────────────────────
    npix = hp.nside2npix(nside) #total number of pixels: 12*n_side**2
    m    = np.arange(npix) # pixel ids
    lon, lat = hp.pix2ang(nside, m, nest=True, lonlat=True) # calculate canonical coordinates

    start = time.time()
    dss   = get_day_dataset(date) # get dataset for specified date
    print(f"download time: {time.time() - start:.1f}s")

    start = time.time()
    # collocate remote sensing data to coordinates of each pixel
    context_df = collocate_coords(
        pd.DataFrame({"lon": lon.flatten(), "lat": lat.flatten(), "time_1d": date}),
        dss,
        date,
    )
    print(f"collocation time: {time.time() - start:.1f}s")
    print(f"Collocated data shape: {context_df.shape}")

    context_df["lon"] = (context_df["lon"] + 180) % 360 - 180 # normalize lon to [-180, 180]
    # do the same preprocessing as in training (add positional/temporal encodings, sea mask, climatologies)
    df = prep_df(context_df, with_target=False, with_log=True)[0]
    df["sst_clim"] += 273.15
    df["sst_anom"]  = df["sst_cci"] - df["sst_clim"]
    df["sss_anom"]  = df["sss_cci"] - df["sss_clim"]
    df["chl_anom"]  = df["chl_globcolour"] - df["chl_clim"]
    df["ssh_anom"]  = df["ssh_sla"] - df["ssh_clim"]
    df["mld_anom"]  = np.log10(df["mld_dens_soda"] + 1e-5) - df["mld_clim"] # mld_clim is in log scale

    # conditioning provided by predictors from training plus marginal sea mask
    context_df = df.loc[:, params["predictors"]+["seamask"]] 
    stats      = {
        "means": params["train_means"],
        "stds":  params["train_stds"],
        "mins":  params["train_mins"],
        "maxs":  params["train_maxs"],
    }
    context_df = normalize(context_df, stats, params["mode"]) # normalize with training statistics
    context_df = context_df.fillna(context_df.mean()) # impute all nans to avoid invalid predictions
    print(f"Preprocessed data shape: {context_df.shape}")

    # add metadata columns (still CPU)
    context_df["healpix_id"] = m
    ring = hp.nest2ring(nside, m)  # map pixel ids to ring order for segmentation in ring order
    context_df["m_rotated"] = m # placeholder for now
    context_df.set_index("healpix_id", inplace=True)

    # ────────────────────────── 1. tensors to GPU ─────────────────────────────────────
    
    # tensort containing predictor columns
    pred_t = torch.as_tensor(
        context_df[params["predictors"]+["seamask"]].values, dtype=dtype, device=device
    )                                                          # (npix, n_pred)

    # sample columns initialised with N(0,1)
    samples_t = torch.randn((npix, n_samples), dtype=dtype, device=device)
    # column set [samples | predictors]
    cds_all_t = torch.cat((samples_t, pred_t), dim=1)          # (npix, n_s+n_pred)

    # coordinates and ring index as tensors
    lon_t   = torch.as_tensor(lon,  dtype=dtype, device=device)
    lat_t   = torch.as_tensor(lat,  dtype=dtype, device=device)
    ring_t  = torch.as_tensor(ring, device=device, dtype=torch.long)

    # Healpix helpers (static)
    fnpix   = npix // 12
    faces_t = torch.arange(12, device=device, dtype=torch.long)
    # x,y coordinates for each pixel for every face (they are the same for all faces)
    x_np, y_np = hp.pix2xyf(nside, np.arange(fnpix), nest=True)[:2] 
    x_t = torch.as_tensor(x_np, device=device, dtype=torch.long)
    y_t = torch.as_tensor(y_np, device=device, dtype=torch.long)

    # ────────────────────────── 2. per-sample diffusion ──────────────────────────────
    
    # initialize scheduler for refining the last few steps
    last_scheduler = DDIMScheduler(
        num_train_timesteps=ddim_scheduler.config.num_train_timesteps,
        beta_schedule=ddim_scheduler.config.beta_schedule,
        clip_sample_range=ddim_scheduler.config.clip_sample_range,
        )
    for i in range(n_samples):
        # select columns from cds_all_t relevant for sample i
        cols_this  = [f"sample_{i}"] + params["predictors"] + ["seamask"]
        colixs     = torch.tensor(
            [i] + list(range(n_samples, n_samples + len(params["predictors"]) + 1)),
            device=device,
            dtype=torch.long,
        )

        cds_t = cds_all_t[:, colixs]                           # (npix, 1+n_pred)
        
        noise_lvl = 40 # noise level where refinement will begin 
        step=-2 # small step size to smooth out boundaries nicely
        noise_scheduler = ddim_scheduler # start with regular scheduler
        t_loop = torch.cat([noise_scheduler.timesteps[::jump], torch.arange(noise_lvl, -1, step).to(device)])
        last_step = len(noise_scheduler.timesteps[::jump]) - 1 # number of denoising steps before refinement
        last_scheduler.set_timesteps(noise_lvl // step, steps=torch.arange(noise_lvl, -1, step), device=device)
        for step_no, t in enumerate(
            tqdm(t_loop, desc=f"sample {i}")
        ):
            ring_mode = (step_no % 3 == 0)
            vert_mode = (step_no % 3 == 2)

            if ring_mode:
                # ───── ring segmentation ──────────────────────────────────────────
                order_t  = torch.argsort(ring_t) # map from nested to ring order
                # rotate pixels in ring order
                ring_rot = torch.as_tensor(
                    do_random_rot(lon[order_t.cpu()], lat[order_t.cpu()], nest=False),
                    device=device,
                    dtype=torch.long,
                )
                
                # map to ring order and rotate
                df_t = cds_t[order_t][ring_rot]                       # (npix, cols)
                # segment reordered and rotated dataset 
                segments_t = (df_t.view(npix // 64, 64, len(cols_this))
                                   .permute(0, 2, 1)
                                   .contiguous())
                
                ds = torch.cat(
                        (segments_t, ),#torch.ones_like(segments_t[:, :1, :])),
                        dim=1,
                     )
                
                # denoise one timestep
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=8096, shuffle=False, num_workers=0
                )

                preds = do_step_loader(model, noise_scheduler, loader, t, device, jump, eta=0.0)
                preds = preds.reshape(npix)
                
                # assign back in correct order
                tmp = torch.empty(npix, dtype=dtype, device=device)
                tmp[ring_rot] = preds.to(dtype)
                cds_t[:, 0]   = tmp[torch.argsort(order_t)]

            else:
                # ───── face segmentation ─────────────────────────────────────────
                
                # rotate all pixels
                m_rot = torch.as_tensor(
                    do_random_rot(lon, lat), device=device, dtype=torch.long
                )
                
                # get pixel ids for all faces in canonical order
                fpix_faces     = (faces_t[:, None] * fnpix +
                                  torch.arange(fnpix, device=device))          # (12,f)
                # rotate the pixels
                fpix_rot_faces = m_rot[fpix_faces]                             # (12,f)
                
                # get the data associated with the roated positions
                df_all_t = cds_t[fpix_rot_faces]                               # (12,f,cols)
                
                # create new tensor with face dimension (12) to index by faces more easily
                fds_all_t = torch.full(
                    (12, nside, nside, len(cols_this)),
                    float("nan"),
                    dtype=dtype,
                    device=device,
                )
                
                # assign it data
                fds_all_t[
                    faces_t.repeat_interleave(fnpix),
                    x_t.repeat(12),
                    y_t.repeat(12),
                ] = df_all_t.reshape(-1, len(cols_this))
                
                # swap axes if segmentation is vertical
                if vert_mode:
                    fds_all_t = fds_all_t.swapaxes(1, 2)      # swap x ↔ y

                n_segments = (12 * nside * nside) // 64
                # reshape faces (which are always powers of 2, so divisble by 64) and permute dimensions to match UNet input shape
                segments_t = (fds_all_t.reshape(-1, len(cols_this))
                                      .view(n_segments, 64, len(cols_this))
                                      .permute(0, 2, 1)
                                      .contiguous())

                ds = torch.cat(
                        (segments_t, ),#torch.ones_like(segments_t[:, :1, :])),
                        dim=1,
                     )
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=8096, shuffle=False, num_workers=0
                )

                preds = do_step_loader(model, noise_scheduler, loader, t, device, jump,eta=0.0)
                preds = preds.reshape(12, nside, nside)          # (face, y, x)
                
                # invert dimensions if needed
                y_idx = x_t if vert_mode else y_t
                x_idx = y_t if vert_mode else x_t
                
                # assign in original shape (without face dimension)
                out_vals = preds[
                    faces_t.repeat_interleave(fnpix),
                    x_idx.repeat(12),
                    y_idx.repeat(12),
                ]
                # assign in rotated order
                cds_t[fpix_rot_faces.reshape(-1), 0] = out_vals
                
                # if we reach the last step of the first inference round, add noise to the image and switch schedulers
                if step_no == last_step:
                    cds_t[:, 0] = noise_scheduler.add_noise(cds_t[:, 0], torch.randn_like(cds_t[:, 0]).to(device), torch.tensor(noise_lvl).to(device))
                    noise_scheduler = last_scheduler

        cds_all_t[:, i] = cds_t[:, 0]

    # ────────────────────────── 3. back to DataFrame ─────────────────────────────────
    sample_cols = [f"sample_{i}" for i in range(n_samples)]
    context_df[sample_cols] = cds_all_t[:, :n_samples].cpu().numpy()
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
nside = 2**10

from diffusers import DDIMScheduler
#model.set_w(1)
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
    beta_schedule=noise_scheduler.config.beta_schedule,
    clip_sample_range=noise_scheduler.config.clip_sample_range,
    #timestep_spacing="trailing"
    )
#timesteps = np.concatenate((np.arange(0, 20, 2), np.arange(40, 1000, 20)))[::-1]
#print(timesteps)
#print(len(timesteps))
# /opt/conda/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddim.py, line 335 for passing inference steps as list
# 1. get previous step value (=t-1)
# idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
# prev_timestep = self.timesteps[idx + 1] if idx+1 < self.num_inference_steps else -1
## prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddim_scheduler.set_timesteps(50, device=device, steps=None)
n=20
df = infer_patch_with_rotations_gpu(model, ddim_scheduler, params, date, nside=nside, jump=None, n_samples=n, device=device)
sample_cols = [f"sample_{i}" for i in range(n)]
# renormalize and add xco2 offset 
df['xco2'] = df['xco2'] * params['train_stds'][11] + params['train_means'][11]
df[sample_cols] = df[sample_cols] * params['train_stds'][0] + params['train_means'][0] + df['xco2'].values[:, np.newaxis]
df.to_parquet(f'{save_path}global.pq')


