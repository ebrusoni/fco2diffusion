from utils import add_src_and_logger
is_renkulab = True
DATA_PATH, logging = add_src_and_logger(is_renkulab, None)
import pandas as pd
import numpy as np
import healpy as hp
from fco2models.utraining import prep_df

def get_nested_patch(patch_ix, patch_size, nside=1024, plot=False):
    patch_pix = np.arange(patch_ix * patch_size, (patch_ix + 1) * patch_size)
    xyf = hp.pix2xyf(nside, patch_pix, nest=True)
    xyf = (xyf[0]  - xyf[0].min(), xyf[1]  - xyf[1].min(), xyf[2])
    lon, lat = hp.pix2ang(nside, patch_pix, nest=True, lonlat=True)

    if plot:
        side = int(np.sqrt(patch_size))
        plot_patch(patch_ix, patch_size, xyf[0] * side + xyf[1], nside=nside)
    return xyf, lon, lat, patch_pix

def plot_patch(patch_ix, patch_size, data, nside=1024):
    npix = hp.nside2npix(nside)
    m = np.full(npix, np.nan)
    patch_pix = np.arange(patch_ix * patch_size, (patch_ix + 1) * patch_size)
    m[patch_pix] = data
    hp.mollview(m, title=f"Patch {patch_ix}", nest=True)
    
def do_rot(lon, lat, zyx, nest=True):
        rot = hp.Rotator(rot=zyx, eulertype="ZYX") # rotate only around the z-axis, so we do not mix equatirial and polar coordinates
        lon_rot, lat_rot = rot(lon, lat, lonlat=True)
        return lon_rot, lat_rot

from fco2dataset.ucollocate import get_day_data, collocate, get_zarr_data
def get_day_dataset(date):
    # get global satellite data for a given date
    # dss = get_day_data(date, save_path='../data/inference/gridded')
    dss = get_zarr_data(str(date.year))
    return dss

def collocate_coords(df, dss, date):
    save_path = '../data/inference/collocated'
    df_collocated = collocate(df, date, save_path=save_path, dss=dss, verbose=False)
    return df_collocated

def segment_sample(data, orientation, segment_len=64, random_offset=0):
    padded_side, _, num_cols = data.shape
    side = padded_side - 2 * segment_len
    if orientation == 'vertical':
        data = np.transpose(data, (1, 0, 2))
    offset = segment_len + random_offset
    segments = data[segment_len:-segment_len, offset:offset + side, :].reshape((side ** 2) // segment_len, segment_len, num_cols)
    segments = np.swapaxes(segments, 1, 2)
    return segments

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

def get_patch_ds(params, patch_ix, patch_size, date, nside, dss=None):
    predictors = params['predictors']
    stats = {
    'means':params['train_means'],
    'stds':params['train_stds'],
    'mins':params['train_mins'],
    'maxs':params['train_maxs']
    }

    # get the patch coordinates
    xyf, lon, lat, patch_pix = get_nested_patch(patch_ix, patch_size, nside=nside)
    #lon, lat = do_rot(lon, lat, (0, 0, 10))
    ring_id = hp.nest2ring(nside, patch_pix)
    if dss is None:
        dss = get_day_dataset(date)
    coords = pd.DataFrame({
        'lon': lon.flatten(),
        'lat': lat.flatten(),
        'x': xyf[0].flatten(),
        'y': xyf[1].flatten(),
        'f': xyf[2].flatten(),
        'patch_pix': patch_pix,
        'ring_pix':ring_id
    })
    coords['time_1d'] = date
    # collocate the data
    context_df = collocate_coords(coords, dss, date)
    #print(context_df.columns)
    #print(context_df.shape)
    context_df['lon'] = (context_df['lon'] + 180) % 360 - 180
    context_df = prep_df(context_df, with_target=False, with_log=False)[0]
    context_df['sst_clim'] += 273.15
    context_df['sst_anom'] = context_df['sst_cci'] - context_df['sst_clim']
    context_df['sss_anom'] = context_df['sss_cci'] - context_df['sss_clim']
    context_df['chl_anom'] = context_df['chl_globcolour'] - context_df['chl_clim']
    context_df['ssh_anom'] = context_df['ssh_sla'] - context_df['ssh_clim']
    context_df['mld_anom'] = np.log10(context_df['mld_dens_soda'] + 1e-5) - context_df['mld_clim']
    context_df = context_df[predictors + ['seamask', 'x', 'y']]
    context_df = normalize(context_df, stats, params['mode'])
    context_df = context_df.fillna(context_df.mean())  # fill NaNs with mean of each column

    height = width = np.sqrt(patch_size).astype(int)
    context_ds = np.zeros((height, width, len(predictors) + 5), dtype=np.float32)
    x = context_df['x'].values.astype(int)
    y = context_df['y'].values.astype(int)
    for i, col in enumerate(predictors):
        context_ds[x, y, i] = context_df[col].values
    # add lat lon
    context_ds[x, y, -2] = lat
    context_ds[x, y, -1] = lon
    context_ds[x, y, -3] = patch_pix.flatten()
    context_ds[x, y, -4] = ring_id.flatten()
    context_ds[x, y, -5] = context_df['seamask'].values

    return context_ds

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
            sample[torch.isnan(sample)] = sample_prev[torch.isnan(sample)]
            samples.append(sample)
            #samples.append(sample.cpu().numpy())
    return torch.cat(samples, axis=0)

import torch
from tqdm import tqdm
def infer_patch(model, noise_scheduler, params, patch_ix, patch_size, date, nside, dss=None, jump=None, n_samples=1, t_loop=None, eta=0, start_sample=None):
    
    # get dataset collocated for the patch
    context_ds = get_patch_ds(params, patch_ix, patch_size, date, nside=nside, dss=dss)
    lat = context_ds[:, :, -2] # later used for plotting
    lon = context_ds[:, :, -1] # later used for plotting
    patch_pix = context_ds[:, :, -3] # later used for plotting
    ring_pix =  context_ds[:, :, -4] 
    
    context_ds = context_ds[:, :, :-4] # model input
    side = np.sqrt(patch_size).astype(int)
    segment_len = 64

    padded_side = side + 2 * segment_len
    padded_context_ds = np.zeros((padded_side, padded_side, context_ds.shape[2]), dtype=np.float32)
    padded_context_ds[segment_len: -segment_len, segment_len: -segment_len, :] = context_ds
    padded_context_ds[segment_len:-segment_len, :segment_len, :] = context_ds[:, :segment_len, :][:, ::-1, :] # just mirror the data for now
    padded_context_ds[segment_len:-segment_len, -segment_len:, :] = context_ds[:, -segment_len:, :][:, ::-1, :]
    

    sample_cols = np.random.randn(padded_side, padded_side, n_samples).astype(np.float32)
    if start_sample is not None:
        sample_cols[segment_len: -segment_len, segment_len: -segment_len, : ] = start_sample#np.repeat(start_sample[:, :, None], n_samples, axis=2) 
    sample_context_ds = np.concatenate([sample_cols, padded_context_ds], axis=2)
    pred_cols = np.arange(n_samples, sample_context_ds.shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    step = 0
    #t_loop = tqdm(noise_scheduler.timesteps[::jump], desc="Denoising steps")
    t_loop = noise_scheduler.timesteps[::jump] if t_loop is None else t_loop
    #print(t_loop)
    # ------------------------------------------------------------------
    # pre-compute some constants once, *outside* the time loop ----------
    cols_per_samp = 1 + len(pred_cols)                       # sample-col + predictors
    n_seg          = patch_size // segment_len               # segments per sample
    order          = np.argsort(ring_pix.flatten())            # ring permutation
    inverse        = np.argsort(order)
    axis_sample    = 2                                       # where the n_samples live
    # ------------------------------------------------------------------
    
    for t in t_loop:
        # ■■■■■■■■■■■■■■■■■■■■  build SEGMENTS for *all* samples at once  ■■■■■■■■■■■■■■■■■■
        #random_offset = np.random.randint(-60, 60)            # one offset shared by all
    
        if step % 3 == 0:                     # ── ring mode (every 3rd step) ──
            # core patch without the padding, shape (side, side, ·)
            core = sample_context_ds[segment_len:-segment_len,
                                     segment_len:-segment_len, :]
            # predictors, same for every sample
            pred_part = core[:, :, pred_cols]                 # (side, side, p)
            # sample columns collected in an extra axis
            samp_part = core[:, :, :n_samples]                # (side, side, n_samples)
    
            # stack  →  (n_samples, side, side, cols_per_samp)
            data = np.concatenate((samp_part.transpose(2,0,1)[..., None],    # (n_samples, side, side, 1)
                                   np.broadcast_to(pred_part,   # (1, side, side, p)
                                                  (n_samples, side, side,
                                                   len(pred_cols)))),
                                  axis=3)
    
            # flatten spatially, apply ring ordering
            ring_ds = data.reshape(n_samples, patch_size, cols_per_samp)
            ring_ds = ring_ds[:, order, :]
    
            # cut into 64-pixel strips and move axes → (n_samples·n_seg, cols, 64)
            segments = (ring_ds
                        .reshape(n_samples, n_seg, segment_len, cols_per_samp)
                        .transpose(0, 1, 3, 2)
                        .reshape(-1, cols_per_samp, segment_len))
            
            #print(f"segments shape: {segments.shape}")
    
        else:
            # one offset per sample   (n_samples,)
            offsets = segment_len + np.random.randint(-64, 64, size=n_samples)
        
            # bring data into shape (n_samples, P, P, cols_per_samp)
            samp_pad = sample_context_ds[:, :, :n_samples].transpose(2, 0, 1)     # (n_s, P, P)
            pred_pad = sample_context_ds[:, :, pred_cols][None, ...]              # (1, P, P, p)
            data     = np.concatenate((samp_pad[..., None],                       # (n_s, P, P, 1)
                                       np.broadcast_to(pred_pad,
                                                      (n_samples, *pred_pad.shape[1:]))),
                                      axis=3)                                     # (n_s, P, P, cols)
        
            if step % 2:                                  # vertical pass
                # swap x ↔ y for every sample
                data = data.transpose(0, 2, 1, 3)         #   (n_s, P, P, cols)
        
            # -------- vectorised slice with per-sample offset ----------------------
            P      = padded_side
            side   = P - 2 * segment_len
            samp_i = np.arange(n_samples)[:, None, None]                         # (n_s,1,1)
        
            rows   = np.arange(segment_len, segment_len + side)[None, :, None]   # (1,side,1)
            cols   = offsets[:, None, None] + np.arange(side)[None, None, :]     # (n_s,1,side)
        
            # broadcast to (n_s, side, side)
            rows_b = np.broadcast_to(rows, (n_samples, side, side))
            cols_b = np.broadcast_to(cols, (n_samples, side, side))
        
            # gather => (n_s, side, side, cols_per_samp)
            sliced = data[samp_i, rows_b, cols_b, :]
        
            # reshape to (n_samples·n_seg, cols_per_samp, 64)
            segments = (sliced
                        .reshape(n_samples, side * side // segment_len,
                                 segment_len, cols_per_samp)
                        .transpose(0, 1, 3, 2)
                        .reshape(-1, cols_per_samp, segment_len)
                        .astype(np.float32))
            
        # ■■■■■■■■■■■■■■■■■■■■■  SEND THROUGH THE MODEL (unchanged)  ■■■■■■■■■■■■■■■■■■■■■■■
        ds        = torch.from_numpy(segments).to(device).float()
        #ds        = torch.cat((ds, torch.ones_like(ds[:, :1, :])), axis=1)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=4096,
                                                 shuffle=False)
        all_samples = do_step_loader(model, noise_scheduler, dataloader,
                                     t, device, jump, eta=eta)           # 1-D tensor
    
        preds = all_samples.reshape(n_samples, -1, segment_len).reshape(n_samples, patch_size)
        
        side = padded_side - 2 * segment_len          # = √patch_size
        rows_core = np.arange(segment_len, segment_len + side)   # common row band
        cols_core = rows_core                         # same numbers but used as columns
        
        # broadcast sample index axis (n_s, side, side)
        samp_idx = np.broadcast_to(np.arange(n_samples)[:, None, None],
                                   (n_samples, side, side))
        
        if step % 3 == 0:                     # ─── ring case (unchanged) ───────────
            imgs = preds[:, inverse].reshape(n_samples, side, side)
            sample_context_ds[segment_len:-segment_len,
                              segment_len:-segment_len,
                              :n_samples] = imgs.transpose(1, 2, 0)
        
        elif step % 2 == 0:                   # ─── horizontal, per-sample offset ───
            imgs = preds.reshape(n_samples, side, side)           # (n_s, y, x)
        
            # index arrays with broadcasting
            rows_idx = np.broadcast_to(rows_core[None, :, None],
                                       (n_samples, side, side))
            cols_idx = offsets[:, None, None] + np.broadcast_to(
                                       np.arange(side)[None, None, :],
                                       (n_samples, side, side))
        
            # scatter back
            sample_context_ds[rows_idx, cols_idx, samp_idx] = imgs
        
        else:                                 # ─── vertical, per-sample offset ─────
            imgs = preds.reshape(n_samples, side, side).transpose(0, 2, 1)  # rotate
        
            rows_idx = offsets[:, None, None] + np.broadcast_to(
                                       np.arange(side)[None, :, None],
                                       (n_samples, side, side))
            cols_idx = np.broadcast_to(cols_core[None, None, :],
                                       (n_samples, side, side))
        
            sample_context_ds[rows_idx, cols_idx, samp_idx] = imgs
            
        step += 1
            
    # remove padding from result
    sample_context_ds = sample_context_ds[segment_len:-segment_len, segment_len:-segment_len, :]
    return np.concatenate([sample_context_ds, lat[:,:, np.newaxis], lon[:,:, np.newaxis], patch_pix[:, :, np.newaxis], ring_pix[:, :, np.newaxis]], axis=2)

import numpy as np
import torch
from tqdm import tqdm


def infer_patch_gpu(
    model,
    ddim_scheduler,
    params,
    patch_ix,
    patch_size,
    date,
    nside,
    dss=None,
    jump=None,
    n_samples=1,
    t_loop=None,
    eta=0,
    start_sample=None,
    *,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,        # use torch.float32 if reproducibility beats speed
):
    """Torch-only version of the original NumPy routine (logic unchanged)."""
    
    # print(device)
    # ───────────────────────── context & padding ───────────────────────────
    context_ds_np = get_patch_ds(params, patch_ix, patch_size, date,
                                 nside=nside, dss=dss).astype(np.float32)

    lat       = context_ds_np[:, :, -2]          # keep on CPU for return
    lon       = context_ds_np[:, :, -1]
    patch_pix = context_ds_np[:, :, -3]
    ring_pix  = context_ds_np[:, :, -4]

    context_ds_np = context_ds_np[:, :, :-4]     # model input only

    core_side     = int(np.sqrt(patch_size))
    segment_len   = 64
    padded_side   = core_side + 2 * segment_len

    # build padded context (mirror padding on x–axis)
    padded_context = torch.zeros((padded_side, padded_side,
                                  context_ds_np.shape[2]),
                                 dtype=dtype, device=device)

    ctx = torch.as_tensor(context_ds_np, dtype=dtype, device=device)
    padded_context[segment_len:-segment_len, segment_len:-segment_len, :] = ctx
    padded_context[segment_len:-segment_len, :segment_len, :]             = \
        torch.flip(ctx[:, :segment_len, :], dims=[1])
    padded_context[segment_len:-segment_len, -segment_len:, :]            = \
        torch.flip(ctx[:, -segment_len:, :], dims=[1])

    # ────────────────────────── initial samples ────────────────────────────
    sample_cols = torch.randn((padded_side, padded_side, n_samples),
                              dtype=dtype, device=device)
    if start_sample is not None:
        samp0 = torch.as_tensor(start_sample, dtype=dtype, device=device)
        sample_cols[segment_len:-segment_len,
                    segment_len:-segment_len, :] = samp0

    sample_context = torch.cat((sample_cols, padded_context), dim=2)

    pred_cols  = torch.arange(n_samples, sample_context.shape[2],
                              device=device)
    cols_per_s = 1 + len(pred_cols)
    n_seg      = patch_size // segment_len

    # ring permutation (static)
    order   = torch.argsort(
                 torch.as_tensor(ring_pix.flatten(), device=device))
    inverse = torch.argsort(order)
    
    last_scheduler = DDIMScheduler(
        num_train_timesteps=ddim_scheduler.config.num_train_timesteps,
        beta_schedule=ddim_scheduler.config.beta_schedule,
        clip_sample_range=ddim_scheduler.config.clip_sample_range,
        #timestep_spacing="trailing"
    )
    
    # timesteps
    if t_loop is None:
        steps = ddim_scheduler.timesteps[::jump]
    else:
        steps = t_loop
    #steps = list(steps)                       # ensure we can iterate twice

    noise_lvl = 40
    step=-2
    noise_scheduler = ddim_scheduler
    last_step = len(steps) - 1
    steps = torch.cat([steps, torch.arange(noise_lvl, -1, step).to(device)])
    last_scheduler.set_timesteps(noise_lvl // step, steps=torch.arange(noise_lvl, -1, step), device=device)

    model.to(device).eval()

    # ───────────────────────────── main loop ───────────────────────────────
    # print(steps)
    for step_no, t in enumerate(steps):
        ring_pass = (step_no % 3 == 0)
        vert_pass = (step_no % 3 == 2)        # i.e. step_no % 2 and not ring

        if ring_pass:
            # ----- build ring segments ------------------------------------
            core = sample_context[segment_len:-segment_len,
                                   segment_len:-segment_len, :]

            pred_part = core[:, :, pred_cols]                    # (P,P,p)
            samp_part = core[:, :, :n_samples]                   # (P,P,n)

            data = torch.cat((samp_part.permute(2, 0, 1)  # (n,P,P)
                                          .unsqueeze(-1),  # (n,P,P,1)
                              pred_part.unsqueeze(0)
                                       .expand(n_samples, -1, -1, -1)),
                             dim=3)                              # (n,P,P,c)

            ring_ds = data.reshape(n_samples, patch_size,
                                   cols_per_s)[:, order, :]

            segments = (ring_ds
                        .reshape(n_samples, n_seg, segment_len, cols_per_s)
                        .permute(0, 1, 3, 2)
                        .reshape(-1, cols_per_s, segment_len)
                        .contiguous())

        else:
            # ----- build vertical / horizontal segments -------------------
            offsets = segment_len + torch.randint(-64, 64,
                                                  (n_samples,),
                                                  device=device)

            samp_pad = sample_context[:, :, :n_samples]           # (P,P,n)
            samp_pad = samp_pad.permute(2, 0, 1)                  # (n,P,P)

            pred_pad = sample_context[:, :, pred_cols]            # (P,P,p)
            pred_pad = pred_pad.unsqueeze(0).expand(n_samples, -1, -1, -1)

            data = torch.cat((samp_pad.unsqueeze(-1), pred_pad),  # (n,P,P,c)
                             dim=3)

            if vert_pass:                 # rotate by swapping x and y
                data = data.transpose(1, 2)

            P          = padded_side
            side_core  = P - 2 * segment_len
            rows_band  = torch.arange(segment_len,
                                      segment_len + side_core,
                                      device=device)             # (side)
            cols_band  = torch.arange(side_core, device=device)   # (side)

            # broadcast indices
            samp_i = torch.arange(n_samples, device=device)[:, None, None]
            rows_i = rows_band[None, :, None].expand(n_samples,
                                                     side_core,
                                                     side_core)
            cols_i = (offsets[:, None, None] +
                      cols_band[None, None, :]).expand(n_samples,
                                                        side_core,
                                                        side_core)

            sliced = data[samp_i, rows_i, cols_i, :]              # (n,s,s,c)

            segments = (sliced
                        .reshape(n_samples,
                                 side_core * side_core // segment_len,
                                 segment_len, cols_per_s)
                        .permute(0, 1, 3, 2)
                        .reshape(-1, cols_per_s, segment_len)
                        .contiguous())

        # ----- model step --------------------------------------------------
        ds_loader = torch.utils.data.DataLoader(segments,
                                                batch_size=4096,
                                                shuffle=False)

        all_samples = do_step_loader(model, noise_scheduler,
                                     ds_loader,
                                     t, device, jump, eta=eta)   # GPU 1-D

        preds = all_samples.to(device).view(n_samples,
                                            -1, segment_len) \
                            .reshape(n_samples, patch_size)

        side_core = padded_side - 2 * segment_len
        rows_core = torch.arange(segment_len,
                                 segment_len + side_core,
                                 device=device)
        cols_core = rows_core

        samp_idx = torch.arange(n_samples, device=device)[:, None, None] \
                              .expand(n_samples, side_core, side_core)

        if ring_pass:
            imgs = preds[:, inverse].view(n_samples, side_core, side_core)
            sample_context[segment_len:-segment_len,
                           segment_len:-segment_len,
                           :n_samples] = imgs.permute(1, 2, 0)

        elif not vert_pass:         # horizontal
            imgs = preds.view(n_samples, side_core, side_core)

            rows_idx = rows_core[None, :, None].expand(n_samples,
                                                       side_core,
                                                       side_core)
            cols_idx = (offsets[:, None, None] +
                        torch.arange(side_core, device=device)
                             [None, None, :]).expand(n_samples,
                                                     side_core,
                                                     side_core)

            sample_context[rows_idx, cols_idx, samp_idx] = imgs

        else:                       # vertical
            imgs = preds.view(n_samples, side_core, side_core).transpose(1, 2)

            rows_idx = (offsets[:, None, None] +
                        torch.arange(side_core, device=device)
                             [None, :, None]).expand(n_samples,
                                                     side_core,
                                                     side_core)
            cols_idx = cols_core[None, None, :].expand(n_samples,
                                                       side_core,
                                                       side_core)

            sample_context[rows_idx, cols_idx, samp_idx] = imgs

        if step_no == last_step:
            sample_context[:, :, :n_samples] = noise_scheduler.add_noise(sample_context[:, :, :n_samples], torch.randn_like(sample_context[:, :, :n_samples]).to(device), torch.tensor(noise_lvl).to(device))
            noise_scheduler = last_scheduler

    # ───────────────────────────── return ─────────────────────────────────
    result = sample_context[segment_len:-segment_len,
                            segment_len:-segment_len, :].cpu().numpy()
    extra  = np.stack((lat, lon, patch_pix, ring_pix), axis=2)    # (P,P,4)
    return np.concatenate((result, extra), axis=2, dtype=np.float32)


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

from diffusers import DDIMScheduler
#model.set_w(1)
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=noise_scheduler.config.num_train_timesteps,
    beta_schedule=noise_scheduler.config.beta_schedule,
    clip_sample_range=noise_scheduler.config.clip_sample_range,
    )
#timesteps = np.concatenate((np.arange(0, 40, 2), np.arange(40, 1000, 20)))[::-1]
#/opt/conda/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddim.py, line 335 for passing inference steps as list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps = 50
ddim_scheduler.set_timesteps(n_steps, device=device)
n=5
nside = 2**10
npix = hp.nside2npix(nside)
face_pixs = npix//12
num_subfaces = 4 # number of subfaces should a power of 4
patch_size = face_pixs // num_subfaces
print(f"Patch size: {patch_size}")

patch_ix = 31
samples = []
start_date ='2022-01-01' 
date_range = pd.date_range(start=start_date, end='2022-03-01', freq='D')
dfs = []
dfs_cond = []
date_range_loop = tqdm(date_range, desc="Processing dates")
start_sample = None
timesteps = ddim_scheduler.timesteps



for date in date_range_loop:
    t_loop = None if date == pd.Timestamp(start_date) else timesteps[n_steps // 2:]
    sample = infer_patch_gpu(model, ddim_scheduler, params, patch_ix, patch_size, date, nside=nside, 
                        dss=None, jump=None, n_samples=n, t_loop=t_loop, start_sample=start_sample)
    
    lat = sample[:, :, -4]
    lon = sample[:, :, -3]
    patch_pix = sample[:, :, -2].astype(int)
    seamask = sample[:, :, -5].astype(bool)
    
    # Base DataFrame with index
    index = pd.MultiIndex.from_arrays([
        patch_pix.flatten().astype(int),
        lat.flatten(),
        lon.flatten()
    ], names=['patch_pix', 'lat', 'lon'])
    
    # Prepare sample columns
    sample_cols = [f"sample_{i}" for i in range(n)] #+ params['predictors']
    sample_data = sample[:, :, :n].reshape(-1, n).copy()
    
    # De-normalize xco2, rescale and apply xco2 correction
    xco2_ix = params['predictors'].index('xco2') + 1
    xco2 = sample[:, :, xco2_ix + n - 1].flatten()
    xco2 = xco2 * params['train_stds'][xco2_ix] + params['train_means'][xco2_ix]
    sample_data[:, :n] = sample_data[:,:n] * params['train_stds'][0] + params['train_means'][0] + xco2[:, np.newaxis]
    
    # Create MultiIndex columns (date, sample_x)
    columns = pd.MultiIndex.from_product([[date.strftime('%Y-%m-%d')], sample_cols])
    
    # Build DataFrame and append
    df = pd.DataFrame(sample_data, index=index, columns=columns)
    dfs = df if date == pd.Timestamp(start_date) else pd.concat([dfs, df], axis=1)
    
    # renoise sample for next day conditioning
    start_sample = torch.from_numpy(sample[:, :, :n])
    start_sample = ddim_scheduler.add_noise(start_sample, torch.randn_like(start_sample), timesteps[n_steps // 2 - 1]).numpy()
    date_range_loop.set_postfix(date=date.strftime('%Y-%m-%d'), shape=df.shape)

dfs["seamask"] = seamask.flatten()
dfs.to_parquet(f'{save_path}papagayo.pq')