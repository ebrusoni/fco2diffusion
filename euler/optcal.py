import pandas as pd 
import numpy as np

sample_cols = [f"mlp_{i}" for i in range(50)]
mlp_test = pd.read_parquet(f"../models/sota_anoms64_big/test_predictions.pq")
pred_mask = pd.read_parquet("../models/anoms_sea_1d/pred_nans.pq")

mlp_test = mlp_test.loc[pred_mask.values]
print(mlp_test.shape)

def scale_s(S, sigma_scale):
    offset = (sigma_scale - 1) * S.mean(axis=1)
    return (sigma_scale * S) - offset[:, None]

def score(covered):
    perfect = np.arange(0.1, 1.0, 0.1)
    return np.mean(np.abs(covered-perfect))

def cal(S, truth):
    # S: (n_rows, n_samp) samples from your model. Replace df.values with your samples.
    #S = df.loc[mask, sample_cols].values            # shape (n_rows, 50) in your real case
    y = truth                # shape (n_rows,)

    levels = np.arange(0.1, 1.0, 0.1)  # 10%,...,90% central coverage

    q_lo = np.quantile(S, (1 - levels)/2, axis=1).T   # shape (n_rows, len(levels))
    q_hi = np.quantile(S, 1 - (1 - levels)/2, axis=1).T

    covered = ((y[:, None] >= q_lo) & (y[:, None] <= q_hi)).mean(axis=0)  # empirical coverage
    return covered

import numpy as np

import numpy as np

def get_best_cal(S, truth, sigma_grid):
    """
    Loop-based search (non-vectorised) for the sigma_scale that minimises the
    mean absolute calibration error. Returns:
      - the best scaling factor,
      - the calibration values (empirical coverages) at each quantile level
        for that best factor,
      - and the error for every candidate sigma.

    Parameters
    ----------
    S : np.ndarray, shape (n_rows, n_samples)
        Samples from the model.
    truth : np.ndarray, shape (n_rows,)
        Ground truth targets.
    sigma_grid : 1D iterable of float
        Candidate sigma_scale values to test.

    Returns
    -------
    best_sigma : float
        Sigma value that minimises the mean absolute calibration error.
    best_covered : np.ndarray, shape (n_levels,)
        Empirical coverage at each level (0.1, 0.2, ..., 0.9) for best_sigma.
    scores : np.ndarray, shape (len(sigma_grid),)
        Mean absolute calibration error for each sigma in `sigma_grid`.
    levels : np.ndarray, shape (n_levels,)
        The coverage levels corresponding to `best_covered` (10%..90%).
    """
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    scores = np.empty_like(sigma_grid, dtype=float)

    best_sigma = None
    best_score = np.inf
    best_covered = None

    for i, sigma in enumerate(sigma_grid):
        S_scaled = scale_s(S, sigma)
        covered = cal(S_scaled, truth)   # empirical coverage per level (0.1..0.9)
        err = score(covered)             # mean absolute calibration error

        scores[i] = err
        if err < best_score:
            best_score = err
            best_sigma = float(sigma)
            best_covered = covered

    levels = np.arange(0.1, 1.0, 0.1)
    return best_sigma, best_covered, scores, levels



truth = mlp_test.fco2rec_uatm.values
S = mlp_test[sample_cols].values
sigma_grid = np.arange(5, 5.5, 0.01)
print(get_best_cal(S, truth, sigma_grid))


