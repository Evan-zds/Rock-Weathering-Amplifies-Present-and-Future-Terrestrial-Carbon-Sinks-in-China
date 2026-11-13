import os
import glob
import warnings
import numpy as np
import rasterio as rio
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore', category=RuntimeWarning)


def calibrate_single_task(args):
    model, ssp, feature, base_dir, output_dir, is_ph = args

    additive_features = ['CO2air', 'Tsur', 'Tsub', 'SMsur', 'SMsub', 'pCO2soil', 'pHsur', 'pHsub']
    multiplicative_features = ['TotPrec', 'ROsur', 'ROsub', 'Evap', 'Trans', 'GPP', 'LAI']

    obs_path = os.path.join(base_dir, 'features', f'annual_{feature}.tif')
    if not os.path.exists(obs_path):
        obs_path = os.path.join(base_dir, 'features', f'{feature}.tif')
    if not os.path.exists(obs_path):
        print(f"❌ 找不到观测文件: {obs_path}，跳过此任务")
        return

    if is_ph:
        ph_dir = os.path.join(base_dir, feature, model)
        hist_matches = glob.glob(os.path.join(ph_dir, '**', f'*{ssp}*history*.tif'), recursive=True)
        fut_matches = glob.glob(os.path.join(ph_dir, '**', f'*{ssp}*future*.tif'), recursive=True)
    else:
        feat_dir = os.path.join(base_dir, 'CMIP6', model, ssp, feature)
        hist_matches = [os.path.join(feat_dir, f'{feature}_{model}_{ssp}_history.tif')]
        fut_matches = [os.path.join(feat_dir, f'{feature}_{model}_{ssp}_future.tif')]

    if len(hist_matches) != 1 or len(fut_matches) != 1:
        return

    hist_path = hist_matches[0]
    fut_path = fut_matches[0]

    with rio.open(obs_path, 'r') as src_obs:
        obs_data = src_obs.read(1).astype(np.float32)
        meta = src_obs.meta.copy()

    with rio.open(hist_path, 'r') as src_hist:
        mod_hist = src_hist.read(1).astype(np.float32)

    with rio.open(fut_path, 'r') as src_fut:
        mod_fut = src_fut.read(1).astype(np.float32)

    valid_mask = ~np.isnan(obs_data) & ~np.isnan(mod_hist) & ~np.isnan(mod_fut)

    calibrated_fut = np.full_like(obs_data, np.nan, dtype=np.float32)
    calibrated_hist = np.full_like(obs_data, np.nan, dtype=np.float32)

    if feature in additive_features:
        delta = mod_fut[valid_mask] - mod_hist[valid_mask]
        calibrated_fut[valid_mask] = obs_data[valid_mask] + delta
        calibrated_hist[valid_mask] = obs_data[valid_mask]

        if feature in ['SMsur', 'SMsub', 'pCO2soil']:
            calibrated_fut[valid_mask] = np.clip(calibrated_fut[valid_mask], a_min=0.0, a_max=None)
            calibrated_hist[valid_mask] = np.clip(calibrated_hist[valid_mask], a_min=0.0, a_max=None)
        elif feature in ['pHsur', 'pHsub']:
            calibrated_fut[valid_mask] = np.clip(calibrated_fut[valid_mask], a_min=3.0, a_max=10.0)
            calibrated_hist[valid_mask] = np.clip(calibrated_hist[valid_mask], a_min=3.0, a_max=10.0)

    elif feature in multiplicative_features:
        threshold = 0.05
        vm_hist = mod_hist[valid_mask]
        vm_fut = mod_fut[valid_mask]
        vm_obs = obs_data[valid_mask]

        normal_idx = vm_hist >= threshold
        low_val_idx = ~normal_idx

        res_fut = np.zeros_like(vm_obs)
        if np.any(normal_idx):
            ratio = vm_fut[normal_idx] / vm_hist[normal_idx]
            ratio = np.clip(ratio, a_min=0.05, a_max=5.0)
            res_fut[normal_idx] = vm_obs[normal_idx] * ratio

        if np.any(low_val_idx):
            res_fut[low_val_idx] = vm_obs[low_val_idx] + (vm_fut[low_val_idx] - vm_hist[low_val_idx])

        calibrated_fut[valid_mask] = np.clip(res_fut, a_min=0.0, a_max=None)
        calibrated_hist[valid_mask] = np.clip(vm_obs, a_min=0.0, a_max=None)

    out_dir = os.path.join(output_dir, model, ssp, feature)
    os.makedirs(out_dir, exist_ok=True)
    meta.update({'dtype': 'float32', 'nodata': np.nan, 'compress': 'lzw'})

    out_fut_path = os.path.join(out_dir, f'{feature}_{model}_{ssp}_future.tif')
    with rio.open(out_fut_path, 'w', **meta) as dst:
        dst.write(calibrated_fut, 1)

    out_hist_path = os.path.join(out_dir, f'{feature}_{model}_{ssp}_history.tif')
    with rio.open(out_hist_path, 'w', **meta) as dst:
        dst.write(calibrated_hist, 1)


if __name__ == '__main__':
    base_dir = r'ChinaHeadwaterDIC'
    output_dir = os.path.join(base_dir, 'CMIP6_processed')

    CMIP6_model_ls = [
        'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'GFDL-ESM4', 'IPSL-CM6A-LR',
        'MIROC-ES2L', 'MPI-ESM1-2-LR', 'NorESM2-LM', 'UKESM1-0-LL'
    ]
    ssp_ls = ['ssp126', 'ssp245', 'ssp585']

    regular_features = [
        'CO2air', 'Tsub', 'TotPrec', 'LAI', 'SMsur',
        'Trans', 'ROsur', 'GPP', 'Evap', 'Tsur',
        'SMsub', 'ROsub', 'pCO2soil'
    ]

    ph_features = ['pHsur', 'pHsub']

    task_args = []

    for model in CMIP6_model_ls:
        for ssp in ssp_ls:
            for feat in regular_features:
                task_args.append((model, ssp, feat, base_dir, output_dir, False))

    for model in CMIP6_model_ls:
        for ssp in ssp_ls:
            for feat in ph_features:
                task_args.append((model, ssp, feat, base_dir, output_dir, True))

    cpu_cores = 32
    with mp.Pool(processes=cpu_cores) as pool:
        list(tqdm(pool.imap_unordered(calibrate_single_task, task_args), total=len(task_args), desc="Calibrating"))

