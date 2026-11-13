import os
import glob
import numpy as np
import rasterio as rio
from tqdm import tqdm
import multiprocessing as mp
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def process_single_month(args):
    in_path, out_path, transfer_factor, mask_array, meta = args
    if os.path.exists(out_path):
        return

    with rio.open(in_path) as src:
        f_dic = src.read(1).astype(np.float32)

    wcs = f_dic * transfer_factor

    wcs[~mask_array] = np.nan

    meta_out = meta.copy()
    meta_out.update({
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw',
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True
    })

    with rio.open(out_path, 'w', **meta_out) as dst:
        dst.write(wcs, 1)


if __name__ == '__main__':
    feat_dir = r"/ChinaHeadwaterDIC/features"
    mask_path = r"/raster/Final_mask.tif"
    data_dir = r"/raster"

    dic_flux_root = r"/ChinaHeadwaterDIC/DIC_flux/History"
    wcs_output_root = r"/ChinaHeadwaterDIC/WCS/History"

    print(">> 正在加载静态底图并预计算综合转换矩阵 (Transfer Factor)...")

    with rio.open(mask_path) as src:
        mask_array = src.read(1).astype(bool)
        meta = src.meta.copy()

    with rio.open(os.path.join(feat_dir, 'annual_fCa.tif')) as src:
        fCa = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
    with rio.open(os.path.join(feat_dir, 'annual_fSi.tif')) as src:
        fSi = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
    with rio.open(os.path.join(feat_dir, 'annual_Kref.tif')) as src:
        Kref = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)

    k_grid = np.where(Kref > 0, 0.03804 / (Kref + 1e-10), 0.0)

    mask_pure_si = (fCa == 0.0) & (fSi > 0.0)
    mask_pure_ca = (fCa > 0.0) & (fSi == 0.0)
    mask_mixed = (fCa > 0.0) & (fSi > 0.0)

    stoich_coef = np.zeros_like(fCa, dtype=np.float32)
    stoich_coef[mask_pure_si] = 1.0
    stoich_coef[mask_pure_ca] = 0.5

    u_mixed = (fCa[mask_mixed] / fSi[mask_mixed]) * k_grid[mask_mixed]
    stoich_coef[mask_mixed] = (1.0 + u_mixed) / (1.0 + 2.0 * u_mixed)

    with rio.open(os.path.join(data_dir, 'K_alk_base.tif')) as src:
        kalk_base = src.read(1).astype(np.float32)
    with rio.open(os.path.join(data_dir, 'K_non_carb_base.tif')) as src:
        knon_base = src.read(1).astype(np.float32)

    eff_knon = np.where(mask_pure_si, 0.0, knon_base)
    alloc = np.clip((1.0 - eff_knon) * stoich_coef, 0.0, 1.0)

    transfer_factor = kalk_base * alloc
    transfer_factor[~mask_array] = np.nan

    ML_model_ls = ['CatBoost', 'ExtraTrees', 'HistGBDT', 'LightGBM', 'RandomForest', 'XGBoost']
    task_args = []

    for model in ML_model_ls:
        model_in_dir = os.path.join(dic_flux_root, model)
        model_out_dir = os.path.join(wcs_output_root, model)
        os.makedirs(model_out_dir, exist_ok=True)

        all_tifs = glob.glob(os.path.join(model_in_dir, "*.tif"))

        monthly_tifs = [f for f in all_tifs if 'mean_DIC_flux' not in os.path.basename(f) and f.endswith('.tif')]

        for in_path in monthly_tifs:
            fname = os.path.basename(in_path)
            out_fname = fname.replace('DIC_flux_', 'WCS_')
            out_path = os.path.join(model_out_dir, out_fname)

            task_args.append((in_path, out_path, transfer_factor, mask_array, meta))


    cpu_cores = min(32, mp.cpu_count())
    with mp.Pool(processes=cpu_cores) as pool:
        list(tqdm(pool.imap_unordered(process_single_month, task_args), total=len(task_args), desc="Converting to WCS"))

