import os
import numpy as np
import rasterio as rio
from tqdm import tqdm
import multiprocessing as mp
import glob


def process(args):
    model, year, month = args

    with rio.open(os.path.join(r"/ChinaHeadwaterDIC/hydrology/ROsur", f"ROsur_{year}_{month:02d}.tif")) as src:
        ROsur_array = src.read(1).astype(np.float32)
    with rio.open(os.path.join(r"/ROsub", f"ROsub_{year}_{month:02d}.tif")) as src:
        ROsub_array = src.read(1).astype(np.float32)
    RO_array = ROsur_array + ROsub_array

    with rio.open(os.path.join(r"/ChinaHeadwaterDIC/DIC_concentration/History", model, f"DIC_concentration_{year}_{month:02d}.tif")) as src:
        DIC_concentration_array = src.read(1).astype(np.float32)
        metadata = src.profile
    DIC_concentration_array = np.clip(DIC_concentration_array, 0, None)
    DIC_flux_array = DIC_concentration_array * RO_array * 1e-3

    output_path = os.path.join(r"/ChinaHeadwaterDIC/DIC_flux/History", model, f"DIC_flux_{year}_{month:02d}.tif")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rio.open(output_path, 'w', **metadata) as dst:
        dst.write(DIC_flux_array, 1)



if __name__ == '__main__':
    model_ls = ['LightGBM', 'XGBoost', 'CatBoost', 'HistGBDT', 'RandomForest', 'ExtraTrees']
    year_ls = list(range(2000, 2021))
    month_ls = list(range(1, 13))
    args = [(model, year, month) for model in model_ls for year in year_ls for month in month_ls]
    with mp.Pool(processes=72) as pool:
        list(tqdm(pool.imap(process, args)))
