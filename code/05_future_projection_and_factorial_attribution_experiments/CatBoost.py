"""S1"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os
import rasterio as rio
import glob
from tqdm import tqdm
import gc


def process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls):
    model, ssp, status = args
    folder_path = r'ChinaHeadwaterDIC'
    output_path = os.path.join(folder_path, 'DIC_concentration', 'CMIP6', experiment, ML_model, model, ssp, f'DIC_concentration_{model}_{ssp}_{status}.tif')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return

    CMIP6_feature_ls = ['CO2air', 'Evap', 'GPP', 'LAI', 'pCO2soil', 'ROsur', 'ROsub', 'ROratio',
                        'SMsur', 'SMsub', 'TotPrec', 'Trans', 'Tsur', 'Tsub', 'pHsur', 'pHsub']
    non_CMIP6_feature_ls = [feature for feature in feature_ls if feature not in CMIP6_feature_ls]
    """Elev, fCa, fSi, lithology, Kref, Esoil, fClaysur, fClaysub, fSiltsur, fSiltsub, fSandsur, fSandsub, SOCsur, SOCsub, GDP, PopDens, fCrop, fBare, fGrass, fForest"""
    assert len(CMIP6_feature_ls + non_CMIP6_feature_ls) == len(feature_ls)

    predict_feature_dict = {}

    for var in non_CMIP6_feature_ls:
        if var in ['GDP', 'PopDens', 'fCrop', 'fBare', 'fGrass', 'fForest']:
            tif_path = os.path.join(folder_path, 'features', f'annual_{var}.tif')
        else:
            tif_path = glob.glob(os.path.join(folder_path, '*', var, f'{var}.tif'))[0]
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            predict_feature_dict[var] = src.read(1).astype(np.float32).flatten()[mask_flat]

    dynamic_vars_for_exp = experiment_ls[experiment]

    for var in CMIP6_feature_ls:
        if var in dynamic_vars_for_exp:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_processed', model, ssp, var, f'{var}_{model}_{ssp}_{status}.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_{status}.tif')
        else:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_processed', model, ssp, var, f'{var}_{model}_{ssp}_history.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_history.tif')
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            arr_flat = src.read(1).astype(np.float32).flatten()[mask_flat]
            if var in ['CO2air', 'LAI', 'pCO2soil', 'ROratio', 'SMsur', 'SMsub', 'Tsur', 'Tsub', 'pHsur', 'pHsub']:
                predict_feature_dict[var] = arr_flat
            else:
                predict_feature_dict[var] = arr_flat / 12.0

    predict_feature_df = pd.DataFrame(predict_feature_dict)[feature_ls]
    pred_valid = optimizer.predict(predict_feature_df)
    pred_valid = np.clip(pred_valid, 0, None)
    predict_label_array = np.full(shape=(rows * columns), fill_value=np.nan, dtype=np.float32)
    predict_label_array[mask_flat] = pred_valid
    predict_label_array = predict_label_array.reshape((rows, columns))

    with rio.open(output_path, 'w', **metadata) as dst:
        dst.write(predict_label_array, 1)
    del predict_feature_dict, predict_feature_df, pred_valid, predict_label_array
    gc.collect()


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pd.set_option('display.max_columns', None)

    data_path = r"Submit_dataset.xlsx"
    raw_df = pd.read_excel(data_path)
    df_monthly = raw_df.groupby(['site_id', 'year', 'month'], as_index=False).mean(numeric_only=True)

    feature_ls = df_monthly.loc[:, 'CO2air': 'fForest'].columns.tolist()
    label = 'DIC (mg/L)'


    """Start"""
    ML_model = 'CatBoost'
    experiment = 'S1'

    best_params = {'iterations': 978, 'depth': 8, 'learning_rate': 0.02841927464540473,
                   'l2_leaf_reg': 5.845420577455735,
                   'subsample': 0.9024744934818496, 'min_data_in_leaf': 53, 'random_strength': 0.526551012551942}
    optimizer = CatBoostRegressor(**best_params, bootstrap_type='Poisson', loss_function='RMSE', task_type="GPU", devices='1', random_state=42, verbose=0)
    """End"""

    experiment_ls = {
        'S1': ['CO2air', 'Tsur', 'Tsub', 'TotPrec'],
        'S2': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans'],
        'S3': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub'],
        'S4': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub', 'GPP', 'LAI', 'pCO2soil']
    }

    optimizer.fit(df_monthly[feature_ls], df_monthly[label])

    mask_array = rio.open(r"raster/mask.tif", 'r').read(1).astype(bool)
    rows, columns = mask_array.shape
    mask_flat = mask_array.flatten()

    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': np.float32,
        'width': columns,
        'height': rows,
        'crs': 'EPSG:4326',
        'transform': rio.transform.from_bounds(west=70, south=15, east=140, north=55, height=rows, width=columns),
        'nodata': np.nan,
        'compress': 'lzw',
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True,
        'interleave': 'band',
        'predictor': 2
    }

    model_ls = ['BCC-CSM2-MR', 'CanESM5', 'CESM2', 'GFDL-ESM4', 'IPSL-CM6A-LR',
                'MIROC-ES2L', 'MPI-ESM1-2-LR', 'NorESM2-LM', 'UKESM1-0-LL']
    ssp_ls = ['ssp126', 'ssp245', 'ssp585']
    if experiment == 'S4':
        status_ls = ['history', 'future']
    else:
        status_ls = ['future']
    args_ls = [(model, ssp, status) for model in model_ls for ssp in ssp_ls for status in status_ls]

    for args in tqdm(args_ls):
        process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls)


"""S2"""
def process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls):
    model, ssp, status = args
    folder_path = r'ChinaHeadwaterDIC'
    output_path = os.path.join(folder_path, 'DIC_concentration', 'CMIP6', experiment, ML_model, model, ssp, f'DIC_concentration_{model}_{ssp}_{status}.tif')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return

    CMIP6_feature_ls = ['CO2air', 'Evap', 'GPP', 'LAI', 'pCO2soil', 'ROsur', 'ROsub', 'ROratio',
                        'SMsur', 'SMsub', 'TotPrec', 'Trans', 'Tsur', 'Tsub', 'pHsur', 'pHsub']
    non_CMIP6_feature_ls = [feature for feature in feature_ls if feature not in CMIP6_feature_ls]
    """Elev, fCa, fSi, lithology, Kref, Esoil, fClaysur, fClaysub, fSiltsur, fSiltsub, fSandsur, fSandsub, SOCsur, SOCsub, GDP, PopDens, fCrop, fBare, fGrass, fForest"""
    assert len(CMIP6_feature_ls + non_CMIP6_feature_ls) == len(feature_ls)

    predict_feature_dict = {}

    for var in non_CMIP6_feature_ls:
        if var in ['GDP', 'PopDens', 'fCrop', 'fBare', 'fGrass', 'fForest']:
            tif_path = os.path.join(folder_path, 'features', f'annual_{var}.tif')
        else:
            tif_path = glob.glob(os.path.join(folder_path, '*', var, f'{var}.tif'))[0]
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            predict_feature_dict[var] = src.read(1).astype(np.float32).flatten()[mask_flat]

    dynamic_vars_for_exp = experiment_ls[experiment]

    for var in CMIP6_feature_ls:
        if var in dynamic_vars_for_exp:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_processed', model, ssp, var, f'{var}_{model}_{ssp}_{status}.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_{status}.tif')
        else:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_processed', model, ssp, var, f'{var}_{model}_{ssp}_history.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_history.tif')
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            arr_flat = src.read(1).astype(np.float32).flatten()[mask_flat]
            if var in ['CO2air', 'LAI', 'pCO2soil', 'ROratio', 'SMsur', 'SMsub', 'Tsur', 'Tsub', 'pHsur', 'pHsub']:
                predict_feature_dict[var] = arr_flat
            else:
                predict_feature_dict[var] = arr_flat / 12.0

    predict_feature_df = pd.DataFrame(predict_feature_dict)[feature_ls]
    pred_valid = optimizer.predict(predict_feature_df)
    pred_valid = np.clip(pred_valid, 0, None)
    predict_label_array = np.full(shape=(rows * columns), fill_value=np.nan, dtype=np.float32)
    predict_label_array[mask_flat] = pred_valid
    predict_label_array = predict_label_array.reshape((rows, columns))

    with rio.open(output_path, 'w', **metadata) as dst:
        dst.write(predict_label_array, 1)
    del predict_feature_dict, predict_feature_df, pred_valid, predict_label_array
    gc.collect()


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pd.set_option('display.max_columns', None)

    data_path = r"Submit_dataset.xlsx"
    raw_df = pd.read_excel(data_path)
    df_monthly = raw_df.groupby(['site_id', 'year', 'month'], as_index=False).mean(numeric_only=True)

    feature_ls = df_monthly.loc[:, 'CO2air': 'fForest'].columns.tolist()
    label = 'DIC (mg/L)'


    """Start"""
    ML_model = 'CatBoost'
    experiment = 'S2'

    best_params = {'iterations': 978, 'depth': 8, 'learning_rate': 0.02841927464540473,
                   'l2_leaf_reg': 5.845420577455735,
                   'subsample': 0.9024744934818496, 'min_data_in_leaf': 53, 'random_strength': 0.526551012551942}
    optimizer = CatBoostRegressor(**best_params, bootstrap_type='Poisson', loss_function='RMSE', task_type="GPU", devices='1', random_state=42, verbose=0)
    """End"""


    experiment_ls = {
        'S1': ['CO2air', 'Tsur', 'Tsub', 'TotPrec'],
        'S2': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans'],
        'S3': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub'],
        'S4': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub', 'GPP', 'LAI', 'pCO2soil']
    }

    optimizer.fit(df_monthly[feature_ls], df_monthly[label])

    mask_array = rio.open(r"raster/mask.tif", 'r').read(1).astype(bool)
    rows, columns = mask_array.shape
    mask_flat = mask_array.flatten()

    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': np.float32,
        'width': columns,
        'height': rows,
        'crs': 'EPSG:4326',
        'transform': rio.transform.from_bounds(west=70, south=15, east=140, north=55, height=rows, width=columns),
        'nodata': np.nan,
        'compress': 'lzw',
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True,
        'interleave': 'band',
        'predictor': 2
    }

    model_ls = ['BCC-CSM2-MR', 'CanESM5', 'CESM2', 'GFDL-ESM4', 'IPSL-CM6A-LR',
                'MIROC-ES2L', 'MPI-ESM1-2-LR', 'NorESM2-LM', 'UKESM1-0-LL']
    ssp_ls = ['ssp126', 'ssp245', 'ssp585']
    if experiment == 'S4':
        status_ls = ['history', 'future']
    else:
        status_ls = ['future']
    args_ls = [(model, ssp, status) for model in model_ls for ssp in ssp_ls for status in status_ls]

    for args in tqdm(args_ls):
        process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls)


"""S3"""

def process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls):
    model, ssp, status = args
    folder_path = r'ChinaHeadwaterDIC'
    output_path = os.path.join(folder_path, 'DIC_concentration', 'CMIP6', experiment, ML_model, model, ssp, f'DIC_concentration_{model}_{ssp}_{status}.tif')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return

    CMIP6_feature_ls = ['CO2air', 'Evap', 'GPP', 'LAI', 'pCO2soil', 'ROsur', 'ROsub', 'ROratio',
                        'SMsur', 'SMsub', 'TotPrec', 'Trans', 'Tsur', 'Tsub', 'pHsur', 'pHsub']
    non_CMIP6_feature_ls = [feature for feature in feature_ls if feature not in CMIP6_feature_ls]
    """Elev, fCa, fSi, lithology, Kref, Esoil, fClaysur, fClaysub, fSiltsur, fSiltsub, fSandsur, fSandsub, SOCsur, SOCsub, GDP, PopDens, fCrop, fBare, fGrass, fForest"""
    assert len(CMIP6_feature_ls + non_CMIP6_feature_ls) == len(feature_ls)

    predict_feature_dict = {}

    for var in non_CMIP6_feature_ls:
        if var in ['GDP', 'PopDens', 'fCrop', 'fBare', 'fGrass', 'fForest']:
            tif_path = os.path.join(folder_path, 'features', f'annual_{var}.tif')
        else:
            tif_path = glob.glob(os.path.join(folder_path, '*', var, f'{var}.tif'))[0]
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            predict_feature_dict[var] = src.read(1).astype(np.float32).flatten()[mask_flat]

    dynamic_vars_for_exp = experiment_ls[experiment]

    for var in CMIP6_feature_ls:
        if var in dynamic_vars_for_exp:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_UBC', model, ssp, var, f'{var}_{model}_{ssp}_{status}.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_{status}.tif')
        else:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_UBC', model, ssp, var, f'{var}_{model}_{ssp}_history.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_history.tif')
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            arr_flat = src.read(1).astype(np.float32).flatten()[mask_flat]
            if var in ['CO2air', 'LAI', 'pCO2soil', 'ROratio', 'SMsur', 'SMsub', 'Tsur', 'Tsub', 'pHsur', 'pHsub']:
                predict_feature_dict[var] = arr_flat
            else:
                predict_feature_dict[var] = arr_flat / 12.0

    predict_feature_df = pd.DataFrame(predict_feature_dict)[feature_ls]
    pred_valid = optimizer.predict(predict_feature_df)
    pred_valid = np.clip(pred_valid, 0, None)
    predict_label_array = np.full(shape=(rows * columns), fill_value=np.nan, dtype=np.float32)
    predict_label_array[mask_flat] = pred_valid
    predict_label_array = predict_label_array.reshape((rows, columns))

    with rio.open(output_path, 'w', **metadata) as dst:
        dst.write(predict_label_array, 1)
    del predict_feature_dict, predict_feature_df, pred_valid, predict_label_array
    gc.collect()


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pd.set_option('display.max_columns', None)

    data_path = r"Submit_dataset.xlsx"
    raw_df = pd.read_excel(data_path)
    df_monthly = raw_df.groupby(['site_id', 'year', 'month'], as_index=False).mean(numeric_only=True)

    feature_ls = df_monthly.loc[:, 'CO2air': 'fForest'].columns.tolist()
    label = 'DIC (mg/L)'


    """Start"""
    ML_model = 'CatBoost'
    experiment = 'S3'

    best_params = {'iterations': 978, 'depth': 8, 'learning_rate': 0.02841927464540473,
                   'l2_leaf_reg': 5.845420577455735,
                   'subsample': 0.9024744934818496, 'min_data_in_leaf': 53, 'random_strength': 0.526551012551942}
    optimizer = CatBoostRegressor(**best_params, bootstrap_type='Poisson', loss_function='RMSE', task_type="GPU", devices='1', random_state=42, verbose=0)
    """End"""


    experiment_ls = {
        'S1': ['CO2air', 'Tsur', 'Tsub', 'TotPrec'],
        'S2': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans'],
        'S3': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub'],
        'S4': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub', 'GPP', 'LAI', 'pCO2soil']
    }

    optimizer.fit(df_monthly[feature_ls], df_monthly[label])

    mask_array = rio.open(r"raster/mask.tif", 'r').read(1).astype(bool)
    rows, columns = mask_array.shape
    mask_flat = mask_array.flatten()

    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': np.float32,
        'width': columns,
        'height': rows,
        'crs': 'EPSG:4326',
        'transform': rio.transform.from_bounds(west=70, south=15, east=140, north=55, height=rows, width=columns),
        'nodata': np.nan,
        'compress': 'lzw',
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True,
        'interleave': 'band',
        'predictor': 2
    }

    model_ls = ['BCC-CSM2-MR', 'CanESM5', 'CESM2', 'GFDL-ESM4', 'IPSL-CM6A-LR',
                'MIROC-ES2L', 'MPI-ESM1-2-LR', 'NorESM2-LM', 'UKESM1-0-LL']
    ssp_ls = ['ssp126', 'ssp245', 'ssp585']
    if experiment == 'S4':
        status_ls = ['history', 'future']
    else:
        status_ls = ['future']
    args_ls = [(model, ssp, status) for model in model_ls for ssp in ssp_ls for status in status_ls]

    for args in tqdm(args_ls):
        process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls)


"""S4"""

def process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls):
    model, ssp, status = args
    folder_path = r'ChinaHeadwaterDIC'
    output_path = os.path.join(folder_path, 'DIC_concentration', 'CMIP6', experiment, ML_model, model, ssp, f'DIC_concentration_{model}_{ssp}_{status}.tif')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return

    CMIP6_feature_ls = ['CO2air', 'Evap', 'GPP', 'LAI', 'pCO2soil', 'ROsur', 'ROsub', 'ROratio',
                        'SMsur', 'SMsub', 'TotPrec', 'Trans', 'Tsur', 'Tsub', 'pHsur', 'pHsub']
    non_CMIP6_feature_ls = [feature for feature in feature_ls if feature not in CMIP6_feature_ls]
    """Elev, fCa, fSi, lithology, Kref, Esoil, fClaysur, fClaysub, fSiltsur, fSiltsub, fSandsur, fSandsub, SOCsur, SOCsub, GDP, PopDens, fCrop, fBare, fGrass, fForest"""
    assert len(CMIP6_feature_ls + non_CMIP6_feature_ls) == len(feature_ls)

    predict_feature_dict = {}

    for var in non_CMIP6_feature_ls:
        if var in ['GDP', 'PopDens', 'fCrop', 'fBare', 'fGrass', 'fForest']:
            tif_path = os.path.join(folder_path, 'features', f'annual_{var}.tif')
        else:
            tif_path = glob.glob(os.path.join(folder_path, '*', var, f'{var}.tif'))[0]
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            predict_feature_dict[var] = src.read(1).astype(np.float32).flatten()[mask_flat]

    dynamic_vars_for_exp = experiment_ls[experiment]

    for var in CMIP6_feature_ls:
        if var in dynamic_vars_for_exp:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_processed', model, ssp, var, f'{var}_{model}_{ssp}_{status}.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_{status}.tif')
        else:
            if var not in ['pHsur', 'pHsub']:
                tif_path = os.path.join(folder_path, 'CMIP6_processed', model, ssp, var, f'{var}_{model}_{ssp}_history.tif')
            else:
                tif_path = os.path.join(folder_path, var, model, ssp, f'{var}_{model}_{ssp}_history.tif')
        assert os.path.exists(tif_path), var

        with rio.open(tif_path) as src:
            arr_flat = src.read(1).astype(np.float32).flatten()[mask_flat]
            if var in ['CO2air', 'LAI', 'pCO2soil', 'ROratio', 'SMsur', 'SMsub', 'Tsur', 'Tsub', 'pHsur', 'pHsub']:
                predict_feature_dict[var] = arr_flat
            else:
                predict_feature_dict[var] = arr_flat / 12.0

    predict_feature_df = pd.DataFrame(predict_feature_dict)[feature_ls]
    pred_valid = optimizer.predict(predict_feature_df)
    pred_valid = np.clip(pred_valid, 0, None)
    predict_label_array = np.full(shape=(rows * columns), fill_value=np.nan, dtype=np.float32)
    predict_label_array[mask_flat] = pred_valid
    predict_label_array = predict_label_array.reshape((rows, columns))

    with rio.open(output_path, 'w', **metadata) as dst:
        dst.write(predict_label_array, 1)
    del predict_feature_dict, predict_feature_df, pred_valid, predict_label_array
    gc.collect()


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pd.set_option('display.max_columns', None)

    data_path = r"Submit_dataset.xlsx"
    raw_df = pd.read_excel(data_path)
    df_monthly = raw_df.groupby(['site_id', 'year', 'month'], as_index=False).mean(numeric_only=True)

    feature_ls = df_monthly.loc[:, 'CO2air': 'fForest'].columns.tolist()
    label = 'DIC (mg/L)'


    """Start"""
    ML_model = 'CatBoost'
    experiment = 'S4'

    best_params = {'iterations': 978, 'depth': 8, 'learning_rate': 0.02841927464540473,
                   'l2_leaf_reg': 5.845420577455735,
                   'subsample': 0.9024744934818496, 'min_data_in_leaf': 53, 'random_strength': 0.526551012551942}
    optimizer = CatBoostRegressor(**best_params, bootstrap_type='Poisson', loss_function='RMSE', task_type="GPU", devices='1', random_state=42, verbose=0)
    """End"""


    experiment_ls = {
        'S1': ['CO2air', 'Tsur', 'Tsub', 'TotPrec'],
        'S2': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans'],
        'S3': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub'],
        'S4': ['CO2air', 'Tsur', 'Tsub', 'TotPrec', 'ROsur', 'ROsub', 'ROratio', 'Evap', 'Trans', 'SMsur', 'SMsub', 'pHsur', 'pHsub', 'GPP', 'LAI', 'pCO2soil']
    }

    optimizer.fit(df_monthly[feature_ls], df_monthly[label])

    mask_array = rio.open(r"raster/mask.tif", 'r').read(1).astype(bool)
    rows, columns = mask_array.shape
    mask_flat = mask_array.flatten()

    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': np.float32,
        'width': columns,
        'height': rows,
        'crs': 'EPSG:4326',
        'transform': rio.transform.from_bounds(west=70, south=15, east=140, north=55, height=rows, width=columns),
        'nodata': np.nan,
        'compress': 'lzw',
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': True,
        'interleave': 'band',
        'predictor': 2
    }

    model_ls = ['BCC-CSM2-MR', 'CanESM5', 'CESM2', 'GFDL-ESM4', 'IPSL-CM6A-LR',
                'MIROC-ES2L', 'MPI-ESM1-2-LR', 'NorESM2-LM', 'UKESM1-0-LL']
    ssp_ls = ['ssp126', 'ssp245', 'ssp585']
    if experiment == 'S4':
        status_ls = ['history', 'future']
    else:
        status_ls = ['future']
    args_ls = [(model, ssp, status) for model in model_ls for ssp in ssp_ls for status in status_ls]

    for args in tqdm(args_ls):
        process_predict_DIC(args, optimizer, mask_flat, rows, columns, metadata, feature_ls, ML_model, experiment, experiment_ls)
