"""History"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import os
import rasterio as rio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_predict_DIC(ML_model):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pd.set_option('display.max_columns', None)

    data_path = r"Submit_dataset.xlsx"
    raw_df = pd.read_excel(data_path)
    df_monthly = raw_df.groupby(['site_id', 'year', 'month'], as_index=False).mean(numeric_only=True)

    feature_ls = df_monthly.loc[:, 'CO2air': 'fForest'].columns.tolist()
    label = 'DIC (mg/L)'
    feature_df = df_monthly[feature_ls]
    label_df = df_monthly[label]


    mask_array = rio.open(r"mask.tif", 'r').read(1).astype(bool)

    if ML_model == 'LightGBM':
        best_params = {'n_estimators': 590, 'max_depth': 7, 'num_leaves': 31, 'learning_rate': 0.045383646844197226,
                       'subsample': 0.6004357984835966, 'colsample_bytree': 0.8679807796459401,
                       'min_child_samples': 17, 'reg_alpha': 0.13271553986071344, 'reg_lambda': 0.06452997894034963}
        optimizer = LGBMRegressor(**best_params, objective='regression', device='gpu', random_state=42, verbose=-1, gpu_device_id=0)
    elif ML_model == 'XGBoost':
        best_params = {'n_estimators': 538, 'max_depth': 10, 'learning_rate': 0.020266007580476524,
                       'subsample': 0.6422345667575905, 'colsample_bytree': 0.8017867311735984,
                       'min_child_weight': 5, 'reg_alpha': 0.0002079475552171627, 'reg_lambda': 8.036177050073895e-05}
        optimizer = XGBRegressor(random_state=42, booster='gbtree', objective='reg:squarederror', tree_method="hist",
                                 device="cuda:0", **best_params)
    elif ML_model == 'CatBoost':
        best_params = {'iterations': 978, 'depth': 8, 'learning_rate': 0.02841927464540473,'l2_leaf_reg': 5.845420577455735,
                       'subsample': 0.9024744934818496, 'min_data_in_leaf': 53, 'random_strength': 0.526551012551942}
        optimizer = CatBoostRegressor(**best_params, bootstrap_type='Poisson', loss_function='RMSE', task_type="GPU",
                                      devices='1', random_state=42, verbose=0)
    elif ML_model == 'HistGBDT':
        best_params = {'max_iter': 794, 'learning_rate': 0.07992772707809999, 'max_depth': 8, 'max_leaf_nodes': 38,
                       'min_samples_leaf': 32, 'l2_regularization': 9.930635343432396, 'max_bins': 225}
        optimizer = HistGradientBoostingRegressor(random_state=42, verbose=0, **best_params)
    elif ML_model == 'RandomForest':
        best_params = {'n_estimators': 540, 'max_depth': 12, 'min_samples_leaf': 10, 'min_samples_split': 20,
                       'max_features': 0.6341145659238546, 'max_samples': 0.8916234643702775}
        optimizer = RandomForestRegressor(random_state=42, verbose=0, **best_params)
    elif ML_model == 'ExtraTrees':
        best_params = {'n_estimators': 532, 'max_depth': 14, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 1.0}
        optimizer = ExtraTreesRegressor(random_state=42, verbose=0, **best_params)
    else:
        raise f"Don't support model: {ML_model}"

    optimizer.fit(feature_df, label_df)

    rows = 4452
    columns = 7792
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

    year_month_feature_ls = ['CO2air', 'TotPrec', 'Tsur', 'Tsub', 'Evap', 'ROsur', 'ROsub', 'ROratio',
                             'Trans', 'SMsur', 'SMsub', 'GPP', 'LAI']
    year_feature_ls = ['GDP', 'PopDens', 'fBare', 'fCrop', 'fForest', 'fGrass', 'pCO2soil']
    permanent_feature_ls = ['fClaysur', 'fClaysub', 'Elev', 'lithology', 'fCa', 'fSi',
                            'Kref', 'SOCsur', 'SOCsub', 'pHsur', 'pHsub', 'Esoil',
                            'fSiltsur', 'fSiltsub', 'fSandsur', 'fSandsub']

    assert len(set(year_month_feature_ls + year_feature_ls + permanent_feature_ls)) == len(set(feature_ls))

    folder_path = r'ChinaHeadwaterDIC'

    predict_feature_df = pd.DataFrame(data=np.zeros(shape=(int(rows * columns), len(feature_ls))),
                                      columns=feature_ls, dtype=np.float32)

    for var in permanent_feature_ls:
        tif_path = sorted(glob.glob(os.path.join(folder_path, '*', var, f'{var}.tif')))
        assert len(tif_path) == 1, var
        with rio.open(tif_path[0]) as src:
            predict_feature_df[var] = src.read(1).astype(np.float32).flatten()
    del var

    for year in tqdm(range(2000, 2021), total=21, desc='Year Updating...'):
        for var in year_feature_ls:
            tif_ls = sorted(glob.glob(os.path.join(folder_path, '*', var, f'{var}_{year}.tif')))
            assert len(tif_ls) == 1
            with rio.open(tif_ls[0]) as src:
                predict_feature_df[var] = src.read(1).astype(np.float32).flatten()
        del var

        for month in range(1, 13):
            output_path = os.path.join(folder_path, 'DIC_concentration', 'History', 'LightGBM',
                                       f'DIC_concentration_{year}_{month:02d}.tif')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            for var in year_month_feature_ls:
                tif_ls = sorted(glob.glob(os.path.join(folder_path, '*', var, f'{var}_{year}_{month:02d}.tif')))
                assert len(tif_ls) == 1
                with rio.open(tif_ls[0]) as src:
                    predict_feature_df[var] = src.read(1).astype(np.float32).flatten()

            del var

            predict_feature_df = predict_feature_df.reindex(columns=feature_ls)
            predict_label_array = optimizer.predict(predict_feature_df).T.reshape((rows, columns))
            predict_label_array = np.clip(predict_label_array, 0, None)
            predict_label_array = np.ma.masked_where(~mask_array, predict_label_array)

            plt.imshow(predict_label_array, cmap='Spectral_r')
            plt.show()

            with rio.open(output_path, 'w', **metadata) as dst:
                dst.write(predict_label_array, 1)


if __name__ == '__main__':
    ML_model = 'LightGBM'
    process_predict_DIC(ML_model)

