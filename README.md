Title: Rock Weathering Amplifies Present and Future Terrestrial Carbon Sinks in China

Authors: Deshuai Zhanga, Ronny Lauerwaldb, Yuanyuan Huangc, Pierre Regnierd, Philippe Ciaise, Yanjun Lif, Xiaosheng Xiaf, Guoping Tanga, Weihua Tanga, Wenping Yuang, *, Quanzhou Gaoa, *, Haicheng Zhanga, *

Submit_dataset.xlsx: The quality-controlled national database of 6,026 in-situ headwater dissolved inorganic carbon measurements across 1,657 headwater catchments (drainage area ≤ 100 km²) in China (2000–2022). It includes unique sample and site identifiers (sample_id, site_id), geographic coordinates (longitude, latitude), sampling dates (year, month, day), drainage areas (Area), hydrochemical parameters (Twater, pHwater, DIC, SO42-), 36 environmental predictors, literature references (reference), and metadata definitions.

The core analytical workflow:
* 01_hydrochemical_reconstruction_speciation: Empirical reconstructions of stream water temperature and water pH, and freshwater carbonate thermodynamic equilibrium (aqueous DIC speciation).

* 02_ML_training_and_cross_validation: Training and hyperparameter configuration of the six tree-based machine-learning models (LightGBM, XGBoost, CatBoost, HistGBDT, RandomForest, and ExtraTrees) for predicting headwater DIC concentration, evaluated via both sample-level 10-fold cross-validation and site-level (grouped) 10-fold cross-validation across multi-year, yearly, and monthly scales.

* 03_SHAP_interpretability_analysis: Multi-model SHapley Additive exPlanations (SHAP) analysis quantifying the relative importance and partial dependence patterns of the 36 environmental predictors for headwater DIC concentration.

* 04_gridded_simulation_and_WCS_uncertainty: CMIP6 delta-change bias correction, 1-km national gridded simulations of headwater DIC concentration and runoff-coupled land-to-river DIC flux, process-based partitioning of weathering-derived carbon sink (WCS) using GLiM lithology-specific kinetics, and ±1σ sensitivity and 10,000-iteration Monte Carlo uncertainty propagation.

* 05_factorial_attribution_experiments: Factorial simulation experiments quantifying the relative contributions of climate, hydrology, soil properties, and vegetation dynamics to projected future changes in national WCS.

System Requirements & Dependencies:
* Python ≥ 3.9
* Core packages: numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost, shap, rasterio, openpyxl, tqdm, matplotlib, seaborn

