# ==================================================================================================
# A Unified MLP-Based ANN Model of MR systems 
# devloped by Jongmin Choi, Jeju National University, Jeju-si, Republic of Korea
# Test Code
# ==================================================================================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
from scipy.interpolate import interp1d

# Suppress TensorFlow/Keras warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("Warning: Training and validation code should be run before the Test code starts")

# 1. Define columns
input_cols_full = [
    'Load_index(-)', 't_R1(s)', 't_R3(s)', 'T_amb(K)', 'T_HHEX_f_o(K)', 
    'Freq(Hz)', 'UF(-)', 'mdot_w(kg/min)', 'T_CHEX_w_i(K)'
]
input_cols_no_index = [col for col in input_cols_full if col != 'Load_index(-)']
output_cols = ['T_span(K)', 'Wdot_m(W)', 'T_f_c(K)', 'Qdot_c(W)', 'COP(-)']

# 2. Load data and split structures
dev_df = pd.read_csv('df_raw_dev.csv')
test_df_raw = pd.read_csv('df_raw_test.csv')

with open('kfold_splits.pkl', 'rb') as f:
    fold_splits = pickle.load(f)

cases = ['case_a', 'case_b', 'case_c', 'case_d']
case_results = {}

# ==============================================================================
# 3. Ensemble evaluation loop on the independent test set for each case
# ==============================================================================
for case_name in cases:
    use_load_index = 'b' in case_name or 'd' in case_name
    transform_data = 'c' in case_name or 'd' in case_name
    curr_input_cols = input_cols_full if use_load_index else input_cols_no_index
    
    fold_test_preds = []
    
    for fold_idx, (train_idx, _) in enumerate(fold_splits, start=1):
        # 1) Load Scaler & Model
        with open(f'models_{case_name}/scaler_x_fold{fold_idx}.pkl', 'rb') as f:
            scaler_x = pickle.load(f)
        with open(f'models_{case_name}/scaler_y_fold{fold_idx}.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
            
        model = load_model(f'models_{case_name}/ann_model_fold{fold_idx}.keras', compile=False)
        
        # 2) Apply In-Fold Transformation to Test Set
        test_fold_df = test_df_raw.copy()
        if transform_data:
            train_fold = dev_df.iloc[train_idx]
            load1_train = train_fold[train_fold['Load_index(-)'] == 1]
            trans_cols = ['mdot_w(kg/min)', 'T_CHEX_w_i(K)', 'Qdot_c(W)', 'COP(-)']
            train_means = load1_train[trans_cols].mean()
            
            mask_no_load = (test_fold_df['Load_index(-)'] == 0)
            for col in trans_cols:
                test_fold_df.loc[mask_no_load, col] = train_means[col]
        
        # 3) Normalize Test dataset
        x_test = test_fold_df[curr_input_cols].values
        x_test_scaled = scaler_x.transform(x_test)
        
        
        # 4) Predict normalized Test dataset
        y_pred_scaled = model(x_test_scaled, training=False).numpy()
        y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
        fold_test_preds.append(y_pred_orig)
    
    # Predict 5-Fold ensemble average
    ensemble_pred = np.mean(fold_test_preds, axis=0)
    pred_df = test_df_raw.copy()
    
    for idx, col in enumerate(output_cols):
        pred_df[f'Pred_{col}'] = ensemble_pred[:, idx]
        
    # 4) Post-processing (Restore physical true zero values for no-load conditions in the proposed transformation model)
    if transform_data:
        mask_noload_test = (pred_df['Load_index(-)'] == 0)
        pred_df.loc[mask_noload_test, 'Pred_Qdot_c(W)'] = 0.0
        pred_df.loc[mask_noload_test, 'Pred_COP(-)'] = 0.0
        
    case_results[case_name] = pred_df
    pred_df.to_csv(f'parity_test_results_{case_name}.csv', index=False, encoding='utf-8-sig')

print("=== Independent Test Set Ensemble Prediction Completed for All Cases ===")