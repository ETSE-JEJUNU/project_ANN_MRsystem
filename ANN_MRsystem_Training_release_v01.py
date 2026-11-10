# ==================================================================================================
# A Unified MLP-Based ANN Model of MR systems 
# devloped by Jongmin Choi, Jeju National University, Jeju-si, Republic of Korea
# ==================================================================================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError

warnings.filterwarnings('ignore', category=UserWarning)

# Randomseed Case Study
SEED = 777 # Default
# SEED = 13
# SEED = 17
# SEED = 42
# SEED = 101
# SEED = 123
# SEED = 503
# SEED = 1009
# SEED = 2026
# SEED = 9999

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------------------
# 1. Load data and define columns
# ------------------------------------------------------------------------------
file_path = "Database_for_release.xlsx" # Database file located in the working directory
df_raw = pd.read_excel(file_path, sheet_name=0)

input_cols_full = [
    'Load_index(-)', 't_R1(s)', 't_R3(s)', 'T_amb(K)', 'T_HHEX_f_o(K)', 
    'Freq(Hz)', 'UF(-)', 'mdot_w(kg/min)', 'T_CHEX_w_i(K)'
]
input_cols_no_index = [col for col in input_cols_full if col != 'Load_index(-)']

output_cols = ['T_span(K)', 'Wdot_m(W)', 'T_f_c(K)', 'Qdot_c(W)', 'COP(-)']

print(f"=== Raw Dataset Loaded: Total {len(df_raw)} samples ===")

# ------------------------------------------------------------------------------
# 2. Isolate independent test set (80% Dev Set / 20% Independent Test Set)
# ------------------------------------------------------------------------------
dev_df, test_df = train_test_split(
    df_raw, 
    test_size=0.2, 
    stratify=df_raw['Load_index(-)'], 
    random_state=SEED
)

dev_df = dev_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

dev_df.to_csv('df_raw_dev.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('df_raw_test.csv', index=False, encoding='utf-8-sig')
print(f"-> Dataset Split Complete: Dev Set ({len(dev_df)}), Test Set ({len(test_df)})")

# ------------------------------------------------------------------------------
# 3. Generate and save 5-fold stratified split indices
# ------------------------------------------------------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(dev_df, dev_df['Load_index(-)']))

with open('kfold_splits.pkl', 'wb') as f:
    pickle.dump(fold_splits, f)

# ------------------------------------------------------------------------------
# 4. Build ANN model
# ------------------------------------------------------------------------------
def build_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        
        # Hidden neurons case 1
        # Dense(8, activation='relu', name='Hidden_1'),
        
        # Hidden neurons case 2
        # Dense(16, activation='relu', name='Hidden_1'),
        # Dense(8, activation='relu', name='Hidden_2'),
        
        # Hidden neurons case 3
        # Dense(32, activation='relu', name='Hidden_1'),
        # Dense(16, activation='relu', name='Hidden_2'),
        # Dense(8, activation='relu', name='Hidden_3'),
        
        # Hidden neurons case 4
        Dense(64, activation='relu', name='Hidden_1'),
        Dense(32, activation='relu', name='Hidden_2'),
        Dense(16, activation='relu', name='Hidden_3'),
        
        # Hidden neurons case 5
        # Dense(128, activation='relu', name='Hidden_1'),
        # Dense(64, activation='relu', name='Hidden_2'),
        # Dense(32, activation='relu', name='Hidden_3'),
        
        Dense(output_dim, activation='linear', name='Output')
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=[RootMeanSquaredError(name='rmse')]
    )
    return model

# ------------------------------------------------------------------------------
# 5. Model training for Case (a) through Case (d)
# ------------------------------------------------------------------------------
cases_config = {
    'case_a': {'use_load_index': False, 'transform_data': False},
    'case_b': {'use_load_index': True,  'transform_data': False},
    'case_c': {'use_load_index': False, 'transform_data': True},
    'case_d': {'use_load_index': True,  'transform_data': True}
}

for case_name, config in cases_config.items():
    print(f"\n==================================================")
    print(f"  Starting Pipeline: [{case_name.upper()}]")
    print(f"==================================================")
    
    # Create save directory
    os.makedirs(f'models_{case_name}', exist_ok=True)
    
    # Determine input columns to use
    curr_input_cols = input_cols_full if config['use_load_index'] else input_cols_no_index
    
    # 5-fold cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        # Create independent dataframes for each fold
        train_fold = dev_df.iloc[train_idx].copy()
        val_fold = dev_df.iloc[val_idx].copy()
        test_fold = test_df.copy()
        
        # ----------------------------------------------------------------------
        # [In-Fold Preprocessing: Data Transformation]
        # ----------------------------------------------------------------------
        if config['transform_data']:
            # 1) Calculate average values using only Load_index=1 data within the training fold
            load1_train = train_fold[train_fold['Load_index(-)'] == 1]
            trans_cols = ['mdot_w(kg/min)', 'T_CHEX_w_i(K)', 'Qdot_c(W)', 'COP(-)']
            train_means = load1_train[trans_cols].mean()
            
            # 2) Replace no-load (Load_index=0) samples in Train, Val, and Test exclusively with the Train average values
            for df_target in [train_fold, val_fold, test_fold]:
                mask_no_load = (df_target['Load_index(-)'] == 0)
                df_target.loc[mask_no_load, trans_cols] = train_means.values
        
        # Extract NumPy arrays
        x_train = train_fold[curr_input_cols].values
        y_train = train_fold[output_cols].values
        x_val   = val_fold[curr_input_cols].values
        y_val   = val_fold[output_cols].values
        
        # ----------------------------------------------------------------------
        # [In-Fold Preprocessing: Feature & Target Scaling]
        # ----------------------------------------------------------------------
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        # Normalize using only the training fold data
        x_train_scaled = scaler_x.fit_transform(x_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        
        # Apply normalization parameters from the training fold data to validation set
        x_val_scaled = scaler_x.transform(x_val)
        y_val_scaled = scaler_y.transform(y_val)
        
        # Save normalization parameters
        with open(f'models_{case_name}/scaler_x_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(scaler_x, f)
        with open(f'models_{case_name}/scaler_y_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
            
        # ----------------------------------------------------------------------
        # Model Training & Early Stopping
        # ----------------------------------------------------------------------
        model = build_model(len(curr_input_cols), len(output_cols))
        model_filename = f'models_{case_name}/ann_model_fold{fold_idx}.keras'
        
        esc = EarlyStopping(monitor='val_loss', patience=20, verbose=0, restore_best_weights=True)
        cp = ModelCheckpoint(filepath=model_filename, monitor='val_loss', verbose=0, save_best_only=True)
        
        history = model.fit(
            x_train_scaled, y_train_scaled,
            epochs=300,
            batch_size=16,
            validation_data=(x_val_scaled, y_val_scaled),
            callbacks=[esc, cp],
            verbose=0
        )
        
        # Save loss curve data
        with open(f'models_{case_name}/history_fold{fold_idx}.pkl', 'wb') as f:
            pickle.dump(history.history, f)
            
        print(f" [{case_name.upper()}] Fold {fold_idx} Trained & Saved -> Val Loss: {min(history.history['val_loss']):.5f}")

print("\n=== ALL CASES TRAINING COMPLETED SUCCESSFULLY ===")