import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Reuse the verified preprocessing pipeline from LSTM baseline
from phase2_lstm import preprocess_data

def get_hybrid_data(csv_path):
    print("Loading and preprocessing data via Phase-2 verified pipeline...")
    data, scalers, annual_df = preprocess_data(csv_path)
    
    print("Extracting spatial features...")
    df = pd.read_csv(csv_path)
    df['LOCATION_ID'] = df['STATE_UT'].astype(str) + '_' + \
                        df['DISTRICT'].astype(str) + '_' + \
                        df['BLOCK'].astype(str) + '_' + \
                        df['VILLAGE'].astype(str) + '_' + \
                        df['LATITUDE'].astype(str) + '_' + \
                        df['LONGITUDE'].astype(str)
                        
    loc_mapping = df.drop_duplicates('LOCATION_ID').set_index('LOCATION_ID')[['LATITUDE', 'LONGITUDE']]
    
    train_seqs = [s for s in data['sequences'] if s['TARGET_YEAR'] <= 2018]
    val_seqs = [s for s in data['sequences'] if 2019 <= s['TARGET_YEAR'] <= 2021]
    test_seqs = [s for s in data['sequences'] if 2022 <= s['TARGET_YEAR'] <= 2023]
    
    S_train_raw = np.array([loc_mapping.loc[s['LOCATION_ID']].values for s in train_seqs])
    S_val_raw = np.array([loc_mapping.loc[s['LOCATION_ID']].values for s in val_seqs])
    S_test_raw = np.array([loc_mapping.loc[s['LOCATION_ID']].values for s in test_seqs])
    
    spatial_scaler = StandardScaler()
    if len(S_train_raw) > 0:
        S_train_scaled = spatial_scaler.fit_transform(S_train_raw)
    else:
        S_train_scaled = np.empty((0, 2))
        
    S_val_scaled = spatial_scaler.transform(S_val_raw) if len(S_val_raw) > 0 else np.empty((0, 2))
    S_test_scaled = spatial_scaler.transform(S_test_raw) if len(S_test_raw) > 0 else np.empty((0, 2))
    
    data['S_train'] = S_train_scaled
    data['S_val'] = S_val_scaled
    data['S_test'] = S_test_scaled
    
    scalers['spatial_scaler'] = spatial_scaler
    
    return data, scalers, annual_df

def build_hybrid_model(temporal_shape, spatial_shape):
    # Temporal branch
    temporal_input = Input(shape=temporal_shape, name="temporal_input")
    x_temp = GRU(64, activation='tanh')(temporal_input)
    x_temp = Dropout(0.2)(x_temp)
    
    # Spatial branch
    spatial_input = Input(shape=spatial_shape, name="spatial_input")
    x_spatial = Dense(16, activation='relu')(spatial_input)
    x_spatial = Dense(8, activation='relu')(x_spatial)
    
    # Fusion
    merged = Concatenate()([x_temp, x_spatial])
    x = Dense(16, activation='relu')(merged)
    output = Dense(1, name="output")(x)
    
    model = Model(inputs=[temporal_input, spatial_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_hybrid():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fixed_real_dataset.csv')
    if not os.path.exists(csv_path):
        csv_path = 'fixed_real_dataset.csv'
        
    data, scalers, annual_df = get_hybrid_data(csv_path)
    
    print(f"Valid locations: {annual_df['LOCATION_ID'].nunique()}")
    print(f"Number of annual sequences: {len(data['sequences'])}")
    print(f"Temporal X shape: {data['X_train'].shape}")
    print(f"Spatial S shape: {data['S_train'].shape}")
    print(f"y shape: {data['y_train'].shape}")
    
    model = build_hybrid_model((5, 4), (2,))
    print(f"Parameter count: {model.count_params()}")
    
    model_save_path = os.path.join(os.path.dirname(__file__), 'phase2_hybrid_model.keras')
    
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    print("Training Hybrid model...")
    history = model.fit(
        [data['X_train'], data['S_train']], data['y_train'],
        validation_data=([data['X_val'], data['S_val']], data['y_val']),
        epochs=50,
        batch_size=32,
        shuffle=False,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    epochs_trained = len(history.epoch)
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
    print(f"Epochs actually trained: {epochs_trained}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss}")
    
    # Save scalers and metadata
    scaler_save_path = os.path.join(os.path.dirname(__file__), 'phase2_hybrid_scalers.pkl')
    metadata = {
        'temporal_features': ['WL(mbgl)', 'Rainfall', 'Temperature', 'Humidity'],
        'spatial_features': ['LATITUDE', 'LONGITUDE'],
        'sequence_length': 5,
        'scalers': scalers
    }
    joblib.dump(metadata, scaler_save_path)
    
    print("Evaluating on test set...")
    best_model = tf.keras.models.load_model(model_save_path)
    
    y_pred_scaled = best_model.predict([data['X_test'], data['S_test']])
    y_pred_raw = scalers['target_scaler'].inverse_transform(y_pred_scaled)
    y_test_raw = data['y_test_raw'].reshape(-1, 1)
    
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    mae = mean_absolute_error(y_test_raw, y_pred_raw)
    r2 = r2_score(y_test_raw, y_pred_raw)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    print("Performing backtesting on all data...")
    X_all_raw = np.vstack((data['X_train'], data['X_val'], data['X_test']))
    S_all_raw = np.vstack((data['S_train'], data['S_val'], data['S_test']))
    y_all_raw = np.vstack((data['y_train'], data['y_val'], data['y_test']))
    
    y_all_pred_scaled = best_model.predict([X_all_raw, S_all_raw])
    y_all_pred_raw = scalers['target_scaler'].inverse_transform(y_all_pred_scaled)
    y_all_true_raw = scalers['target_scaler'].inverse_transform(y_all_raw)
    
    bt_rmse = np.sqrt(mean_squared_error(y_all_true_raw, y_all_pred_raw))
    bt_mae = mean_absolute_error(y_all_true_raw, y_all_pred_raw)
    bt_r2 = r2_score(y_all_true_raw, y_all_pred_raw)
    
    print(f"Backtesting RMSE: {bt_rmse:.4f}")
    print(f"Backtesting MAE: {bt_mae:.4f}")
    print(f"Backtesting R2: {bt_r2:.4f}")
    
    # Verify save/reload consistency
    reloaded_model = tf.keras.models.load_model(model_save_path)
    reloaded_preds_scaled = reloaded_model.predict([data['X_test'], data['S_test']])
    max_diff = np.max(np.abs(y_pred_scaled - reloaded_preds_scaled))
    print(f"Save/reload maximum prediction difference: {max_diff}")
    if np.allclose(y_pred_scaled, reloaded_preds_scaled, atol=1e-5):
        print("Save/reload consistency verified.")
    else:
        print("WARNING: Save/reload consistency failed.")

    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Hybrid Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'phase2_hybrid_loss_plot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_raw, y_pred_raw, alpha=0.5)
    plt.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], 'r--')
    plt.xlabel('Actual WL(mbgl)')
    plt.ylabel('Predicted WL(mbgl)')
    plt.title('Hybrid Actual vs Predicted (Test Set)')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'phase2_hybrid_actual_vs_predicted.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_all_true_raw, y_all_pred_raw, alpha=0.5)
    plt.plot([y_all_true_raw.min(), y_all_true_raw.max()], [y_all_true_raw.min(), y_all_true_raw.max()], 'r--')
    plt.xlabel('Actual WL(mbgl)')
    plt.ylabel('Predicted WL(mbgl)')
    plt.title('Hybrid Backtesting Actual vs Predicted (All Data)')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'phase2_hybrid_backtesting.png'))
    plt.close()

if __name__ == "__main__":
    import time
    start_time = time.time()
    train_hybrid()
    print(f"Total script duration: {time.time() - start_time:.2f} seconds")
