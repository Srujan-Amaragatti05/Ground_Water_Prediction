import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def preprocess_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Extracting YEAR and LOCATION_ID...")
    # Handle Date to Year.
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['YEAR'] = df['Date'].dt.year
    
    # Location ID
    df['LOCATION_ID'] = df['STATE_UT'].astype(str) + '_' + \
                        df['DISTRICT'].astype(str) + '_' + \
                        df['BLOCK'].astype(str) + '_' + \
                        df['VILLAGE'].astype(str) + '_' + \
                        df['LATITUDE'].astype(str) + '_' + \
                        df['LONGITUDE'].astype(str)
                        
    print("Aggregating to annual data...")
    # Features to aggregate
    features = ['WL(mbgl)', 'Rainfall', 'Temperature', 'Humidity']
    # Group by LOCATION_ID and YEAR and calculate mean
    annual_df = df.groupby(['LOCATION_ID', 'YEAR'])[features].mean().reset_index()
    
    # Sort chronologically
    annual_df = annual_df.sort_values(by=['LOCATION_ID', 'YEAR'])
    
    print("Creating sequences...")
    sequences = []
    
    # We will build sequences of 5 years predicting the 6th year WL
    grouped = annual_df.groupby('LOCATION_ID')
    for loc_id, group in grouped:
        group = group.sort_values(by='YEAR').reset_index(drop=True)
        n_years = len(group)
        if n_years < 6:
            continue
            
        years = group['YEAR'].values
        feats = group[features].values # [WL, Rainfall, Temperature, Humidity]
        wl = group['WL(mbgl)'].values
        
        for i in range(n_years - 5):
            # Check for consecutive years: if year at i+5 minus year at i is 5, they are consecutive (since distinct)
            if years[i+5] - years[i] == 5:
                seq_x = feats[i:i+5]
                seq_y = wl[i+5] # Predict next year's WL
                target_year = years[i+5]
                sequences.append({
                    'LOCATION_ID': loc_id,
                    'TARGET_YEAR': target_year,
                    'X': seq_x,
                    'y': seq_y
                })
                    
    print(f"Total annual sequences created: {len(sequences)}")
    
    # Split based on TARGET_YEAR
    train_seqs = [s for s in sequences if s['TARGET_YEAR'] <= 2018]
    val_seqs = [s for s in sequences if 2019 <= s['TARGET_YEAR'] <= 2021]
    test_seqs = [s for s in sequences if 2022 <= s['TARGET_YEAR'] <= 2023]
    
    print(f"Train samples: {len(train_seqs)}")
    print(f"Validation samples: {len(val_seqs)}")
    print(f"Test samples: {len(test_seqs)}")
    
    X_train_raw = np.array([s['X'] for s in train_seqs])
    y_train_raw = np.array([s['y'] for s in train_seqs])
    
    X_val_raw = np.array([s['X'] for s in val_seqs])
    y_val_raw = np.array([s['y'] for s in val_seqs])
    
    X_test_raw = np.array([s['X'] for s in test_seqs])
    y_test_raw = np.array([s['y'] for s in test_seqs])
    test_target_years = np.array([s['TARGET_YEAR'] for s in test_seqs])
    
    # Scaling
    # Features: [WL, Rainfall, Temperature, Humidity]
    
    X_train_2d = X_train_raw.reshape(-1, 4) if len(X_train_raw) > 0 else np.empty((0, 4))
    X_val_2d = X_val_raw.reshape(-1, 4) if len(X_val_raw) > 0 else np.empty((0, 4))
    X_test_2d = X_test_raw.reshape(-1, 4) if len(X_test_raw) > 0 else np.empty((0, 4))
    
    feature_scaler = StandardScaler()
    if len(X_train_2d) > 0:
        X_train_scaled = feature_scaler.fit_transform(X_train_2d).reshape(-1, 5, 4)
    else:
        X_train_scaled = np.empty((0, 5, 4))
        
    X_val_scaled = feature_scaler.transform(X_val_2d).reshape(-1, 5, 4) if len(X_val_2d) > 0 else np.empty((0, 5, 4))
    X_test_scaled = feature_scaler.transform(X_test_2d).reshape(-1, 5, 4) if len(X_test_2d) > 0 else np.empty((0, 5, 4))
    
    target_scaler = StandardScaler()
    if len(y_train_raw) > 0:
        y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1))
    else:
        y_train_scaled = np.empty((0, 1))
        
    y_val_scaled = target_scaler.transform(y_val_raw.reshape(-1, 1)) if len(y_val_raw) > 0 else np.empty((0, 1))
    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)) if len(y_test_raw) > 0 else np.empty((0, 1))
    
    scalers = {'feature_scaler': feature_scaler, 'target_scaler': target_scaler}
    
    data = {
        'X_train': X_train_scaled, 'y_train': y_train_scaled,
        'X_val': X_val_scaled, 'y_val': y_val_scaled,
        'X_test': X_test_scaled, 'y_test': y_test_scaled,
        'y_test_raw': y_test_raw, 'test_target_years': test_target_years,
        'sequences': sequences
    }
    
    return data, scalers, annual_df

def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_model():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fixed_real_dataset.csv')
    if not os.path.exists(csv_path):
        # Fallback if running from root
        csv_path = 'fixed_real_dataset.csv'
        
    data, scalers, annual_df = preprocess_data(csv_path)
    
    print(f"Valid locations: {annual_df['LOCATION_ID'].nunique()}")
    print(f"Number of annual sequences: {len(data['sequences'])}")
    print(f"X shape: {data['X_train'].shape}")
    print(f"y shape: {data['y_train'].shape}")
    
    model = build_model((5, 4))
    print(f"Parameter count: {model.count_params()}")
    
    model_save_path = os.path.join(os.path.dirname(__file__), 'phase2_lstm_model.keras')
    
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    print("Training model...")
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=50,
        batch_size=32,
        shuffle=False,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    epochs_trained = len(history.epoch)
    best_val_loss = min(history.history['val_loss'])
    print(f"Epochs actually trained: {epochs_trained}")
    print(f"Best validation loss: {best_val_loss}")
    
    # Save scalers and metadata
    scaler_save_path = os.path.join(os.path.dirname(__file__), 'phase2_scalers.pkl')
    metadata = {
        'feature_order': ['WL(mbgl)', 'Rainfall', 'Temperature', 'Humidity'],
        'sequence_length': 5,
        'scalers': scalers
    }
    joblib.dump(metadata, scaler_save_path)
    print(f"Scaler saved to: {scaler_save_path}")
    print(f"Model saved to: {model_save_path}")
    
    print("Evaluating on test set...")
    best_model = tf.keras.models.load_model(model_save_path)
    
    y_pred_scaled = best_model.predict(data['X_test'])
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
    y_all_raw = np.vstack((data['y_train'], data['y_val'], data['y_test']))
    
    y_all_pred_scaled = best_model.predict(X_all_raw)
    y_all_pred_raw = scalers['target_scaler'].inverse_transform(y_all_pred_scaled)
    y_all_true_raw = scalers['target_scaler'].inverse_transform(y_all_raw)
    
    bt_rmse = np.sqrt(mean_squared_error(y_all_true_raw, y_all_pred_raw))
    bt_mae = mean_absolute_error(y_all_true_raw, y_all_pred_raw)
    bt_r2 = r2_score(y_all_true_raw, y_all_pred_raw)
    
    print(f"Backtesting RMSE: {bt_rmse:.4f}")
    print(f"Backtesting MAE: {bt_mae:.4f}")
    print(f"Backtesting R2: {bt_r2:.4f}")
    
    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'phase2_loss_plot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_raw, y_pred_raw, alpha=0.5)
    plt.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], 'r--')
    plt.xlabel('Actual WL(mbgl)')
    plt.ylabel('Predicted WL(mbgl)')
    plt.title('Actual vs Predicted (Test Set)')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'phase2_actual_vs_predicted.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_all_true_raw, y_all_pred_raw, alpha=0.5)
    plt.plot([y_all_true_raw.min(), y_all_true_raw.max()], [y_all_true_raw.min(), y_all_true_raw.max()], 'r--')
    plt.xlabel('Actual WL(mbgl)')
    plt.ylabel('Predicted WL(mbgl)')
    plt.title('Backtesting Actual vs Predicted (All Data)')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'phase2_backtesting.png'))
    plt.close()
    
    print("Reload verification completed.")
    return data, history, best_model, scalers

if __name__ == "__main__":
    train_model()
