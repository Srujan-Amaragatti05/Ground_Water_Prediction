"""
backend/dl/phase2_hybrid_final_experiment.py
============================================
Final Controlled One-Shot Improvement Experiment for Phase-2 Hybrid Spatio-Temporal GRU.

Objective:
  Determine whether the Hybrid model can be legitimately improved on held-out test
  data (2022-2023) using technically justified representational enhancements,
  without parameter bloat or endless hyperparameter searching.

Protocol:
  - Exact same official dataset (fixed_real_dataset.csv)
  - Exact same temporal partitions:
      Train: <= 2018 (126,992 sequences)
      Val:   2019-2021 (28,914 sequences)
      Test:  2022-2023 (21,080 sequences)
  - Scalers fit strictly on training data (zero test leakage)
  - Model selection based strictly on validation performance
  - Evaluation of the selected candidate on the untouched test set
  - Tolerance-Based Prediction Accuracy (±0.5m, ±1.0m, ±2.0m)
  - Production artifacts remain completely untouched
"""

from __future__ import annotations

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# Import verified Phase-2 data loading and utilities
from phase2_full_training_report import (
    load_and_preprocess_all,
    calculate_metrics,
    analyze_training_loss,
    get_architecture_details,
    RANDOM_SEED,
    DATASET_PATH,
    REPORT_DIR,
    TEMPORAL_FEATURES,
    SPATIAL_FEATURES,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE
)


# ===========================================================================
# 1. MODEL ARCHITECTURE BUILDERS
# ===========================================================================

def build_baseline_v0(temporal_shape=(5, 4), spatial_shape=(2,)) -> tf.keras.Model:
    """Baseline V0: Standard 2-branch sequential feedforward fusion."""
    t_in = Input(shape=temporal_shape, name="temporal_input")
    x_t = GRU(64, activation='tanh', name="gru_temporal")(t_in)
    x_t = Dropout(0.2, name="dropout_temporal")(x_t)
    
    s_in = Input(shape=spatial_shape, name="spatial_input")
    x_s = Dense(16, activation='relu', name="dense_spatial_1")(s_in)
    x_s = Dense(8, activation='relu', name="dense_spatial_2")(x_s)
    
    merged = Concatenate(name="fusion_concat")([x_t, x_s])
    x_f = Dense(16, activation='relu', name="dense_fusion")(merged)
    out = Dense(1, name="predicted_wl")(x_f)
    
    model = Model(inputs=[t_in, s_in], outputs=out, name="Hybrid_Baseline_V0")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def build_candidate_a_residual_skip(temporal_shape=(5, 4), spatial_shape=(2,)) -> tf.keras.Model:
    """
    Candidate A: Residual Temporal Skip Connection.
    Maintains a direct connection from the 64-D GRU output to the final prediction,
    concatenated with an 8-D spatial conditioning offset. Prevents temporal signal
    bottlenecking through a small 16-D layer.
    """
    t_in = Input(shape=temporal_shape, name="temporal_input")
    x_t = GRU(64, activation='tanh', name="gru_temporal")(t_in)
    x_t = Dropout(0.2, name="dropout_temporal")(x_t)
    
    s_in = Input(shape=spatial_shape, name="spatial_input")
    x_s = Dense(16, activation='relu', name="dense_spatial_1")(s_in)
    x_s = Dense(8, activation='relu', name="dense_spatial_2")(x_s)
    
    # Spatial conditioning representation
    h_spatial = Dense(16, activation='relu', name="spatial_modulation")(Concatenate()([x_t, x_s]))
    
    # Residual skip: Direct GRU temporal features concatenated with spatial modulation
    fused = Concatenate(name="residual_temporal_skip")([x_t, h_spatial])
    out = Dense(1, name="predicted_wl")(fused)
    
    model = Model(inputs=[t_in, s_in], outputs=out, name="Hybrid_Candidate_A_ResidualSkip")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def build_candidate_b_layernorm_skip(temporal_shape=(5, 4), spatial_shape=(2,)) -> tf.keras.Model:
    """
    Candidate B: LayerNormalization + Residual Skip.
    Normalizes feature representations across temporal and spatial trunks before
    fusion to prevent magnitude dominance and stabilize intermediate representations.
    """
    t_in = Input(shape=temporal_shape, name="temporal_input")
    x_t = GRU(64, activation='tanh', name="gru_temporal")(t_in)
    x_t = Dropout(0.2, name="dropout_temporal")(x_t)
    x_t_norm = LayerNormalization(name="layernorm_temporal")(x_t)
    
    s_in = Input(shape=spatial_shape, name="spatial_input")
    x_s = Dense(16, activation='relu', name="dense_spatial_1")(s_in)
    x_s = Dense(8, activation='relu', name="dense_spatial_2")(x_s)
    x_s_norm = LayerNormalization(name="layernorm_spatial")(x_s)
    
    h_spatial = Dense(16, activation='relu', name="spatial_modulation")(Concatenate()([x_t_norm, x_s_norm]))
    fused = Concatenate(name="residual_layernorm_skip")([x_t_norm, h_spatial])
    out = Dense(1, name="predicted_wl")(fused)
    
    model = Model(inputs=[t_in, s_in], outputs=out, name="Hybrid_Candidate_B_LayerNormSkip")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def build_candidate_c_huber_loss(temporal_shape=(5, 4), spatial_shape=(2,)) -> tf.keras.Model:
    """
    Candidate C: Baseline Architecture with Robust Huber Loss.
    Uses Huber loss (delta=1.0) to prevent extreme drawdown outliers from distorting
    gradient updates, while still evaluating final performance in true MSE/RMSE/MAE.
    """
    t_in = Input(shape=temporal_shape, name="temporal_input")
    x_t = GRU(64, activation='tanh', name="gru_temporal")(t_in)
    x_t = Dropout(0.2, name="dropout_temporal")(x_t)
    
    s_in = Input(shape=spatial_shape, name="spatial_input")
    x_s = Dense(16, activation='relu', name="dense_spatial_1")(s_in)
    x_s = Dense(8, activation='relu', name="dense_spatial_2")(x_s)
    
    merged = Concatenate(name="fusion_concat")([x_t, x_s])
    x_f = Dense(16, activation='relu', name="dense_fusion")(merged)
    out = Dense(1, name="predicted_wl")(x_f)
    
    model = Model(inputs=[t_in, s_in], outputs=out, name="Hybrid_Candidate_C_HuberLoss")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=Huber(delta=1.0))
    return model


# ===========================================================================
# 2. MASTER EXPERIMENT RUNNER
# ===========================================================================

def run_final_hybrid_experiment():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 82)
    print("      FINAL PHASE-2 HYBRID SPATIO-TEMPORAL GRU IMPROVEMENT EXPERIMENT")
    print("=" * 82)
    
    # 1. Load data
    data, scalers, annual_df, ds_summary = load_and_preprocess_all(DATASET_PATH)
    
    X_train, y_train, S_train = data['X_train'], data['y_train'], data['S_train']
    X_val,   y_val,   S_val   = data['X_val'],   data['y_val'],   data['S_val']
    X_test,  y_test,  S_test  = data['X_test'],  data['y_test'],  data['S_test']
    
    y_val_raw = data['y_val_raw'].reshape(-1, 1)
    y_test_raw = data['y_test_raw'].reshape(-1, 1)
    
    X_all = np.vstack((X_train, X_val, X_test))
    S_all = np.vstack((S_train, S_val, S_test))
    y_all = np.vstack((y_train, y_val, y_test))
    y_all_true_raw = scalers['target_scaler'].inverse_transform(y_all)
    
    candidates = [
        {
            "key": "baseline_v0",
            "name": "Hybrid Baseline (V0)",
            "builder": build_baseline_v0,
            "description": "Reference architecture: GRU(64) + Spatial(16->8) + Fusion(16->1), MSE"
        },
        {
            "key": "candidate_a",
            "name": "Candidate A (Residual Skip Fusion)",
            "builder": build_candidate_a_residual_skip,
            "description": "Direct temporal skip connection (GRU 64 -> Output) concatenated with spatial modulation (16)"
        },
        {
            "key": "candidate_b",
            "name": "Candidate B (LayerNorm + Residual Skip)",
            "builder": build_candidate_b_layernorm_skip,
            "description": "LayerNormalization on temporal & spatial representations before residual skip fusion"
        },
        {
            "key": "candidate_c",
            "name": "Candidate C (Huber Robust Loss)",
            "builder": build_candidate_c_huber_loss,
            "description": "Baseline V0 architecture trained with Huber loss (delta=1.0) to dampen outlier gradient pull"
        }
    ]
    
    trained_models = {}
    val_performances = {}
    
    print("\n--- PHASE 1: TRAINING ALL CANDIDATES & EVALUATING ON VALIDATION SET ---")
    
    for cand in candidates:
        ckey = cand["key"]
        cname = cand["name"]
        print("\n" + "=" * 82)
        print(f" TRAINING: {cname}")
        print(f" Description: {cand['description']}")
        print("=" * 82)
        
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        model = cand["builder"]((5, 4), (2,))
        arch_info = get_architecture_details(model)
        checkpoint_path = os.path.join(REPORT_DIR, f"final_exp_{ckey}.keras")
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=0),
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
        ]
        
        t0 = time.time()
        history = model.fit(
            [X_train, S_train], y_train,
            validation_data=([X_val, S_val], y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            callbacks=callbacks,
            verbose=1
        )
        duration = round(time.time() - t0, 2)
        epochs_trained = len(history.epoch)
        
        # Load best weights
        best_model = tf.keras.models.load_model(checkpoint_path)
        
        # Evaluate strictly on VALIDATION SET for model selection
        val_pred_scaled = best_model.predict([X_val, S_val], verbose=0)
        val_pred_raw = scalers['target_scaler'].inverse_transform(val_pred_scaled)
        
        val_metrics = calculate_metrics(y_val_raw, val_pred_raw)
        
        loss_analysis = analyze_training_loss(history.history)
        
        print(f"[{cname}] Training Time: {duration}s ({epochs_trained} epochs)")
        print(f"[{cname}] Validation MSE: {val_metrics['mse']:.6f} | RMSE: {val_metrics['rmse']:.4f} mbgl | MAE: {val_metrics['mae']:.4f} mbgl | R^2: {val_metrics['r2']:.4f}")
        
        trained_models[ckey] = {
            "name": cname,
            "description": cand["description"],
            "model": best_model,
            "architecture": arch_info,
            "training_time": duration,
            "epochs_trained": epochs_trained,
            "history": history.history,
            "loss_analysis": loss_analysis,
            "val_metrics": val_metrics,
            "checkpoint_path": checkpoint_path
        }
        val_performances[ckey] = val_metrics["rmse"]

    # =======================================================================
    # 3. MODEL SELECTION STRICTLY BASED ON VALIDATION PERFORMANCE
    # =======================================================================
    print("\n" + "=" * 82)
    print(" PHASE 2: CANDIDATE SELECTION VIA VALIDATION PERFORMANCE (2019-2021)")
    print("=" * 82)
    print(f"{'Model':<35} | {'Val RMSE (mbgl)':<16} | {'Val MAE (mbgl)':<15} | {'Val R^2':<8}")
    print("-" * 80)
    for ckey, cand in trained_models.items():
        vm = cand["val_metrics"]
        print(f"{cand['name']:<35} | {vm['rmse']:<16.4f} | {vm['mae']:<15.4f} | {vm['r2']:<8.4f}")
    print("-" * 80)
    
    # Identify the best candidate among {Candidate A, B, C}
    improvement_candidates = ["candidate_a", "candidate_b", "candidate_c"]
    selected_candidate_key = min(improvement_candidates, key=lambda k: val_performances[k])
    selected_candidate = trained_models[selected_candidate_key]
    baseline_val_rmse = val_performances["baseline_v0"]
    candidate_val_rmse = val_performances[selected_candidate_key]
    
    print(f"\n-> Selected Candidate: {selected_candidate['name']}")
    print(f"-> Selected Candidate Val RMSE: {candidate_val_rmse:.4f} mbgl (vs Baseline V0: {baseline_val_rmse:.4f} mbgl)")
    
    # =======================================================================
    # 4. ONE-SHOT HELD-OUT TEST EVALUATION (2022-2023)
    # =======================================================================
    print("\n" + "=" * 82)
    print(" PHASE 3: EVALUATION ON UNTOUCHED HELD-OUT TEST SET (2022-2023)")
    print("=" * 82)
    
    models_to_test = ["baseline_v0", selected_candidate_key]
    test_evaluations = {}
    
    for k in models_to_test:
        cand = trained_models[k]
        mod = cand["model"]
        
        # Test predictions
        test_pred_scaled = mod.predict([X_test, S_test], verbose=0)
        test_pred_raw = scalers['target_scaler'].inverse_transform(test_pred_scaled)
        
        # Backtest predictions
        bt_pred_scaled = mod.predict([X_all, S_all], verbose=0)
        bt_pred_raw = scalers['target_scaler'].inverse_transform(bt_pred_scaled)
        
        t_met = calculate_metrics(y_test_raw, test_pred_raw)
        bt_met = calculate_metrics(y_all_true_raw, bt_pred_raw)
        
        test_evaluations[k] = {
            "name": cand["name"],
            "test_metrics": t_met,
            "backtest_metrics": bt_met,
            "test_predictions": test_pred_raw,
            "backtest_predictions": bt_pred_raw
        }
        
        print(f"\n[{cand['name']} TEST RESULTS (2022-2023)]")
        print(f"  RMSE: {t_met['rmse']:.4f} mbgl | MAE: {t_met['mae']:.4f} mbgl | R^2: {t_met['r2']:.4f}")
        print(f"  Accuracy +/-0.5m: {t_met['pct_within_0_5m']}% | +/-1.0m: {t_met['pct_within_1_0m']}% | +/-2.0m: {t_met['pct_within_2_0m']}%")
        print(f"[{cand['name']} BACKTEST RESULTS (1999-2023)]")
        print(f"  RMSE: {bt_met['rmse']:.4f} mbgl | MAE: {bt_met['mae']:.4f} mbgl | R^2: {bt_met['r2']:.4f}")

    # =======================================================================
    # 5. DETERMINATION: DID CANDIDATE BEAT BASELINE?
    # =======================================================================
    base_t = test_evaluations["baseline_v0"]["test_metrics"]
    cand_t = test_evaluations[selected_candidate_key]["test_metrics"]
    
    rmse_diff = base_t["rmse"] - cand_t["rmse"]  # positive means candidate has lower RMSE (improvement)
    mae_diff = base_t["mae"] - cand_t["mae"]    # positive means candidate has lower MAE (improvement)
    r2_diff = cand_t["r2"] - base_t["r2"]        # positive means candidate has higher R2 (improvement)
    
    is_genuine_improvement = (rmse_diff > 0.005) and (r2_diff >= 0.0)
    
    print("\n" + "=" * 82)
    print(" VERDICT")
    print("=" * 82)
    print(f"Baseline V0 Test RMSE:             {base_t['rmse']:.4f} mbgl")
    print(f"Selected Candidate Test RMSE:       {cand_t['rmse']:.4f} mbgl (Delta: {rmse_diff:+.4f} mbgl)")
    print(f"Baseline V0 Test MAE:              {base_t['mae']:.4f} mbgl")
    print(f"Selected Candidate Test MAE:        {cand_t['mae']:.4f} mbgl (Delta: {mae_diff:+.4f} mbgl)")
    print(f"Baseline V0 Test R^2:              {base_t['r2']:.4f}")
    print(f"Selected Candidate Test R^2:        {cand_t['r2']:.4f} (Delta: {r2_diff:+.4f})")
    
    if is_genuine_improvement:
        verdict_str = f"GENUINE IMPROVEMENT CONFIRMED. {selected_candidate['name']} outperformed Baseline V0 by {rmse_diff:.4f} mbgl RMSE."
    else:
        verdict_str = f"NO GENUINE IMPROVEMENT FOUND. The controlled one-shot improvement experiment did not produce a genuine improvement over the baseline Hybrid under the existing evaluation protocol."
    print(f"\nVerdict: {verdict_str}")
    print("=" * 82)
    
    # =======================================================================
    # 6. PLOT GENERATION
    # =======================================================================
    print("\n[PLOTS] Generating diagnostic and comparison plots...")
    
    # Plot 1: Validation loss comparison across all candidates
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for ckey, cand in trained_models.items():
        vl = cand["history"]["val_loss"]
        ep = range(1, len(vl) + 1)
        ax.plot(ep, vl, label=f"{cand['name']} (Best: {min(vl):.5f})", lw=2)
    ax.set_title("One-Shot Hybrid Candidates: Validation Loss Convergence", fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(REPORT_DIR, "final_hybrid_candidates_val_loss.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    
    # Plot 2: Selected candidate Actual vs Predicted
    cand_preds = test_evaluations[selected_candidate_key]["test_predictions"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test_raw, cand_preds, alpha=0.35, color='#2ca02c', edgecolors='none', s=18)
    min_v = min(float(y_test_raw.min()), float(cand_preds.min()))
    max_v = max(float(y_test_raw.max()), float(cand_preds.max()))
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='1:1 Ideal Reference')
    ax.set_title(f"{selected_candidate['name']}: Actual vs Predicted (Test 2022-2023)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Actual Ground Water Level (mbgl)", fontsize=10)
    ax.set_ylabel("Predicted Ground Water Level (mbgl)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(REPORT_DIR, "final_hybrid_candidate_actual_vs_predicted.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    
    # Plot 3: Residual Distribution
    resids = y_test_raw.flatten() - cand_preds.flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resids, bins=60, color='#1f77b4', alpha=0.75, edgecolor='black', lw=0.5, density=True)
    m_err = float(np.mean(resids))
    ax.axvline(0, color='black', linestyle='--', lw=1.5, label='Zero Error')
    ax.axvline(m_err, color='red', linestyle='-', lw=1.5, label=f'Mean Error ({m_err:.3f}m)')
    ax.set_title(f"{selected_candidate['name']}: Residual Distribution (Test Set)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Residual: Actual - Predicted (mbgl)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p3 = os.path.join(REPORT_DIR, "final_hybrid_candidate_residuals_distribution.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    
    # Plot 4: Tolerance Accuracy Comparison between Baseline V0 and Selected Candidate
    fig, ax = plt.subplots(figsize=(8, 5))
    acc_labels = ['Acc <= +/-0.5m', 'Acc <= +/-1.0m', 'Acc <= +/-2.0m']
    base_accs = [base_t['pct_within_0_5m'], base_t['pct_within_1_0m'], base_t['pct_within_2_0m']]
    cand_accs = [cand_t['pct_within_0_5m'], cand_t['pct_within_1_0m'], cand_t['pct_within_2_0m']]
    
    x = np.arange(len(acc_labels))
    w = 0.35
    b1 = ax.bar(x - w/2, base_accs, w, label=f"Baseline V0", color='#7f7f7f')
    b2 = ax.bar(x + w/2, cand_accs, w, label=f"{selected_candidate['name']}", color='#1f77b4')
    ax.set_title("Tolerance-Based Prediction Accuracy: Baseline vs Selected Candidate", fontsize=11, fontweight='bold')
    ax.set_ylabel("Percentage of Test Predictions (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(acc_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
    for rect in b2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    p4 = os.path.join(REPORT_DIR, "final_hybrid_tolerance_accuracy_comparison.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    
    # Plot 5: Chronological Backtesting Plot
    cand_bt_preds = test_evaluations[selected_candidate_key]["backtest_predictions"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_all_true_raw, cand_bt_preds, alpha=0.25, color='#ff7f0e', edgecolors='none', s=14)
    min_bt = min(float(y_all_true_raw.min()), float(cand_bt_preds.min()))
    max_bt = max(float(y_all_true_raw.max()), float(cand_bt_preds.max()))
    ax.plot([min_bt, max_bt], [min_bt, max_bt], 'k--', lw=2, label='1:1 Ideal Reference')
    ax.set_title(f"{selected_candidate['name']}: Historical Backtesting (1999-2023)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Actual Historical Water Level (mbgl)", fontsize=10)
    ax.set_ylabel("Predicted Historical Water Level (mbgl)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p5 = os.path.join(REPORT_DIR, "final_hybrid_candidate_backtesting.png")
    plt.savefig(p5, dpi=200)
    plt.close()
    
    # =======================================================================
    # 7. PERSIST RESULTS TO JSON & RESULTS.MD
    # =======================================================================
    results_json_path = os.path.join(REPORT_DIR, "results.json")
    with open(results_json_path, "r", encoding="utf-8") as f:
        master_results = json.load(f)
        
    master_results["hybrid_final_one_shot_experiment"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidates_evaluated": {
            k: {
                "name": v["name"],
                "description": v["description"],
                "parameters": v["architecture"]["total_parameters"],
                "training_time_sec": v["training_time"],
                "epochs_trained": v["epochs_trained"],
                "val_metrics": v["val_metrics"]
            } for k, v in trained_models.items()
        },
        "selected_candidate_via_validation": {
            "key": selected_candidate_key,
            "name": selected_candidate["name"],
            "parameters": selected_candidate["architecture"]["total_parameters"],
            "val_rmse": candidate_val_rmse
        },
        "test_comparison": {
            "baseline_v0": {
                "parameters": trained_models["baseline_v0"]["architecture"]["total_parameters"],
                "test_metrics": base_t,
                "backtest_metrics": test_evaluations["baseline_v0"]["backtest_metrics"]
            },
            "selected_candidate": {
                "parameters": selected_candidate["architecture"]["total_parameters"],
                "test_metrics": cand_t,
                "backtest_metrics": test_evaluations[selected_candidate_key]["backtest_metrics"]
            },
            "metric_deltas_candidate_minus_baseline": {
                "rmse_improvement": round(rmse_diff, 4),
                "mae_improvement": round(mae_diff, 4),
                "r2_improvement": round(r2_diff, 4)
            },
            "is_genuine_improvement": is_genuine_improvement,
            "verdict": verdict_str
        },
        "plots": {
            "candidates_val_loss": p1,
            "candidate_actual_vs_predicted": p2,
            "candidate_residuals": p3,
            "tolerance_accuracy_comparison": p4,
            "candidate_backtesting": p5
        }
    }
    
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    print(f"[EXPORT] Updated: {results_json_path}")
    
    # Append structured section to RESULTS.md
    results_md_path = os.path.join(REPORT_DIR, "RESULTS.md")
    
    md_final = []
    md_final.append("\n---\n")
    md_final.append("# Final One-Shot Hybrid Improvement Experiment")
    md_final.append(f"\n*Executed on: {time.strftime('%Y-%m-%d %H:%M:%S')} under strict zero-leakage protocol.*\n")
    md_final.append("## 1. Motivation & Technical Rationale")
    md_final.append("Rather than expanding parameter count or stacking recurrent layers, this experiment investigated whether targeted architectural and loss formulation refinements could legitimately improve held-out test generalization:")
    md_final.append("- **Baseline V0** (`14,809` params): Reference architecture with standard sequential fusion (`GRU(64) + Spatial(16->8) -> Dense(16) -> Dense(1)`).")
    md_final.append("- **Candidate A: Residual Temporal Skip Connection** (`14,873` params): Provides a direct gradient path from the 64-D GRU output to the prediction layer, concatenated with a 16-D spatial modulation embedding, preventing temporal bottlenecking through the 16-D dense layer.")
    md_final.append("- **Candidate B: LayerNormalization + Residual Skip** (`15,017` params): Normalizes temporal and spatial representations before fusion to prevent feature magnitude dominance.")
    md_final.append(r"- **Candidate C: Robust Huber Loss** (`14,809` params): Baseline V0 architecture trained with Huber loss ($\delta=1.0$) to prevent extreme groundwater drawdown outliers from dominating gradient updates.")
    
    md_final.append("\n## 2. Validation-Based Model Selection (2019–2021)")
    md_final.append("In strict compliance with machine learning best practices, **candidate selection was performed exclusively on the validation set** without inspecting test metrics:\n")
    md_final.append("| Architecture Candidate | Parameters | Epochs | Training Time | Validation MSE | Validation RMSE (mbgl) | Validation MAE (mbgl) | Validation R² |")
    md_final.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for ckey, cand in trained_models.items():
        vm = cand["val_metrics"]
        is_sel = " **(Selected)**" if ckey == selected_candidate_key else ""
        md_final.append(f"| **{cand['name']}{is_sel}** | {cand['architecture']['total_parameters']:,} | {cand['epochs_trained']} | {cand['training_time']}s | {vm['mse']:.6f} | **{vm['rmse']:.4f}** | **{vm['mae']:.4f}** | **{vm['r2']:.4f}** |")
        
    md_final.append(f"\n-> **Selection Verdict**: **{selected_candidate['name']}** achieved the strongest validation performance (`{candidate_val_rmse:.4f} mbgl` RMSE) and was selected for one-shot test evaluation.")
    
    md_final.append("\n## 3. One-Shot Held-Out Test Evaluation (2022–2023)")
    md_final.append("The selected candidate was evaluated once on the untouched held-out test split against Baseline V0:\n")
    md_final.append("| Model | Parameters | Test RMSE (mbgl) | Test MAE (mbgl) | Test R² | Acc $\\pm 0.5$m | Acc $\\pm 1.0$m | Acc $\\pm 2.0$m | Backtest RMSE | Backtest MAE | Backtest R² |")
    md_final.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    md_final.append(f"| **Hybrid Baseline (V0)** | {trained_models['baseline_v0']['architecture']['total_parameters']:,} | **{base_t['rmse']:.4f}** | **{base_t['mae']:.4f}** | **{base_t['r2']:.4f}** | {base_t['pct_within_0_5m']}% | {base_t['pct_within_1_0m']}% | {base_t['pct_within_2_0m']}% | {test_evaluations['baseline_v0']['backtest_metrics']['rmse']:.4f} | {test_evaluations['baseline_v0']['backtest_metrics']['mae']:.4f} | {test_evaluations['baseline_v0']['backtest_metrics']['r2']:.4f} |")
    md_final.append(f"| **{selected_candidate['name']}** | {selected_candidate['architecture']['total_parameters']:,} | **{cand_t['rmse']:.4f}** | **{cand_t['mae']:.4f}** | **{cand_t['r2']:.4f}** | {cand_t['pct_within_0_5m']}% | {cand_t['pct_within_1_0m']}% | {cand_t['pct_within_2_0m']}% | {test_evaluations[selected_candidate_key]['backtest_metrics']['rmse']:.4f} | {test_evaluations[selected_candidate_key]['backtest_metrics']['mae']:.4f} | {test_evaluations[selected_candidate_key]['backtest_metrics']['r2']:.4f} |")
    
    md_final.append("\n## 4. Final Scientific Verdict")
    if is_genuine_improvement:
        md_final.append(f"> **Verdict**: **GENUINE IMPROVEMENT CONFIRMED**. {selected_candidate['name']} achieved a lower test RMSE by `{rmse_diff:.4f} mbgl` and higher test R² by `{r2_diff:+.4f}`, legitimately improving predictive capacity.")
    else:
        md_final.append(f"> **Verdict**: **NO GENUINE IMPROVEMENT OVER BASELINE**. Under the controlled one-shot protocol, the selected candidate achieved `{cand_t['rmse']:.4f} mbgl` RMSE compared to `{base_t['rmse']:.4f} mbgl` for Baseline V0 (Delta: `{rmse_diff:+.4f} mbgl`). Therefore, **Hybrid Baseline (V0) is officially retained** as the authoritative Hybrid architecture.")
        
    md_final.append("\n## 5. Experiment Artifacts")
    md_final.append(f"- [Candidates Validation Loss Convergence](./final_hybrid_candidates_val_loss.png)")
    md_final.append(f"- [Selected Candidate Actual vs Predicted](./final_hybrid_candidate_actual_vs_predicted.png)")
    md_final.append(f"- [Selected Candidate Residual Distribution](./final_hybrid_candidate_residuals_distribution.png)")
    md_final.append(f"- [Tolerance-Based Prediction Accuracy Comparison](./final_hybrid_tolerance_accuracy_comparison.png)")
    md_final.append(f"- [Selected Candidate Historical Backtesting](./final_hybrid_candidate_backtesting.png)")
    
    with open(results_md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(md_final) + "\n")
    print(f"[EXPORT] Appended final experiment section to: {results_md_path}")
    
    print("\n" + "=" * 82)
    print(" EXPERIMENT COMPLETE")
    print("=" * 82)

if __name__ == "__main__":
    try:
        run_final_hybrid_experiment()
    except Exception as exc:
        print(f"\n[ERROR] Experiment aborted: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
