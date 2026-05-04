# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wind power prediction system using a hybrid deep learning architecture (TCN + Mamba + Attention) with DMD (Dynamic Mode Decomposition) post-correction. Forecasts turbine power output (`ROUND(A.POWER,0)` and `YD15`) from 15-minute interval meteorological data.

## Running Training

```bash
cd /root/Windpower_prediction/model

# Full training pipeline (all turbines)
python train.py

# Ablation studies with different modes
python train_ablation.py --mode baseline    # TCN+Mamba only, no fusion, no DMD
python train_ablation.py --mode dmd         # TCN+Mamba + DMD correction
python train_ablation.py --mode fusion      # TCN+Mamba + ResNet future feature fusion
python train_ablation.py --mode full        # TCN+Mamba + fusion + DMD

# Single turbine ablation
python train_ablation.py --mode dmd --turbine 14 --epochs 30 --batch 64

# Hyperparameter search (Optuna Bayesian optimization)
python search.py --trials 20 --turbine 14 --epochs 20
```

Key training parameters (hardcoded in `train.py`): `input_len=480` (120h), `pred_len=96` (24h), `batch_size=128`, `lr=0.00164`, `patience=10`.

## Architecture

**`models.py`** — Core model definitions:
- `MultiTaskModel`: Full model — dilated causal TCN (with residual connections) → MambaBlock → TemporalAttention → concat with ResNet-encoded future features → two task heads (power + YD15)
- `TCNResBlock`: TCN residual block with dilated causal convolutions and shortcut connection
- `MambaBlock`: Positional encoding + stacked Mamba SSM layers (`mamba_ssm` package)
- `ResNet`: Encodes future meteorological features (5 dims) into prediction-length representation
- `DynamicWeightedLoss`: Epoch-dependent weighted Huber loss (ωt linearly from 0.1 to 0.5)
- `EarlyStopping`: Saves best checkpoint

**`dataset.py`** — Data preprocessing (`data_preprocess`) and feature engineering (`feature_engineer`):
- IQR-based outlier removal, 15-min resampling, KNN imputation (k=50)
- Cyclical encoding of month/day/hour/minute as sin/cos pairs
- Invalid data rules: zero wind speed with nonzero power → zero; high wind with nonzero power → zero

**`dataloader.py`** — PyTorch `WindDataset`:
- Train/val/test split: 70%/20%/10%
- Sliding window: `input_len=480`, `stride=76`, `pred_len=96`
- Historical features (13): WINDSPEED, PREPOWER, ROUND(A.WS,1), ROUND(A.POWER,0), YD15, + 8 cyclical (month/day/hour/minute sin/cos)
- Future features (5): WINDSPEED, WINDDIRECTION, TEMPERATURE, HUMIDITY, PRESSURE
- Targets (2): ROUND(A.POWER,0), YD15
- StandardScaler normalization; scalers saved per turbine

**`utils.py`** — DMD post-correction pipeline:
- Build Hankel matrix from prediction error sequence → SVD-based DMD decomposition → reconstruct error → add back to predictions
- Hankel window `K=20`

## Training Pipeline

1. CSV files in `data/` are read per turbine (numbered 11–20)
2. Preprocessing: dedup → outlier removal → resample to 15min → KNN impute → linear interpolate
3. Feature engineering: cyclical time encoding
4. Training: Multi-task model with dynamic weighted Huber loss
5. Validation: DMD fitted on validation residuals to learn error dynamics
6. Test: DMD corrects YD15 predictions using validation-learned dynamics
7. Evaluation: ACC_NRMSE, ACC_NMAE, R², RMSE, MAE

## Key Dependencies

- `mamba_ssm` — Mamba selective state space model (requires CUDA)
- `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `tqdm`

## Output Structure

All outputs are unified under `output/`:
- `output/checkpoints/` — Model weights (`best_model.pth`)
- `output/scaler/` — Per-turbine StandardScaler pkl files
- `output/plots/` — Prediction visualization PNGs
- `output/metrics/` — Per-horizon evaluation CSVs (ablation mode)

## Data Format

CSV files named `{turbine_id}.csv` with columns: DATATIME, WINDSPEED, PREPOWER, WINDDIRECTION, TEMPERATURE, HUMIDITY, PRESSURE, ROUND(A.WS,1), ROUND(A.POWER,0), YD15. Data is at 15-minute intervals starting from late 2021.
