# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wind power prediction system using a hybrid deep learning architecture (TCN + Mamba + Attention) with GAN-based training and DMD (Dynamic Mode Decomposition) post-correction. Forecasts turbine power output (`ROUND(A.POWER,0)` and `YD15`) from 15-minute interval meteorological data.

**Note:** The `transformer/` folder is legacy and should be ignored.

## Running Training

```bash
# Full training pipeline (all turbines)
cd /root/Windpower_prediction/model
python train.py

# Ablation studies with different modes
python train_ablation.py --mode baseline    # Mamba only, no fusion
python train_ablation.py --mode dmd         # Mamba + DMD correction (default pipeline)
python train_ablation.py --mode fusion      # Mamba + future feature fusion (ResNet)
python train_ablation.py --mode gan         # Mamba + GAN discriminator

# Single turbine ablation
python train_ablation.py --mode dmd --turbine 14 --epochs 30 --batch 64
```

Key training parameters (hardcoded in `train.py`): `input_len=480` (120h), `pred_len=96` (24h), `batch_size=128`, `lr=0.001`, `patience=10`.

## Architecture

**`models.py`** — Core model definitions:
- `MultiTaskModel`: Full model — dilated causal TCN → MambaBlock → TemporalAttention → concat with ResNet-encoded future features → two task heads (power + YD15)
- `MambaBlock`: Positional encoding + stacked Mamba SSM layers (`mamba_ssm` package)
- `CausalConv1d` / `ConvBNLayer` / `ResNetBlock`: Dilated causal convolution stack (no future leakage)
- `ResNet`: Encodes future meteorological features (wind speed, direction, temp, humidity, pressure) into prediction-length representation
- `Discriminator`: GAN discriminator for adversarial training
- `DynamicWeightedLoss`: Epoch-dependent weighted Huber loss (unused in current training loop)
- `EarlyStopping`: Saves best checkpoint to `/root/model/checkpoints/best_model.pth`

**`dataset.py`** — Data preprocessing (`data_preprocess`) and feature engineering (`feature_engineer`):
- IQR-based outlier removal, 15-min resampling, KNN imputation (k=50)
- Cyclical encoding of month/day/hour/minute as sin/cos pairs
- Invalid data rules: zero wind speed with nonzero power → zero; high wind with nonzero power → zero; moderate wind with zero power → NaN

**`dataloader.py`** — PyTorch `WindDataset`:
- Train/val/test split: 70%/20%/10%
- Sliding window: `input_len=480`, `stride=76`, `pred_len=96`
- Input features (17): WINDSPEED, PREPOWER, WINDDIRECTION, TEMPERATURE, HUMIDITY, PRESSURE, ROUND(A.WS,1), ROUND(A.POWER,0), YD15, + 8 cyclical features
- Future features (5): WINDSPEED, WINDDIRECTION, TEMPERATURE, HUMIDITY, PRESSURE
- Targets (2): ROUND(A.POWER,0), YD15
- StandardScaler normalization; scalers saved to `/root/model/output/scaler_{tid}_*.pkl`

**`utils.py`** — DMD post-correction pipeline:
- Build Hankel matrix from prediction error sequence → SVD-based DMD decomposition → reconstruct error → add back to predictions
- Hankel window `K=20`

## Training Pipeline

1. CSV files in `data/` are read per turbine (numbered 11–20)
2. Preprocessing: dedup → outlier removal → resample to 15min → KNN impute → linear interpolate
3. Feature engineering: cyclical time encoding
4. Training: GAN framework — generator (MultiTaskModel) trained with 0.1×adversarial + 0.9×MSE loss
5. Test: DMD corrects YD15 predictions using error dynamics reconstruction
6. Evaluation: ACC_NRMSE, ACC_NMAE, R², RMSE, MAE

## Key Dependencies

- `mamba_ssm` — Mamba selective state space model (requires CUDA)
- `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `tqdm`

## Output Paths

- Checkpoints: `/root/model/checkpoints/`
- Scalers: `/root/model/output/scaler_{turbine_id}_*.pkl`
- Plots: `/root/model/plots/`
- Metrics CSV: `/root/model/metrics/`

## Data Format

CSV files named `{turbine_id}.csv` with columns: DATATIME, WINDSPEED, PREPOWER, WINDDIRECTION, TEMPERATURE, HUMIDITY, PRESSURE, ROUND(A.WS,1), ROUND(A.POWER,0), YD15. Data is at 15-minute intervals starting from late 2021.
