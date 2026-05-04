# Multi-task Wind Power Forecasting via Mamba-based Selective State-Space Modeling with Dynamic Mode Decomposition

## Overview

We propose a hybrid forecasting framework that combines deep sequence modeling with physics-inspired error correction for wind power prediction. The model utilizes a dual-branch architecture to extract features from historical and future data separately, fuses them into a unified representation, and employs a Mamba block for efficient global dependency modeling. Predictions are further refined through Dynamic Mode Decomposition (DMD)-based residual correction.

**Key contributions:**

- **Dual-Branch Context Encoding**: A parallel TCN and 1D-ResNet architecture separately encodes historical power trajectories and future meteorological signals, providing task-aligned local representations before fusion.
- **Efficient Long-Range Modeling**: A Mamba-based encoder models global temporal dependencies and cross-variable interactions with linear complexity O(L).
- **Dynamic Residual Correction**: DMD is introduced to model structured residual dynamics and correct systematic forecast bias under non-stationary conditions.

## Architecture

```
Historical Features (X1)          Future Features (X2)
        │                                │
        ▼                                ▼
  ┌──────────┐                    ┌──────────┐
  │   TCN    │                    │  ResNet  │
  │ (dilated │                    │ (1D, 7×7 │
  │  causal) │                    │  + pool) │
  └────┬─────┘                    └────┬─────┘
       │                               │
       ▼                               │
  ┌──────────┐                         │
  │  Mamba   │                         │
  │  Block   │                         │
  └────┬─────┘                         │
       │                               │
       ▼                               │
  ┌──────────┐                         │
  │ Temporal │                         │
  │ Attention│                         │
  └────┬─────┘                         │
       │                               │
       └───────────┬───────────────────┘
                   │
                   ▼
            ┌────────────┐
            │  Concat +  │
            │ Task Heads │
            └─────┬──────┘
                  │
           ┌──────┴──────┐
           ▼             ▼
     ROUND(A.POWER,0)   YD15
                  │
                  ▼
           ┌────────────┐
           │ DMD-based  │
           │ Correction │
           └────────────┘
```

## Project Structure

```
.
├── model/
│   ├── models.py          # Model definitions (MultiTaskModel, TCNResBlock, MambaBlock, ResNet, etc.)
│   ├── dataloader.py      # Dataset class with sliding window and Z-score normalization
│   ├── dataset.py         # Data preprocessing and cyclical feature engineering
│   ├── train.py           # Main training pipeline with DMD correction
│   ├── train_ablation.py  # Ablation study scripts
│   └── utils.py           # DMD decomposition and Hankel matrix utilities
├── data/                  # Turbine CSV files (11.csv ~ 20.csv)
├── checkpoints/           # Model checkpoints
├── output/                # Saved scalers
├── plots/                 # Prediction visualization plots
└── metrics/               # Evaluation metrics CSV
```

## Installation

**Requirements:**

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- mamba-ssm
- scikit-learn, pandas, numpy, matplotlib, tqdm, joblib

```bash
# Create conda environment
conda create -n windpower python=3.12
conda activate windpower

# Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install mamba-ssm (requires CUDA toolkit)
pip install causal-conv1d mamba-ssm

# Install other dependencies
pip install scikit-learn pandas numpy matplotlib tqdm joblib
```

## Usage

### Data Preparation

Place turbine CSV files in the `data/` directory. Each file should be named `{turbine_id}.csv` with the following columns:

| Column | Description |
|--------|-------------|
| DATATIME | Timestamp (15-min intervals) |
| WINDSPEED | Numerical weather prediction wind speed |
| PREPOWER | Previous period cumulative power |
| WINDDIRECTION | Wind direction |
| TEMPERATURE | Ambient temperature |
| HUMIDITY | Relative humidity |
| PRESSURE | Atmospheric pressure |
| ROUND(A.WS,1) | Measured wind speed |
| ROUND(A.POWER,0) | Measured active power (target 1) |
| YD15 | YD15 power indicator (target 2) |

### Training

```bash
cd model

# Full training (all turbines)
python train.py

# Single turbine
python train.py  # modify data_path in __main__ block
```

### Ablation Studies

```bash
cd model

# Baseline: TCN + Mamba (no fusion, no DMD)
python train_ablation.py --mode baseline

# + DMD correction
python train_ablation.py --mode dmd

# + Future feature fusion
python train_ablation.py --mode fusion

# Full model (fusion + DMD)
python train_ablation.py --mode full

# Single turbine with custom hyperparameters
python train_ablation.py --mode full --turbine 14 --epochs 30 --batch 64 --lr 0.0005
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input window L | 480 steps | 5 days of historical data |
| Prediction horizon H | 96 steps | 24 hours forecast |
| Stride | 76 steps | Sliding window stride |
| Batch size | 128 | |
| Learning rate | 0.001 | Adam optimizer |
| Early stopping patience | 10 epochs | |
| Train / Val / Test split | 70% / 20% / 10% | Chronological split |
| DMD Hankel window K | 20 | |
| Dropout | 0.1 | Applied to TCN, ResNet, Mamba, task heads |
| Huber loss delta | 1.0 | |
| Loss weight ωt | 0.1 → 0.5 | Linear increase over training |

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error (MW)
- **MAE**: Mean Absolute Error (MW)
- **ACC-NRMSE**: 1 - NRMSE (Accuracy of Normalized RMSE)
- **ACC-NMAE**: 1 - NMAE (Accuracy of Normalized MAE)
- **R²**: Coefficient of Determination

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
