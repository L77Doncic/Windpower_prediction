# 🌬️ Wind Power Prediction Project

## 📌 Overview
This project focuses on predicting wind power output using machine learning models. The goal is to forecast wind power generation based on historical data and environmental factors. 🚀

## 🏗️ Project Structure
- `model/`: Contains the core code for data loading, preprocessing, model training, and evaluation.
  - `dataloader.py`: Handles dataset loading and preprocessing. 📂
  - `dataset.py`: Implements data preprocessing and feature engineering. 🔧
  - `models.py`: Defines the machine learning models used for prediction. 🤖
  - `train.py`: Script for training the models. 🏋️‍♂️
  - `train_ablation.py`: Script for ablation studies. 🔍
  - `utils.py`: Utility functions for the project. 🛠️
- `data/`: Stores input data files (e.g., `14.csv`). 📊
- `output/`: Contains model outputs and predictions. 📈
- `plots/`: Stores visualization plots. 📉

## 🧹 Data Preprocessing
- **Data Cleaning**: Handles missing values and removes duplicates. 🧽
- **Feature Engineering**: Generates additional features from raw data. 🧩
- **Normalization**: Scales data using `StandardScaler` for model training. ⚖️

## 🤖 Model Training
- **Model Architecture**: Uses a custom neural network with Mamba-based components for time-series prediction. 🧠
- **Training Process**: Includes early stopping and checkpointing to save the best model. ⏱️
- **Evaluation Metrics**: Uses R² score and other metrics to assess model performance. 📊

## 🚀 Usage
1. **Data Preparation**: Place your data files in the `data/` folder. 📂
2. **Training**: Run `train.py` to train the model. 🏋️‍♂️
3. **Prediction**: Use the trained model to generate predictions. 🔮

## 📋 Requirements
- Python 3.x 🐍
- PyTorch 🔥
- scikit-learn 📚
- pandas 🐼
- numpy 🔢

## 📜 License
This project is open-source and available under the MIT License. 🌍