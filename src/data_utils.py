import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """Load Boston Housing dataset"""
    logger.info(f"Loading data from {data_path}")
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = pd.read_csv(data_path, header=None, delimiter=r"\s+", names=column_names)
    logger.info(f"Data shape: {data.shape}")
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    logger.info("Preprocessing data...")
    
    # Remove MEDV outliers (censored at 50)
    data = data[~(data['MEDV'] >= 50.0)]
    logger.info(f"Data shape after removing outliers: {data.shape}")
    
    return data

def feature_engineering(data: pd.DataFrame):
    """Perform feature selection and transformation"""
    logger.info("Performing feature engineering...")
    
    column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    X = data[column_sels]
    y = data['MEDV']
    
    # Log transformation for skewed features
    y = np.log1p(y)
    for col in X.columns:
        if np.abs(X[col].skew()) > 0.3:
            X[col] = np.log1p(X[col])
    
    logger.info(f"Selected features: {column_sels}")
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    logger.info(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale features using MinMaxScaler"""
    logger.info("Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler