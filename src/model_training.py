from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model"""
    logger.info("Training Gradient Boosting Regressor...")
    
    model = GradientBoostingRegressor(
        alpha=0.9,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        min_samples_split=2,
        n_estimators=100,
        random_state=30
    )
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def train_svr(X_train, y_train):
    """Train SVR model"""
    logger.info("Training SVR model...")
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X_train, y_train)
    logger.info("SVR training completed")
    return model

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model"""
    logger.info("Training Decision Tree Regressor...")
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Decision Tree training completed")
    return model

def save_model(model, model_path: str):
    """Save model to disk"""
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    logger.info("Model saved successfully")

def load_model(model_path: str):
    """Load model from disk"""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model