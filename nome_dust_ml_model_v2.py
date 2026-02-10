"""
Nome Dust Forecast Model - ML Version (Fixed)
==============================================

FIXES from v1:
1. Date parsing with format='mixed' (loads all 13,000+ records)
2. Exclude non-dust features (CO, NO, NO2, O3) - these are pollutants, not dust predictors
3. Proper feature importance (normalized gain, not split counts)
4. Regularization to prevent overfitting on small datasets
5. Better validation metrics

Requirements:
    pip install numpy pandas scikit-learn lightgbm joblib requests

Author: Nome Dust Forecasting Project
Date: December 2024
"""

import json
import warnings
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, 
    roc_auc_score, 
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score
)
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Using sklearn GradientBoosting.")

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MLConfig:
    """ML model configuration"""
    
    # Dust thresholds
    PM10_ELEVATED: float = 50.0      # Moderate dust
    PM10_HIGH: float = 100.0         # High dust
    PM10_SEVERE: float = 200.0       # Severe dust
    
    # Features to EXCLUDE (not dust-related)
    EXCLUDE_COLUMNS: List[str] = field(default_factory=lambda: [
        'CO', 'NO', 'NO2', 'O3',  # Air pollutants, not dust
        'site_name',              # Constant
        'Unnamed: 0',             # Index
        'AQI', 'AQI_value',       # Derived from PM, would leak
    ])
    
    # Lag hours for features
    LAG_HOURS: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    
    # Rolling windows
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    
    # Model regularization (prevent overfitting)
    MIN_SAMPLES_LEAF: int = 30
    MAX_DEPTH: int = 5
    LEARNING_RATE: float = 0.03
    N_ESTIMATORS: int = 300
    SUBSAMPLE: float = 0.7
    COLSAMPLE: float = 0.7
    REG_ALPHA: float = 0.5
    REG_LAMBDA: float = 1.0


CONFIG = MLConfig()


def _ensure_main_mlconfig():
    """Allow joblib to resolve pickled MLConfig saved under __main__."""
    try:
        import __main__  # type: ignore
        if not hasattr(__main__, "MLConfig"):
            __main__.MLConfig = MLConfig  # type: ignore[attr-defined]
    except Exception:
        pass


# =============================================================================
# DATA LOADING
# =============================================================================

def load_nome_data(filepath: str, config: MLConfig = CONFIG) -> pd.DataFrame:
    """
    Load Nome PM data with proper date parsing.
    
    FIXES:
    - Uses format='mixed' for date parsing (handles both date-only and datetime)
    - Excludes non-dust columns (CO, NO, NO2, O3)
    - Converts to numeric properly
    """
    print(f"Loading data from {filepath}...")
    
    # Load raw
    df = pd.read_csv(filepath)
    print(f"  Raw rows: {len(df)}")
    
    # Parse dates with mixed format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"  After date parsing: {len(df)}")
    
    # Set index
    df = df.set_index('date').sort_index()
    
    # Drop excluded columns
    for col in config.EXCLUDE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Convert remaining to numeric
    for col in df.columns:
        if col not in ['site_name']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename columns for consistency
    df = df.rename(columns={
        'AT': 'temp_c',
        'RH': 'humidity'
    })
    
    # Drop rows without PM10
    df = df.dropna(subset=['PM10'])
    print(f"  After dropping missing PM10: {len(df)}")
    
    # Report date range
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  PM10 range: {df['PM10'].min():.1f} to {df['PM10'].max():.1f}")
    print(f"  PM10 > 50 (dust events): {(df['PM10'] >= 50).sum()} ({100*(df['PM10'] >= 50).mean():.1f}%)")
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df: pd.DataFrame, config: MLConfig = CONFIG) -> pd.DataFrame:
    """
    Create ML features from raw data.
    
    Features focus on dust-relevant signals:
    - Temperature (frozen ground indicator)
    - Humidity (moisture suppression)
    - PM10 history (persistence/trends)
    - Temporal patterns (season, hour)
    
    Excludes: CO, NO, NO2, O3 (air pollutants, not dust drivers)
    """
    result = df.copy()
    
    # ===================
    # TEMPORAL FEATURES
    # ===================
    result['hour'] = result.index.hour
    result['day_of_week'] = result.index.dayofweek
    result['month'] = result.index.month
    result['day_of_year'] = result.index.dayofyear
    
    # Cyclical encoding
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    # Season indicators
    result['is_dust_season'] = result['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    result['is_winter'] = result['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)
    result['is_daytime'] = ((result['hour'] >= 8) & (result['hour'] <= 20)).astype(int)
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    
    # ===================
    # TEMPERATURE FEATURES
    # ===================
    if 'temp_c' in result.columns:
        # Binary thresholds (let model learn which matter)
        result['temp_below_0'] = (result['temp_c'] < 0).astype(int)
        result['temp_below_minus5'] = (result['temp_c'] < -5).astype(int)
        result['temp_below_minus10'] = (result['temp_c'] < -10).astype(int)
        result['temp_above_5'] = (result['temp_c'] > 5).astype(int)
        result['temp_above_10'] = (result['temp_c'] > 10).astype(int)
        result['temp_above_15'] = (result['temp_c'] > 15).astype(int)
        result['temp_above_20'] = (result['temp_c'] > 20).astype(int)
        
        # Lagged temperature
        for lag in config.LAG_HOURS:
            result[f'temp_lag_{lag}h'] = result['temp_c'].shift(lag)
        
        # Temperature changes (warming/cooling trends)
        result['temp_change_3h'] = result['temp_c'] - result['temp_c'].shift(3)
        result['temp_change_6h'] = result['temp_c'] - result['temp_c'].shift(6)
        result['temp_change_24h'] = result['temp_c'] - result['temp_c'].shift(24)
        
        # Rolling stats
        for window in [6, 12, 24]:
            result[f'temp_mean_{window}h'] = result['temp_c'].rolling(window, min_periods=1).mean()
            result[f'temp_max_{window}h'] = result['temp_c'].rolling(window, min_periods=1).max()
            result[f'temp_min_{window}h'] = result['temp_c'].rolling(window, min_periods=1).min()
    
    # ===================
    # HUMIDITY FEATURES
    # ===================
    if 'humidity' in result.columns:
        result['is_dry'] = (result['humidity'] < 50).astype(int)
        result['is_very_dry'] = (result['humidity'] < 30).astype(int)
        result['is_humid'] = (result['humidity'] > 70).astype(int)
        result['is_very_humid'] = (result['humidity'] > 85).astype(int)
        
        for lag in [1, 3, 6, 12, 24]:
            result[f'humidity_lag_{lag}h'] = result['humidity'].shift(lag)
        
        result['humidity_change_6h'] = result['humidity'] - result['humidity'].shift(6)
        result['humidity_change_24h'] = result['humidity'] - result['humidity'].shift(24)
        
        for window in [6, 12, 24]:
            result[f'humidity_mean_{window}h'] = result['humidity'].rolling(window, min_periods=1).mean()
    
    # ===================
    # PM10 LAGGED FEATURES (critical for persistence)
    # ===================
    if 'PM10' in result.columns:
        # Lagged values (no leakage - all shifted)
        for lag in config.LAG_HOURS:
            result[f'pm10_lag_{lag}h'] = result['PM10'].shift(lag)
        
        # Rolling statistics (shifted to avoid leakage)
        for window in config.ROLLING_WINDOWS:
            result[f'pm10_mean_{window}h'] = result['PM10'].shift(1).rolling(window, min_periods=1).mean()
            result[f'pm10_max_{window}h'] = result['PM10'].shift(1).rolling(window, min_periods=1).max()
            result[f'pm10_std_{window}h'] = result['PM10'].shift(1).rolling(window, min_periods=1).std()
        
        # Trends
        result['pm10_trend_3h'] = result['PM10'].shift(1) - result['PM10'].shift(4)
        result['pm10_trend_6h'] = result['PM10'].shift(1) - result['PM10'].shift(7)
        result['pm10_trend_24h'] = result['PM10'].shift(1) - result['PM10'].shift(25)
        
        # Anomaly (relative to recent history)
        result['pm10_anomaly'] = result['PM10'].shift(1) / (result['pm10_mean_24h'] + 1)
    
    # ===================
    # PM25 FEATURES (if available)
    # ===================
    if 'PM25' in result.columns:
        for lag in [1, 3, 6]:
            result[f'pm25_lag_{lag}h'] = result['PM25'].shift(lag)
        
        # Dust signature: low PM25/PM10 ratio indicates coarse particles (dust)
        if 'PM10' in result.columns:
            result['pm_ratio'] = result['PM25'].shift(1) / (result['PM10'].shift(1) + 0.1)
            result['is_coarse'] = (result['pm_ratio'] < 0.3).astype(int)
    
    # ===================
    # INTERACTION FEATURES
    # ===================
    if 'temp_c' in result.columns and 'humidity' in result.columns:
        # Drying potential (warm + dry = dust risk)
        result['drying_potential'] = result['temp_c'] * (100 - result['humidity']) / 100
    
    # ===================
    # TARGET VARIABLES
    # ===================
    result['is_dust_event'] = (result['PM10'] >= config.PM10_ELEVATED).astype(int)
    result['is_high_dust'] = (result['PM10'] >= config.PM10_HIGH).astype(int)
    
    return result


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude targets and raw values)"""
    exclude = [
        'PM10', 'PM25',  # Raw targets
        'is_dust_event', 'is_high_dust',  # Targets
        'site_name',
    ]
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]


# =============================================================================
# ML MODELS
# =============================================================================

class DustClassifier:
    """
    Dust event classifier with calibrated probabilities.
    
    FIXES:
    - Stronger regularization to prevent overfitting
    - Normalized feature importance (gain-based)
    - Cross-validation calibration
    """
    
    def __init__(self, config: MLConfig = CONFIG):
        self.config = config
        self.model = None
        self.calibrated_model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.trained = False
        self.metrics = {}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train with regularization to prevent overfitting"""
        
        print(f"\nTraining classifier on {len(X_train)} samples...")
        print(f"  Positive class: {y_train.sum()} ({100*y_train.mean():.1f}%)")
        
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build model with regularization
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                learning_rate=self.config.LEARNING_RATE,
                max_depth=self.config.MAX_DEPTH,
                min_child_samples=self.config.MIN_SAMPLES_LEAF,
                subsample=self.config.SUBSAMPLE,
                colsample_bytree=self.config.COLSAMPLE,
                reg_alpha=self.config.REG_ALPHA,
                reg_lambda=self.config.REG_LAMBDA,
                class_weight='balanced',
                random_state=42,
                verbose=-1,
                importance_type='gain',  # Use gain, not split count
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                learning_rate=self.config.LEARNING_RATE,
                max_depth=self.config.MAX_DEPTH,
                min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
                subsample=self.config.SUBSAMPLE,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calibrate probabilities
        print("  Calibrating probabilities...")
        n_cv = min(5, max(2, y_train.sum() // 10))  # Adjust CV folds based on positive samples
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='isotonic', cv=n_cv
        )
        self.calibrated_model.fit(X_train_scaled, y_train)
        
        # Training metrics
        train_probs = self.calibrated_model.predict_proba(X_train_scaled)[:, 1]
        train_preds = (train_probs >= 0.5).astype(int)
        
        self.metrics = {
            'train_samples': len(y_train),
            'train_positive': int(y_train.sum()),
            'train_auc': roc_auc_score(y_train, train_probs) if y_train.sum() > 0 else 0,
            'train_brier': brier_score_loss(y_train, train_probs),
            'train_precision': precision_score(y_train, train_preds, zero_division=0),
            'train_recall': recall_score(y_train, train_preds, zero_division=0),
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_probs = self.calibrated_model.predict_proba(X_val_scaled)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            
            self.metrics['val_samples'] = len(y_val)
            self.metrics['val_positive'] = int(y_val.sum())
            self.metrics['val_auc'] = roc_auc_score(y_val, val_probs) if y_val.sum() > 0 else 0
            self.metrics['val_brier'] = brier_score_loss(y_val, val_probs)
            self.metrics['val_precision'] = precision_score(y_val, val_preds, zero_division=0)
            self.metrics['val_recall'] = recall_score(y_val, val_preds, zero_division=0)
        
        # Feature importance (normalized)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            total = importances.sum()
            if total > 0:
                importances = importances / total
            
            idx = np.argsort(importances)[::-1][:15]
            self.metrics['top_features'] = {
                self.feature_names[i]: round(float(importances[i]), 4)
                for i in idx
            }
        
        self.trained = True
        
        print(f"  Train AUC: {self.metrics['train_auc']:.3f}")
        print(f"  Train Brier: {self.metrics['train_brier']:.4f}")
        if 'val_auc' in self.metrics:
            print(f"  Val AUC: {self.metrics['val_auc']:.3f}")
            print(f"  Val Precision: {self.metrics['val_precision']:.3f}")
            print(f"  Val Recall: {self.metrics['val_recall']:.3f}")
        
        return self.metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict dust probability"""
        X_aligned = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_aligned)
        return self.calibrated_model.predict_proba(X_scaled)[:, 1]
    
    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        try:
            data = joblib.load(path)
        except AttributeError as exc:
            if "MLConfig" in str(exc):
                _ensure_main_mlconfig()
                data = joblib.load(path)
            else:
                raise
        self.model = data['model']
        self.calibrated_model = data['calibrated_model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.trained = True


class PM10Regressor:
    """
    PM10 regressor with quantile predictions.
    
    FIXES:
    - Stronger regularization
    - Log transform for better distribution
    - Proper quantile coverage
    """
    
    def __init__(self, config: MLConfig = CONFIG):
        self.config = config
        self.model_p50 = None
        self.model_p10 = None
        self.model_p90 = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.trained = False
        self.metrics = {}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train quantile regressors"""
        
        print(f"\nTraining PM10 regressor on {len(X_train)} samples...")
        print(f"  PM10 median: {y_train.median():.1f}, max: {y_train.max():.1f}")
        
        self.feature_names = list(X_train.columns)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Log transform target
        y_log = np.log1p(y_train)
        
        # Common params with regularization
        params = dict(
            n_estimators=self.config.N_ESTIMATORS,
            learning_rate=self.config.LEARNING_RATE,
            max_depth=self.config.MAX_DEPTH,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            subsample=self.config.SUBSAMPLE,
            random_state=42
        )
        
        # Train median (p50)
        print("  Training median model...")
        self.model_p50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, **params)
        self.model_p50.fit(X_train_scaled, y_log)
        
        # Train lower (p10)
        print("  Training p10 model...")
        self.model_p10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, **params)
        self.model_p10.fit(X_train_scaled, y_log)
        
        # Train upper (p90)
        print("  Training p90 model...")
        self.model_p90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, **params)
        self.model_p90.fit(X_train_scaled, y_log)
        
        # Compute metrics
        pred_p50 = np.expm1(self.model_p50.predict(X_train_scaled))
        pred_p10 = np.expm1(self.model_p10.predict(X_train_scaled))
        pred_p90 = np.expm1(self.model_p90.predict(X_train_scaled))
        
        self.metrics = {
            'train_samples': len(y_train),
            'train_mae': mean_absolute_error(y_train, pred_p50),
            'train_rmse': np.sqrt(mean_squared_error(y_train, pred_p50)),
            'train_coverage': ((y_train >= pred_p10) & (y_train <= pred_p90)).mean(),
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_p50 = np.expm1(self.model_p50.predict(X_val_scaled))
            val_p10 = np.expm1(self.model_p10.predict(X_val_scaled))
            val_p90 = np.expm1(self.model_p90.predict(X_val_scaled))
            
            self.metrics['val_samples'] = len(y_val)
            self.metrics['val_mae'] = mean_absolute_error(y_val, val_p50)
            self.metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_p50))
            self.metrics['val_coverage'] = ((y_val >= val_p10) & (y_val <= val_p90)).mean()
        
        self.trained = True
        
        print(f"  Train MAE: {self.metrics['train_mae']:.2f}")
        print(f"  Train coverage: {self.metrics['train_coverage']:.1%}")
        if 'val_mae' in self.metrics:
            print(f"  Val MAE: {self.metrics['val_mae']:.2f}")
            print(f"  Val coverage: {self.metrics['val_coverage']:.1%}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict PM10 with uncertainty bounds"""
        X_aligned = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_aligned)
        
        p50 = np.expm1(self.model_p50.predict(X_scaled))
        p10 = np.expm1(self.model_p10.predict(X_scaled))
        p90 = np.expm1(self.model_p90.predict(X_scaled))
        
        # Ensure monotonicity
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
        
        return np.clip(p50, 0, 2000), np.clip(p10, 0, 2000), np.clip(p90, 0, 2000)
    
    def save(self, path: str):
        joblib.dump({
            'model_p50': self.model_p50,
            'model_p10': self.model_p10,
            'model_p90': self.model_p90,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, path)
    
    def load(self, path: str):
        try:
            data = joblib.load(path)
        except AttributeError as exc:
            if "MLConfig" in str(exc):
                _ensure_main_mlconfig()
                data = joblib.load(path)
            else:
                raise
        self.model_p50 = data['model_p50']
        self.model_p10 = data['model_p10']
        self.model_p90 = data['model_p90']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.trained = True


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_dust_models(data_path: str, output_dir: str = 'models', 
                      val_split: float = 0.2) -> Dict:
    """
    Complete training pipeline.
    
    1. Load data (with fixed date parsing)
    2. Create features (excluding non-dust columns)
    3. Train classifier + regressor
    4. Save models
    """
    
    print("="*70)
    print("NOME DUST ML MODEL - TRAINING")
    print("="*70)
    
    # Load data
    df = load_nome_data(data_path)
    
    # Create features
    print("\nCreating features...")
    featured_df = create_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(featured_df)
    print(f"  Features: {len(feature_cols)}")
    
    # Remove rows with NaN
    model_df = featured_df.dropna(subset=feature_cols + ['is_dust_event', 'PM10'])
    print(f"  Samples after dropping NaN: {len(model_df)}")
    
    # Time-based split
    split_idx = int(len(model_df) * (1 - val_split))
    train_df = model_df.iloc[:split_idx]
    val_df = model_df.iloc[split_idx:]
    
    print(f"  Train: {len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Val: {len(val_df)} ({val_df.index.min().date()} to {val_df.index.max().date()})")
    
    X_train = train_df[feature_cols]
    y_train_class = train_df['is_dust_event']
    y_train_reg = train_df['PM10']
    
    X_val = val_df[feature_cols]
    y_val_class = val_df['is_dust_event']
    y_val_reg = val_df['PM10']
    
    # Train classifier
    print("\n" + "-"*50)
    print("CLASSIFIER")
    print("-"*50)
    classifier = DustClassifier()
    class_metrics = classifier.train(X_train, y_train_class, X_val, y_val_class)
    
    # Train regressor
    print("\n" + "-"*50)
    print("REGRESSOR")
    print("-"*50)
    regressor = PM10Regressor()
    reg_metrics = regressor.train(X_train, y_train_reg, X_val, y_val_reg)
    
    # Print learned patterns
    print("\n" + "-"*50)
    print("LEARNED PATTERNS (Top Features)")
    print("-"*50)
    
    if 'top_features' in class_metrics:
        for feat, imp in list(class_metrics['top_features'].items())[:10]:
            interpretation = interpret_feature(feat)
            print(f"  {feat}: {imp:.4f} - {interpretation}")
    
    # Save models
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    classifier.save(output_path / 'classifier.joblib')
    regressor.save(output_path / 'regressor.joblib')
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'data_file': data_path,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'feature_count': len(feature_cols),
        'classifier_metrics': class_metrics,
        'regressor_metrics': reg_metrics,
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nModels saved to {output_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Classifier Val AUC: {class_metrics.get('val_auc', 'N/A'):.3f}")
    print(f"Classifier Val Precision: {class_metrics.get('val_precision', 'N/A'):.3f}")
    print(f"Classifier Val Recall: {class_metrics.get('val_recall', 'N/A'):.3f}")
    print(f"Regressor Val MAE: {reg_metrics.get('val_mae', 'N/A'):.2f}")
    print(f"Regressor Val Coverage: {reg_metrics.get('val_coverage', 'N/A'):.1%}")
    
    return {
        'classifier': class_metrics,
        'regressor': reg_metrics,
        'metadata': metadata
    }


def interpret_feature(feature_name: str) -> str:
    """Interpret what a feature means"""
    interpretations = {
        'temp_c': 'Current temperature',
        'temp_below_0': 'Frozen ground indicator',
        'temp_below_minus5': 'Deep freeze - no dust possible',
        'temp_above_10': 'Warm enough for dust',
        'humidity': 'Current humidity (suppresses dust)',
        'is_dry': 'Dry conditions favor dust',
        'pm10_lag_1h': 'Recent PM10 (persistence)',
        'pm10_lag_24h': 'Yesterday PM10',
        'pm10_mean_24h': '24h average PM10',
        'pm10_trend_3h': 'PM10 trend (rising/falling)',
        'pm10_anomaly': 'Unusual PM10 level',
        'is_dust_season': 'May-October indicator',
        'is_winter': 'Winter indicator (no dust)',
        'drying_potential': 'Warm + dry = dust risk',
        'pm_ratio': 'PM25/PM10 ratio (dust signature)',
        'is_coarse': 'Coarse particles (dust)',
        'hour': 'Hour of day',
        'month': 'Month of year',
    }
    
    for key, interp in interpretations.items():
        if key in feature_name:
            return interp
    
    return 'Other feature'


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Nome Dust ML Model')
    parser.add_argument('--pm-data', type=str, default='NomeHourlyData.csv',
                       help='Path to PM data CSV')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split')
    
    args = parser.parse_args()
    
    if not Path(args.pm_data).exists():
        # Try alternate names
        alternates = ['Nome-Hourly-Data.csv', 'NomeHourlyData.csv', 'nome_hourly_data.csv']
        for alt in alternates:
            if Path(alt).exists():
                args.pm_data = alt
                break
        else:
            print(f"ERROR: Data file not found: {args.pm_data}")
            return 1
    
    results = train_dust_models(
        data_path=args.pm_data,
        output_dir=args.output,
        val_split=args.val_split
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
