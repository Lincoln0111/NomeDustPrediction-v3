"""
Nome Dust Forecast Model - ML Version 3 (Enhanced)
===================================================

IMPROVEMENTS from v2:
1. Enhanced feature engineering (EMA, second-order derivatives, interaction terms)
2. Time-series cross-validation with gap
3. Model ensemble (LightGBM + XGBoost + GradientBoosting)
4. Better extreme value handling with Huber loss
5. Threshold optimization for F1/Recall
6. Feature selection to reduce overfitting
7. Stratified sampling for rare events

Requirements:
    pip install numpy pandas scikit-learn lightgbm xgboost joblib

Author: Nome Dust Forecasting Project
Date: February 2026
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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    GradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    brier_score_loss, 
    roc_auc_score, 
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve
)
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed.")

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MLConfigV3:
    """ML model configuration - Enhanced version"""
    
    # Dust thresholds
    PM10_ELEVATED: float = 50.0      # Moderate dust
    PM10_HIGH: float = 100.0         # High dust
    PM10_SEVERE: float = 200.0       # Severe dust
    
    # Features to EXCLUDE
    EXCLUDE_COLUMNS: List[str] = field(default_factory=lambda: [
        'CO', 'NO', 'NO2', 'O3',
        'site_name',
        'Unnamed: 0',
        'AQI', 'AQI_value',
    ])
    
    # Enhanced lag hours
    LAG_HOURS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 24, 48])
    
    # Enhanced rolling windows
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [3, 6, 12, 24, 48])
    
    # EMA spans for exponential moving averages
    EMA_SPANS: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    
    # Time-series CV settings
    N_SPLITS: int = 5
    GAP_HOURS: int = 24  # Gap between train and validation to prevent leakage
    
    # Model regularization - more aggressive
    MIN_SAMPLES_LEAF: int = 50
    MAX_DEPTH: int = 4
    LEARNING_RATE: float = 0.05
    N_ESTIMATORS: int = 200
    SUBSAMPLE: float = 0.6
    COLSAMPLE: float = 0.6
    REG_ALPHA: float = 1.0
    REG_LAMBDA: float = 2.0
    
    # Ensemble settings
    USE_ENSEMBLE: bool = True
    ENSEMBLE_WEIGHTS: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])  # LGB, XGB, GB
    
    # Threshold optimization
    OPTIMIZE_THRESHOLD: bool = True
    THRESHOLD_METRIC: str = 'f1'  # 'f1', 'recall', 'precision'
    
    # Feature selection
    USE_FEATURE_SELECTION: bool = True
    FEATURE_SELECTION_THRESHOLD: str = 'median'  # Keep features above median importance


CONFIG = MLConfigV3()


def _ensure_main_mlconfig():
    """Allow joblib to resolve pickled MLConfigV3 saved under __main__."""
    try:
        import __main__
        if not hasattr(__main__, "MLConfigV3"):
            __main__.MLConfigV3 = MLConfigV3
    except Exception:
        pass


# =============================================================================
# DATA LOADING
# =============================================================================

def load_nome_data(filepath: str, config: MLConfigV3 = CONFIG) -> pd.DataFrame:
    """Load Nome PM data with proper date parsing."""
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath)
    print(f"  Raw rows: {len(df)}")
    
    # Use format='mixed' to handle both date-only and datetime formats
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"  After date parsing: {len(df)}")
    
    df = df.set_index('date').sort_index()
    
    for col in config.EXCLUDE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    for col in df.columns:
        if col not in ['site_name']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.rename(columns={'AT': 'temp_c', 'RH': 'humidity'})
    df = df.dropna(subset=['PM10'])
    
    print(f"  After dropping missing PM10: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  PM10 range: {df['PM10'].min():.1f} to {df['PM10'].max():.1f}")
    print(f"  PM10 > 50 (dust events): {(df['PM10'] >= 50).sum()} ({100*(df['PM10'] >= 50).mean():.1f}%)")
    
    return df


# =============================================================================
# ENHANCED FEATURE ENGINEERING
# =============================================================================

def create_features_v3(df: pd.DataFrame, config: MLConfigV3 = CONFIG) -> pd.DataFrame:
    """
    Create enhanced ML features.
    
    New features in v3:
    - Exponential Moving Averages (EMA)
    - Second-order derivatives (acceleration)
    - More interaction terms
    - Volatility features
    - Regime indicators
    """
    result = df.copy()
    
    # ===================
    # TEMPORAL FEATURES
    # ===================
    result['hour'] = result.index.hour
    result['day_of_week'] = result.index.dayofweek
    result['month'] = result.index.month
    result['day_of_year'] = result.index.dayofyear
    
    # Cyclical encoding (enhanced)
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    
    # Season indicators
    result['is_dust_season'] = result['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
    result['is_peak_dust'] = result['month'].isin([6, 7, 8]).astype(int)
    result['is_winter'] = result['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)
    result['is_daytime'] = ((result['hour'] >= 8) & (result['hour'] <= 20)).astype(int)
    result['is_peak_hours'] = ((result['hour'] >= 10) & (result['hour'] <= 18)).astype(int)
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    
    # ===================
    # TEMPERATURE FEATURES (Enhanced)
    # ===================
    if 'temp_c' in result.columns:
        # Binary thresholds
        result['temp_below_minus10'] = (result['temp_c'] < -10).astype(int)
        result['temp_below_minus5'] = (result['temp_c'] < -5).astype(int)
        result['temp_below_0'] = (result['temp_c'] < 0).astype(int)
        result['temp_above_5'] = (result['temp_c'] > 5).astype(int)
        result['temp_above_10'] = (result['temp_c'] > 10).astype(int)
        result['temp_above_15'] = (result['temp_c'] > 15).astype(int)
        result['temp_above_20'] = (result['temp_c'] > 20).astype(int)
        
        # Lagged temperature
        for lag in config.LAG_HOURS:
            result[f'temp_lag_{lag}h'] = result['temp_c'].shift(lag)
        
        # Temperature changes (first derivative)
        result['temp_change_1h'] = result['temp_c'] - result['temp_c'].shift(1)
        result['temp_change_3h'] = result['temp_c'] - result['temp_c'].shift(3)
        result['temp_change_6h'] = result['temp_c'] - result['temp_c'].shift(6)
        result['temp_change_24h'] = result['temp_c'] - result['temp_c'].shift(24)
        
        # Temperature acceleration (second derivative)
        result['temp_accel_3h'] = result['temp_change_3h'] - result['temp_change_3h'].shift(3)
        result['temp_accel_6h'] = result['temp_change_6h'] - result['temp_change_6h'].shift(6)
        
        # Rolling stats
        for window in config.ROLLING_WINDOWS:
            result[f'temp_mean_{window}h'] = result['temp_c'].rolling(window, min_periods=1).mean()
            result[f'temp_max_{window}h'] = result['temp_c'].rolling(window, min_periods=1).max()
            result[f'temp_min_{window}h'] = result['temp_c'].rolling(window, min_periods=1).min()
            result[f'temp_std_{window}h'] = result['temp_c'].rolling(window, min_periods=1).std()
        
        # EMA (Exponential Moving Average)
        for span in config.EMA_SPANS:
            result[f'temp_ema_{span}h'] = result['temp_c'].ewm(span=span, adjust=False).mean()
        
        # Temperature volatility
        result['temp_volatility_24h'] = result['temp_c'].rolling(24, min_periods=6).std() / (result['temp_c'].rolling(24, min_periods=6).mean().abs() + 1)
        
        # Freeze-thaw cycles
        result['freeze_thaw'] = ((result['temp_c'].shift(1) < 0) & (result['temp_c'] > 0)).astype(int).rolling(24).sum()
    
    # ===================
    # HUMIDITY FEATURES (Enhanced)
    # ===================
    if 'humidity' in result.columns:
        result['is_very_dry'] = (result['humidity'] < 30).astype(int)
        result['is_dry'] = (result['humidity'] < 50).astype(int)
        result['is_moderate'] = ((result['humidity'] >= 50) & (result['humidity'] < 70)).astype(int)
        result['is_humid'] = (result['humidity'] >= 70).astype(int)
        result['is_very_humid'] = (result['humidity'] > 85).astype(int)
        
        for lag in config.LAG_HOURS:
            result[f'humidity_lag_{lag}h'] = result['humidity'].shift(lag)
        
        result['humidity_change_3h'] = result['humidity'] - result['humidity'].shift(3)
        result['humidity_change_6h'] = result['humidity'] - result['humidity'].shift(6)
        result['humidity_change_24h'] = result['humidity'] - result['humidity'].shift(24)
        
        # Humidity acceleration
        result['humidity_accel_6h'] = result['humidity_change_6h'] - result['humidity_change_6h'].shift(6)
        
        for window in config.ROLLING_WINDOWS:
            result[f'humidity_mean_{window}h'] = result['humidity'].rolling(window, min_periods=1).mean()
            result[f'humidity_min_{window}h'] = result['humidity'].rolling(window, min_periods=1).min()
        
        for span in config.EMA_SPANS:
            result[f'humidity_ema_{span}h'] = result['humidity'].ewm(span=span, adjust=False).mean()
        
        # Drying duration
        result['hours_dry'] = (result['humidity'] < 50).astype(int).groupby((result['humidity'] >= 50).cumsum()).cumsum()
    
    # ===================
    # PM10 LAGGED FEATURES (Enhanced)
    # ===================
    if 'PM10' in result.columns:
        # Lagged values
        for lag in config.LAG_HOURS:
            result[f'pm10_lag_{lag}h'] = result['PM10'].shift(lag)
        
        # Rolling statistics
        for window in config.ROLLING_WINDOWS:
            shifted = result['PM10'].shift(1)
            result[f'pm10_mean_{window}h'] = shifted.rolling(window, min_periods=1).mean()
            result[f'pm10_max_{window}h'] = shifted.rolling(window, min_periods=1).max()
            result[f'pm10_min_{window}h'] = shifted.rolling(window, min_periods=1).min()
            result[f'pm10_std_{window}h'] = shifted.rolling(window, min_periods=1).std()
            result[f'pm10_median_{window}h'] = shifted.rolling(window, min_periods=1).median()
        
        # EMA
        for span in config.EMA_SPANS:
            result[f'pm10_ema_{span}h'] = result['PM10'].shift(1).ewm(span=span, adjust=False).mean()
        
        # Trends (first derivative)
        result['pm10_trend_1h'] = result['PM10'].shift(1) - result['PM10'].shift(2)
        result['pm10_trend_3h'] = result['PM10'].shift(1) - result['PM10'].shift(4)
        result['pm10_trend_6h'] = result['PM10'].shift(1) - result['PM10'].shift(7)
        result['pm10_trend_24h'] = result['PM10'].shift(1) - result['PM10'].shift(25)
        
        # Acceleration (second derivative)
        result['pm10_accel_3h'] = result['pm10_trend_3h'] - result['pm10_trend_3h'].shift(3)
        result['pm10_accel_6h'] = result['pm10_trend_6h'] - result['pm10_trend_6h'].shift(6)
        
        # Anomaly scores
        result['pm10_anomaly'] = result['PM10'].shift(1) / (result['pm10_mean_24h'] + 1)
        result['pm10_zscore'] = (result['PM10'].shift(1) - result['pm10_mean_24h']) / (result['pm10_std_24h'] + 0.1)
        
        # Volatility
        result['pm10_volatility'] = result['pm10_std_24h'] / (result['pm10_mean_24h'] + 1)
        
        # Regime indicators
        result['pm10_regime_high'] = (result['pm10_mean_24h'] > 50).astype(int)
        result['pm10_regime_extreme'] = (result['pm10_max_24h'] > 200).astype(int)
        
        # Recent dust event
        result['had_dust_6h'] = (result['PM10'].shift(1).rolling(6, min_periods=1).max() >= 50).astype(int)
        result['had_dust_24h'] = (result['PM10'].shift(1).rolling(24, min_periods=1).max() >= 50).astype(int)
        result['had_severe_24h'] = (result['PM10'].shift(1).rolling(24, min_periods=1).max() >= 200).astype(int)
    
    # ===================
    # PM25 FEATURES
    # ===================
    if 'PM25' in result.columns:
        for lag in [1, 3, 6, 12, 24]:
            result[f'pm25_lag_{lag}h'] = result['PM25'].shift(lag)
        
        if 'PM10' in result.columns:
            result['pm_ratio'] = result['PM25'].shift(1) / (result['PM10'].shift(1) + 0.1)
            result['pm_ratio_mean_24h'] = result['pm_ratio'].rolling(24, min_periods=1).mean()
            result['is_coarse'] = (result['pm_ratio'] < 0.3).astype(int)
            result['is_fine'] = (result['pm_ratio'] > 0.5).astype(int)
    
    # ===================
    # INTERACTION FEATURES (Enhanced)
    # ===================
    if 'temp_c' in result.columns and 'humidity' in result.columns:
        # Drying potential
        result['drying_potential'] = result['temp_c'] * (100 - result['humidity']) / 100
        result['drying_potential_24h'] = result['drying_potential'].rolling(24, min_periods=1).mean()
        
        # Heat index proxy
        result['heat_humidity_idx'] = result['temp_c'] * result['humidity'] / 100
        
        # Dry-warm conditions
        result['dry_warm'] = ((result['temp_c'] > 10) & (result['humidity'] < 50)).astype(int)
        result['dry_warm_hours'] = result['dry_warm'].rolling(24, min_periods=1).sum()
        
        # Very favorable for dust
        result['dust_favorable'] = (
            (result['temp_c'] > 5) & 
            (result['humidity'] < 60) & 
            (result['is_dust_season'] == 1)
        ).astype(int)
    
    # ===================
    # TEMPORAL INTERACTION
    # ===================
    if 'PM10' in result.columns:
        # Hour * PM10 interaction
        result['pm10_hour_interact'] = result['pm10_lag_1h'] * result['hour'] / 24
        
        # Season * PM10 regime
        result['season_regime'] = result['is_dust_season'] * result.get('pm10_regime_high', 0)
    
    # ===================
    # TARGET VARIABLES
    # ===================
    result['is_dust_event'] = (result['PM10'] >= config.PM10_ELEVATED).astype(int)
    result['is_high_dust'] = (result['PM10'] >= config.PM10_HIGH).astype(int)
    result['is_severe_dust'] = (result['PM10'] >= config.PM10_SEVERE).astype(int)
    
    return result


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude targets and raw values)"""
    exclude = [
        'PM10', 'PM25',
        'is_dust_event', 'is_high_dust', 'is_severe_dust',
        'site_name',
    ]
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]


def select_features(X: pd.DataFrame, y: pd.Series, config: MLConfigV3 = CONFIG) -> List[str]:
    """Select important features using model-based selection."""
    if not config.USE_FEATURE_SELECTION:
        return list(X.columns)
    
    print("  Running feature selection...")
    
    if HAS_LIGHTGBM:
        selector_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbose=-1,
            importance_type='gain'
        )
    else:
        selector_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
    
    selector = SelectFromModel(selector_model, threshold=config.FEATURE_SELECTION_THRESHOLD)
    selector.fit(X.fillna(0), y)
    
    selected = X.columns[selector.get_support()].tolist()
    print(f"  Selected {len(selected)}/{len(X.columns)} features")
    
    return selected


# =============================================================================
# ENHANCED ML MODELS
# =============================================================================

class EnsembleDustClassifier:
    """
    Ensemble dust event classifier with:
    - Multiple base models (LightGBM, XGBoost, GradientBoosting)
    - Soft voting or stacking
    - Calibrated probabilities
    - Optimized threshold
    """
    
    def __init__(self, config: MLConfigV3 = CONFIG):
        self.config = config
        self.models = {}
        self.ensemble = None
        self.calibrated_model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.selected_features = []
        self.trained = False
        self.metrics = {}
        self.optimal_threshold = 0.5
    
    def _build_base_models(self):
        """Build base models for ensemble."""
        models = []
        
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMClassifier(
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
                importance_type='gain',
            )
            models.append(('lgb', lgb_model))
            self.models['lgb'] = lgb_model
        
        if HAS_XGBOOST:
            xgb_model = xgb.XGBClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                learning_rate=self.config.LEARNING_RATE,
                max_depth=self.config.MAX_DEPTH,
                min_child_weight=self.config.MIN_SAMPLES_LEAF,
                subsample=self.config.SUBSAMPLE,
                colsample_bytree=self.config.COLSAMPLE,
                reg_alpha=self.config.REG_ALPHA,
                reg_lambda=self.config.REG_LAMBDA,
                scale_pos_weight=3,  # Handle imbalance
                random_state=42,
                eval_metric='logloss',
            )
            models.append(('xgb', xgb_model))
            self.models['xgb'] = xgb_model
        
        # Gradient Boosting (sklearn)
        gb_model = GradientBoostingClassifier(
            n_estimators=min(200, self.config.N_ESTIMATORS),
            learning_rate=self.config.LEARNING_RATE,
            max_depth=self.config.MAX_DEPTH,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            subsample=self.config.SUBSAMPLE,
            random_state=42
        )
        models.append(('gb', gb_model))
        self.models['gb'] = gb_model
        
        return models
    
    def _optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find optimal classification threshold."""
        if self.config.THRESHOLD_METRIC == 'f1':
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        elif self.config.THRESHOLD_METRIC == 'recall':
            # Find threshold for 90% recall
            for thresh in np.arange(0.1, 0.9, 0.01):
                rec = recall_score(y_true, (y_proba >= thresh).astype(int))
                if rec >= 0.9:
                    return thresh
            return 0.3
        else:
            return 0.5
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train ensemble with time-series aware validation."""
        
        print(f"\nTraining ensemble classifier on {len(X_train)} samples...")
        print(f"  Positive class: {y_train.sum()} ({100*y_train.mean():.1f}%)")
        
        self.feature_names = list(X_train.columns)
        
        # Feature selection
        self.selected_features = select_features(X_train, y_train, self.config)
        X_train_selected = X_train[self.selected_features]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected.fillna(0))
        
        # Build base models
        base_models = self._build_base_models()
        
        if self.config.USE_ENSEMBLE and len(base_models) > 1:
            print(f"  Building ensemble with {len(base_models)} models...")
            
            # Soft voting ensemble
            weights = self.config.ENSEMBLE_WEIGHTS[:len(base_models)]
            weights = [w / sum(weights) for w in weights]  # Normalize
            
            self.ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                weights=weights
            )
            self.ensemble.fit(X_train_scaled, y_train)
        else:
            # Use single best model
            self.ensemble = base_models[0][1]
            self.ensemble.fit(X_train_scaled, y_train)
        
        # Calibrate probabilities
        print("  Calibrating probabilities...")
        n_cv = min(5, max(2, y_train.sum() // 10))
        self.calibrated_model = CalibratedClassifierCV(
            self.ensemble, method='isotonic', cv=n_cv
        )
        self.calibrated_model.fit(X_train_scaled, y_train)
        
        # Get training predictions
        train_probs = self.calibrated_model.predict_proba(X_train_scaled)[:, 1]
        
        # Optimize threshold
        if self.config.OPTIMIZE_THRESHOLD:
            self.optimal_threshold = self._optimize_threshold(y_train.values, train_probs)
            print(f"  Optimal threshold: {self.optimal_threshold:.3f}")
        
        train_preds = (train_probs >= self.optimal_threshold).astype(int)
        
        # Training metrics
        self.metrics = {
            'train_samples': len(y_train),
            'train_positive': int(y_train.sum()),
            'train_auc': roc_auc_score(y_train, train_probs) if y_train.sum() > 0 else 0,
            'train_brier': brier_score_loss(y_train, train_probs),
            'train_precision': precision_score(y_train, train_preds, zero_division=0),
            'train_recall': recall_score(y_train, train_preds, zero_division=0),
            'train_f1': f1_score(y_train, train_preds, zero_division=0),
            'optimal_threshold': self.optimal_threshold,
            'selected_features': len(self.selected_features),
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_selected = X_val[self.selected_features]
            X_val_scaled = self.scaler.transform(X_val_selected.fillna(0))
            val_probs = self.calibrated_model.predict_proba(X_val_scaled)[:, 1]
            val_preds = (val_probs >= self.optimal_threshold).astype(int)
            
            self.metrics['val_samples'] = len(y_val)
            self.metrics['val_positive'] = int(y_val.sum())
            self.metrics['val_auc'] = roc_auc_score(y_val, val_probs) if y_val.sum() > 0 else 0
            self.metrics['val_brier'] = brier_score_loss(y_val, val_probs)
            self.metrics['val_precision'] = precision_score(y_val, val_preds, zero_division=0)
            self.metrics['val_recall'] = recall_score(y_val, val_preds, zero_division=0)
            self.metrics['val_f1'] = f1_score(y_val, val_preds, zero_division=0)
        
        # Feature importance
        self._compute_feature_importance()
        
        self.trained = True
        
        print(f"  Train AUC: {self.metrics['train_auc']:.3f}")
        print(f"  Train F1: {self.metrics['train_f1']:.3f}")
        if 'val_auc' in self.metrics:
            print(f"  Val AUC: {self.metrics['val_auc']:.3f}")
            print(f"  Val F1: {self.metrics['val_f1']:.3f}")
            print(f"  Val Precision: {self.metrics['val_precision']:.3f}")
            print(f"  Val Recall: {self.metrics['val_recall']:.3f}")
        
        return self.metrics
    
    def _compute_feature_importance(self):
        """Compute aggregated feature importance from ensemble."""
        importance_sum = np.zeros(len(self.selected_features))
        count = 0
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_sum += model.feature_importances_
                count += 1
        
        if count > 0:
            importances = importance_sum / count
            total = importances.sum()
            if total > 0:
                importances = importances / total
            
            idx = np.argsort(importances)[::-1][:15]
            self.metrics['top_features'] = {
                self.selected_features[i]: round(float(importances[i]), 4)
                for i in idx
            }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict dust probability."""
        # Align features
        X_selected = X.reindex(columns=self.selected_features, fill_value=0)
        X_scaled = self.scaler.transform(X_selected.fillna(0))
        return self.calibrated_model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with optimized threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.optimal_threshold).astype(int)
    
    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'ensemble': self.ensemble,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'metrics': self.metrics,
            'config': self.config,
            'optimal_threshold': self.optimal_threshold,
        }, path)
    
    def load(self, path: str):
        try:
            data = joblib.load(path)
        except AttributeError as exc:
            if "MLConfigV3" in str(exc):
                _ensure_main_mlconfig()
                data = joblib.load(path)
            else:
                raise
        self.models = data['models']
        self.ensemble = data['ensemble']
        self.calibrated_model = data['calibrated_model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.selected_features = data['selected_features']
        self.metrics = data['metrics']
        self.optimal_threshold = data.get('optimal_threshold', 0.5)
        self.trained = True


class EnsemblePM10Regressor:
    """
    Ensemble PM10 regressor with:
    - Multiple base models for quantile predictions
    - Huber loss for robustness to outliers
    - Better uncertainty quantification
    """
    
    def __init__(self, config: MLConfigV3 = CONFIG):
        self.config = config
        self.model_p50 = None
        self.model_p10 = None
        self.model_p90 = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.selected_features = []
        self.trained = False
        self.metrics = {}
        self.y_train_std = 1.0  # For uncertainty scaling
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train quantile regressors with better extreme value handling."""
        
        print(f"\nTraining ensemble PM10 regressor on {len(X_train)} samples...")
        print(f"  PM10 median: {y_train.median():.1f}, max: {y_train.max():.1f}")
        
        self.feature_names = list(X_train.columns)
        self.y_train_std = y_train.std()
        
        # Feature selection (use same features as classifier)
        self.selected_features = select_features(X_train, (y_train >= 50).astype(int), self.config)
        X_train_selected = X_train[self.selected_features]
        
        X_train_scaled = self.scaler.fit_transform(X_train_selected.fillna(0))
        
        # Log transform for better distribution
        y_log = np.log1p(y_train)
        
        # Common params with stronger regularization
        params = dict(
            n_estimators=min(400, self.config.N_ESTIMATORS),
            learning_rate=self.config.LEARNING_RATE,
            max_depth=self.config.MAX_DEPTH,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            subsample=self.config.SUBSAMPLE,
            random_state=42
        )
        
        # Train median (p50) with Huber loss for robustness
        print("  Training median model (Huber loss)...")
        self.model_p50 = GradientBoostingRegressor(loss='huber', alpha=0.9, **params)
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
        
        # Ensure monotonicity
        pred_p10 = np.minimum(pred_p10, pred_p50)
        pred_p90 = np.maximum(pred_p90, pred_p50)
        
        self.metrics = {
            'train_samples': len(y_train),
            'train_mae': mean_absolute_error(y_train, pred_p50),
            'train_rmse': np.sqrt(mean_squared_error(y_train, pred_p50)),
            'train_coverage': ((y_train >= pred_p10) & (y_train <= pred_p90)).mean(),
            'selected_features': len(self.selected_features),
        }
        
        # Compute metrics by PM10 level
        high_mask = y_train >= 100
        if high_mask.sum() > 10:
            self.metrics['train_mae_high'] = mean_absolute_error(y_train[high_mask], pred_p50[high_mask])
        
        if X_val is not None and y_val is not None:
            X_val_selected = X_val[self.selected_features]
            X_val_scaled = self.scaler.transform(X_val_selected.fillna(0))
            val_p50 = np.expm1(self.model_p50.predict(X_val_scaled))
            val_p10 = np.expm1(self.model_p10.predict(X_val_scaled))
            val_p90 = np.expm1(self.model_p90.predict(X_val_scaled))
            
            val_p10 = np.minimum(val_p10, val_p50)
            val_p90 = np.maximum(val_p90, val_p50)
            
            self.metrics['val_samples'] = len(y_val)
            self.metrics['val_mae'] = mean_absolute_error(y_val, val_p50)
            self.metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_p50))
            self.metrics['val_coverage'] = ((y_val >= val_p10) & (y_val <= val_p90)).mean()
            
            high_mask_val = y_val >= 100
            if high_mask_val.sum() > 10:
                self.metrics['val_mae_high'] = mean_absolute_error(y_val[high_mask_val], val_p50[high_mask_val])
        
        self.trained = True
        
        print(f"  Train MAE: {self.metrics['train_mae']:.2f}")
        print(f"  Train coverage: {self.metrics['train_coverage']:.1%}")
        if 'val_mae' in self.metrics:
            print(f"  Val MAE: {self.metrics['val_mae']:.2f}")
            print(f"  Val RMSE: {self.metrics['val_rmse']:.2f}")
            print(f"  Val coverage: {self.metrics['val_coverage']:.1%}")
            if 'val_mae_high' in self.metrics:
                print(f"  Val MAE (PM10>100): {self.metrics['val_mae_high']:.2f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict PM10 with uncertainty bounds."""
        X_selected = X.reindex(columns=self.selected_features, fill_value=0)
        X_scaled = self.scaler.transform(X_selected.fillna(0))
        
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
            'selected_features': self.selected_features,
            'metrics': self.metrics,
            'y_train_std': self.y_train_std,
        }, path)
    
    def load(self, path: str):
        try:
            data = joblib.load(path)
        except AttributeError as exc:
            if "MLConfigV3" in str(exc):
                _ensure_main_mlconfig()
                data = joblib.load(path)
            else:
                raise
        self.model_p50 = data['model_p50']
        self.model_p10 = data['model_p10']
        self.model_p90 = data['model_p90']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.selected_features = data['selected_features']
        self.metrics = data['metrics']
        self.y_train_std = data.get('y_train_std', 1.0)
        self.trained = True


# =============================================================================
# TIME-SERIES CROSS-VALIDATION
# =============================================================================

def timeseries_cv_train(df: pd.DataFrame, config: MLConfigV3 = CONFIG) -> Tuple[Dict, Dict]:
    """
    Train with time-series cross-validation.
    
    Uses multiple folds with a gap to prevent leakage.
    """
    print("\n" + "="*70)
    print("TIME-SERIES CROSS-VALIDATION")
    print("="*70)
    
    feature_cols = get_feature_columns(df)
    
    # Remove rows with NaN in key columns
    model_df = df.dropna(subset=['is_dust_event', 'PM10'] + feature_cols[:5])
    print(f"Samples for CV: {len(model_df)}")
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS, gap=config.GAP_HOURS)
    
    cv_metrics_class = []
    cv_metrics_reg = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(model_df)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")
        
        train_df = model_df.iloc[train_idx]
        val_df = model_df.iloc[val_idx]
        
        print(f"  Train: {len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()})")
        print(f"  Val: {len(val_df)} ({val_df.index.min().date()} to {val_df.index.max().date()})")
        
        X_train = train_df[feature_cols]
        y_train_class = train_df['is_dust_event']
        y_train_reg = train_df['PM10']
        
        X_val = val_df[feature_cols]
        y_val_class = val_df['is_dust_event']
        y_val_reg = val_df['PM10']
        
        # Train classifier
        classifier = EnsembleDustClassifier(config)
        class_metrics = classifier.train(X_train, y_train_class, X_val, y_val_class)
        cv_metrics_class.append(class_metrics)
        
        # Train regressor
        regressor = EnsemblePM10Regressor(config)
        reg_metrics = regressor.train(X_train, y_train_reg, X_val, y_val_reg)
        cv_metrics_reg.append(reg_metrics)
    
    # Aggregate CV metrics
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    avg_class_metrics = {
        'cv_val_auc_mean': np.mean([m.get('val_auc', 0) for m in cv_metrics_class]),
        'cv_val_auc_std': np.std([m.get('val_auc', 0) for m in cv_metrics_class]),
        'cv_val_f1_mean': np.mean([m.get('val_f1', 0) for m in cv_metrics_class]),
        'cv_val_f1_std': np.std([m.get('val_f1', 0) for m in cv_metrics_class]),
        'cv_val_recall_mean': np.mean([m.get('val_recall', 0) for m in cv_metrics_class]),
        'cv_val_precision_mean': np.mean([m.get('val_precision', 0) for m in cv_metrics_class]),
    }
    
    avg_reg_metrics = {
        'cv_val_mae_mean': np.mean([m.get('val_mae', 0) for m in cv_metrics_reg]),
        'cv_val_mae_std': np.std([m.get('val_mae', 0) for m in cv_metrics_reg]),
        'cv_val_rmse_mean': np.mean([m.get('val_rmse', 0) for m in cv_metrics_reg]),
        'cv_val_coverage_mean': np.mean([m.get('val_coverage', 0) for m in cv_metrics_reg]),
    }
    
    print(f"\nClassifier CV Results:")
    print(f"  Val AUC: {avg_class_metrics['cv_val_auc_mean']:.3f} ± {avg_class_metrics['cv_val_auc_std']:.3f}")
    print(f"  Val F1: {avg_class_metrics['cv_val_f1_mean']:.3f} ± {avg_class_metrics['cv_val_f1_std']:.3f}")
    print(f"  Val Recall: {avg_class_metrics['cv_val_recall_mean']:.3f}")
    print(f"  Val Precision: {avg_class_metrics['cv_val_precision_mean']:.3f}")
    
    print(f"\nRegressor CV Results:")
    print(f"  Val MAE: {avg_reg_metrics['cv_val_mae_mean']:.2f} ± {avg_reg_metrics['cv_val_mae_std']:.2f}")
    print(f"  Val RMSE: {avg_reg_metrics['cv_val_rmse_mean']:.2f}")
    print(f"  Val Coverage: {avg_reg_metrics['cv_val_coverage_mean']:.1%}")
    
    return avg_class_metrics, avg_reg_metrics


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_dust_models_v3(data_path: str, output_dir: str = 'models_v3', 
                         val_split: float = 0.2, run_cv: bool = True) -> Dict:
    """
    Enhanced training pipeline with:
    - Optional time-series CV
    - Feature selection
    - Model ensemble
    - Threshold optimization
    """
    
    print("="*70)
    print("NOME DUST ML MODEL V3 - ENHANCED TRAINING")
    print("="*70)
    
    # Load data
    df = load_nome_data(data_path)
    
    # Create enhanced features
    print("\nCreating enhanced features...")
    featured_df = create_features_v3(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(featured_df)
    print(f"  Total features: {len(feature_cols)}")
    
    # Remove rows with NaN in critical columns
    critical_cols = ['is_dust_event', 'PM10'] + [c for c in feature_cols if 'lag_1h' in c][:3]
    model_df = featured_df.dropna(subset=critical_cols)
    print(f"  Samples after dropping NaN: {len(model_df)}")
    
    # Run cross-validation if requested
    cv_results = None
    if run_cv:
        cv_class, cv_reg = timeseries_cv_train(model_df, CONFIG)
        cv_results = {'classifier': cv_class, 'regressor': cv_reg}
    
    # Final training on all data (with holdout)
    print("\n" + "="*70)
    print("FINAL MODEL TRAINING")
    print("="*70)
    
    # Time-based split
    split_idx = int(len(model_df) * (1 - val_split))
    train_df = model_df.iloc[:split_idx]
    val_df = model_df.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Val: {len(val_df)} ({val_df.index.min().date()} to {val_df.index.max().date()})")
    
    X_train = train_df[feature_cols]
    y_train_class = train_df['is_dust_event']
    y_train_reg = train_df['PM10']
    
    X_val = val_df[feature_cols]
    y_val_class = val_df['is_dust_event']
    y_val_reg = val_df['PM10']
    
    # Train classifier
    print("\n" + "-"*50)
    print("ENSEMBLE CLASSIFIER")
    print("-"*50)
    classifier = EnsembleDustClassifier()
    class_metrics = classifier.train(X_train, y_train_class, X_val, y_val_class)
    
    # Train regressor
    print("\n" + "-"*50)
    print("ENSEMBLE REGRESSOR")
    print("-"*50)
    regressor = EnsemblePM10Regressor()
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
    
    classifier.save(output_path / 'classifier_v3.joblib')
    regressor.save(output_path / 'regressor_v3.joblib')
    
    # Also save as default names for compatibility
    classifier.save(output_path / 'classifier.joblib')
    regressor.save(output_path / 'regressor.joblib')
    
    # Save metadata
    metadata = {
        'version': '3.0',
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'data_file': data_path,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'feature_count': len(feature_cols),
        'selected_features': len(classifier.selected_features),
        'classifier_metrics': class_metrics,
        'regressor_metrics': reg_metrics,
        'cv_results': cv_results,
        'config': {
            'n_estimators': CONFIG.N_ESTIMATORS,
            'learning_rate': CONFIG.LEARNING_RATE,
            'max_depth': CONFIG.MAX_DEPTH,
            'use_ensemble': CONFIG.USE_ENSEMBLE,
            'use_feature_selection': CONFIG.USE_FEATURE_SELECTION,
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nModels saved to {output_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Classifier Val AUC: {class_metrics.get('val_auc', 'N/A'):.3f}")
    print(f"Classifier Val F1: {class_metrics.get('val_f1', 'N/A'):.3f}")
    print(f"Classifier Val Precision: {class_metrics.get('val_precision', 'N/A'):.3f}")
    print(f"Classifier Val Recall: {class_metrics.get('val_recall', 'N/A'):.3f}")
    print(f"Classifier Optimal Threshold: {class_metrics.get('optimal_threshold', 0.5):.3f}")
    print(f"Regressor Val MAE: {reg_metrics.get('val_mae', 'N/A'):.2f}")
    print(f"Regressor Val RMSE: {reg_metrics.get('val_rmse', 'N/A'):.2f}")
    print(f"Regressor Val Coverage: {reg_metrics.get('val_coverage', 'N/A'):.1%}")
    
    if cv_results:
        print(f"\nCross-Validation Results:")
        print(f"  CV AUC: {cv_results['classifier']['cv_val_auc_mean']:.3f} ± {cv_results['classifier']['cv_val_auc_std']:.3f}")
        print(f"  CV MAE: {cv_results['regressor']['cv_val_mae_mean']:.2f} ± {cv_results['regressor']['cv_val_mae_std']:.2f}")
    
    return {
        'classifier': class_metrics,
        'regressor': reg_metrics,
        'metadata': metadata,
        'cv_results': cv_results
    }


def interpret_feature(feature_name: str) -> str:
    """Interpret what a feature means."""
    interpretations = {
        'temp_c': 'Current temperature',
        'temp_below_0': 'Frozen ground indicator',
        'temp_below_minus5': 'Deep freeze - no dust',
        'temp_above_10': 'Warm enough for dust',
        'temp_accel': 'Temperature acceleration',
        'freeze_thaw': 'Freeze-thaw cycles',
        'humidity': 'Current humidity',
        'is_dry': 'Dry conditions',
        'is_very_dry': 'Very dry conditions',
        'hours_dry': 'Consecutive dry hours',
        'pm10_lag_1h': 'Recent PM10 (persistence)',
        'pm10_lag_24h': 'Yesterday PM10',
        'pm10_mean': 'Average PM10',
        'pm10_ema': 'Exponential avg PM10',
        'pm10_trend': 'PM10 trend',
        'pm10_accel': 'PM10 acceleration',
        'pm10_anomaly': 'Unusual PM10 level',
        'pm10_zscore': 'PM10 z-score',
        'pm10_volatility': 'PM10 variability',
        'pm10_regime': 'High PM10 regime',
        'had_dust': 'Recent dust event',
        'is_dust_season': 'May-October',
        'is_peak_dust': 'June-August (peak)',
        'is_winter': 'Winter (no dust)',
        'drying_potential': 'Warm + dry = risk',
        'dry_warm': 'Dry and warm',
        'dust_favorable': 'Favorable for dust',
        'pm_ratio': 'PM25/PM10 ratio',
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
    
    parser = argparse.ArgumentParser(description='Train Nome Dust ML Model V3')
    parser.add_argument('--pm-data', type=str, default='Nome-Hourly-Data.csv',
                       help='Path to PM data CSV')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split')
    parser.add_argument('--no-cv', action='store_true',
                       help='Skip cross-validation')
    
    args = parser.parse_args()
    
    if not Path(args.pm_data).exists():
        alternates = ['Nome-Hourly-Data.csv', 'NomeHourlyData.csv', 'nome_hourly_data.csv']
        for alt in alternates:
            if Path(alt).exists():
                args.pm_data = alt
                break
        else:
            print(f"ERROR: Data file not found: {args.pm_data}")
            return 1
    
    results = train_dust_models_v3(
        data_path=args.pm_data,
        output_dir=args.output,
        val_split=args.val_split,
        run_cv=not args.no_cv
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
