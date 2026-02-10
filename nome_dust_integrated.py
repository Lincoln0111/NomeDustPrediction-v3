import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import joblib

# Import from v2 ML model
from nome_dust_ml_model_v2 import (
    DustClassifier,
    PM10Regressor,
    create_features,
    MLConfig
)

# Import road classification (if available)
try:
    from nome_dust_forecast_production import (
        RoadClassifier,
        DustForecastConfig
    )
    HAS_ROAD_CLASSIFIER = True
except ImportError:
    HAS_ROAD_CLASSIFIER = False
    print("Note: Road classifier not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default EPA AQI breakpoints (24-hour average, µg/m³)
# Format: (conc_low, conc_high, aqi_low, aqi_high, category, color)
DEFAULT_AQI_BREAKPOINTS_PM25 = [
    (0, 12.0, 0, 50, "Good", "GREEN"),
    (12.1, 35.4, 51, 100, "Moderate", "YELLOW"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "ORANGE"),
    (55.5, 150.4, 151, 200, "Unhealthy", "RED"),
    (150.5, 250.4, 201, 300, "Very Unhealthy", "PURPLE"),
    (250.5, 500.4, 301, 500, "Hazardous", "MAROON"),
]

DEFAULT_AQI_BREAKPOINTS_PM10 = [
    (0, 54, 0, 50, "Good", "GREEN"),
    (55, 154, 51, 100, "Moderate", "YELLOW"),
    (155, 254, 101, 150, "Unhealthy for Sensitive Groups", "ORANGE"),
    (255, 354, 151, 200, "Unhealthy", "RED"),
    (355, 424, 201, 300, "Very Unhealthy", "PURPLE"),
    (425, 604, 301, 500, "Hazardous", "MAROON"),
]


@dataclass
class IntegratedConfig:
    """Configuration for integrated forecast system"""
    
    # Location
    NOME_LAT: float = 64.5011
    NOME_LON: float = -165.4064
    NOME_TZ: str = "America/Anchorage"
    
    # Physics override thresholds
    FROZEN_TEMP_THRESHOLD: float = -5.0
    FROZEN_TRANSITION_LOW: float = -5.0
    FROZEN_TRANSITION_HIGH: float = 2.0

    # Risk weighting (retain probability signal without overpowering PM10)
    HOURLY_PM10_WEIGHT: float = 0.6
    HOURLY_UPPER_WEIGHT: float = 0.3
    HOURLY_PROB_WEIGHT: float = 0.1
    DAILY_PM10_MEAN_WEIGHT: float = 0.6
    DAILY_PM10_MAX_WEIGHT: float = 0.3
    DAILY_PROB_WEIGHT: float = 0.1

    # PM2.5 derivation from PM10 (used only when PM2.5 is not available)
    PM25_FROM_PM10_RATIO: float = 0.15

    # AQI breakpoint tables (configurable)
    AQI_BREAKPOINTS_PM25: List[Tuple[float, float, int, int, str, str]] = field(
        default_factory=lambda: DEFAULT_AQI_BREAKPOINTS_PM25.copy()
    )
    AQI_BREAKPOINTS_PM10: List[Tuple[float, float, int, int, str, str]] = field(
        default_factory=lambda: DEFAULT_AQI_BREAKPOINTS_PM10.copy()
    )
    USE_PM10_AQI_WHEN_PM25_DERIVED: bool = True
    
    # Risk thresholds
    PM10_YELLOW_THRESHOLD: float = 50.0
    PM10_RED_THRESHOLD: float = 150.0
    PROB_YELLOW_THRESHOLD: float = 0.3
    PROB_RED_THRESHOLD: float = 0.6
    
    # API settings
    OPEN_METEO_TIMEOUT: int = 15
    WEATHER_CACHE_MINUTES: int = 30


CONFIG = IntegratedConfig()


class RiskLevel(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    PURPLE = "PURPLE"
    MAROON = "MAROON"


# =============================================================================
# PHYSICS OVERRIDE
# =============================================================================

def apply_frozen_ground_override(temp_c: float, ml_probability: float, 
                                  ml_pm10: float, config: IntegratedConfig = CONFIG) -> Tuple[float, float, str]:
    """Apply physics override for frozen ground conditions."""
    if temp_c is None:
        return ml_probability, ml_pm10, "no_temp_data"
    
    try:
        if pd.isna(temp_c):
            return ml_probability, ml_pm10, "no_temp_data"
        temp_c = float(temp_c)
    except (TypeError, ValueError):
        return ml_probability, ml_pm10, "no_temp_data"
    
    if temp_c < config.FROZEN_TEMP_THRESHOLD:
        return 0.0, 0.0, "frozen_ground"
    
    if temp_c < config.FROZEN_TRANSITION_HIGH:
        transition_range = config.FROZEN_TRANSITION_HIGH - config.FROZEN_TRANSITION_LOW
        thaw_factor = (temp_c - config.FROZEN_TRANSITION_LOW) / transition_range
        thaw_factor = max(0.0, min(1.0, thaw_factor))
        return ml_probability * thaw_factor, ml_pm10 * thaw_factor, f"partial_thaw_{thaw_factor:.2f}"
    
    return ml_probability, ml_pm10, "normal"


# =============================================================================
# WEATHER FETCHER
# =============================================================================

class WeatherDataFetcher:
    """Fetches weather data from Open-Meteo"""
    
    def __init__(self, config: IntegratedConfig = CONFIG):
        self.config = config
        self._cache = {}
        self._cache_time = None
        
    def fetch_current_and_forecast(self, hours: int = 24) -> pd.DataFrame:
        """Fetch weather in Nome local timezone"""
        if (self._cache_time and 
            (datetime.now(timezone.utc) - self._cache_time).seconds < 60 * self.config.WEATHER_CACHE_MINUTES and
            'weather' in self._cache):
            return self._cache['weather']
        
        weather_df = self._fetch_open_meteo(hours)
        
        if weather_df is not None and len(weather_df) > 0:
            self._cache['weather'] = weather_df
            self._cache_time = datetime.now(timezone.utc)
        
        return weather_df
    
    def _fetch_open_meteo(self, hours: int) -> Optional[pd.DataFrame]:
        """Fetch from Open-Meteo API"""
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': self.config.NOME_LAT,
            'longitude': self.config.NOME_LON,
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,soil_moisture_0_to_1cm',
            'forecast_days': max(1, hours // 24 + 1),
            'timezone': self.config.NOME_TZ,
            'past_days': 2,
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.config.OPEN_METEO_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            if not hourly:
                return None
            
            timestamps = pd.to_datetime(hourly.get('time', []))
            
            df = pd.DataFrame({
                'temp_c': hourly.get('temperature_2m', []),
                'humidity': hourly.get('relative_humidity_2m', []),
                'wind_speed': [w / 3.6 if w else None for w in hourly.get('wind_speed_10m', [])],
                'precip': hourly.get('precipitation', []),
                'soil_moisture': hourly.get('soil_moisture_0_to_1cm', []),
            }, index=timestamps)
            
            df.index = df.index.tz_localize(self.config.NOME_TZ)
            df.index.name = 'timestamp'
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Open-Meteo fetch failed: {e}")
            return None


# =============================================================================
# INTEGRATED FORECAST
# =============================================================================

class IntegratedDustForecast:
    """Main forecast system combining ML + physics + roads"""
    
    def __init__(self, config: IntegratedConfig = CONFIG):
        self.config = config
        self.classifier = DustClassifier()
        self.regressor = PM10Regressor()
        self.weather_fetcher = WeatherDataFetcher(config)
        self.road_classifier = None
        self.models_loaded = False
        self.roads_loaded = False
        self._pm10_history = pd.Series(dtype=float)
        self.nome_tz = ZoneInfo(config.NOME_TZ)
        
    def load_models(self, model_dir: str):
        """Load trained ML models"""
        model_path = Path(model_dir)
        self.classifier.load(model_path / 'classifier.joblib')
        self.regressor.load(model_path / 'regressor.joblib')
        self.models_loaded = True
        print(f"ML models loaded from {model_dir}")
        print(f"  Classifier features: {len(self.classifier.feature_names)}")
        print(f"  Val AUC: {self.classifier.metrics.get('val_auc', 'N/A'):.3f}")
        
    def load_roads(self, geojson_path: str):
        """Load road network"""
        if not HAS_ROAD_CLASSIFIER:
            return
        road_config = DustForecastConfig()
        self.road_classifier = RoadClassifier(road_config)
        num_roads = self.road_classifier.load_from_geojson(geojson_path)
        self.roads_loaded = True
        print(f"Loaded {num_roads} roads from {geojson_path}")
    
    def generate_forecast(self, hours: int = 24, current_pm10: float = None) -> Dict[str, Any]:
        """Generate hourly dust forecast"""
        if not self.models_loaded:
            return {'error': 'Models not loaded'}
        
        print("Fetching weather data...")
        weather_df = self.weather_fetcher.fetch_current_and_forecast(hours + 48)
        
        if weather_df is None or len(weather_df) == 0:
            return {'error': 'Failed to fetch weather data'}
        
        print(f"  Got {len(weather_df)} hours of weather data")
        
        raw_temp_series = weather_df['temp_c'].copy()
        weather_df = self._add_pm10_to_weather(weather_df, current_pm10)
        
        print("Creating features...")
        feature_df = self._create_features(weather_df)
        feature_df['_raw_temp_c'] = raw_temp_series.reindex(feature_df.index)
        
        print("Generating predictions...")
        predictions = self._generate_predictions(feature_df, hours)
        
        road_forecasts = []
        area_forecasts = []
        if self.roads_loaded and self.road_classifier:
            print("Applying road-specific factors...")
            road_forecasts, area_forecasts = self._apply_road_factors(predictions)
        
        current = self._get_current_conditions(weather_df, raw_temp_series)
        
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'forecast_hours': hours,
            'models': {
                'classifier_val_auc': self.classifier.metrics.get('val_auc'),
                'regressor_val_mae': self.regressor.metrics.get('val_mae'),
            },
            'current_conditions': current,
            'hourly_forecasts': predictions,
            'road_forecasts': road_forecasts[:1000],
            'area_forecasts': area_forecasts,
        }
    
    def generate_daily_forecast(self, days: int = 3, current_pm10: float = None) -> Dict[str, Any]:
        """
        Generate daily dust forecast for next N days.
        
        Aggregates hourly predictions into daily summaries following the logic
        used in Nome-Daily-Data.csv:
        - PM25: Daily mean of hourly values
        - valid_hours_count: Count of non-NA hours
        - AQI: Category based on daily mean PM25
        - AQI_value: Calculated from daily mean PM25
        """
        if not self.models_loaded:
            return {'error': 'Models not loaded'}
        
        hours_needed = (days + 1) * 24
        hourly_result = self.generate_forecast(hours=hours_needed, current_pm10=current_pm10)
        
        if 'error' in hourly_result:
            return hourly_result
        
        hourly_forecasts = hourly_result.get('hourly_forecasts', [])
        
        if not hourly_forecasts:
            return {'error': 'No hourly forecasts generated'}
        
        daily_forecasts = self._aggregate_to_daily(hourly_forecasts, days)
        
        daily_road_forecasts = []
        if hourly_result.get('road_forecasts'):
            daily_road_forecasts = self._aggregate_roads_to_daily(
                hourly_result['road_forecasts'], days
            )
        
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'forecast_days': days,
            'forecast_type': 'daily',
            'models': hourly_result.get('models', {}),
            'current_conditions': hourly_result.get('current_conditions', {}),
            'daily_forecasts': daily_forecasts,
            'daily_road_forecasts': daily_road_forecasts,
            'source': 'aggregated_hourly',
        }
    
    def _aggregate_to_daily(self, hourly_forecasts: List[Dict], days: int) -> List[Dict]:
        """Aggregate hourly forecasts into daily summaries."""
        now_nome = datetime.now(self.nome_tz)
        today = now_nome.date()
        
        daily_groups = defaultdict(list)
        
        for forecast in hourly_forecasts:
            try:
                ts_str = forecast.get('timestamp')
                if ts_str:
                    ts = pd.Timestamp(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize(self.config.NOME_TZ)
                    forecast_date = ts.date()
                    day_offset = (forecast_date - today).days
                    
                    if 0 <= day_offset < days:
                        daily_groups[day_offset].append(forecast)
            except Exception as e:
                print(f"Error grouping forecast: {e}")
                continue
        
        daily_forecasts = []
        
        for day_offset in range(days):
            forecasts = daily_groups.get(day_offset, [])
            forecast_date = today + timedelta(days=day_offset)
            
            if not forecasts:
                daily_forecasts.append({
                    'day_offset': day_offset,
                    'date': forecast_date.isoformat(),
                    'day_name': self._get_day_name(day_offset),
                    'valid_hours_count': 0,
                    'invalid_hours_count': 24,
                    'pm10_mean': None,
                    'pm10_max': None,
                    'pm10_min': None,
                    'pm25_mean': None,
                    'temp_mean_c': None,
                    'temp_max_c': None,
                    'temp_min_c': None,
                    'temp_mean_f': None,
                    'temp_max_f': None,
                    'temp_min_f': None,
                    'risk_level': 'GREEN',
                    'aqi': None,
                    'aqi_category': 'Unknown',
                    'aqi_color': 'GREEN',
                    'dust_probability_mean': None,
                    'frozen_hours': 0,
                    'physics_status': 'unknown',
                })
                continue
            
            pm10_values = [f['pm10_predicted'] for f in forecasts if f.get('pm10_predicted') is not None]
            temp_values = [f['temperature_c'] for f in forecasts if f.get('temperature_c') is not None]
            prob_values = [f['dust_probability'] for f in forecasts if f.get('dust_probability') is not None]
            frozen_count = sum(1 for f in forecasts if f.get('is_frozen', False))
            
            valid_hours = len(pm10_values)
            invalid_hours = 24 - valid_hours
            
            pm10_mean = np.mean(pm10_values) if pm10_values else None
            pm10_max = np.max(pm10_values) if pm10_values else None
            pm10_min = np.min(pm10_values) if pm10_values else None
            pm25_mean = (pm10_mean * self.config.PM25_FROM_PM10_RATIO) if pm10_mean is not None else None
            pm25_is_derived = pm10_mean is not None
            
            temp_mean = np.mean(temp_values) if temp_values else None
            temp_max = np.max(temp_values) if temp_values else None
            temp_min = np.min(temp_values) if temp_values else None
            
            prob_mean = np.mean(prob_values) if prob_values else None
            
            risk_level = self._determine_daily_risk(
                pm10_mean if pm10_mean else 0, 
                pm10_max if pm10_max else 0, 
                prob_mean if prob_mean else 0
            )
            
            aqi, aqi_category, aqi_color = self._calculate_daily_aqi(
                pm25_mean, pm10_mean, pm25_is_derived
            )
            
            if frozen_count == len(forecasts):
                physics_status = 'frozen_ground'
            elif frozen_count > 0:
                physics_status = f'partial_freeze_{frozen_count}h'
            elif temp_mean is not None and temp_mean < self.config.FROZEN_TRANSITION_HIGH:
                physics_status = 'transition_zone'
            else:
                physics_status = 'normal'
            
            daily_forecasts.append({
                'day_offset': day_offset,
                'date': forecast_date.isoformat(),
                'day_name': self._get_day_name(day_offset),
                'valid_hours_count': valid_hours,
                'invalid_hours_count': invalid_hours,
                'pm10_mean': round(pm10_mean, 1) if pm10_mean is not None else None,
                'pm10_max': round(pm10_max, 1) if pm10_max is not None else None,
                'pm10_min': round(pm10_min, 1) if pm10_min is not None else None,
                'pm25_mean': round(pm25_mean, 1) if pm25_mean is not None else None,
                'temp_mean_c': round(temp_mean, 1) if temp_mean is not None else None,
                'temp_max_c': round(temp_max, 1) if temp_max is not None else None,
                'temp_min_c': round(temp_min, 1) if temp_min is not None else None,
                'temp_mean_f': round(temp_mean * 9/5 + 32, 0) if temp_mean is not None else None,
                'temp_max_f': round(temp_max * 9/5 + 32, 0) if temp_max is not None else None,
                'temp_min_f': round(temp_min * 9/5 + 32, 0) if temp_min is not None else None,
                'risk_level': risk_level.value,
                'aqi': round(aqi) if aqi is not None else None,
                'aqi_category': aqi_category,
                'aqi_color': aqi_color,
                'dust_probability_mean': round(prob_mean, 3) if prob_mean is not None else None,
                'frozen_hours': frozen_count,
                'physics_status': physics_status,
            })
        
        return daily_forecasts
    
    def _aggregate_roads_to_daily(self, road_forecasts: List[Dict], days: int) -> List[Dict]:
        """Aggregate road forecasts to daily summaries."""
        road_daily = defaultdict(lambda: defaultdict(list))
        
        for rf in road_forecasts:
            road_id = rf.get('road_id')
            horizon = rf.get('horizon_hours', 0)
            day_offset = horizon // 24
            
            if day_offset < days:
                road_daily[road_id][day_offset].append(rf)
        
        daily_road_forecasts = []
        
        for road_id, day_groups in road_daily.items():
            for day_offset, forecasts in day_groups.items():
                if not forecasts:
                    continue
                
                pm10_values = [f['pm10_predicted'] for f in forecasts]
                pm10_mean = np.mean(pm10_values)
                pm10_max = np.max(pm10_values)
                
                risk = self._determine_daily_risk(pm10_mean, pm10_max, 0.0)
                
                daily_road_forecasts.append({
                    'road_id': road_id,
                    'road_name': forecasts[0].get('road_name'),
                    'area': forecasts[0].get('area'),
                    'day_offset': day_offset,
                    'pm10_mean': round(pm10_mean, 1),
                    'pm10_max': round(pm10_max, 1),
                    'risk_level': risk.value,
                    'road_dust_factor': forecasts[0].get('road_dust_factor', 1.0),
                })
        
        return daily_road_forecasts
    
    def _get_day_name(self, day_offset: int) -> str:
        """Get day name from offset."""
        if day_offset == 0:
            return "Today"
        elif day_offset == 1:
            return "Tomorrow"
        else:
            future_date = datetime.now(self.nome_tz) + timedelta(days=day_offset)
            return future_date.strftime("%A")
    
    def _risk_from_pm10(self, pm10: float) -> RiskLevel:
        """Map PM10 to 6-level EPA AQI risk bins."""
        if pm10 <= 54:
            return RiskLevel.GREEN
        if pm10 <= 154:
            return RiskLevel.YELLOW
        if pm10 <= 254:
            return RiskLevel.ORANGE
        if pm10 <= 354:
            return RiskLevel.RED
        if pm10 <= 424:
            return RiskLevel.PURPLE
        return RiskLevel.MAROON

    def _bump_risk(self, risk: RiskLevel, steps: int) -> RiskLevel:
        """Increase risk by N steps without exceeding the maximum."""
        order = [
            RiskLevel.GREEN, RiskLevel.YELLOW, RiskLevel.ORANGE,
            RiskLevel.RED, RiskLevel.PURPLE, RiskLevel.MAROON
        ]
        idx = order.index(risk)
        return order[min(idx + steps, len(order) - 1)]

    def _determine_daily_risk(self, pm10_mean: float, pm10_max: float, prob_mean: float) -> RiskLevel:
        """Determine daily risk level using PM10 + probability signal."""
        effective_pm10 = (
            self.config.DAILY_PM10_MEAN_WEIGHT * pm10_mean
            + self.config.DAILY_PM10_MAX_WEIGHT * pm10_max
        )
        risk = self._risk_from_pm10(effective_pm10)

        prob_steps = 0
        if prob_mean >= self.config.PROB_RED_THRESHOLD:
            prob_steps = 2
        elif prob_mean >= self.config.PROB_YELLOW_THRESHOLD:
            prob_steps = 1

        if prob_steps > 0 and self.config.DAILY_PROB_WEIGHT > 0:
            scaled = int(round(prob_steps * (self.config.DAILY_PROB_WEIGHT / 0.1)))
            risk = self._bump_risk(risk, max(1, scaled))

        return risk

    def _calculate_aqi_from_breakpoints(
        self,
        concentration: float,
        breakpoints: List[Tuple[float, float, int, int, str, str]]
    ) -> Tuple[Optional[float], str, str]:
        """Calculate AQI, category, and color using breakpoint table."""
        if concentration is None or concentration < 0:
            return None, "Unknown", "GREEN"

        for (c_low, c_high, i_low, i_high, category, color) in breakpoints:
            if concentration <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return aqi, category, color

        return 500, "Hazardous", "MAROON"

    def _calculate_daily_aqi(
        self,
        pm25_mean: float,
        pm10_mean: float,
        pm25_is_derived: bool
    ) -> Tuple[Optional[float], str, str]:
        """Calculate AQI, category, and color from daily PM2.5 mean."""
        aqi25, cat25, col25 = self._calculate_aqi_from_breakpoints(
            pm25_mean, self.config.AQI_BREAKPOINTS_PM25
        )

        if pm25_is_derived and self.config.USE_PM10_AQI_WHEN_PM25_DERIVED:
            aqi10, cat10, col10 = self._calculate_aqi_from_breakpoints(
                pm10_mean, self.config.AQI_BREAKPOINTS_PM10
            )
            if aqi10 is not None and (aqi25 is None or aqi10 > aqi25):
                return aqi10, cat10, col10

        return aqi25, cat25, col25
    
    def _add_pm10_to_weather(self, weather_df: pd.DataFrame, current_pm10: float = None) -> pd.DataFrame:
        """Add PM10 column"""
        df = weather_df.copy()
        if current_pm10 is not None:
            df['PM10'] = current_pm10
        else:
            df['PM10'] = 20.0
        df['PM25'] = df['PM10'] * self.config.PM25_FROM_PM10_RATIO
        return df
    
    def _create_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features"""
        df = weather_df.rename(columns={'temp_c': 'AT', 'humidity': 'RH'})
        return create_features(df)
    
    def _generate_predictions(self, feature_df: pd.DataFrame, max_hours: int) -> List[Dict]:
        """Generate predictions with physics override"""
        predictions = []
        seen_horizons = set()
        
        now_nome = datetime.now(self.nome_tz).replace(minute=0, second=0, microsecond=0)
        
        for feat in self.classifier.feature_names + self.regressor.feature_names:
            if feat not in feature_df.columns:
                feature_df[feat] = 0
        
        for idx, row in feature_df.iterrows():
            try:
                ts = pd.Timestamp(idx)
                
                if ts.tzinfo is None:
                    ts = ts.tz_localize(self.config.NOME_TZ)
                
                delta = ts - now_nome
                horizon = round(delta.total_seconds() / 3600)
                
                if horizon < 0 or horizon > max_hours:
                    continue
                    
                if horizon in seen_horizons:
                    continue
                seen_horizons.add(horizon)
                
                temp_c = row.get('_raw_temp_c', None)
                if pd.isna(temp_c):
                    temp_c = None
                
                X_class = pd.DataFrame([row[self.classifier.feature_names]]).fillna(0)
                X_reg = pd.DataFrame([row[self.regressor.feature_names]]).fillna(0)
                
                ml_prob = float(self.classifier.predict_proba(X_class)[0])
                ml_pm10, ml_lower, ml_upper = self.regressor.predict(X_reg)
                ml_pm10, ml_lower, ml_upper = float(ml_pm10[0]), float(ml_lower[0]), float(ml_upper[0])
                
                adj_prob, adj_pm10, override_reason = apply_frozen_ground_override(
                    temp_c, ml_prob, ml_pm10, self.config
                )

                # HARD PHYSICS OVERRIDES (order matters)
                snow_cover = row.get('snow_cover', row.get('snow_cover_flag', 0))
                ignore_frozen = bool(row.get('ignore_frozen', False))
                is_frozen = temp_c is not None and temp_c < self.config.FROZEN_TEMP_THRESHOLD

                if is_frozen and not ignore_frozen:
                    adj_prob = 0.0
                    adj_pm10 = 0.0
                    override_reason = "frozen_ground"
                elif snow_cover == 1 and not ignore_frozen:
                    adj_prob = 0.0
                    adj_pm10 = 0.0
                    override_reason = "snow_covered"
                
                scale = adj_pm10 / ml_pm10 if ml_pm10 > 0 else (0 if override_reason == "frozen_ground" else 1)
                adj_lower = ml_lower * scale
                adj_upper = ml_upper * scale
                
                risk = self._determine_risk(adj_prob, adj_pm10, adj_upper)
                
                predictions.append({
                    'timestamp': ts.isoformat(),
                    'horizon_hours': horizon,
                    'temperature_c': round(float(temp_c), 1) if temp_c is not None else None,
                    'dust_probability': round(adj_prob, 3),
                    'dust_probability_raw': round(ml_prob, 3),
                    'pm10_predicted': round(adj_pm10, 1),
                    'pm10_predicted_raw': round(ml_pm10, 1),
                    'pm10_lower': round(adj_lower, 1),
                    'pm10_upper': round(adj_upper, 1),
                    'risk_level': risk.value,
                    'physics_override': override_reason,
                    'is_frozen': temp_c is not None and temp_c < self.config.FROZEN_TEMP_THRESHOLD,
                })
                
            except Exception as e:
                print(f"Prediction error at {idx}: {e}")
                continue
        
        predictions.sort(key=lambda x: x['horizon_hours'])
        return predictions
    
    def _determine_risk(self, dust_prob: float, pm10: float, pm10_upper: float) -> RiskLevel:
        """Determine risk level for hourly prediction using PM10 + probability."""
        effective = (
            self.config.HOURLY_PM10_WEIGHT * pm10
            + self.config.HOURLY_UPPER_WEIGHT * pm10_upper
        )
        risk = self._risk_from_pm10(effective)

        prob_steps = 0
        if dust_prob >= self.config.PROB_RED_THRESHOLD:
            prob_steps = 2
        elif dust_prob >= self.config.PROB_YELLOW_THRESHOLD:
            prob_steps = 1

        if prob_steps > 0 and self.config.HOURLY_PROB_WEIGHT > 0:
            scaled = int(round(prob_steps * (self.config.HOURLY_PROB_WEIGHT / 0.1)))
            risk = self._bump_risk(risk, max(1, scaled))

        return risk
    
    def _get_current_conditions(self, weather_df: pd.DataFrame, raw_temp: pd.Series) -> Dict:
        """Get current conditions"""
        now = datetime.now(self.nome_tz).replace(minute=0, second=0, microsecond=0)
        
        if len(weather_df) == 0:
            return {}
        
        try:
            closest_idx = abs(weather_df.index - now).argmin()
        except:
            closest_idx = 0
            
        current = weather_df.iloc[closest_idx]
        temp_c = raw_temp.iloc[closest_idx] if closest_idx < len(raw_temp) else None
        
        if temp_c is not None and not pd.isna(temp_c):
            if temp_c < self.config.FROZEN_TEMP_THRESHOLD:
                physics_status = "frozen_ground"
            elif temp_c < self.config.FROZEN_TRANSITION_HIGH:
                physics_status = "transition_zone"
            else:
                physics_status = "normal"
        else:
            physics_status = "unknown"
        
        return {
            'timestamp': str(weather_df.index[closest_idx]),
            'temperature_c': round(float(temp_c), 1) if pd.notna(temp_c) else None,
            'humidity_pct': round(float(current.get('humidity', 70)), 0),
            'wind_speed_mps': round(float(current.get('wind_speed', 0)), 1) if pd.notna(current.get('wind_speed')) else None,
            'is_frozen': temp_c is not None and temp_c < self.config.FROZEN_TEMP_THRESHOLD,
            'physics_status': physics_status,
        }
    
    def _apply_road_factors(self, predictions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply road-specific factors"""
        road_forecasts = []
        area_aggregates = defaultdict(list)
        
        for pred in predictions:
            base_pm10 = pred['pm10_predicted']
            base_prob = pred['dust_probability']
            horizon = pred['horizon_hours']
            is_frozen = pred.get('is_frozen', False)
            
            for road in self.road_classifier.get_all_roads():
                if is_frozen:
                    adj_pm10, adj_prob = 0.0, 0.0
                    pm10_lower, pm10_upper = 0.0, 0.0
                    risk = RiskLevel.GREEN
                else:
                    adj_pm10 = base_pm10 * road.dust_factor
                    adj_prob = min(1.0, base_prob * (0.5 + 0.5 * road.dust_factor))
                    pm10_lower = pred['pm10_lower'] * road.dust_factor * 0.8
                    pm10_upper = pred['pm10_upper'] * road.dust_factor * 1.2
                    risk = self._determine_risk(adj_prob, adj_pm10, pm10_upper)
                
                road_forecasts.append({
                    'road_id': road.road_id,
                    'road_name': road.name,
                    'area': road.area,
                    'horizon_hours': horizon,
                    'pm10_predicted': round(adj_pm10, 1),
                    'risk_level': risk.value,
                    'road_dust_factor': round(road.dust_factor, 2),
                })
                
                area_aggregates[(road.area, horizon)].append({
                    'pm10': adj_pm10,
                    'risk': risk.value,
                    'importance': road.importance_index
                })
        
        area_forecasts = []
        for (area, horizon), data in area_aggregates.items():
            pm10s = [r['pm10'] for r in data]
            imps = [r.get('importance', 0.5) for r in data]
            weighted = sum(p * i for p, i in zip(pm10s, imps)) / (sum(imps) or 1)
            
            risks = [r['risk'] for r in data]
            maroon = risks.count('MAROON')
            purple = risks.count('PURPLE')
            red = risks.count('RED')
            orange = risks.count('ORANGE')
            yellow = risks.count('YELLOW')

            if maroon > 0:
                area_risk = 'MAROON'
            elif purple > 0:
                area_risk = 'PURPLE'
            elif red > 0:
                area_risk = 'RED'
            elif orange > 0:
                area_risk = 'ORANGE'
            elif yellow > len(risks) * 0.3:
                area_risk = 'YELLOW'
            else:
                area_risk = 'GREEN'
            
            area_forecasts.append({
                'area_name': area,
                'horizon_hours': horizon,
                'pm10_mean': round(np.mean(pm10s), 1),
                'pm10_max': round(np.max(pm10s), 1),
                'pm10_weighted': round(weighted, 1),
                'risk_level': area_risk,
            })
        
        area_forecasts.sort(key=lambda x: (x['horizon_hours'], x['area_name']))
        return road_forecasts, area_forecasts


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Dust Forecast v3 (Daily)')
    parser.add_argument('--models', type=str, default='models')
    parser.add_argument('--roads', type=str, default='nome.geojson')
    parser.add_argument('--days', type=int, default=3, help='Number of days to forecast')
    parser.add_argument('--hours', type=int, default=None, help='Hours for hourly forecast')
    parser.add_argument('--pm10', type=float, default=None)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*70)
    print("NOME DUST FORECAST - INTEGRATED SYSTEM v3 (Daily Forecast)")
    print("="*70)
    
    forecast = IntegratedDustForecast()
    
    if Path(args.models).exists():
        forecast.load_models(args.models)
    else:
        print(f"ERROR: Models not found at {args.models}")
        return 1
    
    if Path(args.roads).exists():
        forecast.load_roads(args.roads)
    
    if args.hours:
        print(f"\nGenerating {args.hours}-hour HOURLY forecast...")
        result = forecast.generate_forecast(hours=args.hours, current_pm10=args.pm10)
        forecast_type = "hourly"
    else:
        print(f"\nGenerating {args.days}-day DAILY forecast...")
        result = forecast.generate_daily_forecast(days=args.days, current_pm10=args.pm10)
        forecast_type = "daily"
    
    if 'error' in result:
        print(f"ERROR: {result['error']}")
        return 1
    
    print("\n" + "="*70)
    print("FORECAST SUMMARY")
    print("="*70)
    
    current = result.get('current_conditions', {})
    print(f"\nCurrent conditions:")
    print(f"  Temperature: {current.get('temperature_c', 'N/A')}°C")
    print(f"  Humidity: {current.get('humidity_pct', 'N/A')}%")
    print(f"  Wind: {current.get('wind_speed_mps', 'N/A')} m/s")
    print(f"  Physics status: {current.get('physics_status', 'N/A')}")
    
    if current.get('is_frozen'):
        print(f"  ❄️  FROZEN GROUND - No dust emission possible")

    risk_emoji = {
        'GREEN': '🟢', 'YELLOW': '🟡', 'ORANGE': '🟠',
        'RED': '🔴', 'PURPLE': '🟣', 'MAROON': '🟤'
    }
    
    if forecast_type == "daily":
        print(f"\n3-DAY DAILY FORECAST:")
        print("-" * 70)
        for day in result.get('daily_forecasts', []):
            frozen = "❄️ " if day.get('physics_status') == 'frozen_ground' else ""
            temp_str = f"{day['temp_min_f']:.0f}°F - {day['temp_max_f']:.0f}°F" if day.get('temp_min_f') else "N/A"
            risk = day.get('risk_level', 'GREEN')
            emoji = risk_emoji.get(risk, '⚪')
            
            print(f"\n  {frozen}{day['day_name']} ({day['date']}):")
            print(f"    Risk Level: {emoji} {risk}")
            print(f"    PM10: {day['pm10_mean']} µg/m³ (range: {day['pm10_min']}-{day['pm10_max']})")
            print(f"    PM2.5: {day['pm25_mean']} µg/m³")
            print(f"    Temperature: {temp_str}")
            print(f"    AQI: {day['aqi']} ({day['aqi_category']}) - {day.get('aqi_color', 'GREEN')}")
            if day['dust_probability_mean']:
                print(f"    Dust Probability: {day['dust_probability_mean']:.1%}")
            print(f"    Valid Hours: {day['valid_hours_count']}/24")
            print(f"    Physics Status: {day['physics_status']}")
    else:
        print(f"\nHourly forecast (first 6 hours):")
        for pred in result.get('hourly_forecasts', [])[:6]:
            frozen = "❄️ " if pred.get('is_frozen') else ""
            override = f" [{pred['physics_override']}]" if pred['physics_override'] != 'normal' else ""
            risk = pred.get('risk_level', 'GREEN')
            emoji = risk_emoji.get(risk, '⚪')
            print(f"  {frozen}+{pred['horizon_hours']:2d}h: {emoji} {risk:6s} "
                  f"PM10={pred['pm10_predicted']:5.1f} prob={pred['dust_probability']:5.1%} "
                  f"T={pred.get('temperature_c', '?')}°C{override}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
    
    print("\n" + "="*70)
    return 0


if __name__ == "__main__":
    exit(main())
