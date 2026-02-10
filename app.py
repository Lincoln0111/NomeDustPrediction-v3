#!/usr/bin/env python3
"""
Nome Dust Risk Prediction API
=============================

Integrates:
- ML-based dust classifier and PM10 regressor (v2)
- Physics override for frozen ground
- Road-specific forecasts
- Real-time weather from Open-Meteo (no API key needed)

Endpoints:
- GET /health - Health check
- GET /weather/{location} - Current weather
- GET /nowcast/latest - Current dust conditions + road forecasts
- GET /forecast/24h - 24-hour forecast with uncertainty
- GET /forecast/{hours} - Custom horizon forecast
- POST /predict - Legacy prediction endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import os
import requests
from datetime import datetime, timezone, timedelta
import logging
from threading import Lock
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# LAZY-LOADED ML SYSTEM
# =============================================================================

_ML_LOCK = Lock()
_ML_FORECAST_SYSTEM = None


def _get_ml_forecast_system():
    """Lazy-load the ML forecast system."""
    global _ML_FORECAST_SYSTEM
    
    if _ML_FORECAST_SYSTEM is not None:
        return _ML_FORECAST_SYSTEM
    
    with _ML_LOCK:
        if _ML_FORECAST_SYSTEM is not None:
            return _ML_FORECAST_SYSTEM
        
        try:
            from nome_dust_integrated import IntegratedDustForecast
            from pathlib import Path
            
            system = IntegratedDustForecast()
            
            # Load models
            model_dir = Path("models")
            if model_dir.exists():
                system.load_models(str(model_dir))
                logger.info("ML models loaded successfully")
            else:
                logger.warning(f"Model directory not found: {model_dir}")
                raise FileNotFoundError("ML models not found. Train first with nome_dust_ml_model_v2.py")
            
            # Load roads
            roads_file = Path("nome.geojson")
            if roads_file.exists():
                system.load_roads(str(roads_file))
                logger.info("Roads loaded successfully")
            else:
                logger.warning("Roads file not found - road-specific forecasts disabled")
            
            _ML_FORECAST_SYSTEM = system
            return _ML_FORECAST_SYSTEM
            
        except Exception as e:
            logger.error(f"Failed to initialize ML system: {e}")
            raise


def _jsonify(obj):
    """Convert complex objects to JSON-serializable format."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if hasattr(obj, 'value') and not isinstance(obj, (str, int, float, bool)):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Nome Dust Risk Prediction API",
    description="ML-powered dust risk prediction for Nome, Alaska",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional OpenWeather key for legacy endpoint
OPENWEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '')

# AQI / PM conversion settings
PM25_FROM_PM10_RATIO = 0.15

# EPA AQI breakpoints (24-hour average)
AQI_BREAKPOINTS_PM25 = [
    (0, 12.0, 0, 50, "Good", "GREEN"),
    (12.1, 35.4, 51, 100, "Moderate", "YELLOW"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "ORANGE"),
    (55.5, 150.4, 151, 200, "Unhealthy", "RED"),
    (150.5, 250.4, 201, 300, "Very Unhealthy", "PURPLE"),
    (250.5, 500.4, 301, 500, "Hazardous", "MAROON"),
]

AQI_BREAKPOINTS_PM10 = [
    (0, 54, 0, 50, "Good", "GREEN"),
    (55, 154, 51, 100, "Moderate", "YELLOW"),
    (155, 254, 101, 150, "Unhealthy for Sensitive Groups", "ORANGE"),
    (255, 354, 151, 200, "Unhealthy", "RED"),
    (355, 424, 201, 300, "Very Unhealthy", "PURPLE"),
    (425, 604, 301, 500, "Hazardous", "MAROON"),
]


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    humidity: float = Field(50, ge=0, le=100)
    traffic_volume: str = Field("Medium")
    days_since_grading: int = Field(10, ge=0)
    days_since_suppressant: int = Field(100, ge=0)
    freeze_thaw_flag: int = Field(0, ge=0, le=1)
    snow_cover_flag: int = Field(0, ge=0, le=1)
    road_loose_flag: int = Field(1, ge=0, le=1)
    construction_activity: str = Field("Low")
    heavy_truck_activity: str = Field("Low")
    atv_activity: str = Field("Low")
    location: str = Field("Nome")


class ManualOverrideRequest(BaseModel):
    """Request model for manual parameter override."""
    traffic_volume: str = Field("Medium", description="Low, Medium, or High")
    days_since_grading: int = Field(7, ge=0, le=60)
    days_since_suppressant: int = Field(30, ge=0, le=120)
    atv_activity: str = Field("Low", description="Low, Medium, or High")
    snow_cover: int = Field(0, ge=0, le=1)
    freeze_thaw: int = Field(0, ge=0, le=1)
    road_loose: int = Field(1, ge=0, le=1)
    construction_activity: str = Field("Low")
    heavy_truck_activity: str = Field("Low")
    ignore_frozen: bool = Field(False, description="If True, ignore frozen ground physics override (for simulation)")


class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: bool
    ml_system_ready: bool
    timestamp: str


# =============================================================================
# WEATHER FETCHING (Open-Meteo - No API Key Required)
# =============================================================================

def get_weather_open_meteo(lat: float = 64.5011, lon: float = -165.4064) -> Dict:
    """Fetch weather from Open-Meteo (free, no API key)."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,precipitation',
            'timezone': 'America/Anchorage',
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        
        return {
            'tavg_c': current.get('temperature_2m', 0),
            'humidity': current.get('relative_humidity_2m', 50),
            'wind_speed_sustained': current.get('wind_speed_10m', 0) / 3.6,  # km/h to m/s
            'wind_gust': current.get('wind_gusts_10m', 0) / 3.6,
            'prcp_mm': current.get('precipitation', 0),
            'description': 'From Open-Meteo',
            'source': 'open-meteo'
        }
    except Exception as e:
        logger.error(f"Open-Meteo fetch failed: {e}")
        return _get_default_weather()


def _get_default_weather() -> Dict:
    """Default weather values."""
    return {
        'tavg_c': -10.0,
        'humidity': 60,
        'wind_speed_sustained': 3.0,
        'wind_gust': 5.0,
        'prcp_mm': 0.0,
        'description': 'Default values',
        'source': 'default'
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Serve the main application page."""
    from fastapi.responses import FileResponse
    return FileResponse("index.html", media_type="text/html")


@app.get("/api")
def api_info():
    """API info endpoint."""
    return {"message": "Nome Dust Risk Prediction API v2.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    ml_ready = False
    try:
        _get_ml_forecast_system()
        ml_ready = True
    except:
        pass
    
    return HealthResponse(
        status="ok",
        message="API is running",
        models_loaded=ml_ready,
        ml_system_ready=ml_ready,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/weather/{location}")
async def get_weather(location: str = "Nome"):
    """Get current weather conditions."""
    try:
        weather = get_weather_open_meteo()
        return {
            "success": True,
            "location": location,
            "weather": weather,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nowcast/latest")
async def nowcast_latest():
    """
    Return latest dust nowcast with ML predictions.
    
    Returns:
        - base: Current conditions (weather, PM10, risk level, physics status)
        - roads: Road-specific forecasts (if roads loaded)
    """
    try:
        system = _get_ml_forecast_system()
        
        # Generate forecast (hour 0 = current)
        result = system.generate_forecast(hours=6)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Extract current conditions (hour 0)
        current = result.get('current_conditions', {})
        hourly = result.get('hourly_forecasts', [])
        
        # Find hour 0 forecast
        hour0 = None
        for h in hourly:
            if h.get('horizon_hours', -1) == 0:
                hour0 = h
                break
        
        if hour0 is None and len(hourly) > 0:
            hour0 = hourly[0]
        
        # Build base response (matches frontend expectations)
        pm10_now = hour0.get('pm10_predicted', 0) if hour0 else 0
        pm25_now = pm10_now * PM25_FROM_PM10_RATIO
        aqi_now, aqi_category_now, aqi_color_now = _calculate_aqi_from_breakpoints(
            pm25_now, AQI_BREAKPOINTS_PM25
        )
        pm10_aqi, pm10_aqi_category, pm10_aqi_color = _calculate_aqi_from_breakpoints(
            pm10_now, AQI_BREAKPOINTS_PM10
        )

        base = {
            'timestamp': current.get('timestamp', datetime.utcnow().isoformat()),
            'temp_c': current.get('temperature_c'),
            'wind': current.get('wind_speed_mps'),
            'hum': current.get('humidity_pct'),
            'vis': None,  # Not available from Open-Meteo current
            'weather_text': current.get('physics_status', 'ML Forecast'),
            
            # PM and risk from ML
            'pm10': pm10_now,
            'pm25': pm25_now,
            'dust_prob': hour0.get('dust_probability', 0) if hour0 else 0,
            'severity': pm10_now,  # Use PM10 as severity proxy
            'severity_norm': pm10_now / 100 if hour0 else 0,
            
            # Risk level
            'risk': hour0.get('risk_level', 'GREEN') if hour0 else 'GREEN',
            'risk_level': hour0.get('risk_level', 'GREEN') if hour0 else 'GREEN',
            
            # AQI-style fields
            'aqi': round(aqi_now) if aqi_now is not None else 0,
            'aqi_category': aqi_category_now,
            'aqi_color': aqi_color_now,
            'pm10_aqi': round(pm10_aqi) if pm10_aqi is not None else 0,
            'pm10_aqi_category': pm10_aqi_category,
            'pm10_aqi_color': pm10_aqi_color,
            'dust_category': hour0.get('risk_level', 'GREEN') if hour0 else 'GREEN',
            'dust_label': _risk_to_label(hour0.get('risk_level', 'GREEN')) if hour0 else 'Good',
            
            # Physics override info
            'physics_override': hour0.get('physics_override', 'normal') if hour0 else 'normal',
            'is_frozen': hour0.get('is_frozen', False) if hour0 else False,
        }
        
        # Road forecasts
        roads = []
        for road in result.get('road_forecasts', []):
            if road.get('horizon_hours', -1) == 0:
                roads.append({
                    'road_id': road.get('road_id'),
                    'road_name': road.get('road_name'),
                    'area': road.get('area'),
                    'severity': road.get('pm10_predicted', 0),
                    'severity_norm': road.get('pm10_predicted', 0) / 100,
                    'risk_level': road.get('risk_level', 'GREEN'),
                    'dust_factor': road.get('road_dust_factor', 1.0),
                })
        
        return {
            "success": True,
            "base": _jsonify(base),
            "roads": _jsonify(roads),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Nowcast failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/24h")
async def forecast_24h():
    """Return 24-hour forecast with uncertainty quantification."""
    return await forecast_custom(24)


@app.get("/forecast/daily")
async def forecast_daily():
    """Return 3-day daily forecast."""
    return await forecast_daily_custom(3)


@app.get("/forecast/daily/{days}")
async def forecast_daily_custom(days: int):
    """
    Return N-day daily forecast.
    
    Aggregates hourly predictions into daily summaries following the logic
    used in Nome-Daily-Data.csv:
    - PM25: Daily mean of hourly values
    - valid_hours_count: Count of valid hours
    - AQI: Category based on daily mean PM25
    
    The frontend expects:
    - daily_forecasts[].day_offset (0=today, 1=tomorrow, etc.)
    - daily_forecasts[].day_name
    - daily_forecasts[].pm10_mean, pm10_max, pm10_min
    - daily_forecasts[].pm25_mean
    - daily_forecasts[].temp_min_f, temp_max_f
    - daily_forecasts[].risk_level
    - daily_forecasts[].aqi, aqi_category
    - daily_forecasts[].physics_status
    """
    if days < 1 or days > 7:
        raise HTTPException(status_code=400, detail="Days must be 1-7")
    
    result = None
    source = "unknown"
    
    # Try ML daily forecast
    try:
        system = _get_ml_forecast_system()
        result = system.generate_daily_forecast(days=days)
        source = "ml_model"
        if 'error' in result:
            logger.warning(f"ML daily forecast returned error: {result['error']}")
            result = None
    except Exception as ml_error:
        logger.warning(f"ML daily forecast unavailable: {ml_error}")
    
    # Try weather-based daily forecast
    if result is None:
        try:
            result = await _generate_weather_based_daily_forecast(days)
            source = "weather_based"
        except Exception as weather_error:
            logger.warning(f"Weather-based daily forecast failed: {weather_error}")
    
    # Fallback to static daily forecast
    if result is None or 'daily_forecasts' not in result:
        logger.info("Using static fallback daily forecast")
        return await _generate_fallback_daily_forecast(days)
    
    try:
        daily_forecasts = result.get('daily_forecasts', [])
        base = result.get('current_conditions', {})
        
        return {
            "success": True,
            "data": {
                "base": _jsonify(base),
                "daily_forecasts": _jsonify(daily_forecasts),
                "forecast_days": days,
                "generated_at": result.get('generated_at', datetime.utcnow().isoformat()),
                "total_days": len(daily_forecasts),
                "source": source,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Daily forecast processing failed: {e}", exc_info=True)
        return await _generate_fallback_daily_forecast(days)


@app.get("/forecast/{hours}")
async def forecast_custom(hours: int):
    """
    Return N-hour forecast with uncertainty.
    
    The frontend expects:
    - forecasts[].horizon_hours
    - forecasts[].pm10_p10, pm10_p50, pm10_p90
    - forecasts[].risk_p50
    - forecasts[].severity_p50
    """
    if hours < 1 or hours > 72:
        raise HTTPException(status_code=400, detail="Hours must be 1-72")
    
    result = None
    source = "unknown"
    
    # Try ML forecast first
    try:
        system = _get_ml_forecast_system()
        result = system.generate_forecast(hours=hours)
        source = "ml_model"
        if 'error' in result:
            logger.warning(f"ML forecast returned error: {result['error']}")
            result = None
    except Exception as ml_error:
        logger.warning(f"ML forecast unavailable: {ml_error}")
    
    # Try weather-based forecast
    if result is None:
        try:
            result = await _generate_weather_based_forecast(hours)
            source = "weather_based"
        except Exception as weather_error:
            logger.warning(f"Weather-based forecast failed: {weather_error}")
    
    # Fallback to static forecast
    if result is None or 'hourly_forecasts' not in result:
        logger.info("Using static fallback forecast")
        return await _generate_fallback_forecast(hours)
    
    try:
        # Transform to frontend format with deduplication
        forecasts_by_hour = {}
        
        for h in result.get('hourly_forecasts', []):
            horizon = h.get('horizon_hours', 0)
            
            if horizon not in forecasts_by_hour:
                forecasts_by_hour[horizon] = {
                    'horizon_hours': horizon,
                    'timestamp': h.get('timestamp'),
                    'pm10_p10': h.get('pm10_lower', 0),
                    'pm10_p50': h.get('pm10_predicted', 0),
                    'pm10_p90': h.get('pm10_upper', 0),
                    'risk_p50': h.get('risk_level', 'GREEN'),
                    'risk_p90': h.get('risk_level', 'GREEN'),
                    'severity_p10': h.get('pm10_lower', 0),
                    'severity_p50': h.get('pm10_predicted', 0),
                    'severity_p90': h.get('pm10_upper', 0),
                    'temperature_c': h.get('temperature_c'),
                    'dust_probability': h.get('dust_probability', 0),
                    'physics_override': h.get('physics_override', 'normal'),
                    'is_frozen': h.get('is_frozen', False),
                }
        
        # Convert to sorted list
        forecasts = [forecasts_by_hour[h] for h in sorted(forecasts_by_hour.keys())]
        
        # Fill missing hours
        existing_hours = set(forecasts_by_hour.keys())
        for h in range(hours + 1):
            if h not in existing_hours:
                forecasts.append({
                    'horizon_hours': h,
                    'timestamp': None,
                    'pm10_p10': 0,
                    'pm10_p50': 0,
                    'pm10_p90': 0,
                    'risk_p50': 'GREEN',
                    'risk_p90': 'GREEN',
                    'severity_p10': 0,
                    'severity_p50': 0,
                    'severity_p90': 0,
                    'temperature_c': None,
                    'dust_probability': 0,
                    'physics_override': 'unknown',
                    'is_frozen': False,
                })
        
        forecasts.sort(key=lambda x: x['horizon_hours'])
        forecasts = [f for f in forecasts if f['horizon_hours'] <= hours]
        
        base = result.get('current_conditions', {})
        
        return {
            "success": True,
            "data": {
                "base": _jsonify(base),
                "forecasts": _jsonify(forecasts),
                "horizon_hours": hours,
                "generated_at": result.get('generated_at', datetime.utcnow().isoformat()),
                "total_forecast_points": len(forecasts),
                "source": source,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Forecast processing failed: {e}", exc_info=True)
        return await _generate_fallback_forecast(hours)


async def _generate_weather_based_forecast(hours: int) -> dict:
    """Generate forecast based on weather data when ML is unavailable."""
    try:
        # Fetch weather from Open-Meteo
        weather = await _fetch_open_meteo_forecast(hours)
        
        hourly_forecasts = []
        now = datetime.utcnow()
        
        for h in range(hours + 1):
            forecast_time = now + timedelta(hours=h)
            
            # Get temperature for this hour
            temp_c = weather.get('hourly_temps', {}).get(h, -10)
            
            # Physics-based prediction
            if temp_c < -5:
                # Frozen ground
                pm10 = 0
                risk = "GREEN"
                physics = "frozen_ground"
                is_frozen = True
            elif temp_c < 2:
                # Partial thaw
                thaw_factor = (temp_c + 5) / 7
                pm10 = 30 * thaw_factor  # Base dust level scaled
                risk = _risk_from_pm10(pm10)
                physics = f"partial_thaw_{thaw_factor:.1f}"
                is_frozen = False
            else:
                # Normal conditions
                pm10 = 30  # Default moderate dust
                risk = _risk_from_pm10(pm10)
                physics = "normal"
                is_frozen = False
            
            hourly_forecasts.append({
                'horizon_hours': h,
                'timestamp': forecast_time.isoformat(),
                'temperature_c': temp_c,
                'pm10_predicted': pm10,
                'pm10_lower': pm10 * 0.5,
                'pm10_upper': pm10 * 1.5,
                'risk_level': risk,
                'dust_probability': min(1.0, pm10 / 100),
                'physics_override': physics,
                'is_frozen': is_frozen,
            })
        
        return {
            'hourly_forecasts': hourly_forecasts,
            'current_conditions': {
                'temp_c': weather.get('current_temp', -10),
                'wind': weather.get('current_wind', 3),
                'hum': weather.get('current_humidity', 65),
            },
            'source': 'weather_based',
            'generated_at': datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Weather-based forecast failed: {e}")
        return await _generate_fallback_forecast_data(hours)


async def _fetch_open_meteo_forecast(hours: int) -> dict:
    """Fetch weather forecast from Open-Meteo."""
    import requests
    
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': 64.5011,
            'longitude': -165.4064,
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m',
            'forecast_days': max(1, hours // 24 + 1),
            'timezone': 'America/Anchorage',
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get('hourly', {})
        temps = hourly.get('temperature_2m', [])
        humidity = hourly.get('relative_humidity_2m', [])
        wind = hourly.get('wind_speed_10m', [])
        
        return {
            'hourly_temps': {i: temps[i] if i < len(temps) else -10 for i in range(hours + 1)},
            'current_temp': temps[0] if temps else -10,
            'current_wind': (wind[0] / 3.6) if wind else 3,  # km/h to m/s
            'current_humidity': humidity[0] if humidity else 65,
        }
        
    except Exception as e:
        logger.warning(f"Open-Meteo fetch failed: {e}")
        return {
            'hourly_temps': {i: -10 for i in range(hours + 1)},
            'current_temp': -10,
            'current_wind': 3,
            'current_humidity': 65,
        }


async def _generate_fallback_forecast(hours: int) -> dict:
    """Generate a minimal valid forecast response."""
    forecasts = []
    now = datetime.utcnow()
    
    for h in range(hours + 1):
        forecasts.append({
            'horizon_hours': h,
            'timestamp': (now + timedelta(hours=h)).isoformat(),
            'pm10_p10': 0,
            'pm10_p50': 0,
            'pm10_p90': 0,
            'risk_p50': 'GREEN',
            'risk_p90': 'GREEN',
            'severity_p10': 0,
            'severity_p50': 0,
            'severity_p90': 0,
            'temperature_c': -10,  # Default cold temp
            'dust_probability': 0,
            'physics_override': 'frozen_ground',
            'is_frozen': True,
        })
    
    return {
        "success": True,
        "data": {
            "base": {},
            "forecasts": forecasts,
            "horizon_hours": hours,
            "generated_at": now.isoformat(),
            "total_forecast_points": len(forecasts),
            "source": "fallback",
        },
        "timestamp": now.isoformat(),
    }


async def _generate_fallback_forecast_data(hours: int) -> dict:
    """Generate fallback forecast data structure."""
    now = datetime.utcnow()
    
    hourly_forecasts = []
    for h in range(hours + 1):
        hourly_forecasts.append({
            'horizon_hours': h,
            'timestamp': (now + timedelta(hours=h)).isoformat(),
            'temperature_c': -10,
            'pm10_predicted': 0,
            'pm10_lower': 0,
            'pm10_upper': 0,
            'risk_level': 'GREEN',
            'dust_probability': 0,
            'physics_override': 'frozen_ground',
            'is_frozen': True,
        })
    
    return {
        'hourly_forecasts': hourly_forecasts,
        'current_conditions': {
            'temp_c': -10,
            'wind': 3,
            'hum': 65,
        },
        'source': 'fallback',
        'generated_at': now.isoformat(),
    }


async def _generate_weather_based_daily_forecast(days: int) -> dict:
    """Generate daily forecast based on weather data when ML is unavailable."""
    try:
        # Fetch weather from Open-Meteo for all days
        weather = await _fetch_open_meteo_forecast(days * 24)
        
        daily_forecasts = []
        now = datetime.utcnow()
        today = now.date()
        
        for d in range(days):
            forecast_date = today + timedelta(days=d)
            
            # Get temperatures for this day (hours d*24 to (d+1)*24)
            day_temps = []
            day_frozen_hours = 0
            
            for h in range(d * 24, (d + 1) * 24):
                temp_c = weather.get('hourly_temps', {}).get(h, -10)
                day_temps.append(temp_c)
                if temp_c < -5:
                    day_frozen_hours += 1
            
            temp_mean = np.mean(day_temps) if day_temps else -10
            temp_max = max(day_temps) if day_temps else -10
            temp_min = min(day_temps) if day_temps else -10
            
            # Physics-based prediction
            if temp_mean < -5:
                pm10_mean = 0
                risk = "GREEN"
                physics = "frozen_ground"
            elif temp_mean < 2:
                thaw_factor = (temp_mean + 5) / 7
                pm10_mean = 30 * thaw_factor
                risk = _risk_from_pm10(pm10_mean)
                physics = f"partial_thaw_{thaw_factor:.2f}"
            else:
                pm10_mean = 30
                risk = _risk_from_pm10(pm10_mean)
                physics = "normal"
            
            pm25_mean = pm10_mean * PM25_FROM_PM10_RATIO
            aqi, aqi_category, aqi_color = _calculate_aqi_category(pm25_mean)
            
            # Get day name
            if d == 0:
                day_name = "Today"
            elif d == 1:
                day_name = "Tomorrow"
            else:
                day_name = forecast_date.strftime("%A")
            
            daily_forecasts.append({
                'day_offset': d,
                'date': forecast_date.isoformat(),
                'day_name': day_name,
                'valid_hours_count': 24,
                'invalid_hours_count': 0,
                'pm10_mean': round(pm10_mean, 1),
                'pm10_max': round(pm10_mean * 1.3, 1),
                'pm10_min': round(pm10_mean * 0.7, 1),
                'pm25_mean': round(pm25_mean, 1),
                'temp_mean_c': round(temp_mean, 1),
                'temp_max_c': round(temp_max, 1),
                'temp_min_c': round(temp_min, 1),
                'temp_mean_f': round(temp_mean * 9/5 + 32, 0),
                'temp_max_f': round(temp_max * 9/5 + 32, 0),
                'temp_min_f': round(temp_min * 9/5 + 32, 0),
                'risk_level': risk,
                'aqi': aqi,
                'aqi_category': aqi_category,
                'aqi_color': aqi_color,
                'dust_probability_mean': min(1.0, pm10_mean / 100),
                'frozen_hours': day_frozen_hours,
                'physics_status': physics,
            })
        
        return {
            'daily_forecasts': daily_forecasts,
            'current_conditions': {
                'temp_c': weather.get('current_temp', -10),
                'wind': weather.get('current_wind', 3),
                'hum': weather.get('current_humidity', 65),
            },
            'source': 'weather_based',
            'generated_at': datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Weather-based daily forecast failed: {e}")
        return await _generate_fallback_daily_forecast_data(days)


async def _generate_fallback_daily_forecast(days: int) -> dict:
    """Generate a minimal valid daily forecast response."""
    now = datetime.utcnow()
    today = now.date()
    
    daily_forecasts = []
    for d in range(days):
        forecast_date = today + timedelta(days=d)
        
        if d == 0:
            day_name = "Today"
        elif d == 1:
            day_name = "Tomorrow"
        else:
            day_name = forecast_date.strftime("%A")
        
        daily_forecasts.append({
            'day_offset': d,
            'date': forecast_date.isoformat(),
            'day_name': day_name,
            'valid_hours_count': 24,
            'invalid_hours_count': 0,
            'pm10_mean': 0,
            'pm10_max': 0,
            'pm10_min': 0,
            'pm25_mean': 0,
            'temp_mean_c': -10,
            'temp_max_c': -8,
            'temp_min_c': -12,
            'temp_mean_f': 14,
            'temp_max_f': 18,
            'temp_min_f': 10,
            'risk_level': 'GREEN',
            'aqi': 0,
            'aqi_category': 'Good',
            'aqi_color': 'GREEN',
            'dust_probability_mean': 0,
            'frozen_hours': 24,
            'physics_status': 'frozen_ground',
        })
    
    return {
        "success": True,
        "data": {
            "base": {},
            "daily_forecasts": daily_forecasts,
            "forecast_days": days,
            "generated_at": now.isoformat(),
            "total_days": len(daily_forecasts),
            "source": "fallback",
        },
        "timestamp": now.isoformat(),
    }


async def _generate_fallback_daily_forecast_data(days: int) -> dict:
    """Generate fallback daily forecast data structure."""
    now = datetime.utcnow()
    today = now.date()
    
    daily_forecasts = []
    for d in range(days):
        forecast_date = today + timedelta(days=d)
        
        if d == 0:
            day_name = "Today"
        elif d == 1:
            day_name = "Tomorrow"
        else:
            day_name = forecast_date.strftime("%A")
        
        daily_forecasts.append({
            'day_offset': d,
            'date': forecast_date.isoformat(),
            'day_name': day_name,
            'valid_hours_count': 24,
            'invalid_hours_count': 0,
            'pm10_mean': 0,
            'pm10_max': 0,
            'pm10_min': 0,
            'pm25_mean': 0,
            'temp_mean_c': -10,
            'temp_max_c': -8,
            'temp_min_c': -12,
            'temp_mean_f': 14,
            'temp_max_f': 18,
            'temp_min_f': 10,
            'risk_level': 'GREEN',
            'aqi': 0,
            'aqi_category': 'Good',
            'aqi_color': 'GREEN',
            'dust_probability_mean': 0,
            'frozen_hours': 24,
            'physics_status': 'frozen_ground',
        })
    
    return {
        'daily_forecasts': daily_forecasts,
        'current_conditions': {
            'temp_c': -10,
            'wind': 3,
            'hum': 65,
        },
        'source': 'fallback',
        'generated_at': now.isoformat(),
    }


def _calculate_aqi_category(pm25_mean: float):
    """Calculate AQI, category, and color from PM2.5 mean."""
    if pm25_mean is None or pm25_mean < 0:
        return 0, "Unknown", "GREEN"

    aqi, category, color = _calculate_aqi_from_breakpoints(pm25_mean, AQI_BREAKPOINTS_PM25)
    return round(aqi) if aqi is not None else 0, category, color


@app.get("/aqi/{location}")
async def get_aqi(location: str = "Nome"):
    """
    Get AQI data - uses ML nowcast.
    """
    try:
        system = _get_ml_forecast_system()
        result = system.generate_forecast(hours=1)
        
        hourly = result.get('hourly_forecasts', [])
        hour0 = hourly[0] if hourly else {}
        
        pm10 = float(hour0.get('pm10_predicted', 0))
        pm25 = pm10 * PM25_FROM_PM10_RATIO
        
        # Convert PM2.5 to US AQI
        aqi, label, color = _calculate_aqi_from_breakpoints(pm25, AQI_BREAKPOINTS_PM25)
        description = _aqi_description(color)
        
        return _jsonify({
            "success": True,
            "location": location,
            "aqi": int(round(aqi)) if aqi is not None else 0,
            "label": label,
            "color": color,
            "description": description,
            "components": {
                "pm2_5": round(pm25, 1),
                "pm10": round(pm10, 1),
                "co": 0,
                "no2": 0,
                "o3": 0,
            },
            "physics_status": str(hour0.get('physics_override', 'normal')),
            "is_frozen": bool(hour0.get('is_frozen', False)),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"AQI fetch failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "aqi": 0,
            "label": "Unknown",
            "color": "#888888",
        }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Legacy prediction endpoint - now uses ML model.
    """
    try:
        system = _get_ml_forecast_system()
        result = system.generate_forecast(hours=1)
        
        hourly = result.get('hourly_forecasts', [])
        hour0 = hourly[0] if hourly else {}
        
        pm10 = float(hour0.get('pm10_predicted', 0))
        risk = str(hour0.get('risk_level', 'GREEN'))
        prob = float(hour0.get('dust_probability', 0))
        
        # Map risk to dust score (0-1 scale for frontend)
        if risk == 'GREEN':
            dust_score = min(0.25, pm10 / 200)
        elif risk == 'YELLOW':
            dust_score = 0.25 + min(0.30, (pm10 - 50) / 300)
        else:
            dust_score = 0.55 + min(0.45, (pm10 - 150) / 300)
        
        # Map to label
        label = 'Green' if risk == 'GREEN' else ('Yellow' if risk == 'YELLOW' else 'Red')
        
        weather = get_weather_open_meteo()
        
        return _jsonify({
            "success": True,
            "dust_score": round(float(dust_score), 3),
            "label": label,
            "confidence": round(float(prob), 3),
            "class_probabilities": {
                "Green": float(1.0 - prob) if risk == 'GREEN' else 0.1,
                "Yellow": float(prob) if risk == 'YELLOW' else 0.2,
                "Red": float(prob) if risk == 'RED' else 0.1,
            },
            "weather": {
                "temperature": round(float(weather['tavg_c']), 1),
                "wind_speed": round(float(weather['wind_speed_sustained']), 1),
                "wind_gust": round(float(weather['wind_gust']), 1),
                "precipitation": round(float(weather['prcp_mm']), 1),
                "snow": 0,
                "humidity": round(float(weather['humidity']), 1),
                "description": weather['description']
            },
            "inputs": {
                "humidity": request.humidity,
                "traffic_volume": request.traffic_volume,
            },
            "ml_prediction": {
                "pm10": round(float(pm10), 1),
                "risk_level": risk,
                "physics_override": str(hour0.get('physics_override', 'normal')),
                "is_frozen": bool(hour0.get('is_frozen', False)),
            }
        })
        
    except Exception as e:
        logger.error(f"Predict failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/manual")
async def predict_manual(request: ManualOverrideRequest):
    """
    Manual override prediction - uses user-specified parameters
    to calculate dust risk based on physics-based rules.
    
    This endpoint is used when the user adjusts Global Settings manually.
    """
    try:
        # Get current weather
        weather = get_weather_open_meteo()
        temp_c = weather.get('tavg_c', -10)
        humidity = weather.get('humidity', 50)
        wind_speed = weather.get('wind_speed_sustained', 3)
        precip = weather.get('prcp_mm', 0)
        
        # Physics override for frozen ground (can be disabled for simulation)
        if request.ignore_frozen:
            # Simulation mode - ignore frozen ground
            is_frozen = False
            physics_override = "simulation_mode"
            thaw_factor = 1.0
        else:
            is_frozen = temp_c < -5.0
            physics_override = "frozen_ground" if temp_c < -5.0 else (
                f"partial_thaw_{(temp_c + 5) / 7:.2f}" if temp_c < 2.0 else "normal"
            )
            thaw_factor = 0.0 if temp_c < -5.0 else (
                (temp_c + 5) / 7.0 if temp_c < 2.0 else 1.0
            )
            thaw_factor = max(0.0, min(1.0, thaw_factor))
        
        # Calculate base dust score from manual parameters
        base_score = _calculate_manual_dust_score(
            traffic_volume=request.traffic_volume,
            days_since_grading=request.days_since_grading,
            days_since_suppressant=request.days_since_suppressant,
            atv_activity=request.atv_activity,
            snow_cover=request.snow_cover,
            freeze_thaw=request.freeze_thaw,
            road_loose=request.road_loose,
            construction_activity=request.construction_activity,
            heavy_truck_activity=request.heavy_truck_activity,
            humidity=humidity,
            wind_speed=wind_speed,
            precip=precip
        )
        
        # Apply physics override (frozen ground suppression)
        adjusted_score = base_score * thaw_factor
        
        # Convert to PM10 estimate (scale: 0-1 score -> 0-300 µg/m³)
        pm10_estimate = adjusted_score * 300
        
        # Determine risk level based on PM10 (EPA PM10 breakpoints)
        risk_level = _risk_from_pm10(pm10_estimate)
        label = _risk_to_label(risk_level)
        pm10_aqi, pm10_aqi_category, pm10_aqi_color = _calculate_aqi_from_breakpoints(
            pm10_estimate, AQI_BREAKPOINTS_PM10
        )
        
        # Calculate road-specific adjustments
        road_forecasts = _calculate_road_forecasts(
            base_score=adjusted_score,
            risk_level=risk_level,
            is_frozen=is_frozen
        )
        
        return {
            "success": True,
            "mode": "manual_override",
            "base": {
                "timestamp": datetime.utcnow().isoformat(),
                "temp_c": round(temp_c, 1),
                "wind": round(wind_speed, 1),
                "hum": round(humidity, 0),
                "weather_text": f"Manual Override - {physics_override}",
                
                # Dust predictions
                "pm10": round(pm10_estimate, 1),
                "pm25": round(pm10_estimate * PM25_FROM_PM10_RATIO, 1),
                "dust_score": round(adjusted_score, 3),
                "dust_prob": round(adjusted_score, 3),
                "severity": round(adjusted_score * 100, 1),
                "severity_norm": round(adjusted_score, 3),
                
                # Risk
                "risk": risk_level,
                "risk_level": risk_level,
                "dust_category": risk_level,
                "dust_label": label,
                
                # AQI estimate
                "aqi": round(_pm25_to_aqi(pm10_estimate * PM25_FROM_PM10_RATIO)),
                "pm10_aqi": round(pm10_aqi) if pm10_aqi is not None else 0,
                "pm10_aqi_category": pm10_aqi_category,
                "pm10_aqi_color": pm10_aqi_color,
                
                # Physics
                "physics_override": physics_override,
                "is_frozen": is_frozen,
                "thaw_factor": round(thaw_factor, 2),
            },
            "roads": road_forecasts,
            "inputs": {
                "traffic_volume": request.traffic_volume,
                "days_since_grading": request.days_since_grading,
                "days_since_suppressant": request.days_since_suppressant,
                "atv_activity": request.atv_activity,
                "snow_cover": request.snow_cover,
                "freeze_thaw": request.freeze_thaw,
            },
            "weather": {
                "temperature": round(temp_c, 1),
                "wind_speed": round(wind_speed, 1),
                "humidity": round(humidity, 0),
                "precipitation": round(precip, 1),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Manual predict failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_manual_dust_score(
    traffic_volume: str,
    days_since_grading: int,
    days_since_suppressant: int,
    atv_activity: str,
    snow_cover: int,
    freeze_thaw: int,
    road_loose: int,
    construction_activity: str,
    heavy_truck_activity: str,
    humidity: float,
    wind_speed: float,
    precip: float
) -> float:
    """
    Calculate dust score from manual parameters using physics-based rules.
    Returns a score from 0.0 to 1.0.
    """
    score = 0.0
    
    # Traffic contribution (0-0.25)
    traffic_map = {"Low": 0.08, "Medium": 0.15, "High": 0.25}
    score += traffic_map.get(traffic_volume, 0.15)
    
    # Days since grading contribution (0-0.15)
    # More days = more dust (loose material accumulates)
    grading_factor = min(1.0, days_since_grading / 30.0)
    score += grading_factor * 0.15
    
    # Days since suppressant contribution (0-0.15)
    # More days = suppressant worn off
    suppressant_factor = min(1.0, days_since_suppressant / 60.0)
    score += suppressant_factor * 0.15
    
    # ATV activity (0-0.10)
    atv_map = {"Low": 0.02, "Medium": 0.05, "High": 0.10}
    score += atv_map.get(atv_activity, 0.02)
    
    # Construction activity (0-0.10)
    construction_map = {"Low": 0.02, "Medium": 0.05, "High": 0.10}
    score += construction_map.get(construction_activity, 0.02)
    
    # Heavy truck activity (0-0.10)
    truck_map = {"Low": 0.02, "Medium": 0.05, "High": 0.10}
    score += truck_map.get(heavy_truck_activity, 0.02)
    
    # Road surface condition
    if road_loose == 1:
        score *= 1.3  # Loose material increases dust
    
    # Snow cover suppression
    if snow_cover == 1:
        score *= 0.1  # Snow almost eliminates dust
    
    # Freeze-thaw cycles
    if freeze_thaw == 1:
        score *= 1.2  # Freeze-thaw loosens material
    
    # Weather modifiers
    # Humidity suppression (high humidity = less dust)
    if humidity > 80:
        score *= 0.5
    elif humidity > 60:
        score *= 0.7
    elif humidity < 40:
        score *= 1.2
    
    # Wind amplification (high wind = more dust)
    if wind_speed > 8:
        score *= 1.5
    elif wind_speed > 5:
        score *= 1.2
    elif wind_speed < 2:
        score *= 0.7
    
    # Precipitation suppression
    if precip > 2:
        score *= 0.2
    elif precip > 0.5:
        score *= 0.5
    
    # Clamp to 0-1 range
    return max(0.0, min(1.0, score))


def _calculate_road_forecasts(
    base_score: float,
    risk_level: str,
    is_frozen: bool
) -> List[Dict]:
    """
    Generate road-specific forecasts based on base score.
    Uses predefined road factors.
    """
    # Road categories with dust factors
    road_types = [
        {"road_id": "front_street", "road_name": "Front Street", "area": "Downtown", "dust_factor": 1.0},
        {"road_id": "bering_street", "road_name": "Bering Street", "area": "Downtown", "dust_factor": 0.9},
        {"road_id": "port_road", "road_name": "Port Road", "area": "Port", "dust_factor": 1.5},
        {"road_id": "snake_river_road", "road_name": "Snake River Road", "area": "Port", "dust_factor": 1.4},
        {"road_id": "airport_road", "road_name": "Airport Road", "area": "Airport", "dust_factor": 1.2},
        {"road_id": "seppala_drive", "road_name": "Seppala Drive", "area": "Airport", "dust_factor": 1.6},
        {"road_id": "greg_kruschek_ave", "road_name": "Greg Kruschek Avenue", "area": "Airport", "dust_factor": 1.6},
        {"road_id": "east_residential", "road_name": "East Residential", "area": "Residential", "dust_factor": 0.8},
        {"road_id": "west_residential", "road_name": "West Residential", "area": "Residential", "dust_factor": 0.8},
        {"road_id": "little_creek_road", "road_name": "Little Creek Road", "area": "West_Residential", "dust_factor": 1.6},
        {"road_id": "center_creek_road", "road_name": "Center Creek Road", "area": "West_Residential", "dust_factor": 1.6},
        {"road_id": "nome_council_road", "road_name": "Nome-Council Road", "area": "Highway", "dust_factor": 1.8},
        {"road_id": "nome_teller_road", "road_name": "Nome-Teller Road", "area": "Highway", "dust_factor": 1.7},
    ]
    
    roads = []
    for road in road_types:
        if is_frozen:
            adj_score = 0.0
            pm10_est = 0.0
            adj_risk = "GREEN"
        else:
            adj_score = min(1.0, base_score * road["dust_factor"])
            pm10_est = adj_score * 300  # Convert to PM10
            
            # Use EPA PM10 breakpoints for consistency
            adj_risk = _risk_from_pm10(pm10_est)
        
        roads.append({
            "road_id": road["road_id"],
            "road_name": road["road_name"],
            "area": road["area"],
            "severity": round(adj_score * 100, 1),
            "severity_norm": round(adj_score, 3),
            "pm10_estimate": round(pm10_est, 1),
            "risk_level": adj_risk,
            "dust_factor": road["dust_factor"],
        })
    
    return roads


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _risk_to_label(risk: str) -> str:
    """Convert risk level to human-readable label."""
    labels = {
        'GREEN': 'Good',
        'YELLOW': 'Moderate',
        'ORANGE': 'Unhealthy for Sensitive Groups',
        'RED': 'Unhealthy',
        'PURPLE': 'Very Unhealthy',
        'MAROON': 'Hazardous',
    }
    return labels.get(risk, 'Unknown')


def _risk_from_pm10(pm10: float) -> str:
    """Map PM10 to EPA 6-level risk scale."""
    if pm10 <= 54:
        return "GREEN"
    if pm10 <= 154:
        return "YELLOW"
    if pm10 <= 254:
        return "ORANGE"
    if pm10 <= 354:
        return "RED"
    if pm10 <= 424:
        return "PURPLE"
    return "MAROON"


def _calculate_aqi_from_breakpoints(concentration, breakpoints):
    """Calculate AQI, category, and color using breakpoint table."""
    if concentration is None or concentration < 0:
        return None, "Unknown", "GREEN"

    for (c_low, c_high, i_low, i_high, category, color) in breakpoints:
        if concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return aqi, category, color

    return 500, "Hazardous", "MAROON"


def _pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 to US AQI."""
    if pm25 <= 12.0:
        return (50 / 12.0) * pm25
    elif pm25 <= 35.4:
        return 51 + ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1)
    elif pm25 <= 55.4:
        return 101 + ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5)
    elif pm25 <= 150.4:
        return 151 + ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5)
    elif pm25 <= 250.4:
        return 201 + ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5)
    else:
        return 301 + ((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5)


def _aqi_category(aqi: float):
    """Get AQI category label, color, description."""
    if aqi <= 50:
        return "Good", "#00E400", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "#FFFF00", "Acceptable air quality."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00", "Sensitive groups may be affected."
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "Everyone may experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "Health alert for everyone."
    else:
        return "Hazardous", "#7E0023", "Emergency conditions."


def _aqi_description(color_name: str) -> str:
    descriptions = {
        "GREEN": "Air quality is satisfactory.",
        "YELLOW": "Acceptable air quality.",
        "ORANGE": "Sensitive groups may be affected.",
        "RED": "Everyone may experience health effects.",
        "PURPLE": "Health alert for everyone.",
        "MAROON": "Emergency conditions."
    }
    return descriptions.get(color_name, "Air quality is satisfactory.")


# =============================================================================
# STATIC FILES & STARTUP
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )


# Mount static files
try:
    app.mount("/", StaticFiles(directory=".", html=True), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
