# Nome Dust Risk Prediction System

A machine learning-powered forecasting system for predicting dust emissions from roads in Nome, Alaska.


## Overview

This system combines **machine learning models** trained on historical PM10 data with **physics-based rules** (frozen ground detection) to provide accurate, real-time dust predictions for Nome's road network.

### Key Features

- 🌡️ **Real-time weather integration** via Open-Meteo API (no API key required)
- 🤖 **ML-powered predictions** using LightGBM classifiers and regressors
- ❄️ **Frozen ground physics override** - automatically detects when dust emission is impossible
- 🗺️ **Interactive map** with road-specific risk levels
- 📊 **24-hour forecasts** with uncertainty quantification
- ⚙️ **Manual override mode** for what-if scenario analysis
- 🧪 **Simulation mode** to test predictions ignoring frozen ground

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [API Endpoints](#api-endpoints)
5. [Frontend Interface](#frontend-interface)
6. [ML Model Details](#ml-model-details)
7. [Physics Override](#physics-override)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Setup

```bash
# Clone or navigate to project directory
cd DustTool

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Files

Ensure these files are in your project directory:

| File | Description |
|------|-------------|
| `app.py` | FastAPI backend server |
| `index.html` | Frontend HTML interface |
| `map_script.js` | Frontend JavaScript logic |
| `nome.geojson` | Road network data |
| `nome_dust_ml_model_v2.py` | ML model training code |
| `nome_dust_integrated_v2.py` | Integrated forecast system |
| `models/` | Directory containing trained models |
| `NomeHourlyData.csv` | Historical PM10 data (for training) |

---

## Quick Start

### 1. Train Models (if not already trained)

```bash
python nome_dust_ml_model_v2.py --pm-data NomeHourlyData.csv
```

This creates the `models/` directory with:
- `classifier.joblib` - Dust event classifier
- `regressor.joblib` - PM10 concentration predictor

### 2. Start the Server

```bash
python app.py
```

Server starts at `http://localhost:8000`

### 3. Open the Interface

Navigate to `http://localhost:8000` in your browser.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Browser)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Control     │  │ Info Panel  │  │ Map (MapLibre GL)       │  │
│  │ Panel       │  │ - Weather   │  │ - Road colors           │  │
│  │ - Settings  │  │ - PM10      │  │ - Click popups          │  │
│  │ - Sliders   │  │ - Risk      │  │ - Heatmap view          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI - app.py)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ /nowcast    │  │ /forecast   │  │ /predict/manual         │  │
│  │ /weather    │  │ /aqi        │  │ /health                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              ML System (nome_dust_integrated_v2.py)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Classifier  │  │ Regressor   │  │ Physics Override        │  │
│  │ (LightGBM)  │  │ (LightGBM)  │  │ (Frozen Ground)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External APIs                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Open-Meteo (Weather Forecast) - No API Key Required     │    │
│  │ https://api.open-meteo.com/v1/forecast                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns system status and model availability.

### Weather

```http
GET /weather/{location}
```

Returns current weather conditions from Open-Meteo.

**Response:**
```json
{
  "success": true,
  "weather": {
    "tavg_c": -19.6,
    "humidity": 63,
    "wind_speed_sustained": 1.7,
    "description": "Clear"
  }
}
```

### Nowcast (Current Conditions)

```http
GET /nowcast/latest
```

Returns current dust prediction with physics override status.

**Response:**
```json
{
  "success": true,
  "base": {
    "timestamp": "2025-12-28T21:00:00",
    "temp_c": -19.6,
    "wind": 1.7,
    "hum": 63,
    "pm10": 0.0,
    "dust_prob": 0.0,
    "risk_level": "GREEN",
    "physics_override": "frozen_ground",
    "is_frozen": true
  },
  "roads": [...]
}
```

### 24-Hour Forecast

```http
GET /forecast/24h
GET /forecast/{hours}
```

Returns hourly forecasts with uncertainty bounds.

**Response:**
```json
{
  "success": true,
  "data": {
    "forecasts": [
      {
        "horizon_hours": 0,
        "temperature_c": -19.6,
        "pm10_p10": 0.0,
        "pm10_p50": 0.0,
        "pm10_p90": 0.0,
        "risk_p50": "GREEN",
        "physics_override": "frozen_ground",
        "is_frozen": true
      }
    ]
  }
}
```

### Manual Override Prediction

```http
POST /predict/manual
```

Calculate predictions using user-specified parameters.

**Request Body:**
```json
{
  "traffic_volume": "High",
  "days_since_grading": 60,
  "days_since_suppressant": 120,
  "atv_activity": "High",
  "snow_cover": 0,
  "ignore_frozen": false
}
```

**Response:** Same format as nowcast with `mode: "manual_override"`.

---

## Frontend Interface

### Panels

| Panel | Location | Description |
|-------|----------|-------------|
| **Global Settings** | Top-left | Adjust parameters for manual override |
| **Info Panel** | Top-right | Shows current conditions, PM10, risk level |
| **Map** | Center | Interactive road map with color-coded risk |
| **Legend** | Bottom-right | Map view toggles and risk level key |
| **Forecast** | Bottom | 24-hour forecast table and chart |

### Global Settings Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Traffic Volume | Low/Medium/High | More traffic = more dust |
| Days Since Grading | 0-60 | Fresh grading = less dust |
| Days Since Suppressant | 0-120 | Recent suppressant = less dust |
| ATV Activity | Low/Medium/High | ATVs disturb road surface |
| Snow Cover | Yes/No | Snow suppresses all dust |
| 🧪 Simulation Mode | On/Off | Ignore frozen ground physics |

### Map Color Coding

| Color | Risk Level | Dust Score |
|-------|------------|------------|
| 🟢 Green | Low | ≤ 0.25 |
| 🟡 Yellow | Moderate | 0.25 - 0.55 |
| 🔴 Red | High | > 0.55 |

### Road Popup Information

When clicking on a road:

```
Nome-Council Road
─────────────────────
Surface: unpaved
Type: secondary
─────────────────────
Dust Score: 0.000
Risk Level: Green
Source: Nowcast
Confidence: 33%
Base Score: 0.000
─────────────────────
❄️ Frozen Ground - No Dust
─────────────────────
🤖 ML Prediction
```

### 24-Hour Forecast Table

| Column | Description |
|--------|-------------|
| Hour | Hours ahead (+0h to +23h) |
| Time | Actual local time (Nome timezone) |
| Temp | Forecasted temperature from Open-Meteo |
| PM10 | Predicted PM10 concentration (µg/m³) |
| Range | Uncertainty range (p10-p90) |
| Risk | Risk badge (Low/Moderate/High) |
| Status | Physics status (❄️ Frozen / 🌡️ Thawing / ✓ Normal) |

### Forecast Summary Stats

| Stat | Description |
|------|-------------|
| **PM10 Range** | Min-Max PM10 across all forecast hours |
| **Avg Severity** | Average dust severity score (0-100) |
| **Risk Hours** | Count of hours in each risk category |
| **Status** | Overall forecast status (❄️ All Frozen / ✓ Active) |

---

## ML Model Details

### Training Data

- **Source:** Nome Hourly PM10 measurements
- **Records:** ~13,000+ hours
- **Features:** 47 engineered features

### Model Architecture

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Classifier | LightGBM (Binary) | Detect dust events (Yes/No) |
| Regressor | LightGBM (Quantile) | Predict PM10 concentration |

### Key Features Used

1. **Temporal:** Hour, day of week, month, season
2. **Lagged PM10:** 1h, 3h, 6h, 12h, 24h previous values
3. **Rolling Stats:** Mean, max, std over 6h/24h windows
4. **Weather:** Temperature, humidity, wind speed
5. **Interactions:** Wind × dry conditions, etc.

### Performance Metrics

| Metric | Value |
|--------|-------|
| Classifier AUC | 0.958 |
| Classifier Precision | 80.9% |
| Classifier Recall | 87.1% |
| Regressor MAE | 27.3 µg/m³ |
| 80% Coverage | 79.6% |

---

## Physics Override

### Frozen Ground Logic

The system applies physics-based rules that override ML predictions:

| Temperature | Status | Dust Factor | Reason |
|-------------|--------|-------------|--------|
| < -5°C | Frozen | 0.0 (none) | Ground frozen solid |
| -5°C to 2°C | Partial Thaw | 0.0 - 1.0 | Gradual transition |
| > 2°C | Normal | 1.0 (full ML) | Ground can emit dust |

### Why This Matters

At temperatures below -5°C:
- Soil moisture is **frozen solid**
- Particles are **bound by ice crystals**
- No mechanical action (traffic, wind) can create airborne dust
- This is **fundamental physics**, not a model limitation

### Simulation Mode

Enable "🧪 Simulation Mode" in the Global Settings to:
- Bypass frozen ground physics
- Test what-if scenarios
- Plan for future conditions
- Train operators during winter

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Server host |

### Physics Thresholds

In `nome_dust_integrated_v2.py`:

```python
FROZEN_TEMP_THRESHOLD = -5.0    # Below this = no dust
FROZEN_TRANSITION_HIGH = 2.0    # Above this = full ML
PM10_YELLOW_THRESHOLD = 50.0    # µg/m³ for Yellow risk
PM10_RED_THRESHOLD = 150.0      # µg/m³ for Red risk
```

---

## Troubleshooting

### Common Issues

#### 1. "Models not loaded" error

```bash
# Train models first
python nome_dust_ml_model_v2.py --pm-data NomeHourlyData.csv

# Verify models exist
ls models/
# Expected: classifier.joblib, regressor.joblib
```

#### 2. All roads show Green but expected dust

Check if ground is frozen:
- Look for "❄️ Frozen Ground" in Info Panel
- Temperature below -5°C = no dust possible
- Enable "🧪 Simulation Mode" to test without physics

#### 3. Road popup shows Yellow but panel shows Green

Update to latest `map_script.js` - older versions didn't apply frozen ground override to all roads.

#### 4. Duplicate hours in forecast table

Update to latest `app.py` and `map_script.js` which include deduplication logic.

#### 5. Weather data not updating

Open-Meteo API may have temporary issues. System uses cached/default values as fallback.

### Debug Commands

Browser console:
```javascript
// Check current data
console.log('Nowcast:', nowcastData);
console.log('Frozen:', nowcastData?.base?.is_frozen);
console.log('Override:', nowcastData?.base?.physics_override);

// Check settings
console.log('Manual mode:', manualOverrideActive);
console.log('Settings:', globalSettings);
```

---

## File Structure

```
DustTool/
├── app.py                          # FastAPI backend server
├── index.html                      # Frontend HTML
├── map_script.js                   # Frontend JavaScript
├── nome.geojson                    # Road network GeoJSON
├── nome_dust_ml_model_v2.py        # ML model training
├── nome_dust_integrated_v2.py      # Integrated forecast system
├── nome_dust_forecast_production.py # Road classification
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── models/                         # Trained model files
│   ├── classifier.joblib
│   └── regressor.joblib
├── NomeHourlyData.csv              # Historical PM10 data
└── NomeDailyData.csv               # Daily aggregated data
```

---

## Dependencies

```txt
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0
requests>=2.31.0
shapely>=2.0.0
pyproj>=3.6.0
rtree>=1.0.0
```

---

## Changelog

### v2.1 (December 2025)
- ✅ Fixed frozen ground physics override for ALL roads
- ✅ Fixed duplicate hours in forecast table
- ✅ Consolidated info panel (merged weather + AQI)
- ✅ Added simulation mode toggle
- ✅ Improved road popup with frozen status indicator
- ✅ Added deduplication to forecast API and frontend

### v2.0 (December 2025)
- Complete ML model rewrite with LightGBM
- Added uncertainty quantification (p10/p50/p90)
- Integrated Open-Meteo weather API (no key required)
- Added 24-hour forecast table view
- Added manual override mode

### v1.0 (Initial Release)
- Basic dust prediction system
- Historical data analysis
- Simple risk classification

---

## License

MIT License - See LICENSE file for details.

---

## Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open a GitHub issue with:
   - Browser console errors
   - Screenshot of the issue
   - Steps to reproduce
