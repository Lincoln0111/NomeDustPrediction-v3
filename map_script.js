
const API_URL = 'http://localhost:8000/predict';
const WEATHER_URL = 'http://localhost:8000/weather/Nome';
const AQI_URL = 'http://localhost:8000/aqi/Nome';
const NOWCAST_URL = 'http://localhost:8000/nowcast/latest';
const MANUAL_OVERRIDE_URL = 'http://localhost:8000/predict/manual';
const FORECAST_URL = 'http://localhost:8000/forecast/daily';
const MAPTILER_KEY = "f5zBB8Hb9R2KXchIgAOK";
const CITY_CENTER = [-165.4064, 64.5011];

const DUST_THRESHOLDS = {
    // Dust_score thresholds (0-1 scale), PM10 ≈ score * 300
    // EPA PM10 breakpoints: 54, 154, 254, 354, 424 µg/m³
    GREEN_MAX: 54 / 300,
    YELLOW_MAX: 154 / 300,
    ORANGE_MAX: 254 / 300,
    RED_MAX: 354 / 300,
    PURPLE_MAX: 424 / 300
};

let map = null;
let originalRoads = null;
let weatherData = { humidity: 50, tavg_c: -7.5, wind_speed_sustained: 5.0 };
let aqiData = null;
let nowcastData = null;
let forecastData = null;

// Global settings with default values
let globalSettings = {
    trafficVolume: 'Medium',
    daysSinceGrading: 7,
    daysSinceSuppressant: 30,
    atvActivity: 'Low',
    snowCover: 0,
    freezeThaw: 0,
    roadLoose: 1,
    construction: 'Low',
    heavyTruck: 'Low',
    ignoreFrozen: false  // Simulation mode - ignore frozen ground physics
};

// Track if user has manually adjusted settings
let manualOverrideActive = false;
let settingsModified = false;

let updateTimeout = null;
let isUpdating = false;
let heatmapVisible = false;

// ============================================================================
// DATETIME DISPLAY
// ============================================================================

function updateDateTime() {
    try {
        const now = new Date();
        const dateStr = now.toLocaleDateString('en-US', {
            timeZone: 'America/Nome',
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        const timeStr = now.toLocaleTimeString('en-US', {
            timeZone: 'America/Nome',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        document.getElementById('local-date').textContent = dateStr;
        document.getElementById('local-time').textContent = timeStr;
    } catch (e) {
        console.error('DateTime error:', e);
    }
}

// ============================================================================
// WEATHER FETCHING
// ============================================================================

async function fetchWeather() {
    try {
        const res = await fetch(WEATHER_URL);
        const data = await res.json();
        
        if (data.success && data.weather) {
            weatherData = data.weather;
            
            // Update individual weather fields
            const tempEl = document.getElementById('temp-value');
            const windEl = document.getElementById('wind-value');
            const humEl = document.getElementById('humidity-value');
            
            const temp = weatherData.tavg_c ?? weatherData.temp_c ?? null;
            const wind = weatherData.wind_speed_sustained ?? weatherData.wind ?? null;
            const hum = weatherData.humidity ?? weatherData.hum ?? null;
            
            if (tempEl && temp !== null) {
                const tempF = (temp * 9/5) + 32;
                tempEl.textContent = `${tempF.toFixed(0)}°F`;
            }
            if (windEl && wind !== null) {
                const windMph = wind * 2.237; // m/s to mph
                windEl.textContent = `${windMph.toFixed(0)} mph`;
            }
            if (humEl && hum !== null) humEl.textContent = `${hum.toFixed(0)}%`;
        }
    } catch (e) {
        console.error('Weather fetch error:', e);
    }
}

async function fetchAQI() {
    // AQI is now handled by nowcast - this function is kept for compatibility
    // but does not update the UI directly
    try {
        const res = await fetch(AQI_URL);
        const data = await res.json();
        if (data.success) {
            aqiData = data;
        }
    } catch (e) {
        console.error('AQI fetch error:', e);
    }
}

// ============================================================================
// NOWCAST FETCHING (Real-time ML predictions)
// ============================================================================

async function fetchNowcast() {
    // Nowcast disabled in favor of daily forecast + live weather
}

// Nowcast summary is now integrated into the main info panel
// This function is kept for compatibility but does nothing
function updateNowcastSummary(text) {
    // No longer needed - info is shown in the consolidated panel
    console.log('Nowcast:', text);
}

function updateFromDailyForecast(day0) {
    if (!day0) return;
    const risk = normalizeRiskLabel(day0.risk_level || 'GREEN');
    const aqiColor = String(day0.aqi_color || '').toUpperCase();
    const pm10 = typeof day0.pm10_mean === 'number' ? day0.pm10_mean : 0;
    const pm25 = typeof day0.pm25_mean === 'number' ? day0.pm25_mean : 0;
    const prob = typeof day0.dust_probability_mean === 'number' ? day0.dust_probability_mean : 0;
    const physicsStatus = day0.physics_status || 'normal';
    const frozenHours = day0.frozen_hours || 0;

    const pm10El = document.getElementById('pm10-value');
    if (pm10El) {
        pm10El.textContent = pm10.toFixed(1);
        pm10El.style.color = getAqiColorByName(aqiColor) || getRiskColor(risk);
    }

    const riskBadge = document.getElementById('risk-badge');
    if (riskBadge) {
        riskBadge.textContent = risk;
        riskBadge.className = 'risk-badge ' + risk.toLowerCase();
    }

    const physicsStatusEl = document.getElementById('physics-status');
    if (physicsStatusEl) {
        if (physicsStatus === 'frozen_ground' || frozenHours === 24) {
            physicsStatusEl.textContent = '❄️ Frozen Ground';
            physicsStatusEl.className = 'status-indicator frozen';
        } else if (physicsStatus.includes('partial_freeze') || frozenHours > 0 || physicsStatus.includes('partial_thaw')) {
            physicsStatusEl.textContent = '🌡️ Partial Thaw';
            physicsStatusEl.className = 'status-indicator simulation';
        } else {
            physicsStatusEl.textContent = '✓ Normal';
            physicsStatusEl.className = 'status-indicator normal';
        }
    }

    const pm25El = document.getElementById('pm25-value');
    const dustProbEl = document.getElementById('dust-prob-value');
    const dataSourceEl = document.getElementById('data-source');

    if (pm25El) pm25El.textContent = `${pm25.toFixed(1)} µg/m³`;
    if (dustProbEl) dustProbEl.textContent = `${(prob * 100).toFixed(0)}%`;
    if (dataSourceEl) dataSourceEl.textContent = 'Forecast';
}

// ============================================================================
// MANUAL OVERRIDE PREDICTION
// ============================================================================

async function fetchManualOverridePrediction() {
    try {
        const requestBody = {
            traffic_volume: globalSettings.trafficVolume,
            days_since_grading: globalSettings.daysSinceGrading,
            days_since_suppressant: globalSettings.daysSinceSuppressant,
            atv_activity: globalSettings.atvActivity,
            snow_cover: globalSettings.snowCover,
            freeze_thaw: globalSettings.freezeThaw,
            road_loose: globalSettings.roadLoose,
            construction_activity: globalSettings.construction,
            heavy_truck_activity: globalSettings.heavyTruck,
            ignore_frozen: globalSettings.ignoreFrozen  // Pass simulation mode flag
        };
        
        const res = await fetch(MANUAL_OVERRIDE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        if (!res.ok) {
            console.error('Manual override API error:', res.status);
            return null;
        }
        
        const data = await res.json();
        
        if (data.success) {
            // Update displays with manual override data
            updateFromManualOverride(data);
            return data;
        } else {
            console.error('Manual override failed:', data);
            return null;
        }
    } catch (e) {
        console.error('Manual override error:', e);
        return null;
    }
}

function updateFromManualOverride(data) {
    const base = data.base || {};
    const risk = normalizeRiskLabel(base.risk_level || base.risk);
    const pm10AqiColor = String(base.pm10_aqi_color || '').toUpperCase();
    const dustScore = base.dust_score || 0;
    const pm10 = base.pm10 || 0;
    const pm25 = base.pm25 || 0;
    const isFrozen = base.is_frozen || false;
    const physicsOverride = base.physics_override || 'normal';
    const thawFactor = base.thaw_factor || 1.0;
    const isSimulation = physicsOverride === 'simulation_mode';
    
    // Update PM10 display
    const pm10El = document.getElementById('pm10-value');
    if (pm10El) {
        pm10El.textContent = pm10.toFixed(1);
        const colorFromAqi = getAqiColorByName(pm10AqiColor);
        const fallbackAqiColor = getAqiColorByName(getAqiColorNameForPM10(pm10));
        pm10El.style.color = colorFromAqi || fallbackAqiColor || getRiskColor(risk);
    }
    
    // Update risk badge
    const riskBadge = document.getElementById('risk-badge');
    if (riskBadge) {
        riskBadge.textContent = risk;
        riskBadge.className = 'risk-badge ' + risk.toLowerCase();
    }
    
    // Update physics status indicator
    const physicsStatus = document.getElementById('physics-status');
    if (physicsStatus) {
        if (isSimulation) {
            physicsStatus.textContent = '🧪 Simulation';
            physicsStatus.className = 'status-indicator simulation';
        } else if (isFrozen) {
            physicsStatus.textContent = '❄️ Frozen Ground';
            physicsStatus.className = 'status-indicator frozen';
        } else if (physicsOverride.includes('partial_thaw')) {
            physicsStatus.textContent = `🌡️ Thaw ${(thawFactor * 100).toFixed(0)}%`;
            physicsStatus.className = 'status-indicator simulation';
        } else {
            physicsStatus.textContent = '✓ Normal';
            physicsStatus.className = 'status-indicator normal';
        }
    }
    
    // Update weather values from response
    const weather = data.weather || {};
    const tempEl = document.getElementById('temp-value');
    const windEl = document.getElementById('wind-value');
    const humEl = document.getElementById('humidity-value');
    
    if (tempEl && weather.temperature !== undefined) {
        const tempF = (weather.temperature * 9/5) + 32;
        tempEl.textContent = `${tempF.toFixed(0)}°F`;
    }
    if (windEl && weather.wind_speed !== undefined) {
        const windMph = weather.wind_speed * 2.237; // m/s to mph
        windEl.textContent = `${windMph.toFixed(0)} mph`;
    }
    if (humEl && weather.humidity !== undefined) {
        humEl.textContent = `${weather.humidity.toFixed(0)}%`;
    }
    
    // Update air quality details
    const pm25El = document.getElementById('pm25-value');
    const dustProbEl = document.getElementById('dust-prob-value');
    const dataSourceEl = document.getElementById('data-source');
    
    if (pm25El) pm25El.textContent = `${pm25.toFixed(1)} µg/m³`;
    if (dustProbEl) dustProbEl.textContent = `${(dustScore * 100).toFixed(0)}%`;
    if (dataSourceEl) {
        dataSourceEl.textContent = isSimulation ? '🧪 Simulation' : '⚙️ Manual Override';
    }
    
    // Store roads data for map update
    if (data.roads) {
        nowcastData = { ...nowcastData, roads: data.roads, base: base };
    }
}

// ============================================================================
// ML PREDICTION (Legacy)
// ============================================================================

async function predictDustML(trafficVolume, context) {
    try {
        const requestBody = {
            humidity: weatherData.humidity || 50,
            traffic_volume: trafficVolume,
            days_since_grading: context.days_since_grading || 7,
            days_since_suppressant: context.days_since_suppressant || 30,
            freeze_thaw_flag: context.freeze_thaw_flag || 0,
            snow_cover_flag: context.snow_cover_flag || 0,
            road_loose_flag: context.road_loose_flag || 1,
            construction_activity: context.construction_activity || 'Low',
            heavy_truck_activity: context.heavy_truck_activity || 'Low',
            atv_activity: context.atv_activity || 'Low',
            location: 'Nome'
        };
        
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        
        if (!res.ok) {
            throw new Error(`API returned status ${res.status}`);
        }
        
        const data = await res.json();
        
        if (data.success) {
            return {
                dustScore: data.dust_score,
                label: data.label,
                confidence: data.confidence,
                probabilities: data.class_probabilities
            };
        } else {
            throw new Error('Prediction failed');
        }
    } catch (e) {
        console.error('ML Prediction error:', e);
        return {
            dustScore: 0.15,
            label: 'Green',
            confidence: 0.33,
            probabilities: { 'Green': 0.33, 'Yellow': 0.33, 'Red': 0.33 }
        };
    }
}

// ============================================================================
// ROAD UTILITIES
// ============================================================================

// Mayor's ground-truth corrections: roads confirmed as high-dust
const MAYOR_DUSTY_ROADS = new Set([
    'Greg Kruschek Avenue',
    'Nome-Council Road',
    'Little Creek Road',
    'Center Creek Road',
    'Seppala Drive',
]);

function applyStreetMultiplier(score, props) {
    // Mayor's confirmed dusty roads get a fixed high multiplier
    const name = props.name || '';
    if (MAYOR_DUSTY_ROADS.has(name)) {
        return Math.min(1.0, score * 2.0);
    }

    let multiplier = 1.0;

    const surface = (props.surface || '').toLowerCase();
    if (surface.includes('unpaved') || surface.includes('gravel') || surface.includes('dirt')) {
        multiplier *= 2.0;
    } else if (surface.includes('asphalt') || surface.includes('paved')) {
        multiplier *= 0.65;
    }

    const highway = (props.highway || '').toLowerCase();
    if (highway.includes('primary') || highway.includes('trunk')) {
        multiplier *= 1.5;
    } else if (highway.includes('secondary')) {
        multiplier *= 1.3;
    } else if (highway.includes('residential') || highway.includes('service')) {
        multiplier *= 0.8;
    }

    return Math.min(1.0, Math.max(0.0, score * multiplier));
}

function classifyRisk(score) {
    // score is in 0-1 range (dust_score)
    // Convert to PM10 for classification: score * 300
    const pm10 = score * 300;
    return classifyRiskByPM10(pm10);
}

function classifyRiskByPM10(pm10) {
    // EPA PM10 breakpoints
    if (pm10 <= 54) return 'Green';
    if (pm10 <= 154) return 'Yellow';
    if (pm10 <= 254) return 'Orange';
    if (pm10 <= 354) return 'Red';
    if (pm10 <= 424) return 'Purple';
    return 'Maroon';
}

function getRiskColor(label) {
    const colors = {
        'Green': '#00E400',
        'GREEN': '#00E400',
        'Good': '#00E400',
        'Yellow': '#FFFF00',
        'YELLOW': '#FFFF00',
        'Moderate': '#FFFF00',
        'Orange': '#FF7E00',
        'ORANGE': '#FF7E00',
        'Unhealthy for Sensitive Groups': '#FF7E00',
        'Red': '#FF0000',
        'RED': '#FF0000',
        'Unhealthy': '#FF0000',
        'Purple': '#8F3F97',
        'PURPLE': '#8F3F97',
        'Very Unhealthy': '#8F3F97',
        'Maroon': '#7E0023',
        'MAROON': '#7E0023',
        'Hazardous': '#7E0023'
    };
    return colors[label] || '#AAAAAA';
}

function getAqiColorByName(name) {
    const colors = {
        'GREEN': '#00E400',
        'YELLOW': '#FFFF00',
        'ORANGE': '#FF7E00',
        'RED': '#FF0000',
        'PURPLE': '#8F3F97',
        'MAROON': '#7E0023'
    };
    return colors[name] || '';
}

function getAqiColorNameForPM10(pm10) {
    if (pm10 <= 54) return 'GREEN';
    if (pm10 <= 154) return 'YELLOW';
    if (pm10 <= 254) return 'ORANGE';
    if (pm10 <= 354) return 'RED';
    if (pm10 <= 424) return 'PURPLE';
    return 'MAROON';
}

function normalizeRoadName(name) {
    return String(name || '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, ' ')
        .trim();
}

function normalizeRiskLabel(riskLevel) {
    const v = String(riskLevel || '').toUpperCase();
    if (v === 'GREEN') return 'Green';
    if (v === 'YELLOW') return 'Yellow';
    if (v === 'ORANGE') return 'Orange';
    if (v === 'RED') return 'Red';
    if (v === 'PURPLE') return 'Purple';
    if (v === 'MAROON') return 'Maroon';
    return 'Green';
}

function buildNowcastRoadIndex(roads) {
    const idx = new Map();
    if (!Array.isArray(roads)) return idx;
    for (const r of roads) {
        if (!r) continue;
        const key = normalizeRoadName(r.road_name || r.road_id);
        if (key) idx.set(key, r);
    }
    return idx;
}

// ============================================================================
// MAP UPDATE LOGIC
// ============================================================================

async function updateAllRoads() {
    if (!map || !originalRoads || isUpdating) {
        return;
    }
    
    isUpdating = true;
    
    try {
        let prediction;
        let isFrozenGround = false;
        let physicsOverride = 'normal';
        
        // Check if ground is frozen from day 0 forecast
        if (forecastData && Array.isArray(forecastData.daily_forecasts)) {
            const day0 = forecastData.daily_forecasts.find(d => d.day_offset === 0) || forecastData.daily_forecasts[0];
            if (day0) {
                isFrozenGround = day0.physics_status === 'frozen_ground' || day0.frozen_hours === 24;
                physicsOverride = day0.physics_status || 'normal';
            }
        }
        
        // If manual override is active, use manual prediction
        if (manualOverrideActive) {
            const manualData = await fetchManualOverridePrediction();
            if (manualData) {
                prediction = {
                    dustScore: manualData.base.dust_score || 0,
                    label: normalizeRiskLabel(manualData.base.risk_level),
                    confidence: manualData.base.dust_score || 0,
                    probabilities: {},
                    isFrozen: manualData.base.is_frozen || false,
                    physicsOverride: manualData.base.physics_override || 'normal'
                };
                nowcastData = { base: manualData.base || {}, roads: manualData.roads || [] };
                // Update frozen status from manual override response
                isFrozenGround = prediction.isFrozen;
                physicsOverride = prediction.physicsOverride;
            } else {
                // Fallback to ML
                prediction = await predictDustML(globalSettings.trafficVolume, {
                    days_since_grading: globalSettings.daysSinceGrading,
                    days_since_suppressant: globalSettings.daysSinceSuppressant,
                    atv_activity: globalSettings.atvActivity,
                    snow_cover_flag: globalSettings.snowCover,
                    freeze_thaw_flag: globalSettings.freezeThaw,
                    road_loose_flag: globalSettings.roadLoose,
                    construction_activity: globalSettings.construction,
                    heavy_truck_activity: globalSettings.heavyTruck
                });
            }
        } else {
            // Use day 0 forecast as base
            let day0 = null;
            if (forecastData && Array.isArray(forecastData.daily_forecasts)) {
                day0 = forecastData.daily_forecasts.find(d => d.day_offset === 0) || forecastData.daily_forecasts[0];
            }

            if (day0) {
                const pm10Max = typeof day0.pm10_max === 'number' ? day0.pm10_max : null;
                const pm10Mean = typeof day0.pm10_mean === 'number' ? day0.pm10_mean : 0;
                const pm10Base = (typeof pm10Max === 'number') ? pm10Max : pm10Mean;
                prediction = {
                    dustScore: Math.max(0, Math.min(1, pm10Base / 300)),
                    label: normalizeRiskLabel(day0.risk_level || 'GREEN'),
                    confidence: typeof day0.dust_probability_mean === 'number' ? day0.dust_probability_mean : 0
                };
            } else {
                // Fallback to ML prediction if forecast missing
                prediction = await predictDustML(globalSettings.trafficVolume, {
                    days_since_grading: globalSettings.daysSinceGrading,
                    days_since_suppressant: globalSettings.daysSinceSuppressant,
                    atv_activity: globalSettings.atvActivity,
                    snow_cover_flag: globalSettings.snowCover,
                    freeze_thaw_flag: globalSettings.freezeThaw,
                    road_loose_flag: globalSettings.roadLoose,
                    construction_activity: globalSettings.construction,
                    heavy_truck_activity: globalSettings.heavyTruck
                });
            }
        }
        
        // CRITICAL: If ground is frozen, override ALL predictions to 0
        // This is physics-based - frozen ground cannot emit dust
        if (isFrozenGround && physicsOverride !== 'simulation_mode') {
            prediction.dustScore = 0;
            prediction.label = 'Green';
            console.log('Frozen ground detected - forcing all dust scores to 0');
        }

        const updatedRoads = JSON.parse(JSON.stringify(originalRoads));
        
        for (let i = 0; i < updatedRoads.features.length; i++) {
            const feature = updatedRoads.features[i];
            
            // Fallback to ML-adjusted score
            let baseScore = prediction.dustScore;
            
            // Apply frozen ground override - NO dust when frozen
            if (isFrozenGround && physicsOverride !== 'simulation_mode') {
                baseScore = 0;
            }
            
            const adjustedScore = applyStreetMultiplier(baseScore, feature.properties);
            
            feature.properties.dustConcentration = adjustedScore;
            feature.properties.dustLabel = classifyRisk(adjustedScore);
            feature.properties.scoreSource = manualOverrideActive ? 'Manual+Adjusted' : 'ML+Adjusted';
            feature.properties.baseMLScore = prediction.dustScore;
            feature.properties.mlLabel = prediction.label;
            feature.properties.confidence = prediction.confidence;
            feature.properties.isFrozen = isFrozenGround;
        }
        
        map.getSource('roads').setData(updatedRoads);
        
        // Update heatmap if visible
        if (heatmapVisible) {
            const heatmapData = await createDustHeatmap();
            if (map.getSource('heatmap')) {
                map.getSource('heatmap').setData(heatmapData);
            }
        }
        
    } catch (error) {
        console.error('Error updating roads:', error);
    } finally {
        isUpdating = false;
    }
}

// ============================================================================
// HEATMAP
// ============================================================================

async function createDustHeatmap() {
    if (!originalRoads) return null;
    
    // Check if ground is frozen
    let isFrozenGround = false;
    let physicsOverride = 'normal';
    if (forecastData && Array.isArray(forecastData.daily_forecasts)) {
        const day0 = forecastData.daily_forecasts.find(d => d.day_offset === 0) || forecastData.daily_forecasts[0];
        if (day0) {
            isFrozenGround = day0.physics_status === 'frozen_ground' || day0.frozen_hours === 24;
            physicsOverride = day0.physics_status || 'normal';
        }
    }
    
    let prediction;
    if (manualOverrideActive) {
        prediction = {
            dustScore: nowcastData?.base?.dust_score || 0.15
        };
        // Check simulation mode
        if (nowcastData?.base?.physics_override === 'simulation_mode') {
            physicsOverride = 'simulation_mode';
        }
    } else {
        let day0 = null;
        if (forecastData && Array.isArray(forecastData.daily_forecasts)) {
            day0 = forecastData.daily_forecasts.find(d => d.day_offset === 0) || forecastData.daily_forecasts[0];
        }
        if (day0) {
            const pm10Max = typeof day0.pm10_max === 'number' ? day0.pm10_max : null;
            const pm10Mean = typeof day0.pm10_mean === 'number' ? day0.pm10_mean : 0;
            const pm10Base = (typeof pm10Max === 'number') ? pm10Max : pm10Mean;
            prediction = {
                dustScore: Math.max(0, Math.min(1, pm10Base / 300))
            };
        } else {
            prediction = await predictDustML(globalSettings.trafficVolume, {
                days_since_grading: globalSettings.daysSinceGrading,
                days_since_suppressant: globalSettings.daysSinceSuppressant,
                atv_activity: globalSettings.atvActivity,
                snow_cover_flag: globalSettings.snowCover,
                freeze_thaw_flag: globalSettings.freezeThaw,
                road_loose_flag: globalSettings.roadLoose,
                construction_activity: globalSettings.construction,
                heavy_truck_activity: globalSettings.heavyTruck
            });
        }
    }
    
    // Force zero if frozen (unless simulation mode)
    if (isFrozenGround && physicsOverride !== 'simulation_mode') {
        prediction.dustScore = 0;
    }
    
    const points = [];
    const bounds = {
        minLat: 64.48,
        maxLat: 64.52,
        minLon: -165.45,
        maxLon: -165.35
    };
    
    const gridSize = 0.0015;
    
    for (let lat = bounds.minLat; lat <= bounds.maxLat; lat += gridSize) {
        for (let lon = bounds.minLon; lon <= bounds.maxLon; lon += gridSize) {
            let nearestDist = Infinity;
            let nearestRoad = null;
            
            originalRoads.features.forEach(road => {
                if (road.geometry && road.geometry.coordinates) {
                    road.geometry.coordinates.forEach(coord => {
                        if (Array.isArray(coord) && coord.length >= 2) {
                            const dist = Math.sqrt(
                                Math.pow(coord[1] - lat, 2) + 
                                Math.pow(coord[0] - lon, 2)
                            );
                            if (dist < nearestDist) {
                                nearestDist = dist;
                                nearestRoad = road;
                            }
                        }
                    });
                }
            });
            
            let dustScore = prediction.dustScore;
            
            if (nearestRoad && nearestDist < 0.003) {
                dustScore = applyStreetMultiplier(dustScore, nearestRoad.properties || {});
                const fadeFactor = 1 - (nearestDist / 0.003);
                dustScore = dustScore * Math.pow(fadeFactor, 0.5);
            } else {
                dustScore = prediction.dustScore * 0.2;
            }
            
            points.push({
                type: 'Feature',
                geometry: {
                    type: 'Point',
                    coordinates: [lon, lat]
                },
                properties: {
                    dustScore: dustScore,
                    label: classifyRisk(dustScore)
                }
            });
        }
    }
    
    return {
        type: 'FeatureCollection',
        features: points
    };
}

async function toggleMapView(showHeatmap) {
    if (!map) return;
    
    heatmapVisible = showHeatmap;
    
    if (showHeatmap) {
        if (!manualOverrideActive && (!forecastData || !Array.isArray(forecastData.daily_forecasts))) {
            await fetchAndRenderForecast();
        }
        if (map.getLayer('roads-layer')) {
            map.setLayoutProperty('roads-layer', 'visibility', 'none');
        }
        
        const heatmapData = await createDustHeatmap();
        if (!heatmapData || !heatmapData.features || heatmapData.features.length === 0) {
            console.warn('Heatmap has no data to display');
            return;
        }
        
        if (map.getSource('heatmap')) {
            map.getSource('heatmap').setData(heatmapData);
        } else {
            map.addSource('heatmap', {
                type: 'geojson',
                data: heatmapData
            });
            
            map.addLayer({
                id: 'heatmap-layer',
                type: 'heatmap',
                source: 'heatmap',
                paint: {
                    'heatmap-weight': [
                        'interpolate',
                        ['linear'],
                        ['get', 'dustScore'],
                        0, 0,
                        0.15, 0.3,
                        0.25, 0.5,
                        0.4, 0.7,
                        0.55, 0.85,
                        1, 1
                    ],
                    'heatmap-intensity': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        10, 0.8,
                        13, 1.2,
                        15, 1.5
                    ],
                    'heatmap-color': [
                        'interpolate',
                        ['linear'],
                        ['heatmap-density'],
                        0, 'rgba(0, 0, 0, 0)',
                        0.1, 'rgba(65, 105, 225, 0.4)',
                        0.2, 'rgba(34, 139, 34, 0.5)',
                        0.35, 'rgba(50, 205, 50, 0.6)',
                        0.5, 'rgba(173, 255, 47, 0.7)',
                        0.6, 'rgba(255, 255, 0, 0.75)',
                        0.7, 'rgba(255, 165, 0, 0.8)',
                        0.8, 'rgba(255, 69, 0, 0.85)',
                        0.9, 'rgba(220, 20, 60, 0.9)',
                        1, 'rgba(139, 0, 0, 0.95)'
                    ],
                    'heatmap-radius': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        10, 20,
                        13, 35,
                        15, 50
                    ],
                    'heatmap-opacity': 0.85
                }
            });
        }
        
        if (map.getLayer('heatmap-layer')) {
            map.setLayoutProperty('heatmap-layer', 'visibility', 'visible');
        }
        
    } else {
        if (map.getLayer('roads-layer')) {
            map.setLayoutProperty('roads-layer', 'visibility', 'visible');
        }
        
        if (map.getLayer('heatmap-layer')) {
            map.setLayoutProperty('heatmap-layer', 'visibility', 'none');
        }
    }
}

// ============================================================================
// GEOJSON LOADING
// ============================================================================

async function loadRoadsFromGeoJSON() {
    try {
        const response = await fetch('nome.geojson');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (e) {
        console.error('Error loading GeoJSON:', e);
        throw e;
    }
}

// ============================================================================
// ROAD CLICK HANDLER
// ============================================================================

function handleRoadClick(e) {
    if (e.features.length === 0) return;
    
    const feature = e.features[0];
    const props = feature.properties;
    const name = props.name || 'Unnamed Road';
    const isFrozen = props.isFrozen || (nowcastData && nowcastData.base && nowcastData.base.is_frozen);
    const modeIndicator = manualOverrideActive ? '⚙️ Manual Override' : '🤖 ML Prediction';
    const frozenIndicator = isFrozen ? '<br><span style="color: #004085; background: #cce5ff; padding: 2px 6px; border-radius: 3px;">❄️ Frozen Ground - No Dust</span>' : '';
    
    const popupHTML = `
        <div style="min-width: 200px;">
            <h4 style="margin: 0 0 8px 0;">${name}</h4>
            <div style="font-size: 0.85em;">
                <strong>ID:</strong> ${props['@id'] || 'Unknown'}<br>
                <strong>Surface:</strong> ${props.surface || 'Unknown'}<br>
                <strong>Type:</strong> ${props.highway || 'Unknown'}<br>
                <hr style="margin: 8px 0;">
                <strong>Risk Level:</strong> <span style="color: ${getRiskColor(props.dustLabel)}; font-weight: 600;">${props.dustLabel}</span><br>
                ${props.confidence ? `<strong>Confidence:</strong> ${(props.confidence * 100).toFixed(0)}%<br>` : ''}
                ${frozenIndicator}
                <hr style="margin: 8px 0;">
                <small>${modeIndicator}</small>
            </div>
        </div>
    `;
    
    new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(popupHTML)
        .addTo(map);
}

// ============================================================================
// UPDATE SCHEDULING
// ============================================================================

function scheduleUpdate() {
    if (updateTimeout) {
        clearTimeout(updateTimeout);
    }
    updateTimeout = setTimeout(() => {
        updateAllRoads();
    }, 500);
}

// Activate manual override when settings change
function activateManualOverride() {
    if (!manualOverrideActive) {
        manualOverrideActive = true;
        settingsModified = true;
        console.log('Manual override activated');
        
        // Update UI to show override is active
        const refreshBtn = document.getElementById('refresh-weather');
        if (refreshBtn) {
            refreshBtn.textContent = '⚙️ Manual Mode Active';
            refreshBtn.style.background = '#ffc107';
            refreshBtn.style.color = '#000';
        }
    }
    scheduleUpdate();
}

// Reset to ML-based predictions
function resetToMLMode() {
    manualOverrideActive = false;
    settingsModified = false;
    console.log('Reset to ML mode');
    
    // Reset button
    const refreshBtn = document.getElementById('refresh-weather');
    if (refreshBtn) {
        refreshBtn.textContent = 'Refresh Data';
        refreshBtn.style.background = '#6c757d';
        refreshBtn.style.color = 'white';
    }
    
    // Re-fetch nowcast
    fetchNowcast();
    scheduleUpdate();
}

// ============================================================================
// MAP INITIALIZATION
// ============================================================================

async function initializeMap() {
    try {
        map = new maplibregl.Map({
            container: 'map',
            style: `https://api.maptiler.com/maps/streets-v2/style.json?key=${MAPTILER_KEY}`,
            center: CITY_CENTER,
            zoom: 13.5
        });
        
        map.on('load', async () => {
            try {
                await fetchWeather();
                await fetchAQI();
                
                const roadsData = await loadRoadsFromGeoJSON();

                // Mayor's ground-truth corrections: exclude roads that don't generate dust
                const excludedRoadIds = new Set([
                    'way/8984498',      // North Star Assoc Access Road
                    'way/8984547',      // Steadman Street (tertiary segment)
                    'way/281391984',    // Steadman Street (residential segment)
                    'way/336122060',    // Unnamed track
                    'way/224036027',    // Unnamed service road
                    'way/8983407',      // Prospect Street (unclassified)
                    'way/1431045620',   // Division Street (north curve only)
                    'way/179519099',    // Unnamed service road parallel to Center Creek Road
                    'way/336012345',    // Unnamed track west of Center Creek area
                    'way/336122071',    // Unnamed track east of downtown
                    'way/93992266',     // Unnamed service road perpendicular to Cemetery Road
                    'way/8982959',      // Unnamed service road
                    'way/336121906',    // Winter Trail (footway)
                    'way/8983454',      // Unnamed track
                    'way/204581971',    // Unnamed track
                    'way/1333259888',   // Unnamed service road
                    'way/204581968',    // Snake River Road
                    'way/204581963',    // Glacier Creek Road
                    'way/8984836',      // Foot Trail (footway)
                    'way/8983569',      // Anvil Rock Road (track)
                    'way/886864859',    // Unnamed track
                    'way/8982860',      // Unnamed service road
                    'way/8984745',      // Unnamed path (ground)
                    'way/8984630',      // Osborne Road
                    'way/8984746',      // Foot Trail (footway)
                    'way/631146518',    // Unnamed service road (unpaved)
                    'way/179519097',    // Unnamed service road
                    'way/8983468',      // Unnamed service road
                    'way/8983668',      // Construction Road (service)
                    'way/179519106',    // Unnamed service road
                    'way/8983244',      // Unnamed service road
                    'way/8982760',      // Lynden Way
                    'way/629858737',    // Unnamed residential (unpaved)
                    'way/631146513',    // Dredge 5 Road
                    'way/204581987',    // Unnamed track
                    'way/1333259883',   // Unnamed service road
                    'way/8982772',      // Anvil Mountain Tower Road
                    'way/204581955',    // Unnamed unclassified (unpaved)
                    'way/8984709',      // Moonlight Springs Road
                    'way/1333259887',   // Unnamed service road
                    'way/204581988',    // Unnamed unclassified (unpaved)
                    'way/204581991',    // Unnamed unclassified (unpaved)
                    'way/1135345366',   // Unnamed service road
                    'way/8983241',      // Unnamed unclassified
                    'way/1023590514',   // Unnamed unclassified
                    'way/1333259884',   // Unnamed service road
                    'way/1333259890',   // Unnamed service road
                ]);
                const excludedRoadNames = new Set([
                    'North Star Assoc Access Road',
                    'Steadman Street',
                    'Prospect Street',
                    'Winter Trail',
                    'Snake River Road',
                    'Moonlight Springs Road',
                    'Foot Trail',
                    'Anvil Rock Road',
                    'Osborne Road',
                    'Construction Road',
                    'Lynden Way',
                    'Dredge 5 Road',
                    'Anvil Mountain Tower Road',
                ]);
                roadsData.features = roadsData.features.filter(f => {
                    const id = f.properties['@id'] || '';
                    const name = f.properties.name || '';
                    return !excludedRoadIds.has(id) && !excludedRoadNames.has(name);
                });

                roadsData.features.forEach((f, idx) => {
                    if (!f.id) {
                        f.id = f.properties['@id'] || `road-${idx}`;
                    }
                });

                originalRoads = roadsData;
                
                map.addSource('roads', {
                    type: 'geojson',
                    data: roadsData
                });
                
                map.addLayer({
                    id: 'roads-layer',
                    type: 'line',
                    source: 'roads',
                    paint: {
                        'line-width': 7,
                        'line-opacity': 0.9,
                        'line-color': [
                            'match',
                            ['get', 'dustLabel'],
                            'Green', '#00E400',
                            'Yellow', '#FFFF00',
                            'Orange', '#FF7E00',
                            'Red', '#FF0000',
                            'Purple', '#8F3F97',
                            'Maroon', '#7E0023',
                            '#00E400'
                        ]
                    }
                });
                
                await fetchAndRenderForecast();
                await updateAllRoads();
                
                map.on('mousemove', 'roads-layer', () => {
                    map.getCanvas().style.cursor = 'pointer';
                });
                
                map.on('mouseleave', 'roads-layer', () => {
                    map.getCanvas().style.cursor = '';
                });
                
                map.on('click', 'roads-layer', handleRoadClick);
                
                // Auto-refresh every 5 minutes (only if not in manual mode)
                setInterval(async () => {
                    if (!manualOverrideActive) {
                        await fetchWeather();
                        await fetchAQI();
                        await fetchAndRenderForecast();
                        scheduleUpdate();
                    }
                }, 300000);
                
            } catch (error) {
                console.error('Error during map setup:', error);
                alert('Failed to initialize map: ' + error.message);
            }
        });
        
    } catch (error) {
        console.error('Map initialization failed:', error);
        alert('Failed to create map: ' + error.message);
    }
}

// ============================================================================
// EVENT LISTENERS SETUP
// ============================================================================

function setupEventListeners() {
    // Helper function to safely add event listeners
    const addListener = (id, event, handler) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener(event, handler);
        }
    };
    
    // Traffic volume - activates manual override
    addListener('global-traffic', 'change', (e) => {
        globalSettings.trafficVolume = e.target.value;
        activateManualOverride();
    });
    
    // Days since grading - activates manual override
    addListener('global-days-grading', 'input', (e) => {
        globalSettings.daysSinceGrading = parseInt(e.target.value);
        const display = document.getElementById('grading-display');
        if (display) display.textContent = e.target.value;
        activateManualOverride();
    });
    
    // Days since suppressant - activates manual override
    addListener('global-days-suppressant', 'input', (e) => {
        globalSettings.daysSinceSuppressant = parseInt(e.target.value);
        const display = document.getElementById('suppressant-display');
        if (display) display.textContent = e.target.value;
        activateManualOverride();
    });
    
    // ATV activity - activates manual override (optional element)
    addListener('global-atv', 'change', (e) => {
        globalSettings.atvActivity = e.target.value;
        activateManualOverride();
    });
    
    // Snow cover - activates manual override
    addListener('global-snow', 'change', (e) => {
        globalSettings.snowCover = parseInt(e.target.value);
        activateManualOverride();
    });
    
    // Simulation mode toggle (ignore frozen ground)
    const ignoreFrozenCheckbox = document.getElementById('ignore-frozen');
    if (ignoreFrozenCheckbox) {
        ignoreFrozenCheckbox.addEventListener('change', (e) => {
            globalSettings.ignoreFrozen = e.target.checked;
            
            // Show/hide warning
            const warning = document.getElementById('simulation-warning');
            if (warning) {
                warning.style.display = e.target.checked ? 'block' : 'none';
            }
            
            // Always activate manual override when toggling simulation
            activateManualOverride();
        });
    }
    
    // Refresh button - resets to ML mode
    addListener('refresh-weather', 'click', async () => {
        if (manualOverrideActive) {
            // If in manual mode, clicking resets to ML mode
            resetToMLMode();
        } else {
            // Normal refresh
            await fetchWeather();
            await fetchAQI();
            await fetchAndRenderForecast();
            scheduleUpdate();
        }
    });
    
    // Map view toggles
    addListener('toggle-dust-map', 'click', () => {
        const dustBtn = document.getElementById('toggle-dust-map');
        const weatherBtn = document.getElementById('toggle-weather-map');
        if (dustBtn) dustBtn.classList.add('active');
        if (weatherBtn) weatherBtn.classList.remove('active');
        toggleMapView(false);
    });
    
    addListener('toggle-weather-map', 'click', () => {
        const dustBtn = document.getElementById('toggle-dust-map');
        const weatherBtn = document.getElementById('toggle-weather-map');
        if (weatherBtn) weatherBtn.classList.add('active');
        if (dustBtn) dustBtn.classList.remove('active');
        toggleMapView(true);
    });
    
    // DateTime update
    updateDateTime();
    setInterval(updateDateTime, 1000);
}

// ============================================================================
// FORECAST
// ============================================================================

function toggleForecast() {
    const panel = document.getElementById('forecast-panel');
    const btn = document.getElementById('toggle-forecast');
    
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.classList.add('active');
        fetchAndRenderForecast();
    } else {
        panel.style.display = 'none';
        btn.classList.remove('active');
    }
}

async function fetchAndRenderForecast() {
    try {
        // Show loading state
        const tableBody = document.getElementById('forecast-table-body');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: #999;">Loading 3-day forecast...</td></tr>';
        }
        
        const response = await fetch(FORECAST_URL);
        
        // Check if response is OK
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            console.error('Forecast API returned error:', result);
            if (tableBody) {
                const errorMsg = result.error || result.detail || 'Unknown error';
                tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 20px; color: #dc3545;">API Error: ${errorMsg}</td></tr>`;
            }
            return;
        }
        
        // Check if we have daily forecast data
        if (!result.data || !result.data.daily_forecasts || result.data.daily_forecasts.length === 0) {
            console.warn('No daily forecast data returned');
            if (tableBody) {
                tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: #856404;">No forecast data available</td></tr>';
            }
            updateDailyForecastStats([]);
            return;
        }
        
        forecastData = result.data;
        console.log(`Daily forecast loaded: ${forecastData.daily_forecasts.length} days, source: ${forecastData.source || 'unknown'}`);
        
        renderDailyForecastChart(forecastData.daily_forecasts);
        updateDailyForecastStats(forecastData.daily_forecasts);
        renderDailyForecastTable(forecastData.daily_forecasts);

        if (!manualOverrideActive) {
            const day0 = forecastData.daily_forecasts.find(d => d.day_offset === 0) || forecastData.daily_forecasts[0];
            updateFromDailyForecast(day0);
        }
        
    } catch (error) {
        console.error('Failed to fetch daily forecast:', error);
        const tableBody = document.getElementById('forecast-table-body');
        if (tableBody) {
            tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 20px; color: #dc3545;">
                Error: ${error.message}<br>
                <small style="color: #666;">Check console for details. Server may need restart.</small>
            </td></tr>`;
        }
    }
}

function renderDailyForecastTable(dailyForecasts) {
    const tableBody = document.getElementById('forecast-table-body');
    if (!tableBody) return;
    
    if (!dailyForecasts || dailyForecasts.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: #999;">No daily forecast data available</td></tr>';
        return;
    }
    
    let html = '';
    dailyForecasts.forEach((day, index) => {
        const pm10Mean = day.pm10_mean || 0;
        const pm10Range = `${day.pm10_min || 0}-${day.pm10_max || 0}`;
        const tempRange = day.temp_min_f !== null && day.temp_max_f !== null 
            ? `${day.temp_min_f}°F - ${day.temp_max_f}°F` 
            : '--';
        const risk = day.risk_level || 'GREEN';
        const physicsStatus = day.physics_status || 'normal';
        const frozenHours = day.frozen_hours || 0;
        
        // Determine status text and icon
        let statusText = '';
        let statusClass = '';
        if (physicsStatus === 'frozen_ground' || frozenHours === 24) {
            statusText = '❄️ Frozen';
            statusClass = 'color: #004085; background: #cce5ff;';
        } else if (physicsStatus.includes('partial_freeze') || frozenHours > 0) {
            statusText = `🌡️ ${frozenHours}h Frozen`;
            statusClass = 'color: #856404; background: #fff3cd;';
        } else if (physicsStatus === 'transition_zone' || physicsStatus.includes('partial_thaw')) {
            statusText = '🌡️ Thawing';
            statusClass = 'color: #856404; background: #fff3cd;';
        } else {
            statusText = '✓ Normal';
            statusClass = 'color: #155724; background: #d4edda;';
        }
        
        // Risk badge colors (6-level EPA)
        const riskLabel = normalizeRiskLabel(risk);
        const riskColors = {
            'GREEN': { bg: '#e6f4ea', fg: '#1e7e34', label: 'Good' },
            'YELLOW': { bg: '#fff3cd', fg: '#856404', label: 'Moderate' },
            'ORANGE': { bg: '#ffe0b2', fg: '#a85d00', label: 'Sensitive' },
            'RED': { bg: '#f8d7da', fg: '#721c24', label: 'Unhealthy' },
            'PURPLE': { bg: '#e2d5f7', fg: '#5a2d82', label: 'Very Unhealthy' },
            'MAROON': { bg: '#f3c6d0', fg: '#6f1d2a', label: 'Hazardous' }
        };
        const riskKey = String(risk || 'GREEN').toUpperCase();
        const rc = riskColors[riskKey] || riskColors.GREEN;
        const riskBadge = `<span style="background: ${rc.bg}; color: ${rc.fg}; padding: 4px 10px; border-radius: 4px; font-weight: 600;">${rc.label}</span>`;
        
        // AQI badge (use server color when available)
        const aqi = day.aqi || 0;
        const aqiCategory = day.aqi_category || 'Good';
        const aqiColor = String(day.aqi_color || '').toUpperCase();
        const aqiColorMap = {
            'GREEN': { bg: '#e6f4ea', fg: '#1e7e34' },
            'YELLOW': { bg: '#fff3cd', fg: '#856404' },
            'ORANGE': { bg: '#ffe0b2', fg: '#a85d00' },
            'RED': { bg: '#f8d7da', fg: '#721c24' },
            'PURPLE': { bg: '#e2d5f7', fg: '#5a2d82' },
            'MAROON': { bg: '#f3c6d0', fg: '#6f1d2a' }
        };
        const ac = aqiColorMap[aqiColor] || aqiColorMap.GREEN;
        const aqiBadge = `<span style="background: ${ac.bg}; color: ${ac.fg}; padding: 2px 6px; border-radius: 3px;">${aqi} ${aqiCategory}</span>`;
        
        // Row background for alternating
        const rowBg = index % 2 === 0 ? '' : 'background: #f8f9fa;';
        
        html += `
            <tr style="${rowBg}">
                <td style="padding: 10px; font-weight: 600; font-size: 1.1em;">
                    ${day.day_name}
                    <br><small style="color: #666; font-weight: normal;">${day.date}</small>
                </td>
                <td style="padding: 10px; text-align: center;">${tempRange}</td>
                <td style="padding: 10px; text-align: center;">
                    <strong style="font-size: 1.1em;">${pm10Mean.toFixed(1)}</strong>
                    <br><small style="color: #666;">${pm10Range} µg/m³</small>
                </td>
                <td style="padding: 10px; text-align: center;">${aqiBadge}</td>
                <td style="padding: 10px; text-align: center;">${riskBadge}</td>
                <td style="padding: 10px; text-align: center;"><span style="padding: 4px 8px; border-radius: 4px; font-size: 0.9em; ${statusClass}">${statusText}</span></td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
}

// Keep old function for backward compatibility
function renderForecastTable(forecasts) {
    renderDailyForecastTable(forecasts);
}

function renderDailyForecastChart(dailyForecasts) {
    const canvas = document.getElementById('forecast-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!dailyForecasts || dailyForecasts.length === 0) {
        ctx.fillStyle = '#999';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No daily forecast data available', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    const padding = { top: 20, right: 30, bottom: 50, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;
    
    const days = dailyForecasts.map(f => f.day_offset);
    const dayNames = dailyForecasts.map(f => f.day_name);
    const pm10Mean = dailyForecasts.map(f => f.pm10_mean || 0);
    const pm10Min = dailyForecasts.map(f => f.pm10_min || 0);
    const pm10Max = dailyForecasts.map(f => f.pm10_max || 0);
    const risks = dailyForecasts.map(f => f.risk_level || 'GREEN');
    
    const allValues = [...pm10Min, ...pm10Max];
    const minPM10 = Math.floor(Math.min(...allValues) / 10) * 10;
    const maxPM10 = Math.max(10, Math.ceil(Math.max(...allValues) / 10) * 10);
    const pmRange = maxPM10 - minPM10 || 10;
    
    const barWidth = chartWidth / (days.length * 2);
    const xScale = (day, offset = 0) => padding.left + (day + 0.5) * (chartWidth / days.length) + offset;
    const yScale = pm10 => padding.top + chartHeight - ((pm10 - minPM10) / pmRange) * chartHeight;
    
    // Background risk zones (per bar)
    const riskBg = {
        'GREEN': 'rgba(0, 228, 0, 0.15)',
        'YELLOW': 'rgba(255, 255, 0, 0.2)',
        'ORANGE': 'rgba(255, 126, 0, 0.2)',
        'RED': 'rgba(255, 0, 0, 0.2)',
        'PURPLE': 'rgba(143, 63, 151, 0.2)',
        'MAROON': 'rgba(126, 0, 35, 0.2)'
    };
    for (let i = 0; i < days.length; i++) {
        const risk = String(risks[i] || 'GREEN').toUpperCase();
        ctx.fillStyle = riskBg[risk] || riskBg.GREEN;
        const x = xScale(i, -barWidth);
        ctx.fillRect(x, padding.top, barWidth * 2, chartHeight);
    }
    
    // Grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (i / 4) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + chartWidth, y);
        ctx.stroke();
        
        const pmValue = maxPM10 - (i / 4) * pmRange;
        ctx.fillStyle = '#666';
        ctx.font = '11px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(pmValue.toFixed(0), padding.left - 8, y + 4);
    }
    
    // Draw bars with error bars
    for (let i = 0; i < days.length; i++) {
        const x = xScale(i);
        const yMean = yScale(pm10Mean[i]);
        const yMin = yScale(pm10Min[i]);
        const yMax = yScale(pm10Max[i]);
        
        // Bar
        const risk = String(risks[i] || 'GREEN').toUpperCase();
        const barColors = {
            'GREEN': '#00E400',
            'YELLOW': '#FFFF00',
            'ORANGE': '#FF7E00',
            'RED': '#FF0000',
            'PURPLE': '#8F3F97',
            'MAROON': '#7E0023'
        };
        ctx.fillStyle = barColors[risk] || barColors.GREEN;
        ctx.fillRect(x - barWidth/2, yMean, barWidth, chartHeight + padding.top - yMean);
        
        // Error bar (min-max range)
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, yMin);
        ctx.lineTo(x, yMax);
        ctx.stroke();
        
        // Error bar caps
        ctx.beginPath();
        ctx.moveTo(x - 5, yMin);
        ctx.lineTo(x + 5, yMin);
        ctx.moveTo(x - 5, yMax);
        ctx.lineTo(x + 5, yMax);
        ctx.stroke();
        
        // PM10 value label
        ctx.fillStyle = '#333';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(pm10Mean[i].toFixed(1), x, yMean - 8);
    }
    
    // X-axis labels (day names)
    ctx.fillStyle = '#333';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    for (let i = 0; i < days.length; i++) {
        const x = xScale(i);
        ctx.fillText(dayNames[i], x, canvas.height - 10);
    }
    
    // Axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();
    
    // Y-axis label
    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#666';
    ctx.font = '11px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('PM10 (µg/m³)', 0, 0);
    ctx.restore();
}

// Keep old function for backward compatibility
function renderForecastChart(forecasts) {
    renderDailyForecastChart(forecasts);
}

function updateDailyForecastStats(dailyForecasts) {
    if (!dailyForecasts || dailyForecasts.length === 0) return;
    
    const pm10_values = dailyForecasts.map(f => f.pm10_mean || 0);
    const minPM10 = Math.min(...pm10_values);
    const maxPM10 = Math.max(...pm10_values);
    
    const avgPM10 = pm10_values.reduce((a, b) => a + b, 0) / pm10_values.length;
    
    const greenCount = dailyForecasts.filter(f => f.risk_level === 'GREEN').length;
    const yellowCount = dailyForecasts.filter(f => f.risk_level === 'YELLOW').length;
    const orangeCount = dailyForecasts.filter(f => f.risk_level === 'ORANGE').length;
    const redCount = dailyForecasts.filter(f => f.risk_level === 'RED').length;
    const purpleCount = dailyForecasts.filter(f => f.risk_level === 'PURPLE').length;
    const maroonCount = dailyForecasts.filter(f => f.risk_level === 'MAROON').length;
    
    // Check if all days are frozen
    const frozenCount = dailyForecasts.filter(f => 
        f.physics_status === 'frozen_ground' || f.frozen_hours === 24
    ).length;
    
    const pmRangeEl = document.getElementById('pm10-range');
    const avgSevEl = document.getElementById('avg-severity');
    const riskHoursEl = document.getElementById('risk-hours');
    const statusEl = document.getElementById('forecast-status');
    
    if (pmRangeEl) pmRangeEl.textContent = `${minPM10.toFixed(1)}-${maxPM10.toFixed(1)} µg/m³`;
    if (avgSevEl) avgSevEl.textContent = `${avgPM10.toFixed(1)} µg/m³`;
    
    if (riskHoursEl) {
        let riskText = `🟢 ${greenCount} day${greenCount !== 1 ? 's' : ''}`;
        if (yellowCount > 0) riskText += ` 🟡 ${yellowCount}`;
        if (orangeCount > 0) riskText += ` 🟠 ${orangeCount}`;
        if (redCount > 0) riskText += ` 🔴 ${redCount}`;
        if (purpleCount > 0) riskText += ` 🟣 ${purpleCount}`;
        if (maroonCount > 0) riskText += ` 🟤 ${maroonCount}`;
        riskHoursEl.textContent = riskText;
    }
    
    if (statusEl) {
        if (frozenCount === dailyForecasts.length) {
            statusEl.innerHTML = '<span style="color: #004085;">❄️ All Days Frozen</span>';
        } else if (frozenCount > 0) {
            statusEl.innerHTML = `<span style="color: #856404;">❄️ ${frozenCount} day${frozenCount !== 1 ? 's' : ''} Frozen</span>`;
        } else {
            statusEl.innerHTML = '<span style="color: #155724;">✓ Active</span>';
        }
    }
}

// Keep old function for backward compatibility  
function updateForecastStats(forecasts) {
    updateDailyForecastStats(forecasts);
}

// Auto-refresh forecast every 15 minutes
setInterval(() => {
    const panel = document.getElementById('forecast-panel');
    if (panel && panel.style.display !== 'none' && !manualOverrideActive) {
        fetchAndRenderForecast();
    }
}, 15 * 60 * 1000);

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initializeMap();
});
