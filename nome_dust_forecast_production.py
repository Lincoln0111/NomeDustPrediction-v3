"""
Nome Dust Forecast Model - Production Version
==============================================

Enhanced dust forecasting with:
1. Road classification from nome.geojson (length, connectivity, importance, dust factor)
2. Weather forecasts (NWS primary, Open-Meteo backup + soil moisture + ET)
3. Road moisture model (physics-based drying with Open-Meteo validation)
4. Road-specific + Area-based predictions

Requirements:
    pip install numpy pandas shapely pyproj requests scikit-learn

Author: Nome Dust Forecasting Project
Date: December 2024
"""

import json
import math
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from collections import defaultdict
from pathlib import Path
import warnings

# Scientific computing
import numpy as np
import pandas as pd

# Geometry libraries
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import nearest_points, unary_union
import pyproj
from pyproj import Transformer

# Optional: for spatial indexing (much faster connectivity calculation)
try:
    from rtree import index as rtree_index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False
    warnings.warn("rtree not installed. Connectivity calculation will be slower.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DustForecastConfig:
    """Centralized configuration for dust forecasting"""
    
    # Nome coordinates
    NOME_LAT: float = 64.5011
    NOME_LON: float = -165.4064
    
    # Coordinate reference systems
    WGS84: str = "EPSG:4326"  # Geographic (lat/lon)
    UTM_ZONE_3N: str = "EPSG:32603"  # UTM Zone 3N (for Nome, Alaska)
    
    # Dust factors by surface type (explicit tags)
    DUST_FACTOR_EXPLICIT_UNPAVED: float = 1.0   # unpaved, gravel, dirt, ground
    DUST_FACTOR_TRACK: float = 0.95             # track (almost certainly dirt)
    DUST_FACTOR_RESIDENTIAL_UNSPEC: float = 0.75  # residential/service unspecified
    DUST_FACTOR_SECONDARY_UNSPEC: float = 0.6   # secondary/tertiary unspecified
    DUST_FACTOR_PRIMARY: float = 0.4            # primary/arterial (resuspension)
    DUST_FACTOR_EXPLICIT_PAVED: float = 0.25    # asphalt/paved
    
    # Weather thresholds
    WIND_ENTRAINMENT_MPS: float = 5.0    # Wind speed to lift dust
    WIND_STRONG_MPS: float = 8.0         # Strong wind
    HUMIDITY_DRY_PCT: float = 50.0       # Below this = dry conditions
    PRECIP_SUPPRESSION_MM: float = 0.5   # Precipitation that suppresses dust
    
    # Road moisture model
    BASE_DRYING_RATE: float = 0.04       # Base hourly drying rate
    TEMP_DRYING_FACTOR: float = 0.02     # Additional drying per °C above 10
    WIND_DRYING_FACTOR: float = 0.015    # Additional drying per m/s above 2
    HUMIDITY_DRYING_FACTOR: float = -0.008  # Reduced drying per % above 50
    PRECIP_SATURATION_MM: float = 5.0    # Precipitation to fully saturate road
    
    # Connectivity settings
    CONNECTIVITY_THRESHOLD_M: float = 50.0  # Distance to consider roads connected
    
    # Forecast settings
    FORECAST_HOURS_SHORT: int = 24
    FORECAST_HOURS_MEDIUM: int = 48
    FORECAST_HOURS_LONG: int = 72
    
    # API settings
    NWS_USER_AGENT: str = "NomeDustForecast/2.0 (contact@example.com)"
    NWS_TIMEOUT: int = 15
    OPEN_METEO_TIMEOUT: int = 15
    
    # Area definitions (bounding boxes: min_lon, min_lat, max_lon, max_lat)
    AREAS: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        "Downtown": (-165.42, 64.495, -165.38, 64.505),
        "Port": (-165.45, 64.490, -165.42, 64.500),
        "Airport": (-165.48, 64.505, -165.42, 64.515),
        "East_Residential": (-165.38, 64.490, -165.35, 64.510),
        "West_Residential": (-165.45, 64.500, -165.42, 64.520),
        "Highways": (-165.60, 64.45, -165.30, 64.60),  # Catch-all for roads outside town
    })

    # Mayor's ground-truth corrections (January 26, 2026)
    # Roads to completely exclude from model and map
    EXCLUDED_ROADS_BY_NAME: frozenset = frozenset({
        "North Star Assoc Access Road",
        "Steadman Street",
        "Prospect Street",
        "Winter Trail",
        "Snake River Road",
        "Moonlight Springs Road",
        "Foot Trail",
        "Anvil Rock Road",
        "Osborne Road",
        "Construction Road",
        "Lynden Way",
        "Dredge 5 Road",
        "Anvil Mountain Tower Road",
    })
    EXCLUDED_ROADS_BY_ID: frozenset = frozenset({
        "way/8984498",      # North Star Assoc Access Road
        "way/8984547",      # Steadman Street (tertiary segment)
        "way/281391984",    # Steadman Street (residential/unpaved segment)
        "way/336122060",    # Unnamed track near North Star Access Road
        "way/224036027",    # Unnamed service road near Steadman Street
        "way/8983407",      # Prospect Street (unclassified)
        "way/1431045620",   # Division Street (north curve only)
        "way/179519099",    # Unnamed service road parallel to Center Creek Road
        "way/336012345",    # Unnamed track west of Center Creek area
        "way/336122071",    # Unnamed track east of downtown
        "way/93992266",     # Unnamed service road perpendicular to Cemetery Road
        "way/8982959",      # Unnamed service road
        "way/336121906",    # Winter Trail (footway)
        "way/8983454",      # Unnamed track
        "way/204581971",    # Unnamed track
        "way/1333259888",   # Unnamed service road
        "way/204581968",    # Snake River Road (unpaved unclassified)
        "way/204581963",    # Glacier Creek Road (residential)
        "way/8984836",      # Foot Trail (footway)
        "way/8983569",      # Anvil Rock Road (track)
        "way/886864859",    # Unnamed track
        "way/8982860",      # Unnamed service road
        "way/8984745",      # Unnamed path (ground surface)
        "way/8984630",      # Osborne Road (unclassified)
        "way/8984746",      # Foot Trail (footway)
        "way/631146518",    # Unnamed service road (unpaved)
        "way/179519097",    # Unnamed service road
        "way/8983468",      # Unnamed service road
        "way/8983668",      # Construction Road (service)
        "way/179519106",    # Unnamed service road
        "way/8983244",      # Unnamed service road
        "way/8982760",      # Lynden Way (residential)
        "way/629858737",    # Unnamed residential (unpaved)
        "way/631146513",    # Dredge 5 Road (residential)
        "way/204581987",    # Unnamed track
        "way/1333259883",   # Unnamed service road
        "way/8982772",      # Anvil Mountain Tower Road (track)
        "way/204581955",    # Unnamed unclassified (unpaved)
        "way/8984709",      # Moonlight Springs Road (unclassified)
        "way/1333259887",   # Unnamed service road
        "way/204581988",    # Unnamed unclassified (unpaved)
        "way/204581991",    # Unnamed unclassified (unpaved)
        "way/1135345366",   # Unnamed service road
        "way/8983241",      # Unnamed unclassified
        "way/1023590514",   # Unnamed unclassified
        "way/1333259884",   # Unnamed service road
        "way/1333259890",   # Unnamed service road
    })

    # Roads confirmed as high-dust by Mayor — override dust_factor
    DUST_FACTOR_OVERRIDES_BY_NAME: Dict[str, float] = field(default_factory=lambda: {
        "Greg Kruschek Avenue": 1.5,
        "Nome-Council Road": 1.5,
        "Little Creek Road": 1.5,
        "Center Creek Road": 1.5,
        "Seppala Drive": 1.5,
    })


CONFIG = DustForecastConfig()


# =============================================================================
# COORDINATE TRANSFORMATION
# =============================================================================

class CoordinateTransformer:
    """Handles coordinate transformations between WGS84 and UTM"""
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
        # WGS84 (geographic) to UTM Zone 3N (projected, meters)
        self.to_utm = Transformer.from_crs(
            config.WGS84, config.UTM_ZONE_3N, always_xy=True
        )
        self.to_wgs84 = Transformer.from_crs(
            config.UTM_ZONE_3N, config.WGS84, always_xy=True
        )
    
    def lonlat_to_utm(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert longitude/latitude to UTM coordinates (meters)"""
        return self.to_utm.transform(lon, lat)
    
    def utm_to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        """Convert UTM coordinates to longitude/latitude"""
        return self.to_wgs84.transform(x, y)
    
    def linestring_to_utm(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Convert a list of lon/lat coordinates to UTM"""
        return [self.lonlat_to_utm(lon, lat) for lon, lat in coords]
    
    def calculate_length_m(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate length of a linestring in meters using UTM projection"""
        utm_coords = self.linestring_to_utm(coords)
        line = LineString(utm_coords)
        return line.length


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class RiskLevel(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class SurfaceType(Enum):
    UNPAVED = "unpaved"
    GRAVEL = "gravel"
    DIRT = "dirt"
    GROUND = "ground"
    PAVED = "paved"
    ASPHALT = "asphalt"
    UNKNOWN = "unknown"


@dataclass
class RoadSegment:
    """Represents a single road segment from GeoJSON"""
    road_id: str
    name: str
    highway_type: str
    surface: SurfaceType
    surface_known: bool  # True if surface was explicitly tagged
    length_m: float      # Length in meters
    length_km: float     # Length in kilometers
    centroid_lat: float
    centroid_lon: float
    area: str
    coordinates: List[Tuple[float, float]]  # Original lon/lat coordinates
    geometry: LineString  # Shapely geometry (UTM)
    geometry_wgs84: LineString  # Shapely geometry (WGS84)
    
    # Computed properties
    dust_factor: float = 0.0
    connectivity_score: float = 0.0
    importance_index: float = 0.0
    dust_potential: float = 0.0  # dust_factor × length × connectivity
    connected_roads: List[str] = field(default_factory=list)


@dataclass
class RoadMoistureState:
    """Tracks moisture state for a road surface category"""
    moisture_index: float = 0.5  # 0 = bone dry, 1 = saturated
    hours_since_precip: int = 0
    last_precip_mm: float = 0.0
    cumulative_drying: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WeatherForecast:
    """Single hour weather forecast"""
    timestamp: datetime
    temp_c: float
    wind_speed_mps: float
    wind_gust_mps: Optional[float]
    wind_direction_deg: Optional[float]
    humidity_pct: float
    precip_mm: float
    precip_probability: Optional[float]
    visibility_km: Optional[float]
    soil_moisture: Optional[float]  # From Open-Meteo (m³/m³)
    evapotranspiration: Optional[float]  # From Open-Meteo (mm)
    source: str  # "NWS" or "OpenMeteo"


@dataclass
class RoadDustForecast:
    """Dust forecast for a single road"""
    road_id: str
    road_name: str
    area: str
    timestamp: datetime
    horizon_hours: int
    
    # Predictions
    dust_potential: float  # 0-100 scale
    dust_potential_lower: float  # 10th percentile
    dust_potential_upper: float  # 90th percentile
    risk_level: RiskLevel
    
    # Contributing factors
    surface_dust_factor: float
    moisture_index: float
    wind_factor: float
    emission_potential: float  # (1 - moisture) ^ 1.5
    
    # Weather context
    wind_speed_mps: float
    precip_mm: float
    humidity_pct: float


@dataclass
class AreaDustForecast:
    """Aggregated dust forecast for an area"""
    area_name: str
    timestamp: datetime
    horizon_hours: int
    
    # Aggregated predictions
    mean_dust_potential: float
    max_dust_potential: float
    weighted_dust_potential: float  # Weighted by road importance
    risk_level: RiskLevel
    
    # Road breakdown
    total_roads: int
    high_risk_roads: int
    roads_at_risk: List[str]  # Names of roads at YELLOW or RED
    
    # Weather context
    wind_speed_mps: float
    precip_mm: float


# =============================================================================
# ROAD CLASSIFICATION FROM GEOJSON
# =============================================================================

class RoadClassifier:
    """
    Extracts and classifies roads from nome.geojson.
    
    Computes:
    - road_length_km: Accurate length using UTM projection
    - connectivity_score: Based on road network topology
    - road_importance_index: Weighted combination of type, length, connectivity
    - dust_factor: Based on surface type (known or inferred)
    - dust_potential: dust_factor × length × connectivity
    """
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
        self.transformer = CoordinateTransformer(config)
        self.roads: Dict[str, RoadSegment] = {}
        self.areas: Dict[str, List[str]] = defaultdict(list)  # area -> road_ids
        self.connectivity_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Spatial index for fast connectivity lookup
        self._spatial_index = None
        self._endpoint_index: Dict[int, Tuple[str, bool]] = {}  # idx -> (road_id, is_start)
        
    def load_from_geojson(self, geojson_path: str) -> int:
        """
        Load roads from GeoJSON file.
        
        Args:
            geojson_path: Path to nome.geojson
            
        Returns:
            Number of roads loaded
        """
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        # First pass: parse all roads
        for feature in data.get('features', []):
            road = self._parse_feature(feature)
            if road:
                self.roads[road.road_id] = road
                self.areas[road.area].append(road.road_id)
        
        # Build spatial index and compute connectivity
        self._build_spatial_index()
        self._compute_connectivity()
        
        # Compute importance and dust potential
        self._compute_importance()
        self._compute_dust_potential()
        
        return len(self.roads)
    
    def _parse_feature(self, feature: Dict) -> Optional[RoadSegment]:
        """Parse a single GeoJSON feature into a RoadSegment"""
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        
        # Skip non-LineString features
        if geom.get('type') != 'LineString':
            return None
        
        highway_type = props.get('highway', '')
        if not highway_type or highway_type in ['footway', 'path', 'cycleway', 'steps']:
            return None  # Skip pedestrian paths
        
        road_id = props.get('@id', feature.get('id', str(hash(json.dumps(feature)))))
        name = props.get('name', f"Unnamed {highway_type}")

        # Skip roads excluded by Mayor's ground-truth corrections
        if road_id in self.config.EXCLUDED_ROADS_BY_ID or name in self.config.EXCLUDED_ROADS_BY_NAME:
            return None

        # Parse surface
        surface_str = props.get('surface', '').lower()
        surface, surface_known = self._parse_surface(surface_str, highway_type)
        
        # Get coordinates
        coordinates = geom.get('coordinates', [])
        if not coordinates or len(coordinates) < 2:
            return None
        
        # Convert to tuples
        coords = [(c[0], c[1]) for c in coordinates]
        
        # Calculate geometry
        length_m = self.transformer.calculate_length_m(coords)
        length_km = length_m / 1000.0
        
        # Create Shapely geometries
        geometry_wgs84 = LineString(coords)
        utm_coords = self.transformer.linestring_to_utm(coords)
        geometry_utm = LineString(utm_coords)
        
        # Calculate centroid
        centroid = geometry_wgs84.centroid
        centroid_lon, centroid_lat = centroid.x, centroid.y
        
        # Assign area
        area = self._assign_area(centroid_lon, centroid_lat)
        
        # Calculate dust factor
        dust_factor = self._calculate_dust_factor(surface, surface_known, highway_type)

        # Apply Mayor's ground-truth corrections for confirmed dusty roads
        if name in self.config.DUST_FACTOR_OVERRIDES_BY_NAME:
            dust_factor = self.config.DUST_FACTOR_OVERRIDES_BY_NAME[name]

        return RoadSegment(
            road_id=road_id,
            name=name,
            highway_type=highway_type,
            surface=surface,
            surface_known=surface_known,
            length_m=length_m,
            length_km=length_km,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            area=area,
            coordinates=coords,
            geometry=geometry_utm,
            geometry_wgs84=geometry_wgs84,
            dust_factor=dust_factor
        )
    
    def _parse_surface(self, surface_str: str, highway_type: str) -> Tuple[SurfaceType, bool]:
        """Parse surface string into SurfaceType and whether it was explicitly tagged"""
        if not surface_str:
            return SurfaceType.UNKNOWN, False
        
        surface_map = {
            'unpaved': SurfaceType.UNPAVED,
            'gravel': SurfaceType.GRAVEL,
            'dirt': SurfaceType.DIRT,
            'ground': SurfaceType.GROUND,
            'paved': SurfaceType.PAVED,
            'asphalt': SurfaceType.ASPHALT,
            'concrete': SurfaceType.PAVED,
            'cobblestone': SurfaceType.PAVED,
        }
        
        for key, surface_type in surface_map.items():
            if key in surface_str:
                return surface_type, True
        
        return SurfaceType.UNKNOWN, False
    
    def _calculate_dust_factor(self, surface: SurfaceType, surface_known: bool, 
                                highway_type: str) -> float:
        """
        Calculate dust factor based on surface type and highway classification.
        
        Hierarchy:
        - Explicit unpaved/gravel/dirt/ground: 1.0
        - Track: 0.95 (almost certainly dirt)
        - Residential/service (unspecified): 0.75 (many gravel, low traffic)
        - Secondary/tertiary (unspecified): 0.6 (mixed but connected)
        - Primary/arterial: 0.4 (resuspension + winter debris)
        - Explicit paved/asphalt: 0.25 (shoulders + sanding)
        """
        # Explicit surface tags take precedence
        if surface_known:
            if surface in [SurfaceType.UNPAVED, SurfaceType.GRAVEL, 
                          SurfaceType.DIRT, SurfaceType.GROUND]:
                return self.config.DUST_FACTOR_EXPLICIT_UNPAVED
            elif surface in [SurfaceType.PAVED, SurfaceType.ASPHALT]:
                return self.config.DUST_FACTOR_EXPLICIT_PAVED
        
        # Infer from highway type (preserving UNKNOWN status)
        if highway_type == 'track':
            return self.config.DUST_FACTOR_TRACK
        elif highway_type in ['residential', 'service']:
            return self.config.DUST_FACTOR_RESIDENTIAL_UNSPEC
        elif highway_type in ['secondary', 'tertiary', 'unclassified', 'secondary_link', 'tertiary_link']:
            return self.config.DUST_FACTOR_SECONDARY_UNSPEC
        elif highway_type in ['primary', 'trunk', 'primary_link', 'trunk_link']:
            return self.config.DUST_FACTOR_PRIMARY
        
        # Default for unknown
        return self.config.DUST_FACTOR_RESIDENTIAL_UNSPEC
    
    def _assign_area(self, lon: float, lat: float) -> str:
        """Assign a point to an area based on bounding boxes"""
        for area_name, bbox in self.config.AREAS.items():
            if area_name == "Highways":  # Skip highways as catch-all
                continue
            min_lon, min_lat, max_lon, max_lat = bbox
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                return area_name
        return "Highways"  # Default for roads outside defined areas
    
    def _build_spatial_index(self):
        """Build spatial index for efficient endpoint proximity queries"""
        if HAS_RTREE:
            self._spatial_index = rtree_index.Index()
            idx = 0
            for road_id, road in self.roads.items():
                # Index start point
                start = road.geometry.coords[0]
                self._spatial_index.insert(idx, (start[0], start[1], start[0], start[1]))
                self._endpoint_index[idx] = (road_id, True)
                idx += 1
                
                # Index end point
                end = road.geometry.coords[-1]
                self._spatial_index.insert(idx, (end[0], end[1], end[0], end[1]))
                self._endpoint_index[idx] = (road_id, False)
                idx += 1
        else:
            # Fallback: store endpoints in a list
            self._endpoints_list = []
            for road_id, road in self.roads.items():
                start = road.geometry.coords[0]
                end = road.geometry.coords[-1]
                self._endpoints_list.append((road_id, start[0], start[1], True))
                self._endpoints_list.append((road_id, end[0], end[1], False))
    
    def _compute_connectivity(self):
        """
        Compute connectivity score based on road network topology.
        
        Uses spatial index to find roads with endpoints within 50m of each other.
        """
        threshold = self.config.CONNECTIVITY_THRESHOLD_M
        connections = defaultdict(set)
        
        if HAS_RTREE and self._spatial_index:
            # Fast method using R-tree spatial index
            for road_id, road in self.roads.items():
                for point in [road.geometry.coords[0], road.geometry.coords[-1]]:
                    # Query nearby endpoints
                    bbox = (
                        point[0] - threshold,
                        point[1] - threshold,
                        point[0] + threshold,
                        point[1] + threshold
                    )
                    nearby = list(self._spatial_index.intersection(bbox))
                    
                    for idx in nearby:
                        other_id, _ = self._endpoint_index[idx]
                        if other_id != road_id:
                            # Verify distance
                            other_road = self.roads[other_id]
                            for other_point in [other_road.geometry.coords[0], 
                                               other_road.geometry.coords[-1]]:
                                dist = math.sqrt(
                                    (point[0] - other_point[0])**2 + 
                                    (point[1] - other_point[1])**2
                                )
                                if dist < threshold:
                                    connections[road_id].add(other_id)
                                    connections[other_id].add(road_id)
                                    break
        else:
            # Slow method: O(n²) pairwise comparison
            for road_id, road in self.roads.items():
                for other_id, other_road in self.roads.items():
                    if road_id >= other_id:
                        continue
                    
                    # Check all endpoint pairs
                    for p1 in [road.geometry.coords[0], road.geometry.coords[-1]]:
                        for p2 in [other_road.geometry.coords[0], other_road.geometry.coords[-1]]:
                            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            if dist < threshold:
                                connections[road_id].add(other_id)
                                connections[other_id].add(road_id)
                                break
                        else:
                            continue
                        break
        
        # Normalize connectivity score (0-1 based on max connections)
        max_connections = max(len(c) for c in connections.values()) if connections else 1
        
        for road_id, road in self.roads.items():
            num_connections = len(connections.get(road_id, set()))
            road.connectivity_score = num_connections / max_connections if max_connections > 0 else 0
            road.connected_roads = list(connections.get(road_id, set()))
            self.connectivity_graph[road_id] = road.connected_roads
    
    def _compute_importance(self):
        """
        Compute road importance index based on:
        - Highway type (major roads more important)
        - Length (longer roads more important)
        - Connectivity (well-connected roads more important)
        """
        # Highway type importance weights
        type_weights = {
            'primary': 1.0,
            'trunk': 1.0,
            'secondary': 0.8,
            'tertiary': 0.6,
            'unclassified': 0.5,
            'residential': 0.4,
            'service': 0.3,
            'track': 0.2,
            'secondary_link': 0.7,
            'tertiary_link': 0.5,
            'primary_link': 0.9,
        }
        
        # Get max length for normalization
        max_length = max(r.length_km for r in self.roads.values()) if self.roads else 1
        
        for road in self.roads.values():
            type_weight = type_weights.get(road.highway_type, 0.3)
            length_weight = road.length_km / max_length if max_length > 0 else 0
            connectivity_weight = road.connectivity_score
            
            # Weighted combination
            road.importance_index = (
                0.4 * type_weight +
                0.3 * length_weight +
                0.3 * connectivity_weight
            )
    
    def _compute_dust_potential(self):
        """Compute base dust potential: dust_factor × length × connectivity"""
        for road in self.roads.values():
            # Ensure minimum connectivity multiplier
            connectivity_mult = 0.5 + 0.5 * road.connectivity_score
            
            road.dust_potential = (
                road.dust_factor * 
                road.length_km * 
                connectivity_mult
            )
    
    def get_roads_by_area(self, area: str) -> List[RoadSegment]:
        """Get all roads in a specific area"""
        return [self.roads[rid] for rid in self.areas.get(area, [])]
    
    def get_all_roads(self) -> List[RoadSegment]:
        """Get all roads"""
        return list(self.roads.values())
    
    def get_statistics(self) -> Dict:
        """Get summary statistics about road classification"""
        stats = {
            'total_roads': len(self.roads),
            'total_length_km': sum(r.length_km for r in self.roads.values()),
            'by_area': {},
            'by_surface': defaultdict(int),
            'by_surface_known': {'known': 0, 'inferred': 0},
            'by_highway_type': defaultdict(int),
            'dust_factor_distribution': {
                'high (>0.8)': 0,
                'medium (0.5-0.8)': 0,
                'low (<0.5)': 0,
            },
            'connectivity': {
                'mean_connections': 0,
                'max_connections': 0,
                'isolated_roads': 0,
            }
        }
        
        total_connections = 0
        max_connections = 0
        
        for road in self.roads.values():
            # By area
            if road.area not in stats['by_area']:
                stats['by_area'][road.area] = {'count': 0, 'length_km': 0, 'high_dust': 0}
            stats['by_area'][road.area]['count'] += 1
            stats['by_area'][road.area]['length_km'] += road.length_km
            if road.dust_factor >= 0.9:
                stats['by_area'][road.area]['high_dust'] += 1
            
            # By surface
            stats['by_surface'][road.surface.value] += 1
            
            # Surface known vs inferred
            if road.surface_known:
                stats['by_surface_known']['known'] += 1
            else:
                stats['by_surface_known']['inferred'] += 1
            
            # By highway type
            stats['by_highway_type'][road.highway_type] += 1
            
            # Dust factor distribution
            if road.dust_factor > 0.8:
                stats['dust_factor_distribution']['high (>0.8)'] += 1
            elif road.dust_factor >= 0.5:
                stats['dust_factor_distribution']['medium (0.5-0.8)'] += 1
            else:
                stats['dust_factor_distribution']['low (<0.5)'] += 1
            
            # Connectivity stats
            num_conn = len(road.connected_roads)
            total_connections += num_conn
            max_connections = max(max_connections, num_conn)
            if num_conn == 0:
                stats['connectivity']['isolated_roads'] += 1
        
        stats['connectivity']['mean_connections'] = total_connections / len(self.roads) if self.roads else 0
        stats['connectivity']['max_connections'] = max_connections
        
        return stats
    
    def export_to_geojson(self, output_path: str):
        """Export classified roads to GeoJSON with computed properties"""
        features = []
        
        for road in self.roads.values():
            feature = {
                'type': 'Feature',
                'properties': {
                    'road_id': road.road_id,
                    'name': road.name,
                    'highway_type': road.highway_type,
                    'surface': road.surface.value,
                    'surface_known': road.surface_known,
                    'length_km': round(road.length_km, 4),
                    'area': road.area,
                    'dust_factor': round(road.dust_factor, 3),
                    'connectivity_score': round(road.connectivity_score, 3),
                    'importance_index': round(road.importance_index, 3),
                    'dust_potential': round(road.dust_potential, 3),
                    'connected_roads_count': len(road.connected_roads),
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': road.coordinates
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'generated': datetime.now(timezone.utc).isoformat(),
                'total_roads': len(self.roads),
                'total_length_km': round(sum(r.length_km for r in self.roads.values()), 2),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return output_path


# =============================================================================
# WEATHER FORECAST CLIENTS
# =============================================================================

class NWSForecastClient:
    """
    National Weather Service Forecast API client.
    Primary source for weather forecasts.
    
    API Documentation: https://www.weather.gov/documentation/services-web-api
    """
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
        self.headers = {"User-Agent": config.NWS_USER_AGENT}
        self._gridpoint_url = None
        self._gridpoint_cache_time = None
        
    def _get_gridpoint_url(self) -> str:
        """Get the gridpoint forecast URL for Nome"""
        # Cache gridpoint URL for 24 hours
        if self._gridpoint_url and self._gridpoint_cache_time:
            if datetime.now(timezone.utc) - self._gridpoint_cache_time < timedelta(hours=24):
                return self._gridpoint_url
        
        points_url = f"https://api.weather.gov/points/{self.config.NOME_LAT},{self.config.NOME_LON}"
        
        try:
            response = requests.get(points_url, headers=self.headers, 
                                   timeout=self.config.NWS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            self._gridpoint_url = data['properties']['forecastHourly']
            self._gridpoint_cache_time = datetime.now(timezone.utc)
            return self._gridpoint_url
            
        except Exception as e:
            print(f"NWS gridpoint lookup failed: {e}")
            # Fallback URL for Nome (PAFG office)
            return "https://api.weather.gov/gridpoints/PAFG/45,157/forecast/hourly"
    
    def fetch_hourly_forecast(self, hours: int = 72) -> List[WeatherForecast]:
        """
        Fetch hourly weather forecast from NWS.
        
        Args:
            hours: Number of hours to fetch (max ~156 from NWS)
            
        Returns:
            List of WeatherForecast objects
        """
        try:
            url = self._get_gridpoint_url()
            response = requests.get(url, headers=self.headers, 
                                   timeout=self.config.NWS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            periods = data.get('properties', {}).get('periods', [])
            
            for period in periods[:hours]:
                forecast = self._parse_nws_period(period)
                if forecast:
                    forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            print(f"NWS forecast fetch failed: {e}")
            return []
    
    def _parse_nws_period(self, period: Dict) -> Optional[WeatherForecast]:
        """Parse a single NWS forecast period"""
        try:
            # Parse timestamp
            start_time = period.get('startTime', '')
            ts = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            
            # Parse temperature (NWS gives in Fahrenheit by default)
            temp_value = period.get('temperature', 32)
            temp_unit = period.get('temperatureUnit', 'F')
            if temp_unit == 'F':
                temp_c = (temp_value - 32) * 5 / 9
            else:
                temp_c = temp_value
            
            # Parse wind speed (NWS format: "5 to 10 mph" or "10 mph")
            wind_str = period.get('windSpeed', '0 mph')
            wind_mps = self._parse_wind_speed(wind_str)
            
            # Parse wind direction
            wind_dir_str = period.get('windDirection', '')
            wind_dir = self._parse_wind_direction(wind_dir_str)
            
            # Parse humidity (may not always be present)
            humidity = period.get('relativeHumidity', {}).get('value', 70)
            if humidity is None:
                humidity = 70  # Default
            
            # Parse precipitation probability
            precip_prob = period.get('probabilityOfPrecipitation', {}).get('value', 0)
            if precip_prob is None:
                precip_prob = 0
            
            # Estimate precipitation amount from probability and short forecast
            short_forecast = period.get('shortForecast', '').lower()
            precip_mm = self._estimate_precip(precip_prob, short_forecast)
            
            return WeatherForecast(
                timestamp=ts,
                temp_c=temp_c,
                wind_speed_mps=wind_mps,
                wind_gust_mps=None,  # NWS hourly doesn't always include gusts
                wind_direction_deg=wind_dir,
                humidity_pct=humidity,
                precip_mm=precip_mm,
                precip_probability=precip_prob,
                visibility_km=None,
                soil_moisture=None,
                evapotranspiration=None,
                source="NWS"
            )
            
        except Exception as e:
            print(f"Error parsing NWS period: {e}")
            return None
    
    def _parse_wind_speed(self, wind_str: str) -> float:
        """Parse NWS wind speed string to m/s"""
        try:
            wind_str = wind_str.lower().replace('mph', '').strip()
            if 'to' in wind_str:
                parts = wind_str.split('to')
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                mph = (low + high) / 2
            else:
                mph = float(wind_str) if wind_str else 0
            
            # Convert mph to m/s
            return mph * 0.44704
            
        except:
            return 0.0
    
    def _parse_wind_direction(self, dir_str: str) -> Optional[float]:
        """Parse wind direction string to degrees"""
        directions = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        return directions.get(dir_str.upper())
    
    def _estimate_precip(self, prob: float, short_forecast: str) -> float:
        """Estimate precipitation amount from probability and forecast text"""
        # Check for precipitation keywords
        heavy_keywords = ['heavy rain', 'heavy snow', 'thunderstorm', 'downpour']
        moderate_keywords = ['rain', 'snow', 'showers', 'drizzle']
        light_keywords = ['slight chance', 'isolated', 'patchy']
        
        base_amount = 0.0
        
        if prob >= 80:
            base_amount = 2.0
        elif prob >= 60:
            base_amount = 1.0
        elif prob >= 40:
            base_amount = 0.5
        elif prob >= 20:
            base_amount = 0.1
        
        # Adjust based on forecast text
        for keyword in heavy_keywords:
            if keyword in short_forecast:
                return base_amount * 2.0
        
        for keyword in light_keywords:
            if keyword in short_forecast:
                return base_amount * 0.5
        
        return base_amount


class OpenMeteoClient:
    """
    Open-Meteo API client.
    Provides soil moisture, evapotranspiration, and backup weather forecasts.
    
    API Documentation: https://open-meteo.com/en/docs
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
    
    def fetch_forecast_with_soil(self, hours: int = 72) -> List[WeatherForecast]:
        """
        Fetch weather forecast including soil moisture and evapotranspiration.
        
        Args:
            hours: Number of hours to fetch
            
        Returns:
            List of WeatherForecast objects
        """
        params = {
            'latitude': self.config.NOME_LAT,
            'longitude': self.config.NOME_LON,
            'hourly': ','.join([
                'temperature_2m',
                'relative_humidity_2m',
                'dew_point_2m',
                'wind_speed_10m',
                'wind_gusts_10m',
                'wind_direction_10m',
                'precipitation',
                'precipitation_probability',
                'visibility',
                'soil_moisture_0_to_1cm',
                'soil_moisture_1_to_3cm',
                'soil_temperature_0cm',
                'evapotranspiration',
                'et0_fao_evapotranspiration',
            ]),
            'forecast_days': max(1, (hours // 24) + 1),
            'timezone': 'America/Anchorage',
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, 
                                   timeout=self.config.OPEN_METEO_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_hourly_data(data, hours)
            
        except Exception as e:
            print(f"Open-Meteo forecast fetch failed: {e}")
            return []
    
    def fetch_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data for training.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with hourly historical data
        """
        params = {
            'latitude': self.config.NOME_LAT,
            'longitude': self.config.NOME_LON,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join([
                'temperature_2m',
                'relative_humidity_2m',
                'wind_speed_10m',
                'wind_gusts_10m',
                'wind_direction_10m',
                'precipitation',
                'soil_moisture_0_to_1cm',
                'evapotranspiration',
            ]),
            'timezone': 'America/Anchorage',
        }
        
        try:
            response = requests.get(self.HISTORICAL_URL, params=params, 
                                   timeout=60)  # Longer timeout for historical
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            if not hourly:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly.get('time', [])),
                'temp_c': hourly.get('temperature_2m', []),
                'humidity_pct': hourly.get('relative_humidity_2m', []),
                'wind_speed_mps': [w / 3.6 if w is not None else None 
                                   for w in hourly.get('wind_speed_10m', [])],
                'wind_gust_mps': [w / 3.6 if w is not None else None 
                                  for w in hourly.get('wind_gusts_10m', [])],
                'wind_direction_deg': hourly.get('wind_direction_10m', []),
                'precip_mm': hourly.get('precipitation', []),
                'soil_moisture': hourly.get('soil_moisture_0_to_1cm', []),
                'evapotranspiration': hourly.get('evapotranspiration', []),
            })
            
            return df.set_index('timestamp')
            
        except Exception as e:
            print(f"Open-Meteo historical fetch failed: {e}")
            return pd.DataFrame()
    
    def _parse_hourly_data(self, data: Dict, max_hours: int) -> List[WeatherForecast]:
        """Parse Open-Meteo hourly response"""
        hourly = data.get('hourly', {})
        if not hourly:
            return []
        
        times = hourly.get('time', [])
        forecasts = []
        
        for i, time_str in enumerate(times[:max_hours]):
            try:
                # Parse timestamp
                ts = datetime.fromisoformat(time_str)
                if ts.tzinfo is None:
                    # Assume local timezone (America/Anchorage)
                    ts = ts.replace(tzinfo=timezone(timedelta(hours=-9)))
                
                # Get values with None handling
                def safe_get(key, default=None):
                    val = hourly.get(key, [])
                    return val[i] if i < len(val) and val[i] is not None else default
                
                # Wind speed from km/h to m/s
                wind_kmh = safe_get('wind_speed_10m', 0)
                wind_mps = wind_kmh / 3.6 if wind_kmh else 0
                
                gust_kmh = safe_get('wind_gusts_10m')
                gust_mps = gust_kmh / 3.6 if gust_kmh else None
                
                forecast = WeatherForecast(
                    timestamp=ts,
                    temp_c=safe_get('temperature_2m', 0),
                    wind_speed_mps=wind_mps,
                    wind_gust_mps=gust_mps,
                    wind_direction_deg=safe_get('wind_direction_10m'),
                    humidity_pct=safe_get('relative_humidity_2m', 70),
                    precip_mm=safe_get('precipitation', 0),
                    precip_probability=safe_get('precipitation_probability'),
                    visibility_km=safe_get('visibility'),
                    soil_moisture=safe_get('soil_moisture_0_to_1cm'),
                    evapotranspiration=safe_get('evapotranspiration'),
                    source="OpenMeteo"
                )
                forecasts.append(forecast)
                
            except Exception as e:
                print(f"Error parsing Open-Meteo hour {i}: {e}")
                continue
        
        return forecasts


class WeatherForecastManager:
    """
    Manages weather forecasts from multiple sources.
    Uses NWS as primary, Open-Meteo as backup and for soil data.
    """
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
        self.nws_client = NWSForecastClient(config)
        self.open_meteo_client = OpenMeteoClient(config)
        self._cached_forecasts: List[WeatherForecast] = []
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=30)
    
    def get_forecasts(self, hours: int = 72, force_refresh: bool = False) -> List[WeatherForecast]:
        """
        Get weather forecasts, using cache if available.
        
        Strategy:
        1. Try NWS first (primary source)
        2. Always get Open-Meteo for soil moisture/ET
        3. Merge soil data from Open-Meteo into NWS data
        4. If NWS fails, use Open-Meteo alone
        """
        # Check cache
        if not force_refresh and self._cache_valid():
            return self._cached_forecasts[:hours]
        
        # Try NWS first
        print("Fetching NWS forecast...")
        nws_forecasts = self.nws_client.fetch_hourly_forecast(hours)
        
        # Always get Open-Meteo for soil moisture/ET
        print("Fetching Open-Meteo forecast (soil moisture + ET)...")
        open_meteo_forecasts = self.open_meteo_client.fetch_forecast_with_soil(hours)
        
        # Merge data
        if nws_forecasts:
            print(f"Using NWS as primary ({len(nws_forecasts)} hours)")
            forecasts = self._merge_forecasts(nws_forecasts, open_meteo_forecasts)
        elif open_meteo_forecasts:
            print(f"Falling back to Open-Meteo ({len(open_meteo_forecasts)} hours)")
            forecasts = open_meteo_forecasts
        else:
            print("WARNING: No weather data available from any source")
            forecasts = []
        
        # Update cache
        self._cached_forecasts = forecasts
        self._cache_time = datetime.now(timezone.utc)
        
        return forecasts
    
    def _cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_time or not self._cached_forecasts:
            return False
        return datetime.now(timezone.utc) - self._cache_time < self._cache_duration
    
    def _merge_forecasts(self, nws: List[WeatherForecast], 
                         open_meteo: List[WeatherForecast]) -> List[WeatherForecast]:
        """Merge NWS forecasts with Open-Meteo soil data"""
        if not open_meteo:
            return nws
        
        # Index Open-Meteo by hour
        om_by_hour = {}
        for f in open_meteo:
            # Normalize to hour
            hour_key = f.timestamp.replace(minute=0, second=0, microsecond=0)
            om_by_hour[hour_key] = f
        
        # Merge soil data into NWS
        for nws_forecast in nws:
            hour_key = nws_forecast.timestamp.replace(minute=0, second=0, microsecond=0)
            
            # Try exact match first
            om = om_by_hour.get(hour_key)
            
            # If no exact match, try nearby hours (within 2 hours)
            if not om:
                for delta in [1, -1, 2, -2]:
                    nearby_key = hour_key + timedelta(hours=delta)
                    if nearby_key in om_by_hour:
                        om = om_by_hour[nearby_key]
                        break
            
            if om:
                nws_forecast.soil_moisture = om.soil_moisture
                nws_forecast.evapotranspiration = om.evapotranspiration
                
                # Fill in missing NWS data from Open-Meteo
                if nws_forecast.visibility_km is None:
                    nws_forecast.visibility_km = om.visibility_km
                if nws_forecast.wind_gust_mps is None:
                    nws_forecast.wind_gust_mps = om.wind_gust_mps
        
        return nws


# =============================================================================
# ROAD MOISTURE MODEL
# =============================================================================

class RoadMoistureModel:
    """
    Physics-based road surface moisture model.
    
    Combines:
    1. Open-Meteo soil moisture as baseline validation
    2. Physics-based drying model using precipitation, temp, wind, humidity
    3. Blending between approaches for robust estimates
    
    Tracks moisture state by surface category (unpaved roads dry differently than paved).
    """
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
        
        # Track moisture state by surface category
        self.moisture_states: Dict[str, RoadMoistureState] = {
            'unpaved': RoadMoistureState(moisture_index=0.3),
            'gravel': RoadMoistureState(moisture_index=0.3),
            'paved': RoadMoistureState(moisture_index=0.2),
            'unknown': RoadMoistureState(moisture_index=0.3),
        }
        
        # Precipitation history for "hours since significant precip"
        self._precip_history: List[Tuple[datetime, float]] = []
        
        # Open-Meteo soil moisture history for validation
        self._om_soil_history: List[Tuple[datetime, float]] = []
    
    def update_from_weather(self, weather: WeatherForecast):
        """Update moisture state based on weather observation/forecast"""
        
        # Track precipitation history
        self._precip_history.append((weather.timestamp, weather.precip_mm))
        
        # Keep only last 72 hours
        cutoff = weather.timestamp - timedelta(hours=72)
        self._precip_history = [(t, p) for t, p in self._precip_history if t > cutoff]
        
        # Track Open-Meteo soil moisture
        if weather.soil_moisture is not None:
            self._om_soil_history.append((weather.timestamp, weather.soil_moisture))
            self._om_soil_history = [(t, m) for t, m in self._om_soil_history if t > cutoff]
        
        # Update each surface type
        for surface_type, state in self.moisture_states.items():
            self._update_surface_moisture(state, weather, surface_type)
    
    def _update_surface_moisture(self, state: RoadMoistureState, 
                                  weather: WeatherForecast, surface_type: str):
        """Update moisture for a single surface type"""
        
        # 1. Add moisture from precipitation
        if weather.precip_mm > 0:
            precip_contribution = min(
                1.0 - state.moisture_index,
                weather.precip_mm / self.config.PRECIP_SATURATION_MM
            )
            state.moisture_index = min(1.0, state.moisture_index + precip_contribution)
            state.hours_since_precip = 0
            state.last_precip_mm = weather.precip_mm
        else:
            state.hours_since_precip += 1
        
        # 2. Calculate drying rate
        drying_rate = self._calculate_drying_rate(
            temp_c=weather.temp_c,
            wind_mps=weather.wind_speed_mps,
            humidity_pct=weather.humidity_pct,
            et=weather.evapotranspiration,
            surface_type=surface_type
        )
        
        # 3. Apply drying
        state.moisture_index = max(0.0, state.moisture_index - drying_rate)
        state.cumulative_drying += drying_rate
        state.last_update = weather.timestamp
        
        # 4. Validate against Open-Meteo soil moisture (if available)
        if weather.soil_moisture is not None:
            self._validate_with_open_meteo(state, weather.soil_moisture)
    
    def _calculate_drying_rate(self, temp_c: float, wind_mps: float, 
                                humidity_pct: float, et: Optional[float],
                                surface_type: str) -> float:
        """
        Calculate hourly drying rate based on atmospheric conditions.
        
        Physics:
        - Higher temperature → faster evaporation
        - Higher wind → faster evaporation (convective removal)
        - Higher humidity → slower evaporation (reduced vapor pressure gradient)
        - Frozen conditions → minimal drying
        """
        # Base drying rate (hourly)
        drying_rate = self.config.BASE_DRYING_RATE
        
        # Temperature factor
        if temp_c > 10:
            # Warmer = faster drying
            drying_rate += self.config.TEMP_DRYING_FACTOR * (temp_c - 10)
        elif temp_c < 0:
            # Frozen conditions = very slow drying
            drying_rate *= 0.2
        elif temp_c < 5:
            # Cold but not frozen = slow drying
            drying_rate *= 0.5
        
        # Wind factor
        if wind_mps > 2:
            # Wind increases evaporation
            drying_rate += self.config.WIND_DRYING_FACTOR * (wind_mps - 2)
        
        # Humidity factor
        if humidity_pct > 50:
            # High humidity reduces evaporation
            drying_rate += self.config.HUMIDITY_DRYING_FACTOR * (humidity_pct - 50)
        elif humidity_pct < 30:
            # Very dry air accelerates evaporation
            drying_rate += 0.01 * (30 - humidity_pct)
        
        # Surface type adjustment
        surface_factors = {
            'unpaved': 1.0,      # Baseline
            'gravel': 1.1,      # Slightly faster (better drainage)
            'paved': 1.4,       # Faster (less absorption, water runs off)
            'unknown': 1.0,     # Assume unpaved behavior
        }
        drying_rate *= surface_factors.get(surface_type, 1.0)
        
        # Validate/adjust with Open-Meteo ET if available
        if et is not None and et > 0:
            # ET is in mm/hour - convert to drying rate
            # 5mm = full saturation, so 1mm ET ≈ 0.2 drying
            et_based_rate = et / self.config.PRECIP_SATURATION_MM
            
            # Blend: 60% physics model, 40% ET measurement
            # (ET is measured evaporation, but for grass not roads)
            drying_rate = 0.6 * drying_rate + 0.4 * et_based_rate
        
        # Clamp to reasonable range (0% to 15% per hour)
        return max(0.0, min(0.15, drying_rate))
    
    def _validate_with_open_meteo(self, state: RoadMoistureState, om_moisture: float):
        """
        Validate and gently correct our physics model against Open-Meteo soil moisture.
        
        Open-Meteo gives soil_moisture_0_to_1cm in m³/m³, typically 0.0 to 0.4+
        We need to normalize this to our 0-1 moisture index scale.
        """
        # Normalize Open-Meteo (typical range 0-0.4 m³/m³ → 0-1)
        # 0.4 m³/m³ is quite wet soil
        om_normalized = min(1.0, om_moisture / 0.4)
        
        # Check for significant deviation
        deviation = abs(state.moisture_index - om_normalized)
        
        if deviation > 0.3:
            # Large deviation - blend more aggressively toward Open-Meteo
            # (our physics model might be drifting)
            state.moisture_index = 0.5 * state.moisture_index + 0.5 * om_normalized
        elif deviation > 0.15:
            # Moderate deviation - gentle correction
            state.moisture_index = 0.7 * state.moisture_index + 0.3 * om_normalized
        # Small deviation (<0.15) - trust our physics model
    
    def get_moisture_index(self, surface_type: SurfaceType) -> float:
        """Get current moisture index for a surface type"""
        if surface_type in [SurfaceType.UNPAVED, SurfaceType.DIRT, SurfaceType.GROUND]:
            return self.moisture_states['unpaved'].moisture_index
        elif surface_type == SurfaceType.GRAVEL:
            return self.moisture_states['gravel'].moisture_index
        elif surface_type in [SurfaceType.PAVED, SurfaceType.ASPHALT]:
            return self.moisture_states['paved'].moisture_index
        else:
            return self.moisture_states['unknown'].moisture_index
    
    def get_emission_potential(self, surface_type: SurfaceType, temp_c: float) -> float:
        if temp_c < 0:
            return 0.0  # Frozen ground = no dust emission
        moisture = self.get_moisture_index(surface_type)
        return (1 - moisture) ** 1.5

    
    def get_hours_since_significant_precip(self) -> int:
        """Get hours since precipitation >= 0.5mm"""
        if not self._precip_history:
            return 999  # Unknown - assume very dry
        
        now = self._precip_history[-1][0]
        
        for ts, precip in reversed(self._precip_history):
            if precip >= self.config.PRECIP_SUPPRESSION_MM:
                return int((now - ts).total_seconds() / 3600)
        
        return len(self._precip_history)
    
    def get_state_summary(self) -> Dict[str, Dict]:
        """Get summary of moisture state for all surface types"""
        return {
            surface: {
                'moisture_index': round(state.moisture_index, 3),
                'hours_since_precip': state.hours_since_precip,
                'emission_potential': round((1 - state.moisture_index) ** 1.5, 3),
                'last_precip_mm': round(state.last_precip_mm, 1),
                'cumulative_drying': round(state.cumulative_drying, 3),
            }
            for surface, state in self.moisture_states.items()
        }
    
    def reset(self):
        """Reset moisture model to initial state"""
        for state in self.moisture_states.values():
            state.moisture_index = 0.3
            state.hours_since_precip = 0
            state.last_precip_mm = 0.0
            state.cumulative_drying = 0.0
        self._precip_history.clear()
        self._om_soil_history.clear()


# =============================================================================
# DUST FORECAST ENGINE
# =============================================================================

class DustForecastEngine:
    """
    Main engine for generating road-specific and area-based dust forecasts.
    
    Combines:
    - Road classification (surface type, length, connectivity, importance)
    - Weather forecasts (NWS + Open-Meteo)
    - Road moisture model (physics + Open-Meteo validation)
    
    Outputs:
    - Individual road forecasts (451 roads × N hours)
    - Area-aggregated forecasts (6 areas × N hours)
    """
    
    def __init__(self, config: DustForecastConfig = CONFIG):
        self.config = config
        self.road_classifier = RoadClassifier(config)
        self.weather_manager = WeatherForecastManager(config)
        self.moisture_model = RoadMoistureModel(config)
        self._initialized = False
    
    def initialize(self, geojson_path: str) -> int:
        """
        Initialize with road data from GeoJSON.
        
        Args:
            geojson_path: Path to nome.geojson
            
        Returns:
            Number of roads loaded
        """
        print(f"Loading roads from {geojson_path}...")
        num_roads = self.road_classifier.load_from_geojson(geojson_path)
        self._initialized = True
        
        stats = self.road_classifier.get_statistics()
        print(f"Loaded {num_roads} roads ({stats['total_length_km']:.1f} km total)")
        print(f"  Surface known: {stats['by_surface_known']['known']}")
        print(f"  Surface inferred: {stats['by_surface_known']['inferred']}")
        
        return num_roads
    
    def generate_forecast(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate complete dust forecast for all roads and areas.
        
        Args:
            hours: Forecast horizon (24, 48, or 72)
            
        Returns:
            Dict with:
            - timestamp: Generation time
            - forecast_hours: Requested horizon
            - current_conditions: Current weather
            - road_forecasts: List of per-road forecasts
            - area_forecasts: List of per-area aggregated forecasts
            - road_statistics: Summary of road classification
            - moisture_state: Current road moisture model state
        """
        if not self._initialized:
            raise ValueError("Engine not initialized. Call initialize() first.")
        
        # Get weather forecasts
        weather_forecasts = self.weather_manager.get_forecasts(hours)
        
        if not weather_forecasts:
            return {
                'error': 'No weather data available',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'forecast_hours': hours,
            }
        
        print(f"Generating {hours}-hour forecast for {len(self.road_classifier.roads)} roads...")
        
        # Reset moisture model for fresh forecast
        self.moisture_model.reset()
        
        # Generate road-specific forecasts for each hour
        road_forecasts = []
        
        for hour_idx, weather in enumerate(weather_forecasts):
            # Update moisture model with this hour's weather
            self.moisture_model.update_from_weather(weather)
            
            # Generate forecast for each road at this hour
            for road in self.road_classifier.get_all_roads():
                forecast = self._generate_road_forecast(road, weather, hour_idx + 1)
                road_forecasts.append(forecast)
        
        # Aggregate to area forecasts
        area_forecasts = self._aggregate_to_areas(road_forecasts, weather_forecasts)
        
        # Current conditions (first hour)
        current_weather = weather_forecasts[0]
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'forecast_hours': hours,
            'weather_source': current_weather.source,
            'current_conditions': self._format_current_conditions(current_weather),
            'road_forecasts': [self._format_road_forecast(f) for f in road_forecasts],
            'area_forecasts': [self._format_area_forecast(f) for f in area_forecasts],
            'road_statistics': self.road_classifier.get_statistics(),
            'moisture_state': self.moisture_model.get_state_summary(),
        }
    
    def _generate_road_forecast(self, road: RoadSegment, weather: WeatherForecast,
                                 horizon_hours: int) -> RoadDustForecast:
        """Generate dust forecast for a single road at a specific hour"""
        
        # Get moisture-based emission potential
        moisture_index = self.moisture_model.get_moisture_index(road.surface)
        emission_potential = self.moisture_model.get_emission_potential(road.surface, weather.temp_c)
        
        # Calculate wind factor (entrainment)
        wind_factor = self._calculate_wind_factor(weather.wind_speed_mps, weather.wind_gust_mps)
        
        # Calculate base dust potential
        # Combines: surface type × emission potential × wind × importance
        dust_potential = (
            road.dust_factor *          # Surface type factor (0.25-1.0)
            emission_potential *        # Moisture-based (0-1)
            wind_factor *              # Wind entrainment (0-1)
            (0.5 + 0.5 * road.importance_index)  # Importance scaling
        ) * 100  # Scale to 0-100
        
        # Add uncertainty bounds
        # Uncertainty increases with:
        # 1. Forecast horizon (further = more uncertain)
        # 2. Unknown surface type (inferred = more uncertain)
        # 3. Low connectivity (isolated roads = less data)
        
        base_uncertainty = 0.2  # ±20% base
        horizon_uncertainty = (horizon_hours / 72) * 0.3  # Up to ±30% at 72h
        surface_uncertainty = 0.1 if not road.surface_known else 0.0  # ±10% if inferred
        
        total_uncertainty = base_uncertainty + horizon_uncertainty + surface_uncertainty
        
        dust_potential_lower = dust_potential * (1 - total_uncertainty)
        dust_potential_upper = dust_potential * (1 + total_uncertainty)
        
        # Clamp to valid range
        dust_potential = np.clip(dust_potential, 0, 100)
        dust_potential_lower = np.clip(dust_potential_lower, 0, 100)
        dust_potential_upper = np.clip(dust_potential_upper, 0, 100)
        
        # Determine risk level (conservative - use upper bound influence)
        risk_level = self._determine_risk_level(dust_potential, dust_potential_upper)
        
        return RoadDustForecast(
            road_id=road.road_id,
            road_name=road.name,
            area=road.area,
            timestamp=weather.timestamp,
            horizon_hours=horizon_hours,
            dust_potential=float(dust_potential),
            dust_potential_lower=float(dust_potential_lower),
            dust_potential_upper=float(dust_potential_upper),
            risk_level=risk_level,
            surface_dust_factor=road.dust_factor,
            moisture_index=moisture_index,
            wind_factor=wind_factor,
            emission_potential=emission_potential,
            wind_speed_mps=weather.wind_speed_mps,
            precip_mm=weather.precip_mm,
            humidity_pct=weather.humidity_pct,
        )
    
    def _calculate_wind_factor(self, wind_mps: float, gust_mps: Optional[float]) -> float:
        """
        Calculate wind contribution to dust entrainment.
        
        Uses sustained wind speed with gust consideration.
        """
        # Use gust if available and significant
        effective_wind = wind_mps
        if gust_mps and gust_mps > wind_mps * 1.3:
            # Gusts contribute to dust lifting
            effective_wind = 0.7 * wind_mps + 0.3 * gust_mps
        
        # Piecewise linear wind factor
        if effective_wind < 2:
            # Very light wind - minimal dust
            return effective_wind / 2 * 0.1
        elif effective_wind < self.config.WIND_ENTRAINMENT_MPS:
            # Below entrainment threshold - low dust
            return 0.1 + (effective_wind - 2) / 3 * 0.3
        elif effective_wind < self.config.WIND_STRONG_MPS:
            # Entrainment zone - moderate dust
            return 0.4 + (effective_wind - 5) / 3 * 0.3
        else:
            # Strong wind - high dust (capped at 1.0)
            return min(1.0, 0.7 + (effective_wind - 8) / 10 * 0.3)
    
    def _determine_risk_level(self, dust_potential: float, upper_bound: float) -> RiskLevel:
        """
        Determine risk level from dust potential.
        
        Uses conservative estimate (weighted toward upper bound).
        """
        # Conservative: 70% median, 30% upper bound
        effective = 0.7 * dust_potential + 0.3 * upper_bound
        
        if effective >= 50:
            return RiskLevel.RED
        elif effective >= 25:
            return RiskLevel.YELLOW
        else:
            return RiskLevel.GREEN
    
    def _aggregate_to_areas(self, road_forecasts: List[RoadDustForecast],
                            weather_forecasts: List[WeatherForecast]) -> List[AreaDustForecast]:
        """Aggregate road forecasts to area-level forecasts"""
        
        # Group forecasts by (area, hour)
        by_area_hour = defaultdict(list)
        for rf in road_forecasts:
            key = (rf.area, rf.horizon_hours)
            by_area_hour[key].append(rf)
        
        area_forecasts = []
        
        for (area, hour), forecasts in by_area_hour.items():
            if not forecasts:
                continue
            
            # Get weather for this hour
            weather = weather_forecasts[hour - 1] if hour <= len(weather_forecasts) else weather_forecasts[-1]
            
            # Calculate aggregated metrics
            potentials = [f.dust_potential for f in forecasts]
            
            # Get importance for weighting
            importances = []
            for f in forecasts:
                road = self.road_classifier.roads.get(f.road_id)
                importances.append(road.importance_index if road else 0.5)
            
            # Weighted average by importance
            total_importance = sum(importances) or 1
            weighted_potential = sum(p * i for p, i in zip(potentials, importances)) / total_importance
            
            # Count high risk roads
            high_risk = [f for f in forecasts if f.risk_level in [RiskLevel.YELLOW, RiskLevel.RED]]
            red_risk = [f for f in forecasts if f.risk_level == RiskLevel.RED]
            
            # Determine area risk level
            if len(red_risk) > 0:
                area_risk = RiskLevel.RED
            elif len(high_risk) > len(forecasts) * 0.3:
                area_risk = RiskLevel.YELLOW
            elif weighted_potential >= 25:
                area_risk = RiskLevel.YELLOW
            else:
                area_risk = RiskLevel.GREEN
            
            area_forecasts.append(AreaDustForecast(
                area_name=area,
                timestamp=weather.timestamp,
                horizon_hours=hour,
                mean_dust_potential=np.mean(potentials),
                max_dust_potential=np.max(potentials),
                weighted_dust_potential=weighted_potential,
                risk_level=area_risk,
                total_roads=len(forecasts),
                high_risk_roads=len(high_risk),
                roads_at_risk=[f.road_name for f in sorted(high_risk, 
                              key=lambda x: x.dust_potential, reverse=True)[:10]],
                wind_speed_mps=weather.wind_speed_mps,
                precip_mm=weather.precip_mm,
            ))
        
        return area_forecasts
    
    def _format_current_conditions(self, weather: WeatherForecast) -> Dict:
        """Format current conditions for JSON output"""
        return {
            'timestamp': weather.timestamp.isoformat(),
            'temperature_c': round(weather.temp_c, 1),
            'wind_speed_mps': round(weather.wind_speed_mps, 1),
            'wind_gust_mps': round(weather.wind_gust_mps, 1) if weather.wind_gust_mps else None,
            'wind_direction_deg': weather.wind_direction_deg,
            'humidity_pct': round(weather.humidity_pct, 0),
            'precipitation_mm': round(weather.precip_mm, 1),
            'precipitation_probability': weather.precip_probability,
            'visibility_km': round(weather.visibility_km, 1) if weather.visibility_km else None,
            'soil_moisture': round(weather.soil_moisture, 3) if weather.soil_moisture else None,
            'evapotranspiration_mm': round(weather.evapotranspiration, 2) if weather.evapotranspiration else None,
            'source': weather.source,
        }
    
    def _format_road_forecast(self, forecast: RoadDustForecast) -> Dict:
        """Format road forecast for JSON output"""
        return {
            'road_id': forecast.road_id,
            'road_name': forecast.road_name,
            'area': forecast.area,
            'timestamp': forecast.timestamp.isoformat(),
            'horizon_hours': forecast.horizon_hours,
            'dust_potential': round(forecast.dust_potential, 1),
            'dust_potential_lower': round(forecast.dust_potential_lower, 1),
            'dust_potential_upper': round(forecast.dust_potential_upper, 1),
            'risk_level': forecast.risk_level.value,
            'factors': {
                'surface_dust_factor': round(forecast.surface_dust_factor, 2),
                'moisture_index': round(forecast.moisture_index, 3),
                'wind_factor': round(forecast.wind_factor, 3),
                'emission_potential': round(forecast.emission_potential, 3),
            },
            'weather': {
                'wind_speed_mps': round(forecast.wind_speed_mps, 1),
                'precip_mm': round(forecast.precip_mm, 1),
                'humidity_pct': round(forecast.humidity_pct, 0),
            }
        }
    
    def _format_area_forecast(self, forecast: AreaDustForecast) -> Dict:
        """Format area forecast for JSON output"""
        return {
            'area_name': forecast.area_name,
            'timestamp': forecast.timestamp.isoformat(),
            'horizon_hours': forecast.horizon_hours,
            'dust_potential': {
                'mean': round(forecast.mean_dust_potential, 1),
                'max': round(forecast.max_dust_potential, 1),
                'weighted': round(forecast.weighted_dust_potential, 1),
            },
            'risk_level': forecast.risk_level.value,
            'roads': {
                'total': forecast.total_roads,
                'high_risk': forecast.high_risk_roads,
                'at_risk_names': forecast.roads_at_risk,
            },
            'weather': {
                'wind_speed_mps': round(forecast.wind_speed_mps, 1),
                'precip_mm': round(forecast.precip_mm, 1),
            }
        }
    
    def get_road_summary(self) -> List[Dict]:
        """Get summary of all roads with their properties and dust factors"""
        return [
            {
                'road_id': road.road_id,
                'name': road.name,
                'highway_type': road.highway_type,
                'surface': road.surface.value,
                'surface_known': road.surface_known,
                'length_m': round(road.length_m, 1),
                'length_km': round(road.length_km, 4),
                'area': road.area,
                'dust_factor': round(road.dust_factor, 3),
                'connectivity_score': round(road.connectivity_score, 3),
                'importance_index': round(road.importance_index, 3),
                'dust_potential_base': round(road.dust_potential, 3),
                'connected_roads_count': len(road.connected_roads),
                'centroid': {
                    'lat': round(road.centroid_lat, 6),
                    'lon': round(road.centroid_lon, 6),
                }
            }
            for road in self.road_classifier.get_all_roads()
        ]
    
    def export_roads_geojson(self, output_path: str) -> str:
        """Export classified roads to GeoJSON"""
        return self.road_classifier.export_to_geojson(output_path)
    


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nome Dust Forecast Model')
    parser.add_argument('--geojson', type=str, default='nome.geojson',
                       help='Path to nome.geojson file')
    parser.add_argument('--hours', type=int, default=24,
                       help='Forecast horizon (24, 48, or 72)')
    parser.add_argument('--export-roads', type=str, default=None,
                       help='Export classified roads to GeoJSON')
    parser.add_argument('--output', type=str, default=None,
                       help='Output forecast to JSON file')
    
    args = parser.parse_args()
    
    # Check for geojson file
    if not Path(args.geojson).exists():
        print(f"ERROR: GeoJSON file not found: {args.geojson}")
        return 1
    
    print("=" * 70)
    print("Nome Dust Forecast Model - Production Version")
    print("=" * 70)
    
    # Initialize engine
    engine = DustForecastEngine()
    num_roads = engine.initialize(args.geojson)
    
    # Print statistics
    stats = engine.road_classifier.get_statistics()
    print(f"\n--- Road Statistics ---")
    print(f"Total roads: {stats['total_roads']}")
    print(f"Total length: {stats['total_length_km']:.1f} km")
    
    print(f"\nBy surface type:")
    for surface, count in sorted(stats['by_surface'].items()):
        print(f"  {surface}: {count}")
    
    print(f"\nBy area:")
    for area, info in sorted(stats['by_area'].items()):
        print(f"  {area}: {info['count']} roads, {info['length_km']:.1f} km, "
              f"{info['high_dust']} high-dust")
    
    print(f"\nConnectivity:")
    print(f"  Mean connections: {stats['connectivity']['mean_connections']:.1f}")
    print(f"  Max connections: {stats['connectivity']['max_connections']}")
    print(f"  Isolated roads: {stats['connectivity']['isolated_roads']}")
    
    # Export roads if requested
    if args.export_roads:
        print(f"\nExporting classified roads to {args.export_roads}...")
        engine.export_roads_geojson(args.export_roads)
        print("Done.")
    
    # Generate forecast
    print(f"\n--- Generating {args.hours}-Hour Forecast ---")
    
    try:
        forecast = engine.generate_forecast(hours=args.hours)
        
        if 'error' in forecast:
            print(f"Forecast error: {forecast['error']}")
        else:
            print(f"Forecast generated successfully!")
            print(f"  Weather source: {forecast.get('weather_source', 'N/A')}")
            print(f"  Road forecasts: {len(forecast['road_forecasts'])}")
            print(f"  Area forecasts: {len(forecast['area_forecasts'])}")
            
            # Show current conditions
            current = forecast.get('current_conditions', {})
            if current:
                print(f"\nCurrent conditions:")
                print(f"  Temperature: {current.get('temperature_c', 'N/A')}°C")
                print(f"  Wind: {current.get('wind_speed_mps', 'N/A')} m/s")
                print(f"  Humidity: {current.get('humidity_pct', 'N/A')}%")
                print(f"  Soil moisture: {current.get('soil_moisture', 'N/A')}")
            
            # Show area summary for hour 1
            print(f"\nArea forecast summary (hour 1):")
            hour1_areas = [a for a in forecast['area_forecasts'] if a['horizon_hours'] == 1]
            for area in sorted(hour1_areas, key=lambda x: x['dust_potential']['weighted'], reverse=True):
                print(f"  {area['area_name']}: {area['risk_level']} "
                      f"(weighted: {area['dust_potential']['weighted']:.1f}, "
                      f"max: {area['dust_potential']['max']:.1f})")
            
            # Save to file if requested
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(forecast, f, indent=2, default=str)
                print(f"\nForecast saved to {args.output}")
                
    except Exception as e:
        print(f"Forecast generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())