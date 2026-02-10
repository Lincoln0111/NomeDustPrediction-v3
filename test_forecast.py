#!/usr/bin/env python3
"""
Test script for the dust forecasting system.
Demonstrates how to generate and display multi-hour ahead forecasts.
"""

import sys
from pathlib import Path
from nome_dust_forecast_production import DustNowcastSystem, CONFIG

def test_forecast(horizon=12):
    """Test the forecasting system with a specified horizon."""
    print("=" * 80)
    print("DUST NOWCAST & FORECAST SYSTEM - TEST")
    print("=" * 80)
    
    # Initialize system
    system = DustNowcastSystem()
    
    # Load historical data
    pm_path = Path("Nome-Hourly-Data.csv")
    wx_candidates = [
        Path("70200026617.csv"),
        Path("70200026617 (1).csv"),
    ]
    wx_existing = [p for p in wx_candidates if p.exists()]
    
    if not pm_path.exists():
        print(f"ERROR: Missing PM file: {pm_path.resolve()}")
        return False
    
    if not wx_existing:
        print("ERROR: No NOAA weather CSV found")
        return False
    
    wx_path1 = wx_existing[0]
    wx_path2 = wx_existing[1] if len(wx_existing) > 1 else None
    
    print(f"\n📁 Data Sources:")
    print(f"   PM: {pm_path}")
    print(f"   Weather 1: {wx_path1}")
    if wx_path2:
        print(f"   Weather 2: {wx_path2}")
    
    # Initialize and load data
    print("\n🔄 Loading historical data...")
    system.initialize_from_files(
        str(pm_path),
        str(wx_path1),
        str(wx_path2) if wx_path2 else None
    )
    print(f"   ✓ Loaded {len(system.data)} records")
    
    # Train pattern classifier
    print("\n🧠 Training pattern classifier...")
    system.train_ml_until_latest()
    print("   ✓ Classifier trained")
    
    # Train forecaster
    print("\n🔮 Training quantile forecast models...")
    metrics = system.train_forecaster()
    print(f"   ✓ Forecaster trained")
    print(f"   • MAE (p50): {metrics['mae_p50']:.2f} µg/m³")
    print(f"   • Coverage (p10-p90): {metrics['coverage']:.1%}")
    print(f"   • Training samples: {metrics['samples']}")
    
    # Generate current nowcast
    print("\n📍 Current Nowcast:")
    base, roads = system.nowcast_latest(retrain=False)
    print(f"   Timestamp: {base['timestamp']}")
    print(f"   PM10: {base['pm10']:.1f} µg/m³")
    print(f"   PM2.5: {base['pm25']:.1f} µg/m³")
    print(f"   Severity: {base['severity']:.1f}/100")
    print(f"   Risk Level: {base['risk'].value}")
    print(f"   Dust Probability: {base['dust_prob']:.1%}")
    print(f"   Wind: {base['wind']:.1f} m/s")
    print(f"   Visibility: {base['vis']:.1f} km")
    print(f"   Humidity: {base['hum']:.0f}%")
    
    # Generate forecast
    print(f"\n🔮 {horizon}-Hour Forecast:")
    print("-" * 80)
    forecast_result = system.forecast_timeline(horizon=horizon, retrain=False)
    
    if 'error' in forecast_result:
        print(f"ERROR: {forecast_result['error']}")
        return False
    
    forecasts = forecast_result['forecasts']
    
    # Display forecast table
    print(f"{'Hour':>4} | {'PM10 (p10/p50/p90)':^25} | {'Severity (p10/p50/p90)':^30} | {'Risk':^8}")
    print("-" * 80)
    
    for f in forecasts:
        h = f['horizon_hours']
        pm10_str = f"{f['pm10_p10']:.1f}/{f['pm10_p50']:.1f}/{f['pm10_p90']:.1f}"
        sev_str = f"{f['severity_p10']:.1f}/{f['severity_p50']:.1f}/{f['severity_p90']:.1f}"
        risk_str = f"{f['risk_p50']}"
        
        print(f"{h:>4} | {pm10_str:^25} | {sev_str:^30} | {risk_str:^8}")
    
    # Summary statistics
    print("\n📊 Forecast Summary:")
    pm10_p50_values = [f['pm10_p50'] for f in forecasts]
    severity_p50_values = [f['severity_p50'] for f in forecasts]
    
    print(f"   PM10 Range (p50): {min(pm10_p50_values):.1f} - {max(pm10_p50_values):.1f} µg/m³")
    print(f"   Severity Range (p50): {min(severity_p50_values):.1f} - {max(severity_p50_values):.1f}/100")
    
    risk_counts = {}
    for f in forecasts:
        risk = f['risk_p50']
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print(f"   Risk Distribution:")
    for risk, count in sorted(risk_counts.items()):
        pct = 100 * count / len(forecasts)
        print(f"      {risk}: {count}/{len(forecasts)} ({pct:.0f}%)")
    
    # Highlight any concerning periods
    yellow_periods = [f for f in forecasts if f['risk_p50'] == 'YELLOW']
    red_periods = [f for f in forecasts if f['risk_p50'] == 'RED']
    
    if red_periods:
        print(f"\n⚠️  RED ALERT: {len(red_periods)} hour(s) with high dust risk:")
        for f in red_periods[:3]:  # Show first 3
            print(f"      Hour +{f['horizon_hours']}: PM10={f['pm10_p50']:.1f}, Severity={f['severity_p50']:.1f}")
    elif yellow_periods:
        print(f"\n⚠️  CAUTION: {len(yellow_periods)} hour(s) with moderate dust risk:")
        for f in yellow_periods[:3]:  # Show first 3
            print(f"      Hour +{f['horizon_hours']}: PM10={f['pm10_p50']:.1f}, Severity={f['severity_p50']:.1f}")
    else:
        print(f"\n✅ All forecast hours show GREEN (low) risk")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    
    if horizon < 1 or horizon > 72:
        print("ERROR: Forecast horizon must be between 1 and 72 hours")
        sys.exit(1)
    
    success = test_forecast(horizon)
    sys.exit(0 if success else 1)
