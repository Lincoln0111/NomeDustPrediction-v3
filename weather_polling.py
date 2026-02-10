import requests
from datetime import datetime, timezone

USER_AGENT = "NomeDustNowcast/1.0 (ruthvikadasaraju@gmail.com)"
STATION = "PAOM"
URL = f"https://api.weather.gov/stations/{STATION}/observations/latest"

def _val(x):
    # NWS returns {"value": <number or null>, ...}
    return None if x is None else x.get("value")

def fetch_latest_weather():
    r = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    props = r.json()["properties"]

    ts = props["timestamp"]  # ISO string in UTC
    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    temp_c = _val(props.get("temperature"))
    wind_kmh = _val(props.get("windSpeed"))
    gust_kmh = _val(props.get("windGust"))
    wind_dir = _val(props.get("windDirection"))
    vis_m = _val(props.get("visibility"))
    rh = _val(props.get("relativeHumidity"))

    # unit conversions
    wind_mps = None if wind_kmh is None else wind_kmh / 3.6
    gust_mps = None if gust_kmh is None else gust_kmh / 3.6
    vis_km = None if vis_m is None else vis_m / 1000.0

    # precipitation often missing here; keep None (or 0.0 if you prefer)
    precip_3h_mm = _val(props.get("precipitationLast3Hours"))

    return {
        "timestamp_utc": ts,
        "temp_c": temp_c,
        "wind_speed_mps": wind_mps,
        "wind_gust_mps": gust_mps,
        "wind_dir_deg": wind_dir,
        "visibility_km": vis_km,
        "humidity_pct": rh,
        "precip_3h_mm": precip_3h_mm,
        "text": props.get("textDescription"),
    }

if __name__ == "__main__":
    print(fetch_latest_weather())
