import requests

# Coordinates of Tamil Nadu Districts for real-time weather mapping
DISTRICT_COORDINATES = {
    "COIMBATORE": {"lat": 11.0168, "lon": 76.9558},
    "CHENNAI": {"lat": 13.0827, "lon": 80.2707},
    "MADURAI": {"lat": 9.9252, "lon": 78.1198},
    "ARIYALUR": {"lat": 11.1378, "lon": 79.0722},
    "ERODE": {"lat": 11.3410, "lon": 77.7172},
    "CUDDALORE": {"lat": 11.7480, "lon": 79.7680},
    "DHARMAPURI": {"lat": 12.1275, "lon": 78.1582},
    "DINDIGUL": {"lat": 10.3673, "lon": 77.9803},
    "KANCHIPURAM": {"lat": 12.8342, "lon": 79.7036},
    "KANNYAKUMARI": {"lat": 8.0883, "lon": 77.5385},
    "KANNIYAKUMARI": {"lat": 8.0883, "lon": 77.5385},
    "KARUR": {"lat": 10.9601, "lon": 78.0766},
    "KRISHNAGIRI": {"lat": 12.5186, "lon": 78.2138},
    "NAGAPATTINAM": {"lat": 10.7656, "lon": 79.8424},
    "NAMAKKAL": {"lat": 11.2189, "lon": 78.1674},
    "PERAMBALUR": {"lat": 11.2342, "lon": 78.8787},
    "PUDUKKOTTAI": {"lat": 10.3796, "lon": 78.8208},
    "SALEM": {"lat": 11.6643, "lon": 78.1460},
    "THANJAVUR": {"lat": 10.7870, "lon": 79.1378},
    "THENI": {"lat": 10.0104, "lon": 77.4768},
    "TIRUCHIRAPPALLI": {"lat": 10.7905, "lon": 78.7047},
    "TIRUNELVELI": {"lat": 8.7139, "lon": 77.7567},
    "VELLORE": {"lat": 12.9165, "lon": 79.1325},
    "VILLUPURAM": {"lat": 11.9401, "lon": 79.4861},
    "VIRUDHUNAGAR": {"lat": 9.5680, "lon": 77.9624},
    "TAMIL NADU": {"lat": 11.1271, "lon": 78.6569}  # Center coordinates
}

def get_realtime_weather(district_name: str) -> dict:
    """Fetch real-time weather parameters (temp, humidity, precipitation) for a district from Open-Meteo."""
    name_upper = str(district_name).upper().strip()
    coords = DISTRICT_COORDINATES.get(name_upper, DISTRICT_COORDINATES["TAMIL NADU"])
    
    lat = coords["lat"]
    lon = coords["lon"]
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=Asia/Kolkata"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            return {
                "district": district_name,
                "latitude": lat,
                "longitude": lon,
                "temperature": current.get("temperature_2m"),  # °C
                "humidity": current.get("relative_humidity_2m"),  # %
                "precipitation": current.get("precipitation"),  # mm
                "wind_speed": current.get("wind_speed_10m"),  # km/h
                "source": "Open-Meteo API (Real-Time)"
            }
    except Exception as e:
        print(f"Failed to fetch weather for {district_name}: {e}")
        
    # Return sensible seasonal fallback if API fails
    return {
        "district": district_name,
        "latitude": lat,
        "longitude": lon,
        "temperature": 31.5,
        "humidity": 65.0,
        "precipitation": 0.2,
        "wind_speed": 12.0,
        "source": "Climatological Fallback Data"
    }
