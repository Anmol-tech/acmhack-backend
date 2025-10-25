"""
Weather enrichment using Open-Meteo API.
Fetches historical weather data for crash hotspot locations.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class WeatherEnricher:
    """Enrich hotspot data with weather information using Open-Meteo API."""
    
    def __init__(self, base_url: str = "https://api.open-meteo.com/v1/forecast"):
        """
        Initialize weather enricher.
        
        Args:
            base_url: Open-Meteo API endpoint
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def fetch_weather(self, latitude: float, longitude: float, 
                     date: Optional[str] = None) -> Dict:
        """
        Fetch weather data for a location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            date: Date string (YYYY-MM-DD), defaults to recent past
            
        Returns:
            Dictionary with weather parameters
        """
        # If no date provided, use recent date (7 days ago for historical-like data)
        if date is None:
            date_obj = datetime.now() - timedelta(days=7)
            date = date_obj.strftime('%Y-%m-%d')
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': 'temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m,cloud_cover',
            'forecast_days': 1
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('current', {})
            
            # Extract and normalize weather data
            weather_data = {
                'temperature_c': current.get('temperature_2m', 20.0),
                'precipitation_mm': current.get('precipitation', 0.0),
                'rain_mm': current.get('rain', 0.0),
                'wind_speed_kmh': current.get('wind_speed_10m', 0.0),
                'wind_gusts_kmh': current.get('wind_gusts_10m', 0.0),
                'cloud_cover_percent': current.get('cloud_cover', 0),
                'fetch_timestamp': datetime.now().isoformat()
            }
            
            return weather_data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Weather API error for ({latitude}, {longitude}): {e}")
            # Return default weather data on error
            return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """Return default weather data for fallback."""
        return {
            'temperature_c': 18.0,
            'precipitation_mm': 0.0,
            'rain_mm': 0.0,
            'wind_speed_kmh': 10.0,
            'wind_gusts_kmh': 15.0,
            'cloud_cover_percent': 30,
            'fetch_timestamp': datetime.now().isoformat()
        }
    
    def enrich_hotspots(self, hotspots: List[Dict], 
                        rate_limit_delay: float = 0.5) -> List[Dict]:
        """
        Enrich multiple hotspots with weather data.
        
        Args:
            hotspots: List of hotspot dictionaries
            rate_limit_delay: Delay between API calls (seconds)
            
        Returns:
            Enriched hotspots with weather data
        """
        enriched = []
        
        for idx, hotspot in enumerate(hotspots, start=1):
            lat = hotspot.get('latitude', 0)
            lon = hotspot.get('longitude', 0)
            
            logger.info(f"Fetching weather for hotspot {idx}/{len(hotspots)}: {hotspot.get('location_name')}")
            
            weather = self.fetch_weather(lat, lon)
            
            # Merge weather data into hotspot
            enriched_hotspot = {**hotspot, 'weather': weather}
            enriched.append(enriched_hotspot)
            
            # Rate limiting
            if idx < len(hotspots):
                time.sleep(rate_limit_delay)
        
        logger.info(f"Enriched {len(enriched)} hotspots with weather data")
        return enriched
    
    def categorize_weather_risk(self, weather: Dict) -> str:
        """
        Categorize weather conditions into risk levels.
        
        Args:
            weather: Weather data dictionary
            
        Returns:
            Risk category: 'high', 'medium', or 'low'
        """
        rain = weather.get('rain_mm', 0)
        wind_gusts = weather.get('wind_gusts_kmh', 0)
        temp = weather.get('temperature_c', 20)
        
        # High risk: Heavy rain, strong winds, or near-freezing temps
        if rain > 5.0 or wind_gusts > 50 or temp < 2:
            return 'high'
        
        # Medium risk: Light rain, moderate winds, or cold temps
        if rain > 1.0 or wind_gusts > 30 or temp < 10:
            return 'medium'
        
        # Low risk: Clear conditions
        return 'low'
