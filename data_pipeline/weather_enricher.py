"""
Fast Weather enrichment with fallback strategies:
1. Mock/Synthetic weather (no API calls - recommended for high volume)
2. Open-Meteo API with aggressive caching and rate limiting
3. Weatherstack API (paid alternative)

Set WEATHER_MODE environment variable:
- WEATHER_MODE=mock (default) - synthetic weather, instant
- WEATHER_MODE=openmeteo - Open-Meteo API (free but rate limited)
- WEATHER_MODE=weatherstack - Weatherstack API (requires WEATHERSTACK_API_KEY env var)
"""
import logging
import time
import os
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isnan

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Configuration from environment
WEATHER_MODE = os.environ.get("WEATHER_MODE", "mock").lower()
WEATHERSTACK_API_KEY = os.environ.get("WEATHERSTACK_API_KEY", "")


def _mk_session(max_pool: int = 64, retries: int = 6, backoff: float = 0.3) -> requests.Session:
    """Session with connection pooling + retries (429/5xx) and jitter."""
    sess = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff,                # 0.3, 0.6, 1.2, ...
        raise_on_status=False,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(pool_connections=max_pool, pool_maxsize=max_pool, max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    # Keep-alive & short timeouts help throughput
    sess.headers.update({
        "Connection": "keep-alive",
        "Accept": "application/json",
        "User-Agent": "acousticshield-weather/1.0"
    })
    return sess


class _QPSLimiter:
    """Simple token-bucket-ish limiter: allows ~qps per second shared across threads."""
    def __init__(self, qps: float):
        self.qps = max(0.1, qps)
        self._per = 1.0 / self.qps
        self._next = 0.0

    def wait(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self._per


class WeatherEnricher:
    """Enrich hotspot data with weather information using multiple strategies."""

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        max_workers: int = 16,
        qps: float = 10.0,
        dedup_precision: int = 3,   # ~110m resolution (0.001 deg)
        timeout_s: float = 6.0,
        mode: Optional[str] = None,  # override WEATHER_MODE env var
    ):
        """
        Args:
            base_url: Open-Meteo API endpoint
            max_workers: concurrent request workers
            qps: global queries per second cap to be polite and stable
            dedup_precision: round(lat/lon) to this many decimals to avoid duplicate fetches
            timeout_s: per-request timeout
            mode: 'mock', 'openmeteo', or 'weatherstack' (overrides env var)
        """
        self.mode = (mode or WEATHER_MODE).lower()
        self.base_url = base_url
        self.session = _mk_session(max_pool=max_workers * 2) if self.mode != "mock" else None
        self.max_workers = max(1, max_workers)
        self.qps_limiter = _QPSLimiter(qps) if self.mode != "mock" else None
        self.dedup_precision = int(dedup_precision)
        self.timeout_s = timeout_s
        
        logger.info(f"WeatherEnricher initialized with mode: {self.mode}")
        
        if self.mode == "weatherstack" and not WEATHERSTACK_API_KEY:
            logger.warning("WEATHERSTACK_API_KEY not set, falling back to mock mode")
            self.mode = "mock"

    def _generate_mock_weather(self, latitude: float, longitude: float, date: Optional[str] = None) -> Dict:
        """
        Generate realistic synthetic weather based on location and date.
        Uses deterministic randomization (seeded by lat/lon) for consistency.
        """
        # Use lat/lon as seed for deterministic results
        seed_str = f"{latitude:.3f},{longitude:.3f}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        # Base patterns (can be adjusted based on latitude for realism)
        # Temperate zone patterns
        base_temp = 15.0 + rng.uniform(-10, 15)  # 5-30°C range
        
        # Precipitation: weighted toward dry conditions (most days are dry)
        rain_chance = rng.random()
        if rain_chance < 0.7:  # 70% dry
            rain = 0.0
            precipitation = 0.0
            cloud = rng.randint(0, 40)
        elif rain_chance < 0.9:  # 20% light rain
            rain = rng.uniform(0.5, 3.0)
            precipitation = rain
            cloud = rng.randint(60, 85)
        else:  # 10% heavy rain
            rain = rng.uniform(3.0, 15.0)
            precipitation = rain
            cloud = rng.randint(85, 100)
        
        # Wind correlates somewhat with rain
        wind_base = 5.0 + rng.uniform(0, 20)
        if rain > 5.0:
            wind_base += rng.uniform(10, 20)
        
        wind_gusts = wind_base + rng.uniform(5, 15)
        
        return {
            "temperature_c": round(base_temp, 1),
            "precipitation_mm": round(precipitation, 1),
            "rain_mm": round(rain, 1),
            "wind_speed_kmh": round(wind_base, 1),
            "wind_gusts_kmh": round(wind_gusts, 1),
            "cloud_cover_percent": int(cloud),
            "fetch_timestamp": datetime.now().isoformat(),
            "source": "synthetic"
        }

    def _fetch_weatherstack(self, latitude: float, longitude: float, date: Optional[str] = None) -> Dict:
        """Fetch from Weatherstack API (paid service)."""
        try:
            if self.qps_limiter:
                self.qps_limiter.wait()
            
            url = "http://api.weatherstack.com/current"
            params = {
                "access_key": WEATHERSTACK_API_KEY,
                "query": f"{latitude},{longitude}",
            }
            
            resp = self.session.get(url, params=params, timeout=self.timeout_s)
            if resp.status_code >= 400:
                logger.warning(f"Weatherstack API status {resp.status_code}: {resp.text[:200]}")
                return self._get_default_weather()
            
            data = resp.json()
            current = data.get("current", {})
            
            return {
                "temperature_c": float(current.get("temperature", 20.0)),
                "precipitation_mm": float(current.get("precip", 0.0)),
                "rain_mm": float(current.get("precip", 0.0)),
                "wind_speed_kmh": float(current.get("wind_speed", 0.0)),
                "wind_gusts_kmh": float(current.get("wind_speed", 0.0)) * 1.5,  # estimate
                "cloud_cover_percent": int(current.get("cloudcover", 0)),
                "fetch_timestamp": datetime.now().isoformat(),
                "source": "weatherstack"
            }
        except Exception as e:
            logger.warning(f"Weatherstack API error for ({latitude},{longitude}): {e}")
            return self._get_default_weather()

    def _params(self, latitude: float, longitude: float, date: Optional[str]) -> Dict:
        # If no date provided, use recent past (7 days ago) – Open-Meteo `current` is near real-time,
        # but keeping the same behavior as your original code.
        if date is None:
            _ = datetime.now() - timedelta(days=7)  # placeholder for future historical use
        return {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m,cloud_cover",
            "forecast_days": 1,
            "timezone": "UTC",
        }

    def _normalize(self, data: Dict) -> Dict:
        current = data.get("current", {}) if isinstance(data, dict) else {}
        def _num(v, default):
            try:
                if v is None or (isinstance(v, float) and isnan(v)):
                    return default
                return float(v)
            except Exception:
                return default

        return {
            "temperature_c":     _num(current.get("temperature_2m"), 20.0),
            "precipitation_mm":  _num(current.get("precipitation"), 0.0),
            "rain_mm":           _num(current.get("rain"), 0.0),
            "wind_speed_kmh":    _num(current.get("wind_speed_10m"), 0.0),
            "wind_gusts_kmh":    _num(current.get("wind_gusts_10m"), 0.0),
            "cloud_cover_percent": int(_num(current.get("cloud_cover"), 0)),
            "fetch_timestamp":   datetime.now().isoformat(),
        }

    def _get_default_weather(self) -> Dict:
        return {
            "temperature_c": 18.0,
            "precipitation_mm": 0.0,
            "rain_mm": 0.0,
            "wind_speed_kmh": 10.0,
            "wind_gusts_kmh": 15.0,
            "cloud_cover_percent": 30,
            "fetch_timestamp": datetime.now().isoformat(),
            "source": "default"
        }

    def fetch_weather(self, latitude: float, longitude: float, date: Optional[str] = None) -> Dict:
        """Single fetch with mode selection."""
        if self.mode == "mock":
            return self._generate_mock_weather(latitude, longitude, date)
        elif self.mode == "weatherstack":
            return self._fetch_weatherstack(latitude, longitude, date)
        else:  # openmeteo
            try:
                if self.qps_limiter:
                    self.qps_limiter.wait()
                resp = self.session.get(self.base_url, params=self._params(latitude, longitude, date), timeout=self.timeout_s)
                if resp.status_code >= 400:
                    logger.warning(f"Open-Meteo API status {resp.status_code} for ({latitude},{longitude}): {resp.text[:200]}")
                    # Fall back to mock on rate limit
                    logger.info("Falling back to mock weather due to API error")
                    return self._generate_mock_weather(latitude, longitude, date)
                return self._normalize(resp.json())
            except requests.exceptions.RequestException as e:
                logger.warning(f"Open-Meteo API error for ({latitude},{longitude}): {e}")
                logger.info("Falling back to mock weather due to exception")
                return self._generate_mock_weather(latitude, longitude, date)

    def _key(self, lat: float, lon: float) -> Tuple[float, float]:
        return (round(lat, self.dedup_precision), round(lon, self.dedup_precision))

    def enrich_hotspots(self, hotspots: List[Dict], rate_limit_delay: float = 0.0) -> List[Dict]:
        """
        Enrich multiple hotspots with weather data.
        - Mock mode: instant, deterministic synthetic weather
        - API modes: parallel requests with deduplication
        """
        if not hotspots:
            return []

        # Mock mode is instant - no need for concurrency
        if self.mode == "mock":
            logger.info(f"Generating synthetic weather for {len(hotspots)} hotspots (instant)")
            enriched = []
            for h in hotspots:
                lat = float(h.get("latitude", 0.0))
                lon = float(h.get("longitude", 0.0))
                weather = self._generate_mock_weather(lat, lon)
                enriched.append({**h, "weather": weather})
            logger.info(f"Enriched {len(enriched)} hotspots with synthetic weather")
            return enriched

        # API modes: de-duplicate and use concurrency
        coord_to_idx = {}
        for i, h in enumerate(hotspots):
            lat = float(h.get("latitude", 0.0))
            lon = float(h.get("longitude", 0.0))
            coord_to_idx.setdefault(self._key(lat, lon), []).append(i)

        unique_coords = list(coord_to_idx.keys())

        results_cache: Dict[Tuple[float, float], Dict] = {}
        def _task(latlon):
            lat, lon = latlon
            return latlon, self.fetch_weather(lat, lon)

        logger.info(f"Fetching weather for {len(unique_coords)} unique coordinates "
                    f"(from {len(hotspots)} hotspots) with {self.max_workers} workers in {self.mode} mode...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_task, latlon) for latlon in unique_coords]
            for fut in as_completed(futures):
                latlon, weather = fut.result()
                results_cache[latlon] = weather

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # Map back to all hotspots
        enriched = []
        for i, h in enumerate(hotspots, start=1):
            key = self._key(float(h.get("latitude", 0.0)), float(h.get("longitude", 0.0)))
            weather = results_cache.get(key, self._get_default_weather())
            enriched.append({**h, "weather": weather})

        logger.info(f"Enriched {len(enriched)} hotspots with weather data")
        return enriched

    def categorize_weather_risk(self, weather: Dict) -> str:
        rain = float(weather.get("rain_mm", 0))
        wind_gusts = float(weather.get("wind_gusts_kmh", 0))
        temp = float(weather.get("temperature_c", 20))
        if rain > 5.0 or wind_gusts > 50 or temp < 2:
            return "high"
        if rain > 1.0 or wind_gusts > 30 or temp < 10:
            return "medium"
        return "low"
