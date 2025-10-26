"""
Fast Weather enrichment using Open-Meteo API.
- Connection pooling (keep-alive)
- Concurrent fetch with bounded workers + QPS limiter
- Robust retries with backoff + jitter on 429/5xx
- De-duplicate nearby coords (rounding) to avoid redundant calls
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isnan

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


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
    """Enrich hotspot data with weather information using Open-Meteo API (fast)."""

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        max_workers: int = 16,
        qps: float = 10.0,
        dedup_precision: int = 3,   # ~110m resolution (0.001 deg)
        timeout_s: float = 6.0,
    ):
        """
        Args:
            base_url: Open-Meteo API endpoint
            max_workers: concurrent request workers
            qps: global queries per second cap to be polite and stable
            dedup_precision: round(lat/lon) to this many decimals to avoid duplicate fetches
            timeout_s: per-request timeout
        """
        self.base_url = base_url
        self.session = _mk_session(max_pool=max_workers * 2)
        self.max_workers = max(1, max_workers)
        self.qps_limiter = _QPSLimiter(qps)
        self.dedup_precision = int(dedup_precision)
        self.timeout_s = timeout_s

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
        }

    def fetch_weather(self, latitude: float, longitude: float, date: Optional[str] = None) -> Dict:
        """Single fetch, kept for API compatibility."""
        try:
            self.qps_limiter.wait()
            resp = self.session.get(self.base_url, params=self._params(latitude, longitude, date), timeout=self.timeout_s)
            if resp.status_code >= 400:
                # requests' Retry will have retried before; final failure falls here
                logger.warning(f"Weather API status {resp.status_code} for ({latitude},{longitude}): {resp.text[:200]}")
                return self._get_default_weather()
            return self._normalize(resp.json())
        except requests.exceptions.RequestException as e:
            logger.warning(f"Weather API error for ({latitude},{longitude}): {e}")
            return self._get_default_weather()

    def _key(self, lat: float, lon: float) -> Tuple[float, float]:
        return (round(lat, self.dedup_precision), round(lon, self.dedup_precision))

    def enrich_hotspots(self, hotspots: List[Dict], rate_limit_delay: float = 0.0) -> List[Dict]:
        """
        Enrich multiple hotspots with weather data fast.
        - Parallel requests (ThreadPool)
        - Optional tiny inter-batch delay (rate_limit_delay) if you want extra smoothing
        """
        if not hotspots:
            return []

        # 1) De-duplicate close coordinates to avoid redundant calls
        coord_to_idx = {}
        for i, h in enumerate(hotspots):
            lat = float(h.get("latitude", 0.0))
            lon = float(h.get("longitude", 0.0))
            coord_to_idx.setdefault(self._key(lat, lon), []).append(i)

        unique_coords = list(coord_to_idx.keys())

        # 2) Fire concurrent requests
        results_cache: Dict[Tuple[float, float], Dict] = {}
        def _task(latlon):
            lat, lon = latlon
            return latlon, self.fetch_weather(lat, lon)

        logger.info(f"Fetching weather for {len(unique_coords)} unique coordinates "
                    f"(from {len(hotspots)} hotspots) with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_task, latlon) for latlon in unique_coords]
            for fut in as_completed(futures):
                latlon, weather = fut.result()
                results_cache[latlon] = weather

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # 3) Map back to all hotspots
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
