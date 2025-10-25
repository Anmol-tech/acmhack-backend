"""
Synthesize risk events from enriched hotspot data.
Creates diverse risk scenarios for audio training data generation.
"""
import logging
import random
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskEventSynthesizer:
    """Synthesize risk events from crash hotspots and weather data."""
    
    # Risk event types aligned with audio scenarios
    RISK_TYPES = [
        'Normal',
        'TireSkid',
        'EmergencyBraking',
        'CollisionImminent'
    ]
    
    # Time of day categories
    TIME_CATEGORIES = ['morning', 'afternoon', 'evening', 'night']
    
    def __init__(self, seed: int = 42):
        """
        Initialize synthesizer.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def synthesize_events(self, enriched_hotspots: List[Dict], 
                         events_per_hotspot: int = 4) -> List[Dict]:
        """
        Synthesize risk events from enriched hotspot data.
        
        Args:
            enriched_hotspots: List of hotspots with weather data
            events_per_hotspot: Number of events to generate per hotspot
            
        Returns:
            List of synthesized risk event dictionaries
        """
        events = []
        event_id = 1
        
        for hotspot in enriched_hotspots:
            weather = hotspot.get('weather', {})
            weather_risk = self._assess_weather_risk(weather)
            
            # Generate multiple events per hotspot with different risk levels
            for _ in range(events_per_hotspot):
                event = self._create_event(
                    event_id=event_id,
                    hotspot=hotspot,
                    weather=weather,
                    weather_risk=weather_risk
                )
                events.append(event)
                event_id += 1
        
        logger.info(f"Synthesized {len(events)} risk events")
        return events
    
    def _create_event(self, event_id: int, hotspot: Dict, 
                     weather: Dict, weather_risk: str) -> Dict:
        """Create a single risk event with enhanced metadata."""
        # Select risk type with weighted probability based on weather and crash history
        crash_severity = hotspot.get('severity_score', 0)
        collision_type = hotspot.get('most_common_collision_type', 'Unknown')
        
        risk_type = self._select_risk_type(weather_risk, collision_type, crash_severity)
        
        # Generate temporal context
        time_category = random.choice(self.TIME_CATEGORIES)
        timestamp = self._generate_timestamp(time_category)
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(
            risk_type, weather_risk, hotspot.get('crash_count', 1), crash_severity
        )
        
        # Build event with rich metadata
        event = {
            'event_id': f"evt_{event_id:05d}",
            'hotspot_rank': hotspot.get('rank', 0),
            'location_name': hotspot.get('location_name', 'Unknown'),
            'latitude': hotspot.get('latitude', 0),
            'longitude': hotspot.get('longitude', 0),
            'crash_count': hotspot.get('crash_count', 0),
            'risk_type': risk_type,
            'risk_score': risk_score,
            'severity_score': crash_severity,
            'time_category': time_category,
            'timestamp': timestamp,
            'weather_risk': weather_risk,
            'crash_characteristics': {
                'collision_type': collision_type,
                'primary_factor': hotspot.get('primary_factor', 'Unknown'),
                'road_condition': hotspot.get('most_common_road_condition', 'Dry'),
                'lighting': hotspot.get('most_common_lighting', 'Daylight'),
                'total_injuries': hotspot.get('total_injuries', 0),
                'total_fatalities': hotspot.get('total_fatalities', 0),
                'speeding_rate': hotspot.get('speeding_rate', 0),
            },
            'weather_conditions': {
                'temperature_c': weather.get('temperature_c', 20),
                'rain_mm': weather.get('rain_mm', 0),
                'wind_speed_kmh': weather.get('wind_speed_kmh', 0),
                'cloud_cover_percent': weather.get('cloud_cover_percent', 0)
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'source': 'synthetic',
                'data_version': '2.0'
            }
        }
        
        return event
    
    def _select_risk_type(self, weather_risk: str, collision_type: str = 'Unknown', 
                         severity: float = 0) -> str:
        """Select risk type based on weather, collision type, and severity."""
        # Base weights
        if weather_risk == 'high':
            weights = [0.1, 0.3, 0.35, 0.25]
        elif weather_risk == 'medium':
            weights = [0.3, 0.3, 0.25, 0.15]
        else:
            weights = [0.5, 0.25, 0.15, 0.1]
        
        # Adjust based on collision type
        if collision_type in ['Head-On', 'Broadside']:
            # More severe collision types - increase higher risk categories
            weights[2] *= 1.3  # EmergencyBraking
            weights[3] *= 1.5  # CollisionImminent
        elif collision_type == 'Rear End':
            # Common in sudden stops
            weights[2] *= 1.2  # EmergencyBraking
        elif collision_type in ['Sideswipe', 'Hit Object']:
            # Often involves loss of control
            weights[1] *= 1.3  # TireSkid
        
        # Adjust based on severity score
        if severity > 200:
            weights[3] *= 1.3  # High severity area
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return random.choices(self.RISK_TYPES, weights=weights)[0]
    
    def _assess_weather_risk(self, weather: Dict) -> str:
        """Assess weather risk level."""
        rain = weather.get('rain_mm', 0)
        wind = weather.get('wind_speed_kmh', 0)
        temp = weather.get('temperature_c', 20)
        
        if rain > 5.0 or wind > 50 or temp < 2:
            return 'high'
        elif rain > 1.0 or wind > 30 or temp < 10:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_score(self, risk_type: str, weather_risk: str, 
                             crash_count: int, severity_score: float = 0) -> float:
        """Calculate numerical risk score (0-100) with severity consideration."""
        # Base score by risk type
        base_scores = {
            'Normal': 20,
            'TireSkid': 50,
            'EmergencyBraking': 70,
            'CollisionImminent': 90
        }
        base = base_scores.get(risk_type, 50)
        
        # Weather multiplier
        weather_mult = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        weather_factor = weather_mult.get(weather_risk, 1.0)
        
        # Crash history factor (logarithmic scaling)
        crash_factor = min(1.0 + (crash_count / 100.0), 1.5)
        
        # Severity factor
        severity_factor = min(1.0 + (severity_score / 1000.0), 1.3)
        
        score = base * weather_factor * crash_factor * severity_factor
        return min(round(score, 2), 100.0)
    
    def _generate_timestamp(self, time_category: str) -> str:
        """Generate realistic timestamp for given time category."""
        base_date = datetime.now() - timedelta(days=random.randint(1, 30))
        
        hour_ranges = {
            'morning': (6, 11),
            'afternoon': (12, 17),
            'evening': (18, 21),
            'night': (22, 5)
        }
        
        hour_start, hour_end = hour_ranges.get(time_category, (12, 17))
        
        if hour_start > hour_end:  # Handle night wrap-around
            hour = random.choice(list(range(hour_start, 24)) + list(range(0, hour_end + 1)))
        else:
            hour = random.randint(hour_start, hour_end)
        
        minute = random.randint(0, 59)
        timestamp = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return timestamp.isoformat()
    
    def get_event_distribution(self, events: List[Dict]) -> Dict:
        """Get distribution statistics of generated events."""
        risk_type_counts = {}
        weather_risk_counts = {}
        time_counts = {}
        
        for event in events:
            risk_type = event.get('risk_type')
            risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
            
            weather_risk = event.get('weather_risk')
            weather_risk_counts[weather_risk] = weather_risk_counts.get(weather_risk, 0) + 1
            
            time_cat = event.get('time_category')
            time_counts[time_cat] = time_counts.get(time_cat, 0) + 1
        
        return {
            'total_events': len(events),
            'risk_type_distribution': risk_type_counts,
            'weather_risk_distribution': weather_risk_counts,
            'time_distribution': time_counts
        }
