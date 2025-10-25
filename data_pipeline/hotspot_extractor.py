"""
Extract crash hotspots from GeoJSON data with enhanced property extraction.
Identifies top crash locations by road/street name with rich metadata.
"""
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HotspotExtractor:
    """Extract and rank crash hotspots from GeoJSON crash data with comprehensive attributes."""
    
    def __init__(self, geojson_data: Dict):
        """
        Initialize with crash GeoJSON data.
        
        Args:
            geojson_data: GeoJSON FeatureCollection with crash locations and properties
        """
        self.geojson_data = geojson_data
        self.crash_records = self._parse_geojson(geojson_data)
        logger.info(f"Loaded {len(self.crash_records)} crash records")
    
    def _parse_geojson(self, geojson_data: Dict) -> List[Dict]:
        """Parse GeoJSON format extracting comprehensive crash properties."""
        records = []
        for feature in geojson_data.get('features', []):
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})
            coords = geom.get('coordinates', [0, 0])
            
            # Helper function to safely get and strip string values
            def get_str(key: str, default: str = '') -> str:
                val = props.get(key)
                return val.strip() if val and isinstance(val, str) else default
            
            # Helper function to safely get integer values
            def get_int(key: str, default: int = 0) -> int:
                val = props.get(key)
                if val is None or val == '':
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default
            
            # Helper function to safely get float values
            def get_float(key: str, default: float = 0.0) -> float:
                val = props.get(key)
                if val is None or val == '':
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            # Extract all available properties with proper data types
            record = {
                # Location (Double/Float)
                'latitude': get_float('LATITUDE', coords[1] if len(coords) >= 2 else 0),
                'longitude': get_float('LONGITUDE', coords[0] if len(coords) >= 2 else 0),
                'street_a': get_str('INTASTREETNAME'),
                'street_b': get_str('INTBSTREETNAME'),
                
                # Crash characteristics
                'collision_type': get_str('COLLISIONTYPE', 'Unknown'),
                'primary_factor': get_str('PRIMARYCOLLISIONFACTOR', 'Unknown'),
                'vehicle_count': get_int('VEHICLECOUNT'),
                'narrative': get_str('NARRATIVE'),
                
                # Environmental conditions (String)
                'weather': get_str('WEATHER', 'Clear'),
                'lighting': get_str('LIGHTING', 'Daylight'),
                'road_surface': get_str('ROADWAYSURFACE', 'Dry'),
                'road_condition': get_str('ROADWAYCONDITION', 'No Unusual Conditions'),
                
                # Injuries and severity (Integer)
                'minor_injuries': get_int('MINORINJURIES'),
                'moderate_injuries': get_int('MODERATEINJURIES'),
                'severe_injuries': get_int('SEVEREINJURIES'),
                'fatal_injuries': get_int('FATALINJURIES'),
                
                # Flags and violations (String - Y/N or blank)
                'speeding_flag': get_str('SPEEDINGFLAG'),
                'hit_and_run_flag': get_str('HITANDRUNFLAG'),
                'driver_intoxicated': get_str('VEHICLEDRIVERINTOXICATED'),
                
                # Temporal data
                'hour': get_int('HOUR', 12),
                'day_of_week': get_str('DAYOFWEEKNAME'),
                'month': get_str('MONTHNAME'),
                'year': get_int('YEAR', 2020),
                
                # Infrastructure (String)
                'intersection_type': get_str('INTERSECTIONTYPE'),
                'traffic_control': get_str('TRAFFICCONTROL', 'No Controls Present'),
                'traffic_control_type': get_str('INTTRAFFICCONTROLTYPE'),
                
                # Additional context (String)
                'vehicle_damage': get_str('VEHICLEDAMAGE'),
                'pedestrian_action': get_str('PEDESTRIANACTION', 'No Pedestrians Involved'),
                
                # Numeric identifiers (Integer)
                'intersection_number': get_int('INTERSECTIONNUMBERINT'),
                'object_id': get_int('OBJECTID'),
            }
            records.append(record)
        return records
    
    def extract_top_hotspots(self, top_n: int = 25) -> List[Dict]:
        """
        Extract top N crash hotspot locations with rich metadata.
        
        Args:
            top_n: Number of top hotspots to return
            
        Returns:
            List of hotspot dictionaries with detailed crash statistics
        """
        # Group crashes by intersection/location
        location_data = defaultdict(lambda: {
            'crashes': [],
            'total_count': 0,
            'coordinates': None,
            'collision_types': Counter(),
            'weather_conditions': Counter(),
            'lighting_conditions': Counter(),
            'road_conditions': Counter(),
            'primary_factors': Counter(),
            'total_injuries': 0,
            'total_fatalities': 0,
            'speeding_incidents': 0,
            'hit_and_run_incidents': 0,
        })
        
        for record in self.crash_records:
            # Create location key from intersection streets
            street_a = record.get('street_a', '')
            street_b = record.get('street_b', '')
            
            if street_a and street_b:
                location_key = f"{street_a} & {street_b}"
            elif street_a:
                location_key = street_a
            elif street_b:
                location_key = street_b
            else:
                location_key = record.get('street_name', 'Unknown Location')
            
            # Aggregate data
            loc_data = location_data[location_key]
            loc_data['crashes'].append(record)
            loc_data['total_count'] += 1
            
            # Store first valid coordinates
            if loc_data['coordinates'] is None:
                lat, lon = record.get('latitude', 0), record.get('longitude', 0)
                if lat != 0 and lon != 0:
                    loc_data['coordinates'] = {'latitude': lat, 'longitude': lon}
            
            # Aggregate statistics
            loc_data['collision_types'][record.get('collision_type', 'Unknown')] += 1
            loc_data['weather_conditions'][record.get('weather', 'Clear')] += 1
            loc_data['lighting_conditions'][record.get('lighting', 'Daylight')] += 1
            loc_data['road_conditions'][record.get('road_surface', 'Dry')] += 1
            loc_data['primary_factors'][record.get('primary_factor', 'Unknown')] += 1
            
            # Count injuries and fatalities
            loc_data['total_injuries'] += (
                record.get('minor_injuries', 0) +
                record.get('moderate_injuries', 0) +
                record.get('severe_injuries', 0)
            )
            loc_data['total_fatalities'] += record.get('fatal_injuries', 0)
            
            # Flags
            if record.get('speeding_flag'):
                loc_data['speeding_incidents'] += 1
            if record.get('hit_and_run_flag'):
                loc_data['hit_and_run_incidents'] += 1
        
        # Sort by crash count and get top N
        sorted_locations = sorted(
            location_data.items(),
            key=lambda x: x[1]['total_count'],
            reverse=True
        )[:top_n]
        
        # Build hotspot records
        hotspots = []
        for idx, (location_name, data) in enumerate(sorted_locations, start=1):
            coords = data['coordinates'] or {'latitude': 37.3382, 'longitude': -121.8863}
            
            hotspot = {
                'rank': idx,
                'location_name': location_name,
                'crash_count': data['total_count'],
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'most_common_collision_type': data['collision_types'].most_common(1)[0][0] if data['collision_types'] else 'Unknown',
                'most_common_weather': data['weather_conditions'].most_common(1)[0][0] if data['weather_conditions'] else 'Clear',
                'most_common_lighting': data['lighting_conditions'].most_common(1)[0][0] if data['lighting_conditions'] else 'Daylight',
                'most_common_road_condition': data['road_conditions'].most_common(1)[0][0] if data['road_conditions'] else 'Dry',
                'primary_factor': data['primary_factors'].most_common(1)[0][0] if data['primary_factors'] else 'Unknown',
                'total_injuries': data['total_injuries'],
                'total_fatalities': data['total_fatalities'],
                'speeding_rate': data['speeding_incidents'] / data['total_count'] if data['total_count'] > 0 else 0,
                'hit_and_run_rate': data['hit_and_run_incidents'] / data['total_count'] if data['total_count'] > 0 else 0,
                'severity_score': self._calculate_severity_score(data),
                'raw_crashes': data['crashes'][:5]  # Keep sample crashes for context
            }
            hotspots.append(hotspot)
        
        logger.info(f"Extracted {len(hotspots)} hotspots with enhanced metadata")
        return hotspots
    
    def _calculate_severity_score(self, location_data: Dict) -> float:
        """Calculate severity score based on injuries, fatalities, and other factors."""
        score = (
            location_data['total_fatalities'] * 100 +
            location_data['total_injuries'] * 10 +
            location_data['total_count'] * 1 +
            location_data['speeding_incidents'] * 5 +
            location_data['hit_and_run_incidents'] * 3
        )
        return round(score, 2)
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics of crash data."""
        total_crashes = len(self.crash_records)
        
        if total_crashes == 0:
            return {'total_crashes': 0}
        
        # Aggregate stats
        collision_types = Counter()
        weather_conditions = Counter()
        total_injuries = 0
        total_fatalities = 0
        speeding_count = 0
        
        for record in self.crash_records:
            collision_types[record.get('collision_type', 'Unknown')] += 1
            weather_conditions[record.get('weather', 'Clear')] += 1
            total_injuries += (
                record.get('minor_injuries', 0) +
                record.get('moderate_injuries', 0) +
                record.get('severe_injuries', 0)
            )
            total_fatalities += record.get('fatal_injuries', 0)
            if record.get('speeding_flag'):
                speeding_count += 1
        
        return {
            'total_crashes': total_crashes,
            'total_injuries': total_injuries,
            'total_fatalities': total_fatalities,
            'speeding_involved_pct': round(speeding_count / total_crashes * 100, 2) if total_crashes > 0 else 0,
            'most_common_collision_types': dict(collision_types.most_common(5)),
            'weather_distribution': dict(weather_conditions.most_common(5)),
            'avg_injuries_per_crash': round(total_injuries / total_crashes, 2) if total_crashes > 0 else 0
        }
