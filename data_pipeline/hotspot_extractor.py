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
            
            # Extract all available properties with fallbacks
            record = {
                # Location
                'latitude': coords[1] if len(coords) >= 2 else 0,
                'longitude': coords[0] if len(coords) >= 2 else 0,
                'street_a': (props.get('INTASTREETNAME') or props.get('street_name') or '').strip(),
                'street_b': (props.get('INTBSTREETNAME') or props.get('street_b') or '').strip(),
                
                # Crash characteristics
                'collision_type': (props.get('COLLISIONTYPE') or props.get('collision_type') or 'Unknown').strip(),
                'primary_factor': (props.get('PRIMARYCOLLISIONFACTOR') or props.get('primary_factor') or 'Unknown').strip(),
                'vehicle_count': int(props.get('VEHICLECOUNT') or props.get('vehicle_count') or 0),
                'narrative': (props.get('NARRATIVE') or props.get('narrative') or '').strip(),
                
                # Environmental conditions
                'weather': (props.get('WEATHER') or props.get('weather') or 'Clear').strip(),
                'lighting': (props.get('LIGHTING') or props.get('lighting') or 'Daylight').strip(),
                'road_surface': (props.get('ROADWAYSURFACE') or props.get('road_surface') or 'Dry').strip(),
                'road_condition': (props.get('ROADWAYCONDITION') or props.get('road_condition') or 'No Unusual Conditions').strip(),
                
                # Injuries and severity
                'minor_injuries': int(props.get('MINORINJURIES') or props.get('minor_injuries') or 0),
                'moderate_injuries': int(props.get('MODERATEINJURIES') or props.get('moderate_injuries') or 0),
                'severe_injuries': int(props.get('SEVEREINJURIES') or props.get('severe_injuries') or 0),
                'fatal_injuries': int(props.get('FATALINJURIES') or props.get('fatal_injuries') or 0),
                
                # Flags and violations
                'speeding_flag': (props.get('SPEEDINGFLAG') or props.get('speeding_flag') or '').strip(),
                'hit_and_run_flag': (props.get('HITANDRUNFLAG') or props.get('hit_and_run_flag') or '').strip(),
                'driver_intoxicated': (props.get('VEHICLEDRIVERINTOXICATED') or props.get('driver_intoxicated') or '').strip(),
                
                # Temporal data
                'hour': int(props.get('HOUR') or props.get('hour') or 12),
                'day_of_week': (props.get('DAYOFWEEKNAME') or props.get('day_of_week') or '').strip(),
                'month': (props.get('MONTHNAME') or props.get('month') or '').strip(),
                'year': int(props.get('YEAR') or props.get('year') or 2020),
                
                # Infrastructure
                'intersection_type': (props.get('INTERSECTIONTYPE') or props.get('intersection_type') or '').strip(),
                'traffic_control': (props.get('TRAFFICCONTROL') or props.get('traffic_control') or 'No Controls Present').strip(),
                'traffic_control_type': (props.get('INTTRAFFICCONTROLTYPE') or props.get('traffic_control_type') or '').strip(),
                
                # Additional context
                'vehicle_damage': (props.get('VEHICLEDAMAGE') or props.get('vehicle_damage') or '').strip(),
                'pedestrian_action': (props.get('PEDESTRIANACTION') or props.get('pedestrian_action') or 'No Pedestrians Involved').strip(),
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
