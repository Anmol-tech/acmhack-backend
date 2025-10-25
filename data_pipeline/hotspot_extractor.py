"""
Extract crash hotspots from GeoJSON data.
Identifies top crash locations by road/street name.
"""
import logging
from collections import Counter
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class HotspotExtractor:
    """Extract and rank crash hotspots from GeoJSON crash data."""
    
    def __init__(self, geojson_data: Dict):
        """
        Initialize with crash GeoJSON data.
        
        Args:
            geojson_data: GeoJSON FeatureCollection with crash locations
        """
        self.geojson_data = geojson_data
        self.features = geojson_data.get('features', [])
        logger.info(f"Loaded {len(self.features)} crash records")
    
    def extract_top_hotspots(self, top_n: int = 25) -> List[Dict]:
        """
        Extract top N crash hotspot locations.
        
        Args:
            top_n: Number of top hotspots to return
            
        Returns:
            List of hotspot dictionaries with road name, crash count, and representative coordinates
        """
        # Count crashes per road/street
        road_crashes = []
        road_coords = {}  # Store first occurrence coordinates
        
        for feature in self.features:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            
            # Extract road identifier (try multiple common field names)
            road_name = (
                properties.get('street_name') or 
                properties.get('road_name') or 
                properties.get('primary_rd') or
                properties.get('location') or
                'Unknown Road'
            )
            
            # Get coordinates
            coords = geometry.get('coordinates', [0, 0])
            if geometry.get('type') == 'Point' and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
            else:
                lon, lat = 0, 0
            
            road_crashes.append(road_name)
            
            # Store first occurrence coordinates for this road
            if road_name not in road_coords and lat != 0 and lon != 0:
                road_coords[road_name] = {'latitude': lat, 'longitude': lon}
        
        # Count and rank
        crash_counter = Counter(road_crashes)
        top_roads = crash_counter.most_common(top_n)
        
        # Build hotspot records
        hotspots = []
        for idx, (road_name, crash_count) in enumerate(top_roads, start=1):
            coords = road_coords.get(road_name, {'latitude': 37.3382, 'longitude': -121.8863})  # Default to San Jose
            
            hotspot = {
                'rank': idx,
                'road_name': road_name,
                'crash_count': crash_count,
                'latitude': coords['latitude'],
                'longitude': coords['longitude']
            }
            hotspots.append(hotspot)
        
        logger.info(f"Extracted {len(hotspots)} hotspots")
        return hotspots
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of crash data."""
        total_crashes = len(self.features)
        
        # Extract unique roads
        roads = set()
        for feature in self.features:
            props = feature.get('properties', {})
            road = (
                props.get('street_name') or 
                props.get('road_name') or 
                props.get('primary_rd') or
                'Unknown'
            )
            roads.add(road)
        
        return {
            'total_crashes': total_crashes,
            'unique_roads': len(roads),
            'avg_crashes_per_road': total_crashes / len(roads) if roads else 0
        }
