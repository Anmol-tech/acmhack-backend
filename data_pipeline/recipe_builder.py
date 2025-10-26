"""
Audio recipe builder for Acoustic Shield training data.
Creates audio generation specifications from risk events.
"""
import logging
import random
from typing import Dict, List

logger = logging.getLogger(__name__)


class RecipeBuilder:
    """Build audio generation recipes from risk events."""
    
    # Audio parameters for each risk type
    AUDIO_SPECS = {
        'Normal': {
            'ambient_level': 0.3,
            'engine_intensity': 0.5,
            'tire_noise': 0.2,
            'alert_level': 0.0,
            'duration_seconds': 5.0,
            'sample_rate': 22050
        },
        'TireSkid': {
            'ambient_level': 0.3,
            'engine_intensity': 0.7,
            'tire_noise': 0.9,
            'alert_level': 0.4,
            'duration_seconds': 5.0,
            'sample_rate': 22050
        },
        'EmergencyBraking': {
            'ambient_level': 0.4,
            'engine_intensity': 0.8,
            'tire_noise': 0.8,
            'alert_level': 0.7,
            'duration_seconds': 5.0,
            'sample_rate': 22050
        },
        'CollisionImminent': {
            'ambient_level': 0.5,
            'engine_intensity': 0.9,
            'tire_noise': 1.0,
            'alert_level': 1.0,
            'duration_seconds': 5.0,
            'sample_rate': 22050
        }
    }
    
    def __init__(self):
        """Initialize recipe builder."""
        pass
    
    def build_recipes(self, risk_events: List[Dict], recipes_per_event: int = 1) -> List[Dict]:
        """
        Build audio generation recipes from risk events.
        
        Args:
            risk_events: List of risk event dictionaries
            recipes_per_event: Number of recipe variations to generate per event (default: 1)
            
        Returns:
            List of audio recipe dictionaries
        """
        recipes = []
        
        for event in risk_events:
            # Generate multiple recipe variations per event
            for variation_idx in range(recipes_per_event):
                recipe = self._create_recipe(event, variation_idx)
                recipes.append(recipe)
        
        logger.info(f"Built {len(recipes)} audio recipes from {len(risk_events)} events ({recipes_per_event} per event)")
        return recipes
    
    def _create_recipe(self, event: Dict, variation_idx: int = 0) -> Dict:
        """Create a single audio recipe from a risk event with enhanced parameters."""
        risk_type = event.get('risk_type', 'Normal')
        audio_spec = self.AUDIO_SPECS.get(risk_type, self.AUDIO_SPECS['Normal']).copy()
        
        # Apply random variations for diversity
        variation_factor = 1.0 + (variation_idx * 0.05)  # 0%, 5%, 10% variations
        noise_factor = random.uniform(0.95, 1.05)  # Add small random noise
        
        # Modify audio parameters based on weather
        weather_conditions = event.get('weather_conditions', {})
        weather_risk = event.get('weather_risk', 'low')
        
        # Adjust ambient based on rain
        rain = weather_conditions.get('rain_mm', 0)
        if rain > 0:
            audio_spec['ambient_level'] += min(rain * 0.05, 0.3)
        
        # Adjust based on weather risk
        if weather_risk == 'high':
            audio_spec['tire_noise'] = min(audio_spec['tire_noise'] * 1.2 * noise_factor, 1.0)
            audio_spec['ambient_level'] = min(audio_spec['ambient_level'] * 1.3 * noise_factor, 1.0)
        
        # Apply variation factor to create diversity
        audio_spec['engine_intensity'] = min(audio_spec['engine_intensity'] * variation_factor * noise_factor, 1.0)
        audio_spec['tire_noise'] = min(audio_spec['tire_noise'] * variation_factor * noise_factor, 1.0)
        
        # Adjust based on crash characteristics
        crash_chars = event.get('crash_characteristics', {})
        collision_type = crash_chars.get('collision_type', 'Unknown')
        
        # Collision-type specific adjustments
        if collision_type in ['Head-On', 'Broadside']:
            audio_spec['alert_level'] = min(audio_spec['alert_level'] * 1.3, 1.0)
        elif 'Rear End' in collision_type:
            audio_spec['engine_intensity'] = min(audio_spec['engine_intensity'] * 1.2, 1.0)
        
        # Build recipe with folder structure
        recipe = {
            'event_id': event.get('event_id'),
            'recipe_id': f"recipe_{event.get('event_id')}_v{variation_idx}",
            'risk_type': risk_type,
            'risk_score': event.get('risk_score', 50),
            'audio_parameters': audio_spec,
            'context': {
                'location_name': event.get('location_name'),
                'time_category': event.get('time_category'),
                'weather_risk': weather_risk,
                'weather_conditions': weather_conditions,
                'collision_type': collision_type,
                'primary_factor': crash_chars.get('primary_factor', 'Unknown'),
                'road_condition': crash_chars.get('road_condition', 'Dry'),
            },
            'output': {
                'filename': f"{event.get('event_id')}_v{variation_idx}_{risk_type.lower()}.wav",
                'folder': risk_type,  # Organize by risk type folder
                'format': 'wav',
                'channels': 1
            },
            'metadata': {
                'source_event': event.get('event_id'),
                'variation_index': variation_idx,
                'hotspot_rank': event.get('hotspot_rank'),
                'crash_count': event.get('crash_count'),
                'severity_score': event.get('severity_score', 0),
                'total_injuries': crash_chars.get('total_injuries', 0),
                'total_fatalities': crash_chars.get('total_fatalities', 0),
            }
        }
        
        return recipe
    
    def group_recipes_by_risk_type(self, recipes: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group recipes by risk type for balanced dataset creation.
        
        Args:
            recipes: List of recipe dictionaries
            
        Returns:
            Dictionary mapping risk types to recipe lists
        """
        grouped = {risk_type: [] for risk_type in self.AUDIO_SPECS.keys()}
        
        for recipe in recipes:
            risk_type = recipe.get('risk_type', 'Normal')
            if risk_type in grouped:
                grouped[risk_type].append(recipe)
        
        # Log distribution
        for risk_type, recipe_list in grouped.items():
            logger.info(f"{risk_type}: {len(recipe_list)} recipes")
        
        return grouped
    
    def get_recipe_summary(self, recipes: List[Dict]) -> Dict:
        """Get summary statistics of recipes."""
        risk_type_counts = {}
        total_duration = 0
        
        for recipe in recipes:
            risk_type = recipe.get('risk_type', 'Normal')
            risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
            
            duration = recipe.get('audio_parameters', {}).get('duration_seconds', 5.0)
            total_duration += duration
        
        return {
            'total_recipes': len(recipes),
            'risk_type_distribution': risk_type_counts,
            'total_audio_duration_seconds': total_duration,
            'total_audio_duration_minutes': round(total_duration / 60, 2)
        }
