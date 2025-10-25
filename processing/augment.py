"""
SageMaker Processing script for audio data augmentation.
Generates WAV files from audio recipes using synthetic audio generation.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import wavfile
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioGenerator:
    """Generate synthetic audio based on recipes."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize audio generator.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def generate_audio(self, recipe: Dict) -> np.ndarray:
        """
        Generate audio from recipe.
        
        Args:
            recipe: Audio recipe dictionary
            
        Returns:
            Audio samples as numpy array
        """
        params = recipe.get('audio_parameters', {})
        duration = params.get('duration_seconds', 5.0)
        sample_rate = params.get('sample_rate', self.sample_rate)
        
        # Generate base components
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # 1. Ambient noise (pink noise)
        ambient_level = params.get('ambient_level', 0.3)
        ambient = self._generate_pink_noise(n_samples) * ambient_level
        
        # 2. Engine sound (low frequency rumble)
        engine_intensity = params.get('engine_intensity', 0.5)
        engine = self._generate_engine_sound(t, engine_intensity)
        
        # 3. Tire noise (higher frequency)
        tire_noise = params.get('tire_noise', 0.2)
        tire = self._generate_tire_noise(t, tire_noise)
        
        # 4. Alert/warning sounds
        alert_level = params.get('alert_level', 0.0)
        alert = self._generate_alert_sound(t, alert_level)
        
        # Mix components
        audio = ambient + engine + tire + alert
        
        # Normalize to prevent clipping
        audio = self._normalize(audio)
        
        return audio
    
    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        white = np.random.randn(n_samples)
        # Simple approximation of pink noise using IIR filter
        b, a = signal.butter(1, 0.5, btype='low')
        pink = signal.filtfilt(b, a, white)
        return pink * 0.1
    
    def _generate_engine_sound(self, t: np.ndarray, intensity: float) -> np.ndarray:
        """Generate engine rumble sound."""
        # Low frequency oscillation (40-80 Hz)
        freq1 = 50 + np.random.randn() * 5
        freq2 = 70 + np.random.randn() * 5
        
        engine = (
            np.sin(2 * np.pi * freq1 * t) * 0.6 +
            np.sin(2 * np.pi * freq2 * t) * 0.4
        )
        
        # Add some variation
        modulation = 1 + 0.2 * np.sin(2 * np.pi * 2 * t)
        engine = engine * modulation * intensity * 0.2
        
        return engine
    
    def _generate_tire_noise(self, t: np.ndarray, intensity: float) -> np.ndarray:
        """Generate tire/road noise."""
        if intensity < 0.3:
            # Normal tire noise
            noise = np.random.randn(len(t)) * 0.05
        else:
            # Skidding sound (higher intensity, more irregular)
            noise = np.random.randn(len(t)) * 0.2
            # Add high-frequency component for skid
            skid_freq = 3000 + np.random.randn() * 500
            skid = np.sin(2 * np.pi * skid_freq * t) * 0.3
            noise = noise + skid * (intensity - 0.3) / 0.7
        
        return noise * intensity
    
    def _generate_alert_sound(self, t: np.ndarray, level: float) -> np.ndarray:
        """Generate alert/warning sound."""
        if level < 0.1:
            return np.zeros(len(t))
        
        # Beeping alert (800-1200 Hz)
        alert_freq = 1000
        beep_rate = 2 + level * 3  # Faster beeping for higher alert
        
        # Create beeping pattern
        beep_pattern = signal.square(2 * np.pi * beep_rate * t) * 0.5 + 0.5
        alert = np.sin(2 * np.pi * alert_freq * t) * beep_pattern * level * 0.3
        
        return alert
    
    def _normalize(self, audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio to target level."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio * (target_level / max_val)
        return audio


def load_recipes(recipe_dir: str) -> List[Dict]:
    """Load all recipe JSON files from directory."""
    recipe_path = Path(recipe_dir)
    recipes = []
    
    for json_file in recipe_path.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    recipes.extend(data)
                else:
                    recipes.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Loaded {len(recipes)} recipes from {recipe_dir}")
    return recipes


def generate_wav_files(recipes: List[Dict], output_dir: str):
    """Generate WAV files from recipes, organized by risk type folders."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create folders for each risk type
    risk_types = ['Normal', 'TireSkid', 'EmergencyBraking', 'CollisionImminent']
    for risk_type in risk_types:
        (output_path / risk_type).mkdir(exist_ok=True)
    
    generator = AudioGenerator()
    successful = 0
    failed = 0
    
    for idx, recipe in enumerate(recipes, start=1):
        try:
            # Generate audio
            audio = generator.generate_audio(recipe)
            
            # Get output information
            output_info = recipe.get('output', {})
            filename = output_info.get('filename', f'audio_{idx:05d}.wav')
            folder = output_info.get('folder', 'Normal')  # Get risk type folder
            
            # Create path with folder structure
            risk_folder = output_path / folder
            risk_folder.mkdir(exist_ok=True)
            output_file = risk_folder / filename
            
            # Get sample rate
            sample_rate = recipe.get('audio_parameters', {}).get('sample_rate', 22050)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Write WAV file
            wavfile.write(str(output_file), sample_rate, audio_int16)
            
            successful += 1
            
            if idx % 10 == 0:
                logger.info(f"Generated {idx}/{len(recipes)} audio files")
                
        except Exception as e:
            logger.error(f"Failed to generate audio for recipe {idx}: {e}")
            failed += 1
    
    logger.info(f"Audio generation complete: {successful} successful, {failed} failed")
    
    # Log file distribution by risk type
    for risk_type in risk_types:
        count = len(list((output_path / risk_type).glob('*.wav')))
        logger.info(f"  {risk_type}: {count} files")


def main():
    """Main processing function for SageMaker."""
    parser = argparse.ArgumentParser(description='Generate audio training data from recipes')
    parser.add_argument('--recipe-dir', type=str, default='/opt/ml/processing/input',
                       help='Directory containing recipe JSON files')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output',
                       help='Directory for output WAV files')
    
    args = parser.parse_args()
    
    logger.info("Starting audio generation processing job")
    logger.info(f"Recipe directory: {args.recipe_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load recipes
    recipes = load_recipes(args.recipe_dir)
    
    if not recipes:
        logger.error("No recipes found!")
        sys.exit(1)
    
    # Generate WAV files
    generate_wav_files(recipes, args.output_dir)
    
    logger.info("Processing job completed successfully")


if __name__ == '__main__':
    main()
