"""
AWS Bedrock-Enhanced Audio Generation for Acoustic Shield.
Uses Bedrock AI to optimize parameters, then generates actual WAV files.
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
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSynthesizer:
    """Synthesize WAV audio from parameters."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def generate_audio(self, params: Dict) -> np.ndarray:
        """Generate audio samples from parameters."""
        duration = params.get('duration_seconds', 5.0)
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Extract parameters
        ambient_level = params.get('ambient_level', 0.3)
        engine_intensity = params.get('engine_intensity', 0.5)
        tire_noise = params.get('tire_noise', 0.2)
        alert_level = params.get('alert_level', 0.0)
        
        # Generate components
        ambient = self._pink_noise(n_samples) * ambient_level
        engine = self._engine_sound(t, engine_intensity)
        tire = self._tire_sound(t, tire_noise)
        alert = self._alert_sound(t, alert_level)
        
        # Mix and normalize
        audio = ambient + engine + tire + alert
        return self._normalize(audio)
    
    def _pink_noise(self, n: int) -> np.ndarray:
        """Generate pink noise."""
        white = np.random.randn(n)
        b, a = signal.butter(1, 0.5, btype='low')
        return signal.filtfilt(b, a, white) * 0.1
    
    def _engine_sound(self, t: np.ndarray, intensity: float) -> np.ndarray:
        """Generate engine rumble (40-80 Hz)."""
        freq1, freq2 = 50 + np.random.randn() * 5, 70 + np.random.randn() * 5
        engine = np.sin(2 * np.pi * freq1 * t) * 0.6 + np.sin(2 * np.pi * freq2 * t) * 0.4
        modulation = 1 + 0.2 * np.sin(2 * np.pi * 2 * t)
        return engine * modulation * intensity * 0.2
    
    def _tire_sound(self, t: np.ndarray, intensity: float) -> np.ndarray:
        """Generate tire/road noise."""
        if intensity < 0.3:
            return np.random.randn(len(t)) * 0.05 * intensity
        else:
            # Skid sound
            noise = np.random.randn(len(t)) * 0.2
            skid_freq = 3000 + np.random.randn() * 500
            skid = np.sin(2 * np.pi * skid_freq * t) * 0.3
            return (noise + skid * (intensity - 0.3) / 0.7) * intensity
    
    def _alert_sound(self, t: np.ndarray, level: float) -> np.ndarray:
        """Generate alert beeps (800-1200 Hz)."""
        if level < 0.1:
            return np.zeros(len(t))
        beep_rate = 2 + level * 3
        beep_pattern = signal.square(2 * np.pi * beep_rate * t) * 0.5 + 0.5
        return np.sin(2 * np.pi * 1000 * t) * beep_pattern * level * 0.3
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to prevent clipping."""
        max_val = np.abs(audio).max()
        return audio * (0.8 / max_val) if max_val > 0 else audio


class BedrockAudioGenerator:
    """Use Bedrock AI to enhance audio parameters, then generate WAV files."""
    
    def __init__(self, region_name: str = None):
        """
        Initialize Bedrock audio generator.
        
        Args:
            region_name: AWS region name (auto-detected if None)
        """
        self.region_name = region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region_name)
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.synthesizer = AudioSynthesizer()
        logger.info(f"Initialized Bedrock client in region: {self.region_name}")
    
    def enhance_parameters(self, recipe: Dict) -> Dict:
        """
        Use Bedrock AI to enhance audio parameters based on context.
        
        Args:
            recipe: Audio recipe dictionary
            
        Returns:
            Enhanced parameters dictionary
        """
        context = recipe.get('context', {})
        params = recipe.get('audio_parameters', {})
        
        risk_type = recipe.get('risk_type', 'Normal')
        collision_type = context.get('collision_type', 'Unknown')
        road_condition = context.get('road_condition', 'Normal')
        
        prompt = f"""You are an audio engineering expert. Given this vehicle safety scenario, optimize the audio generation parameters.

Scenario:
- Risk: {risk_type}
- Collision Type: {collision_type}
- Road: {road_condition}

Current Parameters:
- engine_intensity: {params.get('engine_intensity', 0.5)}
- tire_noise: {params.get('tire_noise', 0.2)}
- alert_level: {params.get('alert_level', 0.0)}
- ambient_level: {params.get('ambient_level', 0.3)}

Respond with ONLY a JSON object containing optimized values (0.0-1.0):
{{"engine_intensity": 0.X, "tire_noise": 0.X, "alert_level": 0.X, "ambient_level": 0.X}}"""

        try:
            response = self.invoke_bedrock_model(prompt)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                enhanced = json.loads(json_match.group())
                # Merge with original params
                params.update(enhanced)
                logger.debug(f"Enhanced params: {enhanced}")
        except Exception as e:
            logger.warning(f"Bedrock enhancement failed, using original params: {e}")
        
        return params
    
    def generate_audio_description(self, recipe: Dict) -> str:
        """
        Generate AI audio description using Bedrock.
        
        Args:
            recipe: Audio recipe dictionary
            
        Returns:
            AI-generated text-to-audio prompt
        """
        context = recipe.get('context', {})
        params = recipe.get('audio_parameters', {})
        
        risk_type = recipe.get('risk_type', 'Normal')
        location = context.get('location_name', 'Unknown location')
        collision_type = context.get('collision_type', 'Unknown')
        primary_factor = context.get('primary_factor', 'Unknown')
        road_condition = context.get('road_condition', 'Normal')
        
        prompt = f"""You are an expert in creating realistic vehicle safety audio scenarios. Generate a detailed text-to-audio prompt for creating a {params.get('duration_seconds', 5.0)}-second audio clip.

Scenario Context:
- Risk Level: {risk_type}
- Location: {location}
- Collision Type: {collision_type}
- Primary Cause: {primary_factor}
- Road Condition: {road_condition}

Audio Characteristics:
- Engine intensity: {params.get('engine_intensity', 0.5) * 100:.0f}% (low rumble, 40-80Hz)
- Tire noise: {params.get('tire_noise', 0.2) * 100:.0f}% (road friction, skid if >50%)
- Alert level: {params.get('alert_level', 0.0) * 100:.0f}% (warning beeps, 800-1200Hz)
- Ambient: {params.get('ambient_level', 0.3) * 100:.0f}% (environmental sounds)

Generate a concise, vivid prompt (100-150 words) for a text-to-audio AI model that describes:
1. Vehicle engine sounds (rumble, acceleration, braking)
2. Tire sounds (rolling, skidding, traction loss)
3. Alert/warning sounds (if risk is elevated)
4. Environmental ambience (wind, road noise)
5. Temporal progression (how sounds change during the 5 seconds)

Format as a single detailed paragraph suitable for AudioLDM, MusicGen, or Stable Audio models. No additional commentary."""

        return self.invoke_bedrock_model(prompt)
    
    def invoke_bedrock_model(self, prompt: str) -> str:
        """
        Invoke Bedrock Claude model.
        
        Args:
            prompt: User prompt
            
        Returns:
            Model response text
        """
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def generate_wav_file(self, recipe: Dict, output_path: Path, use_ai_enhancement: bool = True) -> bool:
        """
        Generate WAV file with optional Bedrock AI enhancement.
        
        Args:
            recipe: Audio recipe dictionary
            output_path: Path to save WAV file
            use_ai_enhancement: Whether to use Bedrock to optimize parameters
            
        Returns:
            True if successful
        """
        try:
            # Get base parameters
            params = recipe.get('audio_parameters', {}).copy()
            
            # Optionally enhance with Bedrock AI
            if use_ai_enhancement:
                try:
                    params = self.enhance_parameters(recipe)
                except Exception as e:
                    logger.warning(f"AI enhancement failed, using base params: {e}")
            
            # Generate audio samples
            audio = self.synthesizer.generate_audio(params)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Get sample rate
            sample_rate = params.get('sample_rate', 22050)
            
            # Write WAV file
            wavfile.write(str(output_path), sample_rate, audio_int16)
            
            logger.debug(f"Generated WAV: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate WAV: {e}")
            return False
    
    def generate_audio_spec(self, recipe: Dict, output_path: Path) -> bool:
        """
        Generate audio specification file with AI description.
        
        Args:
            recipe: Audio recipe dictionary
            output_path: Path to save the specification JSON
            
        Returns:
            True if successful
        """
        try:
            # Generate AI audio description
            logger.debug(f"Generating AI description for {recipe.get('event_id')}")
            audio_prompt = self.generate_audio_description(recipe)
            
            # Create comprehensive specification
            spec = {
                'recipe_id': recipe.get('recipe_id'),
                'event_id': recipe.get('event_id'),
                'risk_type': recipe.get('risk_type'),
                'audio_generation_prompt': audio_prompt,
                'technical_parameters': recipe.get('audio_parameters', {}),
                'context': recipe.get('context', {}),
                'output': recipe.get('output', {}),
                'metadata': {
                    'generator': 'AWS Bedrock',
                    'model': self.model_id,
                    'region': self.region_name,
                    'format': 'text-to-audio-spec',
                    'instructions': 'Use audio_generation_prompt with AudioLDM, MusicGen, or Stable Audio'
                }
            }
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(spec, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate spec: {e}")
            return False


def load_recipes(recipe_dir: str) -> List[Dict]:
    """Load all recipe JSON files from directory."""
    recipe_path = Path(recipe_dir)
    recipes = []
    
    for recipe_file in recipe_path.glob('*.json'):
        try:
            with open(recipe_file, 'r') as f:
                recipe = json.load(f)
                recipes.append(recipe)
        except Exception as e:
            logger.error(f"Failed to load recipe {recipe_file}: {e}")
    
    logger.info(f"Loaded {len(recipes)} recipes from {recipe_dir}")
    return recipes


def generate_wav_files(recipes: List[Dict], output_dir: str, region_name: str = None, use_ai: bool = False):
    """
    Generate WAV audio files with optional Bedrock AI enhancement.
    
    Args:
        recipes: List of audio recipe dictionaries
        output_dir: Directory for output WAV files
        region_name: AWS region for Bedrock (optional)
        use_ai: Whether to use Bedrock AI to enhance parameters
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create folders for risk types
    risk_types = ['Normal', 'TireSkid', 'EmergencyBraking', 'CollisionImminent']
    for risk_type in risk_types:
        (output_path / risk_type).mkdir(exist_ok=True)
    
    generator = BedrockAudioGenerator(region_name=region_name) if use_ai else None
    synthesizer = AudioSynthesizer()
    
    successful = 0
    failed = 0
    
    for idx, recipe in enumerate(recipes, start=1):
        try:
            # Get output info
            output_info = recipe.get('output', {})
            filename = output_info.get('filename', f'audio_{idx:05d}.wav')
            folder = output_info.get('folder', 'Normal')
            
            # Save to risk-type folder
            risk_folder = output_path / folder
            wav_file = risk_folder / filename
            
            # Generate WAV
            if generator and use_ai:
                success = generator.generate_wav_file(recipe, wav_file, use_ai_enhancement=True)
            else:
                # Direct synthesis without AI
                params = recipe.get('audio_parameters', {})
                audio = synthesizer.generate_audio(params)
                audio_int16 = (audio * 32767).astype(np.int16)
                sample_rate = params.get('sample_rate', 22050)
                wavfile.write(str(wav_file), sample_rate, audio_int16)
                success = True
            
            if success:
                successful += 1
            else:
                failed += 1
            
            if idx % 10 == 0:
                logger.info(f"Generated {idx}/{len(recipes)} WAV files")
                
        except Exception as e:
            logger.error(f"Failed for recipe {idx}: {e}")
            failed += 1
    
    logger.info(f"\nGeneration complete: {successful} successful, {failed} failed")
    
    # Show distribution
    for risk_type in risk_types:
        count = len(list((output_path / risk_type).glob('*.wav')))
        logger.info(f"  {risk_type}: {count} WAV files")


def generate_audio_specifications(recipes: List[Dict], output_dir: str, region_name: str = None):
    """
    Generate AI audio specifications using Bedrock.
    
    Creates JSON files with AI-generated prompts for text-to-audio models.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create folders for risk types
    risk_types = ['Normal', 'TireSkid', 'EmergencyBraking', 'CollisionImminent']
    for risk_type in risk_types:
        (output_path / risk_type).mkdir(exist_ok=True)
    
    generator = BedrockAudioGenerator(region_name=region_name)
    
    successful = 0
    failed = 0
    
    for idx, recipe in enumerate(recipes, start=1):
        try:
            # Get output info
            output_info = recipe.get('output', {})
            filename = output_info.get('filename', f'audio_{idx:05d}.wav')
            folder = output_info.get('folder', 'Normal')
            
            # Create spec filename
            spec_filename = filename.replace('.wav', '_audio_spec.json')
            
            # Save to risk-type folder
            risk_folder = output_path / folder
            spec_file = risk_folder / spec_filename
            
            # Generate using Bedrock
            if generator.generate_audio_spec(recipe, spec_file):
                successful += 1
            else:
                failed += 1
            
            if idx % 10 == 0:
                logger.info(f"Generated {idx}/{len(recipes)} audio specifications")
                
        except Exception as e:
            logger.error(f"Failed for recipe {idx}: {e}")
            failed += 1
    
    logger.info(f"\nGeneration complete: {successful} successful, {failed} failed")
    
    # Show distribution
    for risk_type in risk_types:
        count = len(list((output_path / risk_type).glob('*_audio_spec.json')))
        logger.info(f"  {risk_type}: {count} specifications")
    
    logger.info("\n" + "="*70)
    logger.info("📝 AI Audio Specifications Generated!")
    logger.info("="*70)
    logger.info("\nTo convert to WAV files, use text-to-audio models:")
    logger.info("  • AudioLDM: pip install audioldm")
    logger.info("  • MusicGen: https://github.com/facebookresearch/audiocraft")
    logger.info("  • Stable Audio: https://stability.ai/stable-audio")
    logger.info("\nEach JSON contains 'audio_generation_prompt' field.")
    logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate WAV audio files with optional Bedrock AI enhancement'
    )
    parser.add_argument(
        '--recipe-dir',
        type=str,
        default='/opt/ml/processing/input',
        help='Directory with recipe JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/opt/ml/processing/output',
        help='Output directory for WAV files'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=None,
        help='AWS region for Bedrock (only if using --use-ai)'
    )
    parser.add_argument(
        '--use-ai',
        action='store_true',
        default=True,
        help='Use Bedrock AI to enhance audio parameters (enabled by default)'
    )
    parser.add_argument(
        '--no-ai',
        action='store_true',
        help='Disable AI enhancement for faster generation'
    )
    
    args = parser.parse_args()
    
    # Override use_ai if --no-ai is specified
    if args.no_ai:
        args.use_ai = False
    
    logger.info("="*70)
    logger.info("🎵 BEDROCK AI AUDIO GENERATOR")
    logger.info("="*70)
    logger.info(f"Recipe dir: {args.recipe_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"AI Enhancement: {'✅ ENABLED' if args.use_ai else '⚠️  DISABLED'}")
    logger.info(f"Region: {args.region or 'auto-detect'}")
    logger.info("="*70)
    
    # Load recipes
    recipes = load_recipes(args.recipe_dir)
    
    if not recipes:
        logger.error("No recipes found!")
        sys.exit(1)
    
    # Generate WAV files
    generate_wav_files(recipes, args.output_dir, args.region, args.use_ai)
    
    logger.info("✅ WAV generation complete!")


if __name__ == '__main__':
    main()
