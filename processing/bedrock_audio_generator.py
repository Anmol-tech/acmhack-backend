"""
AWS Bedrock-Enhanced Audio Generation for Acoustic Shield.
Uses Bedrock AI to optimize parameters, then generates actual WAV files.
"""

import argparse
import json
import logging
import os
import sys
import time
import random
import io
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import wavfile
from scipy import signal
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioSynthesizer:
    """Synthesize WAV audio from parameters with optimized vectorized operations."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        # Pre-generate reusable filters for efficiency
        self._pink_filter = signal.butter(1, 0.5, btype="low")

    def generate_audio(self, params: Dict) -> np.ndarray:
        """Generate audio samples from parameters using optimized numpy operations."""
        duration = params.get("duration_seconds", 5.0)
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(
            0, duration, n_samples, dtype=np.float32
        )  # Use float32 for speed

        # Extract parameters
        ambient_level = params.get("ambient_level", 0.3)
        engine_intensity = params.get("engine_intensity", 0.5)
        tire_noise = params.get("tire_noise", 0.2)
        alert_level = params.get("alert_level", 0.0)

        # Generate components in parallel (numpy vectorization)
        ambient = self._pink_noise(n_samples) * ambient_level
        engine = self._engine_sound(t, engine_intensity)
        tire = self._tire_sound(t, tire_noise)
        alert = self._alert_sound(t, alert_level)

        # Mix and normalize (single pass)
        audio = ambient + engine + tire + alert
        return self._normalize(audio)

    def _pink_noise(self, n: int) -> np.ndarray:
        """Generate pink noise using pre-computed filter."""
        white = np.random.randn(n).astype(np.float32)
        b, a = self._pink_filter
        return signal.filtfilt(b, a, white) * 0.1

    def _engine_sound(self, t: np.ndarray, intensity: float) -> np.ndarray:
        """Generate engine rumble (40-80 Hz) - vectorized."""
        freq1, freq2 = 50 + np.random.randn() * 5, 70 + np.random.randn() * 5
        # Vectorized sine generation
        engine = (
            np.sin(2 * np.pi * freq1 * t) * 0.6 + np.sin(2 * np.pi * freq2 * t) * 0.4
        )
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

    # Available models with their capabilities
    AVAILABLE_MODELS = {
        "claude-3.5-sonnet-v2": {
            "id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "max_tokens": 4096,
            "cost": "medium",
            "performance": "excellent",
            "features": ["code_gen", "reasoning", "rag", "multilingual"],
        },
        "claude-3.5-haiku": {
            "id": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "max_tokens": 4096,
            "cost": "low",
            "performance": "fast",
            "features": ["chat", "code_gen", "rag"],
        },
        "nova-pro": {
            "id": "amazon.nova-pro-v1:0",
            "max_tokens": 5000,
            "cost": "low",
            "performance": "good",
            "features": ["code_gen", "reasoning", "rag", "multimodal"],
        },
        "nova-lite": {
            "id": "amazon.nova-lite-v1:0",
            "max_tokens": 5000,
            "cost": "very_low",
            "performance": "fast",
            "features": ["chat", "rag"],
        },
    }

    def __init__(
        self,
        region_name: str = None,
        rate_limit_delay: float = 0.5,
        model: str = "claude-3.5-sonnet-v2",
    ):
        """
        Initialize Bedrock audio generator.

        Args:
            region_name: AWS region name (auto-detected if None)
            rate_limit_delay: Minimum seconds between Bedrock API calls (per instance)
            model: Model to use ('claude-3.5-sonnet-v2', 'claude-3.5-haiku', 'nova-pro', 'nova-lite')
        """
        self.region_name = region_name or os.environ.get(
            "AWS_DEFAULT_REGION", "us-east-1"
        )

        # Select model
        if model not in self.AVAILABLE_MODELS:
            logger.warning(
                f"Unknown model '{model}', defaulting to claude-3.5-sonnet-v2"
            )
            model = "claude-3.5-sonnet-v2"

        model_config = self.AVAILABLE_MODELS[model]
        self.model_name = model
        self.model_id = model_config["id"]
        self.max_tokens = model_config["max_tokens"]
        self.is_anthropic = model.startswith("claude")

        # Configure boto3 with custom retry strategy
        config = Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            read_timeout=60,
            connect_timeout=60,
        )

        self.bedrock_client = boto3.client(
            "bedrock-runtime", region_name=self.region_name, config=config
        )
        # Use latest Claude 3.5 Sonnet v2 for better performance
        self.synthesizer = AudioSynthesizer()
        self.rate_limit_delay = rate_limit_delay
        self.last_api_call = 0
        self.throttle_count = 0
        self.max_consecutive_throttles = 5

        logger.info(f"Initialized Bedrock client in region: {self.region_name}")
        logger.info(f"Using model: {model} ({self.model_id})")
        logger.info(f"Rate limit delay: {rate_limit_delay}s between calls")

    def enhance_parameters(self, recipe: Dict) -> Dict:
        """
        Use Bedrock AI to enhance audio parameters based on context.

        Args:
            recipe: Audio recipe dictionary

        Returns:
            Enhanced parameters dictionary
        """
        context = recipe.get("context", {})
        params = recipe.get("audio_parameters", {}).copy()

        risk_type = recipe.get("risk_type", "Normal")
        collision_type = context.get("collision_type", "Unknown")
        road_condition = context.get("road_condition", "Normal")

        # Cache key based on risk type and major factors
        cache_key = f"{risk_type}_{collision_type}_{road_condition}"

        # Check cache first (dramatically speeds up processing)
        if not hasattr(self, "_param_cache"):
            self._param_cache = {}

        if cache_key in self._param_cache:
            cached_params = self._param_cache[cache_key].copy()
            params.update(cached_params)
            logger.debug(f"Using cached params for {cache_key}")
            return params

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

            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                enhanced = json.loads(json_match.group())
                # Cache for future use
                self._param_cache[cache_key] = enhanced.copy()
                # Merge with original params
                params.update(enhanced)
                logger.debug(f"Enhanced and cached params for {cache_key}: {enhanced}")
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
        context = recipe.get("context", {})
        params = recipe.get("audio_parameters", {})

        risk_type = recipe.get("risk_type", "Normal")
        location = context.get("location_name", "Unknown location")
        collision_type = context.get("collision_type", "Unknown")
        primary_factor = context.get("primary_factor", "Unknown")
        road_condition = context.get("road_condition", "Normal")

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

    def invoke_bedrock_model(self, prompt: str, max_retries: int = 5) -> str:
        """
        Invoke Bedrock Claude model with exponential backoff and jitter.

        Args:
            prompt: User prompt
            max_retries: Maximum number of retry attempts

        Returns:
            Model response text
        """
        # Rate limiting: ensure minimum delay between calls
        elapsed = time.time() - self.last_api_call
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)

        for attempt in range(max_retries):
            try:
                # Format request based on model type
                if self.is_anthropic:
                    # Claude format
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    }
                else:
                    # Amazon Nova format
                    request_body = {
                        "messages": [{"role": "user", "content": [{"text": prompt}]}],
                        "inferenceConfig": {
                            "max_new_tokens": self.max_tokens,
                            "temperature": 0.7,
                        },
                    }

                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id, body=json.dumps(request_body)
                )

                self.last_api_call = time.time()
                self.throttle_count = 0  # Reset on success

                response_body = json.loads(response["body"].read())

                # Extract text based on model type
                if self.is_anthropic:
                    return response_body["content"][0]["text"]
                else:
                    # Nova format
                    return response_body["output"]["message"]["content"][0]["text"]

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                if error_code == "ThrottlingException":
                    self.throttle_count += 1

                    # If too many consecutive throttles, give up to avoid wasting time
                    if self.throttle_count >= self.max_consecutive_throttles:
                        logger.warning(
                            f"Hit throttle limit {self.max_consecutive_throttles} times, giving up on AI enhancement"
                        )
                        raise

                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        base_delay = 2**attempt
                        jitter = random.uniform(0, base_delay * 0.5)
                        wait_time = base_delay + jitter

                        logger.warning(
                            f"Throttled (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"Bedrock API throttled after {max_retries} attempts"
                        )
                        raise
                else:
                    logger.error(f"Bedrock API error: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error invoking Bedrock: {e}")
                raise

        raise Exception(f"Failed after {max_retries} retries")

    def generate_wav_file(
        self, recipe: Dict, output_path: Path, use_ai_enhancement: bool = True
    ) -> bool:
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
            params = recipe.get("audio_parameters", {}).copy()

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
            sample_rate = params.get("sample_rate", 22050)

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
                "recipe_id": recipe.get("recipe_id"),
                "event_id": recipe.get("event_id"),
                "risk_type": recipe.get("risk_type"),
                "audio_generation_prompt": audio_prompt,
                "technical_parameters": recipe.get("audio_parameters", {}),
                "context": recipe.get("context", {}),
                "output": recipe.get("output", {}),
                "metadata": {
                    "generator": "AWS Bedrock",
                    "model": self.model_id,
                    "region": self.region_name,
                    "format": "text-to-audio-spec",
                    "instructions": "Use audio_generation_prompt with AudioLDM, MusicGen, or Stable Audio",
                },
            }

            # Save as JSON
            with open(output_path, "w") as f:
                json.dump(spec, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to generate spec: {e}")
            return False


def load_recipes(recipe_dir: str) -> List[Dict]:
    """Load all recipe JSON files from directory."""
    recipe_path = Path(recipe_dir)
    recipes = []

    for recipe_file in recipe_path.glob("*.json"):
        try:
            with open(recipe_file, "r") as f:
                recipe = json.load(f)
                recipes.append(recipe)
        except Exception as e:
            logger.error(f"Failed to load recipe {recipe_file}: {e}")

    logger.info(f"Loaded {len(recipes)} recipes from {recipe_dir}")
    return recipes


def upload_wav_to_s3(
    audio_data: np.ndarray,
    sample_rate: int,
    s3_key: str,
    bucket: str,
    region: str = None,
) -> bool:
    """
    Upload WAV audio data directly to S3.

    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Audio sample rate
        s3_key: S3 key (path) for the file
        bucket: S3 bucket name
        region: AWS region (optional)

    Returns:
        True if successful
    """
    try:
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Write to in-memory buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)

        # Upload to S3
        s3_client = boto3.client("s3", region_name=region)
        s3_client.put_object(
            Bucket=bucket, Key=s3_key, Body=buffer.getvalue(), ContentType="audio/wav"
        )

        logger.debug(f"Uploaded to s3://{bucket}/{s3_key}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def generate_wav_files(
    recipes: List[Dict],
    output_dir: str,
    region_name: str = None,
    use_ai: bool = False,
    model: str = "claude-3.5-sonnet-v2",
    s3_bucket: str = None,
    s3_prefix: str = "train/",
):
    """
    Generate WAV audio files with optional Bedrock AI enhancement.

    Args:
        recipes: List of audio recipe dictionaries
        output_dir: Directory for output WAV files (local or ignored if s3_bucket provided)
        region_name: AWS region for Bedrock (optional)
        use_ai: Whether to use Bedrock AI to enhance parameters
        model: Bedrock model to use (default: claude-3.5-sonnet-v2)
        s3_bucket: S3 bucket to upload files (if None, saves locally)
        s3_prefix: S3 prefix/folder (default: train/)
    """
    # Determine if we're using S3 or local storage
    use_s3 = s3_bucket is not None

    if use_s3:
        logger.info(f"Output mode: S3 (s3://{s3_bucket}/{s3_prefix})")
    else:
        logger.info(f"Output mode: Local ({output_dir})")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Create folders for risk types
        risk_types = ["Normal", "TireSkid", "EmergencyBraking", "CollisionImminent"]
        for risk_type in risk_types:
            (output_path / risk_type).mkdir(exist_ok=True)

    # Calculate per-instance rate limit with optimization
    # Bedrock has ~10 TPS limit, with aggressive caching we can increase parallelism
    # Most API calls will hit cache after first few, so start conservative then speed up
    rate_limit_delay = (
        1.5  # Reduced from 3.0s - caching reduces actual API calls by ~95%
    )

    generator = (
        BedrockAudioGenerator(
            region_name=region_name, rate_limit_delay=rate_limit_delay, model=model
        )
        if use_ai
        else None
    )
    synthesizer = AudioSynthesizer()

    successful = 0
    failed = 0
    ai_enhanced = 0
    fallback = 0
    cached = 0

    logger.info(f"Starting generation with AI={'enabled' if use_ai else 'disabled'}")
    if use_ai:
        logger.info(f"Rate limit: {rate_limit_delay}s delay between Bedrock calls")
        logger.info(
            f"Caching enabled: Repeated risk scenarios will use cached AI params"
        )

    for idx, recipe in enumerate(recipes, start=1):
        try:
            # Get output info
            output_info = recipe.get("output", {})
            filename = output_info.get("filename", f"audio_{idx:05d}.wav")
            folder = output_info.get("folder", "Normal")

            # Prepare file path
            if use_s3:
                s3_key = f"{s3_prefix}{folder}/{filename}"
                wav_file = None  # No local file needed
            else:
                risk_folder = output_path / folder
                wav_file = risk_folder / filename
                s3_key = None

            # Generate audio data
            params = recipe.get("audio_parameters", {})
            sample_rate = params.get("sample_rate", 22050)

            # Generate WAV with AI or direct synthesis
            if generator and use_ai:
                try:
                    # Track if we used cache
                    cache_size_before = len(getattr(generator, "_param_cache", {}))

                    # Get AI-enhanced parameters
                    enhanced_params = generator.enhance_parameters(recipe)
                    audio = synthesizer.generate_audio(enhanced_params)

                    cache_size_after = len(getattr(generator, "_param_cache", {}))

                    # Upload to S3 or save locally
                    if use_s3:
                        success = upload_wav_to_s3(
                            audio, sample_rate, s3_key, s3_bucket, region_name
                        )
                    else:
                        audio_int16 = (audio * 32767).astype(np.int16)
                        wavfile.write(str(wav_file), sample_rate, audio_int16)
                        success = True

                    if success:
                        ai_enhanced += 1
                        if cache_size_after == cache_size_before:
                            cached += 1  # Used cached parameters

                except Exception as e:
                    # Fallback to non-AI generation on any error
                    logger.warning(
                        f"AI enhancement failed for {filename}, using fallback: {e}"
                    )
                    audio = synthesizer.generate_audio(params)

                    if use_s3:
                        success = upload_wav_to_s3(
                            audio, sample_rate, s3_key, s3_bucket, region_name
                        )
                    else:
                        audio_int16 = (audio * 32767).astype(np.int16)
                        wavfile.write(str(wav_file), sample_rate, audio_int16)
                        success = True
                    fallback += 1
            else:
                # Direct synthesis without AI
                audio = synthesizer.generate_audio(params)

                if use_s3:
                    success = upload_wav_to_s3(
                        audio, sample_rate, s3_key, s3_bucket, region_name
                    )
                else:
                    audio_int16 = (audio * 32767).astype(np.int16)
                    wavfile.write(str(wav_file), sample_rate, audio_int16)
                    success = True

            if success:
                successful += 1
            else:
                failed += 1

            if idx % 100 == 0:
                cache_hit_rate = (cached / ai_enhanced * 100) if ai_enhanced > 0 else 0
                logger.info(
                    f"Progress: {idx}/{len(recipes)} WAV files (AI: {ai_enhanced}, Cached: {cached} [{cache_hit_rate:.1f}%], Fallback: {fallback}, Failed: {failed})"
                )

        except Exception as e:
            logger.error(f"Failed for recipe {idx}: {e}")
            failed += 1

    logger.info(f"\nGeneration complete: {successful} successful, {failed} failed")
    if use_ai:
        cache_hit_rate = (cached / ai_enhanced * 100) if ai_enhanced > 0 else 0
        logger.info(f"  AI enhanced: {ai_enhanced}")
        logger.info(
            f"  Cache hits: {cached} ({cache_hit_rate:.1f}% - {cached} fewer API calls!)"
        )
        logger.info(f"  Fallback (no AI): {fallback}")
        logger.info(
            f"  Unique AI parameter sets: {len(getattr(generator, '_param_cache', {}))}"
        )

    # Show distribution
    risk_types = ["Normal", "TireSkid", "EmergencyBraking", "CollisionImminent"]
    if use_s3:
        # Count files in S3
        try:
            s3_client = boto3.client("s3", region_name=region_name)
            for risk_type in risk_types:
                prefix = f"{s3_prefix}{risk_type}/"
                response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
                count = len(
                    [
                        obj
                        for obj in response.get("Contents", [])
                        if obj["Key"].endswith(".wav")
                    ]
                )
                logger.info(
                    f"  {risk_type}: {count} WAV files in s3://{s3_bucket}/{prefix}"
                )
        except Exception as e:
            logger.warning(f"Could not count S3 files: {e}")
    else:
        # Count local files
        for risk_type in risk_types:
            count = len(list((output_path / risk_type).glob("*.wav")))
            logger.info(f"  {risk_type}: {count} WAV files")


def generate_audio_specifications(
    recipes: List[Dict], output_dir: str, region_name: str = None
):
    """
    Generate AI audio specifications using Bedrock.

    Creates JSON files with AI-generated prompts for text-to-audio models.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create folders for risk types
    risk_types = ["Normal", "TireSkid", "EmergencyBraking", "CollisionImminent"]
    for risk_type in risk_types:
        (output_path / risk_type).mkdir(exist_ok=True)

    generator = BedrockAudioGenerator(region_name=region_name)

    successful = 0
    failed = 0

    for idx, recipe in enumerate(recipes, start=1):
        try:
            # Get output info
            output_info = recipe.get("output", {})
            filename = output_info.get("filename", f"audio_{idx:05d}.wav")
            folder = output_info.get("folder", "Normal")

            # Create spec filename
            spec_filename = filename.replace(".wav", "_audio_spec.json")

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
        count = len(list((output_path / risk_type).glob("*_audio_spec.json")))
        logger.info(f"  {risk_type}: {count} specifications")

    logger.info("\n" + "=" * 70)
    logger.info("📝 AI Audio Specifications Generated!")
    logger.info("=" * 70)
    logger.info("\nTo convert to WAV files, use text-to-audio models:")
    logger.info("  • AudioLDM: pip install audioldm")
    logger.info("  • MusicGen: https://github.com/facebookresearch/audiocraft")
    logger.info("  • Stable Audio: https://stability.ai/stable-audio")
    logger.info("\nEach JSON contains 'audio_generation_prompt' field.")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate WAV audio files with optional Bedrock AI enhancement"
    )
    parser.add_argument(
        "--recipe-dir",
        type=str,
        default="/opt/ml/processing/input",
        help="Directory with recipe JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/output",
        help="Output directory for WAV files",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region for Bedrock (only if using --use-ai)",
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        default=True,
        help="Use Bedrock AI to enhance audio parameters (enabled by default)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI enhancement for faster generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3.5-sonnet-v2",
        choices=["claude-3.5-sonnet-v2", "claude-3.5-haiku", "nova-pro", "nova-lite"],
        help="Bedrock model to use (default: claude-3.5-sonnet-v2)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket to upload WAV files (e.g., acousticshield-ml). If not specified, saves locally.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="train/",
        help="S3 prefix/folder for WAV files (default: train/)",
    )

    args = parser.parse_args()

    # Override use_ai if --no-ai is specified
    if args.no_ai:
        args.use_ai = False

    logger.info("=" * 70)
    logger.info("🎵 BEDROCK AI AUDIO GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Recipe dir: {args.recipe_dir}")
    if args.s3_bucket:
        logger.info(f"Output: s3://{args.s3_bucket}/{args.s3_prefix}")
    else:
        logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"AI Enhancement: {'✅ ENABLED' if args.use_ai else '⚠️  DISABLED'}")
    if args.use_ai:
        logger.info(f"Model: {args.model}")
    logger.info(f"Region: {args.region or 'auto-detect'}")
    logger.info("=" * 70)

    # Load recipes
    recipes = load_recipes(args.recipe_dir)

    if not recipes:
        logger.error("No recipes found!")
        sys.exit(1)

    # Generate WAV files
    generate_wav_files(
        recipes,
        args.output_dir,
        args.region,
        args.use_ai,
        model=args.model if args.use_ai else None,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )

    logger.info("✅ WAV generation complete!")
    if args.s3_bucket:
        logger.info(f"📦 Files uploaded to s3://{args.s3_bucket}/{args.s3_prefix}")


if __name__ == "__main__":
    main()
