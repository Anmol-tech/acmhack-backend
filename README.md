# Acoustic Shield - SageMaker Data Pipeline

Complete data pipeline for generating audio training data from crash hotspot analysis.

## Overview

This pipeline:
1. **Extracts** top 25 crash hotspots from GeoJSON data
2. **Enriches** locations with weather data (Open-Meteo API)
3. **Synthesizes** risk events based on crash history and weather
4. **Builds** audio generation recipes for 4 risk types
5. **Generates** WAV audio files (with optional Bedrock AI enhancement)

## Architecture

```
data_pipeline/
├── hotspot_extractor.py    # Extract top crash locations
├── weather_enricher.py     # Fetch weather data (Open-Meteo)
├── risk_event_synth.py     # Create synthetic risk events
├── recipe_builder.py       # Build audio generation specs
└── s3_utils.py            # S3 operations (region-agnostic)

processing/
└── bedrock_audio_generator.py  # Generate AI-enhanced WAV files (Bedrock + synthesis)

notebooks/
└── 01_build_training_data.ipynb  # Orchestration notebook
```

## Risk Types

- **Normal**: Standard driving conditions
- **TireSkid**: Slippery conditions, potential loss of traction
- **EmergencyBraking**: Sudden stop required
- **CollisionImminent**: Immediate crash danger

## Configuration

All parameters are region-agnostic and configurable:

```python
RAW_BUCKET = 'acousticshield-raw'
ML_BUCKET = 'acousticshield-ml'
CRASH_FILE_KEY = 'crash_hotspots/sanjose_crashes.geojson'
SAGEMAKER_ROLE = 'role-sagemaker-processing'
```

## S3 Structure

```
acousticshield-raw/
├── crash_hotspots/
│   └── sanjose_crashes.geojson
├── risk_events/
│   └── risk_events.json
└── prompts/
    └── audio_recipes.json

acousticshield-ml/
└── train/
    ├── Normal/
    │   ├── evt_00001_normal.wav
    │   └── ...
    ├── TireSkid/
    │   ├── evt_00026_tireskid.wav
    │   └── ...
    ├── EmergencyBraking/
    │   └── ...
    └── CollisionImminent/
        └── ...
```

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
aws configure
```

### 3. Run Pipeline

Open `notebooks/01_build_training_data.ipynb` and run all cells.

The notebook will:
- Extract crash hotspots
- Fetch weather data
- Generate risk events
- Create audio recipes
- Launch SageMaker processing job
- Generate WAV audio files in S3

### 4. Verify Outputs

The pipeline will output:
- Risk events JSON: `s3://acousticshield-raw/risk_events/`
- Audio recipes JSON: `s3://acousticshield-raw/prompts/`
- Training WAV files: `s3://acousticshield-ml/train/{RiskType}/`

## SageMaker Processing Job

Uses PyTorch CPU container:
- Instance: `ml.m5.xlarge`
- Image: `pytorch-training:2.0.0-cpu-py310`
- Script: `processing/bedrock_audio_generator.py`

The processing job:
1. Reads recipe JSON from S3
2. **Uses Bedrock AI** to analyze scenario and optimize parameters
3. Generates WAV audio files (16-bit PCM, 22.05kHz)
4. Writes AI-enhanced WAV files to S3 (organized by risk type)

## AI-Enhanced Audio Generation

**Default mode uses Bedrock AI for intelligent audio synthesis:**

### How It Works
1. **Bedrock Claude** analyzes scenario context:
   - Risk level (Normal → CollisionImminent)
   - Collision type (Head-On, Rear End, Sideswipe, etc.)
   - Road conditions (wet, icy, dry)
   
2. **AI optimizes parameters** intelligently:
   - Engine intensity (0.0-1.0)
   - Tire noise levels (normal vs skid)
   - Alert urgency (beep frequency/intensity)
   - Ambient levels

3. **Local synthesis** generates WAV:
   - Pink noise for ambient
   - 40-80 Hz engine rumble
   - Tire/road friction sounds
   - 800-1200 Hz warning beeps

### Output Format
- **Format**: 16-bit PCM WAV
- **Sample Rate**: 22.05 kHz
- **Duration**: 5 seconds per event
- **Channels**: Mono

### Fast Mode (Optional)
To disable AI enhancement for faster generation:
```python
arguments=['--region', REGION, '--no-ai']  # Disable AI, use base parameters
```

### Why This Approach?

**Bedrock can't generate audio directly** (it's an LLM service), but we use it brilliantly:
- ✅ AI analyzes context and optimizes parameters
- ✅ Local synthesis creates actual WAV files
- ✅ Best of both: AI intelligence + guaranteed WAV output
- ✅ No dependency on external text-to-audio models

## Region Handling

All components are **region-agnostic**:
- Bucket regions auto-detected from S3
- SageMaker sessions use detected region
- No hard-coded regional endpoints

## Dependencies

- `boto3`: AWS SDK
- `sagemaker`: SageMaker Python SDK
- `numpy`: Numerical operations
- `scipy`: Audio signal processing
- `requests`: HTTP client (Open-Meteo API)

## API Keys

**No API keys required!** Open-Meteo API is free and doesn't require authentication.

## IAM Roles

Required roles:
- `role-sagemaker-processing`: SageMaker execution role
  - S3 read/write access (s3:GetObject, s3:PutObject)
  - CloudWatch logs access (logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents)
  - **Bedrock invoke model permissions** (`bedrock:InvokeModel` on `anthropic.claude-3-sonnet-20240229-v1:0`)

## Next Steps

After pipeline completion:
1. **Review AI-enhanced WAV files**: Download and listen to generated audio samples
2. **Verify quality**: Check that risk types sound appropriately different (AI should make them more distinct)
3. **Analyze distribution**: Ensure balanced dataset across 4 risk types
4. **Build classifier**: Train audio classification model (CNN, LSTM, or Transformer)
5. **Train & validate**: Use generated dataset with train/val/test split
6. **Deploy model**: Real-time vehicle safety inference in production

## License

Copyright 2025 Acoustic Shield Team
