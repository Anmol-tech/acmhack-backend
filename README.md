# Acoustic Shield - SageMaker Data Pipeline

Complete data pipeline for generating audio training data from crash hotspot analysis.

## Overview

This pipeline:
1. **Extracts** top 25 crash hotspots from GeoJSON data
2. **Enriches** locations with weather data (Open-Meteo API)
3. **Synthesizes** risk events based on crash history and weather
4. **Builds** audio generation recipes for 4 risk types
5. **Generates** WAV training files using SageMaker Processing

## Architecture

```
data_pipeline/
├── hotspot_extractor.py    # Extract top crash locations
├── weather_enricher.py     # Fetch weather data (Open-Meteo)
├── risk_event_synth.py     # Create synthetic risk events
├── recipe_builder.py       # Build audio generation specs
└── s3_utils.py            # S3 operations (region-agnostic)

processing/
└── augment.py             # SageMaker processing script

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
    ├── evt_00001_normal.wav
    ├── evt_00002_tireskid.wav
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
- Generate WAV files in S3

### 4. Verify Outputs

The pipeline will output:
- Risk events JSON: `s3://acousticshield-raw/risk_events/`
- Audio recipes JSON: `s3://acousticshield-raw/prompts/`
- Training WAV files: `s3://acousticshield-ml/train/`

## SageMaker Processing Job

Uses PyTorch CPU container:
- Instance: `ml.m5.xlarge`
- Image: `pytorch-training:2.0.1-cpu-py310`
- Script: `processing/augment.py`

The processing job:
1. Reads recipe JSON from S3
2. Generates synthetic audio (5 seconds per event)
3. Applies audio parameters (engine, tire noise, alerts)
4. Writes WAV files to S3

## Audio Generation

Each audio file contains:
- **Ambient noise**: Pink noise based on weather
- **Engine sound**: Low-frequency rumble (40-80 Hz)
- **Tire noise**: Road friction sounds (intensifies for skids)
- **Alert sounds**: Beeping warnings (800-1200 Hz)

Parameters are dynamically adjusted based on:
- Risk type (Normal → CollisionImminent)
- Weather conditions (rain, wind, temperature)
- Time of day (morning, afternoon, evening, night)

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
  - S3 read/write access
  - CloudWatch logs access

## Output Format

### Risk Event JSON
```json
{
  "event_id": "evt_00001",
  "risk_type": "TireSkid",
  "risk_score": 65.5,
  "road_name": "Main St",
  "weather_risk": "high",
  "time_category": "evening"
}
```

### Audio Recipe JSON
```json
{
  "recipe_id": "recipe_evt_00001",
  "risk_type": "TireSkid",
  "audio_parameters": {
    "ambient_level": 0.4,
    "engine_intensity": 0.7,
    "tire_noise": 0.9,
    "alert_level": 0.4,
    "duration_seconds": 5.0,
    "sample_rate": 22050
  },
  "output": {
    "filename": "evt_00001_tireskid.wav"
  }
}
```

## Troubleshooting

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### S3 Access Errors
Check AWS credentials and IAM permissions:
```bash
aws sts get-caller-identity
```

### SageMaker Job Failures
Check CloudWatch logs in AWS Console:
- Navigate to SageMaker → Processing Jobs
- Click on failed job → View logs

## Next Steps

After pipeline completion:
1. Validate WAV file quality
2. Build audio classification model
3. Train on generated dataset
4. Deploy for real-time inference

## License

Copyright 2025 Acoustic Shield Team
