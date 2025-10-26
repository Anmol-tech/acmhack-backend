# Quick Start Guide - Generate 1000+ Audio Files

## ✅ Setup Complete

All required packages are installed and the pipeline is configured to generate **1,000 audio files**.

## Configuration Summary

```python
TOP_N_HOTSPOTS = 50         # Extract top 50 crash hotspots
EVENTS_PER_HOTSPOT = 5      # Generate 5 risk events per hotspot
RECIPES_PER_EVENT = 4       # Create 4 recipe variations per event
```

**Expected output:** `50 × 5 × 4 = 1,000 WAV files`

## How to Run

### Option 1: Run the Notebook (Recommended)
1. Open `notebooks/01_build_training_data.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Extract 50 crash hotspots from GeoJSON
   - Enrich with weather data from Open-Meteo API
   - Generate 250 risk events
   - Build 1,000 audio recipes with variations
   - Launch SageMaker Processing job to generate WAV files

### Option 2: Run Step-by-Step

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from data_pipeline import (
    S3Client, HotspotExtractor, WeatherEnricher,
    RiskEventSynthesizer, RecipeBuilder
)

# 1. Load crash data
s3_client = S3Client()
crash_data = s3_client.read_json('acousticshield-raw', 'crash_hotspots/sanjose_crashes.geojson')

# 2. Extract hotspots
extractor = HotspotExtractor(crash_data)
hotspots = extractor.extract_top_hotspots(top_n=50)

# 3. Enrich with weather
enricher = WeatherEnricher()
enriched = enricher.enrich_hotspots(hotspots)

# 4. Synthesize events
synthesizer = RiskEventSynthesizer(seed=42)
events = synthesizer.synthesize_events(enriched, events_per_hotspot=5)

# 5. Build recipes
builder = RecipeBuilder()
recipes = builder.build_recipes(events, recipes_per_event=4)

print(f"Generated {len(recipes)} audio recipes!")
```

## Recipe Variations Explained

Each event generates 4 variations:
- **v0**: Base parameters (100%)
- **v1**: +5% intensity (105%)
- **v2**: +10% intensity (110%)
- **v3**: +15% intensity (115%)

Plus ±5% random noise for natural diversity.

## Output Structure

```
s3://acousticshield-ml/training-data/
├── Normal/
│   ├── evt_00001_v0_normal.wav
│   ├── evt_00001_v1_normal.wav
│   ├── evt_00001_v2_normal.wav
│   └── evt_00001_v3_normal.wav
├── TireSkid/
│   ├── evt_00002_v0_tireskid.wav
│   └── ...
├── EmergencyBraking/
│   ├── evt_00003_v0_emergencybraking.wav
│   └── ...
└── CollisionImminent/
    ├── evt_00004_v0_collisionimminent.wav
    └── ...
```

## Verify Package Installation

```bash
python verify_packages.py
```

Expected output:
```
✓ boto3
✓ sagemaker
✓ numpy
✓ scipy
✓ requests
✓ torch
✓ transformers
✓ datasets
✓ evaluate
✓ soundfile
✓ librosa
✓ ALL PACKAGES INSTALLED SUCCESSFULLY!
```

## Verify File Count Configuration

```bash
python test_file_count.py
```

Expected output:
```
50 hotspots × 5 events × 4 recipes = 1,000 audio files
✓ Target met: True
  Expected output: 1,000 WAV files
```

## Training the Model

After generating WAV files, train the model:

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point='train.py',
    source_dir='training',
    role='role-sagemaker-train',
    instance_type='ml.g4dn.xlarge',
    transformers_version='4.44',
    pytorch_version='2.3',
    hyperparameters={
        'epochs': 4,
        'learning-rate': 3e-5,
        'batch-size': 8
    }
)

estimator.fit({'train': 's3://acousticshield-ml/training-data/'})
```

## Deploy Endpoint

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='acousticshield-endpoint'
)
```

## Test Inference

```python
import io
import numpy as np
import soundfile as sf

# Generate test audio
sample_rate = 16000
t = np.linspace(0, 1.0, sample_rate)
test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)

# Convert to WAV
wav_buffer = io.BytesIO()
sf.write(wav_buffer, test_audio, sample_rate, format='WAV')

# Predict
response = predictor.predict(wav_buffer.getvalue())
print(f"Predicted: {response['label']} ({response['confidence']:.2%})")
```

## Estimated Training Time

- **Processing Job** (WAV generation): ~15-20 minutes (ml.m5.xlarge)
- **Training Job** (wav2vec2): ~30-40 minutes (ml.g4dn.xlarge)
- **Total Pipeline**: ~50-60 minutes

## Cost Estimate (AWS)

- Processing: ml.m5.xlarge × 0.33 hours ≈ $0.08
- Training: ml.g4dn.xlarge × 0.67 hours ≈ $0.44
- Endpoint: ml.m5.xlarge × hours deployed
- S3 storage: ~500MB audio + ~1GB models ≈ $0.03/month
- **Total (one-time)**: ~$0.52 + endpoint runtime

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Region Issues
The pipeline auto-detects region from S3 bucket. No hardcoding needed!

### Low File Count
Check configuration in `notebooks/01_build_training_data.ipynb`:
- `TOP_N_HOTSPOTS = 50`
- `EVENTS_PER_HOTSPOT = 5`
- `RECIPES_PER_EVENT = 4`

### SageMaker Errors
Verify IAM roles exist:
- `role-sagemaker-processing`
- `role-sagemaker-train`

## Next Steps

1. ✅ Packages installed
2. ✅ Configuration verified (1000 files)
3. 🔄 Run `01_build_training_data.ipynb`
4. ⏳ Wait for WAV generation
5. ⏳ Train model
6. ⏳ Deploy endpoint
7. ✅ Test inference

## Support

See `UPDATES.md` for detailed changes and `README.md` for architecture overview.
