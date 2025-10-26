# SageMaker Training & Deployment - Acoustic Shield

Complete guide for training and deploying the audio classification model on AWS SageMaker.

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Training Script](#training-script)
- [Inference Script](#inference-script)
- [Deployment Notebook](#deployment-notebook)
- [Configuration Options](#configuration-options)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)

## 🎯 Overview

This implementation fine-tunes **facebook/wav2vec2-base** for 4-class audio classification:
- **Normal** - Normal driving conditions
- **TireSkid** - Tire screeching/loss of traction
- **EmergencyBraking** - Sudden braking event
- **CollisionImminent** - Imminent collision scenario

## ✅ Prerequisites

### AWS Resources
- ✅ S3 buckets created:
  - `s3://acousticshield-raw` - Raw data
  - `s3://acousticshield-ml` - Training data & models
- ✅ IAM role: `role-sagemaker-train` with permissions:
  - S3 read/write access
  - SageMaker full access
  - CloudWatch logs access
- ✅ Training data organized in audiofolder format:
  ```
  s3://acousticshield-ml/train/
  ├── Normal/*.wav
  ├── TireSkid/*.wav
  ├── EmergencyBraking/*.wav
  └── CollisionImminent/*.wav
  ```

### Local Environment
```bash
pip install boto3 sagemaker soundfile numpy
```

## 🚀 Quick Start

### Option 1: Use Jupyter Notebook (Recommended)

```bash
cd notebooks
jupyter notebook 02_train_and_deploy.ipynb
```

Run all cells to:
1. Configure training parameters
2. Launch SageMaker training job
3. Deploy real-time endpoint
4. Test inference

### Option 2: Use Python Script

```python
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

# Auto-detect region
s3 = boto3.client('s3')
bucket_location = s3.get_bucket_location(Bucket='acousticshield-ml')['LocationConstraint']
region = bucket_location if bucket_location else 'us-east-1'

# Initialize session
session = sagemaker.Session(boto3.Session(region_name=region))

# Get IAM role
iam = boto3.client('iam', region_name=region)
role_arn = iam.get_role(RoleName='role-sagemaker-train')['Role']['Arn']

# Create estimator
estimator = HuggingFace(
    entry_point='train.py',
    source_dir='training',
    role=role_arn,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    transformers_version='4.44',
    pytorch_version='2.3',
    py_version='py311',
    hyperparameters={
        'epochs': 4,
        'learning-rate': 3e-5,
        'batch-size': 8,
    },
    output_path='s3://acousticshield-ml/models/',
    sagemaker_session=session,
)

# Train
estimator.fit({'train': 's3://acousticshield-ml/train/'})

# Deploy
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='acousticshield-endpoint'
)

print(f"✅ Endpoint deployed: {predictor.endpoint_name}")
```

## 📄 Training Script

**File**: `training/train.py`

### Key Features
- Loads data using HuggingFace `audiofolder` format
- Auto-resamples all audio to 16 kHz
- Fine-tunes `facebook/wav2vec2-base` 
- Computes accuracy + macro-F1 metrics
- Saves model + feature extractor
- Supports optional validation set

### Arguments
```python
--epochs              # Training epochs (default: 4)
--learning-rate       # Learning rate (default: 3e-5)
--batch-size          # Batch size per device (default: 8)
--warmup-steps        # Warmup steps (default: 500)
--max-audio-length    # Max samples (default: 80000 = 5 sec @ 16kHz)
```

### Training Process
1. Load audiofolder dataset from S3
2. Split train/val (90/10) if no validation set
3. Resample audio to 16 kHz
4. Extract wav2vec2 features
5. Fine-tune classification head
6. Evaluate each epoch
7. Save best model (by F1 score)
8. Upload to S3

### Output Structure
```
/opt/ml/model/
├── config.json
├── pytorch_model.bin
├── preprocessor_config.json
├── label_info.json
└── special_tokens_map.json
```

## 🔮 Inference Script

**File**: `training/inference.py`

### Key Features
- Accepts `audio/wav` format (any sample rate)
- Auto-resamples to 16 kHz if needed
- Converts stereo → mono automatically
- Returns JSON with predictions
- Minimal dependencies (transformers, soundfile, torch)

### Input Format
```
Content-Type: audio/wav
Body: <WAV file bytes>
```

### Output Format
```json
{
  "label": "TireSkid",
  "confidence": 0.8523,
  "probs": {
    "Normal": 0.0523,
    "TireSkid": 0.8523,
    "EmergencyBraking": 0.0834,
    "CollisionImminent": 0.0120
  }
}
```

### Handler Functions
- `model_fn(model_dir)` - Load model once at startup
- `input_fn(request_body, content_type)` - Parse WAV bytes
- `predict_fn(input_data, model)` - Run inference
- `output_fn(prediction, accept)` - Serialize JSON

## 📓 Deployment Notebook

**File**: `notebooks/02_train_and_deploy.ipynb`

### Structure
1. **Configuration** - Set S3 paths, role, hyperparameters
2. **Initialize Session** - Auto-detect region
3. **Create Estimator** - Configure HuggingFace training
4. **Start Training** - Launch training job (~30-40 min)
5. **Deploy Endpoint** - Deploy real-time endpoint (~5-8 min)
6. **Test Endpoint** - Smoke test with sample audio
7. **Save Info** - Export endpoint details
8. **Cleanup** - Optional endpoint deletion

### Customization
All parameters at top of notebook:
```python
# S3 paths
TRAIN_S3 = "s3://acousticshield-ml/train/"
VAL_S3 = "s3://acousticshield-ml/val/"  # Or None

# Hyperparameters
EPOCHS = 4
LEARNING_RATE = 3e-5
BATCH_SIZE = 8

# Instances
TRAIN_INSTANCE_TYPE = "ml.g4dn.xlarge"
ENDPOINT_INSTANCE_TYPE = "ml.m5.xlarge"
```

## ⚙️ Configuration Options

### Change Training Duration
```python
EPOCHS = 8  # Train longer for better accuracy
```

### Change Learning Rate
```python
LEARNING_RATE = 5e-5  # Higher for faster convergence
LEARNING_RATE = 1e-5  # Lower for fine-tuning
```

### Change Batch Size
```python
BATCH_SIZE = 16  # Larger batch (needs more memory)
BATCH_SIZE = 4   # Smaller batch (if OOM errors)
```

### Skip Validation Set
```python
VAL_S3 = None  # Will split from train data (90/10)
```

### Use Different Instance
```python
TRAIN_INSTANCE_TYPE = "ml.p3.2xlarge"     # Faster GPU
TRAIN_INSTANCE_TYPE = "ml.g5.xlarge"       # Latest GPU
ENDPOINT_INSTANCE_TYPE = "ml.c5.xlarge"    # CPU optimized
```

## 🧪 Testing

### Test with boto3
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

with open('test.wav', 'rb') as f:
    audio_bytes = f.read()

response = runtime.invoke_endpoint(
    EndpointName='acousticshield-endpoint-20241025-120000',
    ContentType='audio/wav',
    Accept='application/json',
    Body=audio_bytes
)

result = json.loads(response['Body'].read())
print(f"Predicted: {result['label']} ({result['confidence']:.2%})")
```

### Test with Predictor
```python
from sagemaker.predictor import Predictor
from sagemaker.serializers import DataSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = Predictor(
    endpoint_name='acousticshield-endpoint-20241025-120000',
    serializer=DataSerializer(content_type='audio/wav'),
    deserializer=JSONDeserializer()
)

with open('test.wav', 'rb') as f:
    result = predictor.predict(f.read())

print(result)
```

### Generate Test Audio
```python
import numpy as np
import soundfile as sf

# Generate 1 second sine wave
sr = 16000
t = np.linspace(0, 1, sr)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)
sf.write('test.wav', audio, sr)
```

## 🐛 Troubleshooting

### Training Job Fails

**Problem**: "ResourceLimitExceeded" error
```
Solution: Request limit increase for ml.g4dn.xlarge instances
```

**Problem**: OOM (Out of Memory) error
```python
# Reduce batch size
BATCH_SIZE = 4

# Or enable gradient accumulation
GRADIENT_ACCUMULATION = 2  # Effective batch = 4 * 2 = 8
```

**Problem**: Training too slow
```python
# Use larger batch size
BATCH_SIZE = 16

# Use faster GPU
TRAIN_INSTANCE_TYPE = "ml.p3.2xlarge"
```

### Endpoint Deployment Fails

**Problem**: "MemoryError" at inference
```python
# Use instance with more memory
ENDPOINT_INSTANCE_TYPE = "ml.m5.xlarge"  # 16 GB RAM
```

**Problem**: Cold start latency too high
```python
# Use GPU instance for faster inference
ENDPOINT_INSTANCE_TYPE = "ml.g4dn.xlarge"
```

### Inference Issues

**Problem**: "Invalid audio format" error
```
Ensure audio is WAV format. Check with:
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

**Problem**: Poor accuracy
```python
# Train longer
EPOCHS = 8

# Use larger model
# (Requires custom training script modification)
```

### Region Issues

**Problem**: "NoSuchBucket" error
```
Check bucket exists in same region as training job
```

## 💰 Cost Estimation

### Training (One-Time)
| Instance | Duration | Cost/Hour | Total |
|----------|----------|-----------|-------|
| ml.g4dn.xlarge | ~40 min | $0.736 | ~$0.49 |
| ml.p3.2xlarge | ~20 min | $3.825 | ~$1.28 |

### Endpoint (Ongoing)
| Instance | Cost/Hour | Daily | Monthly |
|----------|-----------|-------|---------|
| ml.m5.xlarge | $0.23 | $5.52 | $165.60 |
| ml.c5.xlarge | $0.204 | $4.90 | $146.88 |
| ml.g4dn.xlarge | $0.736 | $17.66 | $529.92 |

### Storage
- Model artifacts: ~1 GB → ~$0.02/month
- Training data: ~500 MB → ~$0.01/month

### Cost Optimization Tips
1. **Delete endpoint when not in use** - Redeploy in 5-8 minutes
2. **Use CPU for inference** - 3x cheaper than GPU
3. **Enable auto-scaling** - Scale to 0 during off-hours
4. **Use Spot instances for training** - Save up to 70%

## 📊 Expected Performance

### Training Metrics (4 epochs, 1000 samples)
- **Accuracy**: 85-92%
- **Macro F1**: 0.82-0.90
- **Training time**: 30-40 minutes
- **Best epoch**: Usually 3-4

### Inference Performance
- **Latency**: 200-400ms (CPU), 50-100ms (GPU)
- **Throughput**: 2-5 req/sec (single instance)
- **Cold start**: 3-5 seconds

### Model Size
- **Total**: ~380 MB
- **wav2vec2 base**: 95M parameters
- **Classification head**: 4 classes

## 🔗 Useful Links

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [wav2vec2 Model Card](https://huggingface.co/facebook/wav2vec2-base)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)

## 📞 Support

For issues or questions:
1. Check CloudWatch logs: `/aws/sagemaker/TrainingJobs`
2. Review SageMaker documentation
3. Contact AWS support for quota increases

---

**Last Updated**: October 25, 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
