# SageMaker Training & Deployment - Quick Reference

## 🚀 Quick Commands

### Start Training Job
```python
from sagemaker.huggingface import HuggingFace
import boto3

# Get role ARN
iam = boto3.client('iam')
role_arn = iam.get_role(RoleName='role-sagemaker-train')['Role']['Arn']

# Create estimator
estimator = HuggingFace(
    entry_point='train.py',
    source_dir='training',
    role=role_arn,
    instance_type='ml.g4dn.xlarge',
    transformers_version='4.44',
    pytorch_version='2.3',
    py_version='py311',
    hyperparameters={'epochs': 4, 'learning-rate': 3e-5, 'batch-size': 8}
)

# Train
estimator.fit({'train': 's3://acousticshield-ml/train/'})
```

### Deploy Endpoint
```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='acousticshield-endpoint'
)
```

### Test Endpoint
```python
with open('test.wav', 'rb') as f:
    result = predictor.predict(f.read())
print(result['label'], result['confidence'])
```

### Delete Endpoint
```python
predictor.delete_endpoint()
```

## 📋 Configuration Reference

### S3 Paths
```python
TRAIN_S3 = "s3://acousticshield-ml/train/"
VAL_S3 = "s3://acousticshield-ml/val/"  # Optional
MODEL_OUTPUT = "s3://acousticshield-ml/models/"
```

### Hyperparameters
```python
EPOCHS = 4              # Training epochs
LEARNING_RATE = 3e-5    # Learning rate
BATCH_SIZE = 8          # Batch size per device
WARMUP_STEPS = 500      # Warmup steps
```

### Instance Types

**Training (GPU)**
- `ml.g4dn.xlarge` - Recommended, $0.736/hr
- `ml.p3.2xlarge` - Faster, $3.825/hr

**Endpoint (CPU)**
- `ml.m5.xlarge` - Recommended, $0.23/hr
- `ml.c5.xlarge` - CPU-optimized, $0.204/hr

## 🎯 Endpoint Input/Output

### Input
```
Content-Type: audio/wav
Body: <WAV file bytes>
```

### Output
```json
{
  "label": "TireSkid",
  "confidence": 0.85,
  "probs": {
    "Normal": 0.05,
    "TireSkid": 0.85,
    "EmergencyBraking": 0.08,
    "CollisionImminent": 0.02
  }
}
```

## 🔧 Common Modifications

### Train Longer
```python
hyperparameters={'epochs': 8}
```

### Reduce Batch Size (OOM)
```python
hyperparameters={'batch-size': 4}
```

### No Validation Set
```python
estimator.fit({'train': TRAIN_S3})  # Will auto-split 90/10
```

### Use Validation Set
```python
estimator.fit({'train': TRAIN_S3, 'validation': VAL_S3})
```

## 📊 Expected Results

- **Training Time**: 30-40 minutes
- **Deployment Time**: 5-8 minutes
- **Accuracy**: 85-92%
- **F1 Score**: 0.82-0.90

## 💰 Costs

- **Training**: ~$0.50 (one-time)
- **Endpoint**: ~$0.23/hour (ongoing)
- **Storage**: ~$0.02/month

## 🐛 Quick Fixes

**OOM Error**
```python
hyperparameters={'batch-size': 4, 'gradient-accumulation-steps': 2}
```

**Slow Training**
```python
instance_type='ml.p3.2xlarge'  # Faster GPU
```

**High Endpoint Latency**
```python
instance_type='ml.g4dn.xlarge'  # GPU inference
```

## 📞 Files

- `training/train.py` - Training script
- `training/inference.py` - Inference handler
- `notebooks/02_train_and_deploy.ipynb` - Full notebook
- `training/README.md` - Detailed documentation

---

**Quick Start**: Open `notebooks/02_train_and_deploy.ipynb` and run all cells!
