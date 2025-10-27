# Acoustic Shield: AI-Powered Vehicle Sound Classification for Emergency Response

## 🎯 Inspiration

Every second counts in emergency situations. When a vehicle collision is about to occur, the sounds produced—tire screeches, collision impacts, emergency braking—contain critical information that could save lives. We were inspired by the challenge of creating an intelligent system that could **hear danger before it's too late**.

Traditional collision detection systems rely on cameras and sensors, but what if we could leverage the **acoustic signature** of dangerous driving scenarios? This led us to build **Acoustic Shield**: an AI-powered audio classification system that can identify and categorize vehicle-related emergency sounds in real-time.

## 🏗️ What We Built

Acoustic Shield is a complete **end-to-end machine learning pipeline** deployed on AWS that:

1. **Generates synthetic training data** representing different vehicle emergency scenarios
2. **Trains a deep learning model** (wav2vec2) to classify audio into 4 categories:
   - `Normal` - Regular driving conditions
   - `TireSkid` - Sudden tire skidding sounds
   - `EmergencyBraking` - Hard braking events
   - `CollisionImminent` - Sounds indicating imminent collision

3. **Deploys a production-ready REST API** on AWS SageMaker for real-time inference
4. **Processes audio streams** at 16 kHz sampling rate with sub-second latency

### Technical Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Audio Input    │─────▶│  SageMaker       │─────▶│  Classification │
│  (WAV/Stream)   │      │  Endpoint        │      │  Results        │
│  16 kHz         │      │  wav2vec2        │      │  + Confidence   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         ▲                        ▲                         │
         │                        │                         │
         │                 ┌──────┴──────┐                 ▼
         │                 │  S3 Bucket  │         ┌───────────────┐
         │                 │  - Training │         │  Alert System │
         └─────────────────│  - Models   │         │  (Future)     │
                           └─────────────┘         └───────────────┘
```

## 🛠️ How We Built It

### 1. **Data Pipeline Engineering**

We created a sophisticated synthetic audio generation system:

- **Recipe-based synthesis**: Designed a flexible recipe system that generates audio events with configurable parameters (frequency, duration, amplitude)
- **Variation engine**: Built-in randomization ($\pm 5\%$) ensures model robustness
- **Scalability**: Generated **1000+ audio files** across 4 classes with pagination support for AWS S3
- **Weather enrichment**: Integrated weather data API for contextual information

```python
# Example: Recipe variation formula
f_{actual} = f_{base} \times (1 + \mathcal{U}(-0.05, 0.05))
```

Where $\mathcal{U}(a, b)$ represents uniform random distribution between $a$ and $b$.

### 2. **Machine Learning Model**

- **Base Model**: Facebook's wav2vec2-base (300M parameters)
- **Fine-tuning Strategy**: 
  - Learning rate: $\alpha = 5 \times 10^{-5}$
  - Batch size: 16 (optimized for GPU memory)
  - Training epochs: 1 (hackathon speed optimization)
  - Warmup steps: 50
  
- **Data Split**: 80/20 train-validation split with stratified sampling
- **Evaluation Metrics**: 
  - Accuracy
  - F1-score (macro-averaged)
  - Per-class precision/recall

### 3. **AWS Infrastructure**

**Training Pipeline:**
- AWS SageMaker Training Jobs with GPU instances (ml.g4dn.xlarge)
- Custom training script using HuggingFace Transformers
- Automatic hyperparameter tuning and model checkpointing
- CloudWatch integration for real-time monitoring

**Inference Pipeline:**
- SageMaker Real-time Endpoints with auto-scaling
- Custom inference handler supporting audio/wav content type
- Sub-second latency ($< 500ms$ for typical 1-3 second audio clips)
- JSON response format for easy API integration

**Storage & Organization:**
```
s3://acousticshield-ml/
├── train/              # Original training data
├── train_split/        # 80% training set
├── val/                # 20% validation set
└── models/             # Trained model artifacts
```

### 4. **Jupyter Notebook Workflow**

Created comprehensive notebooks for:
- **Data Generation**: `01_build_training_data.ipynb`
- **Training & Deployment**: `02_train_and_deploy.ipynb`

## 💡 What We Learned

### Technical Learnings

1. **Audio Processing at Scale**
   - Learned the importance of consistent sampling rates (16 kHz)
   - Discovered that audio resampling can significantly impact model accuracy
   - Understood the trade-offs between audio quality and processing speed

2. **AWS SageMaker Deep Dive**
   - Mastered SageMaker's HuggingFace container ecosystem
   - Learned about instance quotas and how to handle `ResourceLimitExceeded` errors
   - Discovered the importance of custom inference code for production deployment

3. **Model Optimization**
   - Learned that 1 epoch can be sufficient for demo-quality models in hackathons
   - Discovered the impact of batch size on GPU utilization (8 → 16 = 2x faster)
   - Understood the trade-off between model accuracy and training time

4. **Data Engineering**
   - Learned to handle S3 pagination for large datasets (>1000 files)
   - Discovered the importance of data validation and stratified splitting
   - Understood the value of synthetic data when real-world data is limited

### Hackathon-Specific Lessons

- **Time management is critical**: We pivoted from 4 epochs to 1 epoch training to meet demo deadlines
- **Infrastructure over perfection**: Getting a working endpoint is more valuable than perfect accuracy
- **Error handling matters**: Added comprehensive error messages and troubleshooting guides
- **Document everything**: Created multiple README files for future reference

## 🚧 Challenges We Faced

### 1. **AWS Quota Limitations**

**Challenge**: Hit instance quota limits on multiple GPU instance types:
- `ml.g5.xlarge`: 0 quota (new account limitation)
- `ml.p3.2xlarge`: 1 quota, but already in use from previous interrupted job

**Solution**: 
- Created an "Emergency Stop" cell to clean up stuck training jobs
- Documented 5+ alternative instance types with availability likelihood
- Switched to `ml.g4dn.xlarge` (most reliable for new accounts)

### 2. **Training Job Interruption**

**Challenge**: Accidentally interrupted training at 1.67/4 epochs with keyboard interrupt, losing all progress and model artifacts.

**Solution**:
- Learned that SageMaker training jobs continue even after notebook interruption
- Implemented proper job monitoring and graceful stopping procedures
- Optimized to 1-epoch training (10-15 min) for hackathon speed

### 3. **Audio Format Compatibility**

**Challenge**: Pre-trained models from HuggingFace don't support `audio/wav` content type without custom inference code.

**Solution**:
- Wrote custom `inference.py` handler supporting direct audio/wav input
- Implemented automatic audio resampling to 16 kHz
- Added comprehensive error handling for various audio formats

### 4. **Version Compatibility Issues**

**Challenge**: 
- Transformers 4.44 not supported by SageMaker
- PyTorch 2.3 incompatible with Transformers 4.28
- Parameter name changes (`eval_strategy` → `evaluation_strategy`)

**Solution**:
- Documented compatible versions: Transformers 4.28 + PyTorch 2.0
- Created detailed version matrix in configuration comments
- Fixed deprecated parameter names in training script

### 5. **Data at Scale**

**Challenge**: S3 `list_objects_v2` has 1000-file limit, causing incomplete data splits.

**Solution**:
- Implemented S3 paginator for unlimited file handling
- Added progress tracking for large dataset operations
- Ensured proper stratification across all files

## 📊 Results & Impact

### Model Performance
- **Training Time**: ~15-20 minutes (1 epoch on ml.g4dn.xlarge)
- **Inference Latency**: <500ms per audio clip
- **Expected Accuracy**: 60-75% (sufficient for hackathon demo)

### Infrastructure Efficiency
- **Cost-Optimized**: ~$0.25 for training, $0.23/hour for inference
- **Scalable**: Can handle 1000s of concurrent requests with auto-scaling
- **Production-Ready**: Complete CI/CD pipeline with error handling

### Real-World Applications
1. **Smart City Safety**: Deploy in urban areas to detect accidents in real-time
2. **Fleet Management**: Monitor commercial vehicles for emergency events
3. **Insurance**: Automated accident detection and reporting
4. **Emergency Response**: Alert first responders before 911 calls

## 🔮 What's Next

1. **Real-World Data Collection**: Partner with fleet operators to collect actual vehicle sound data
2. **Multi-Modal Integration**: Combine audio with video and sensor data
3. **Edge Deployment**: Optimize model for on-device inference (TensorFlow Lite/ONNX)
4. **Temporal Analysis**: Detect sequences of events (skid → brake → collision)
5. **Alert System**: Real-time notifications to emergency services

## 🎓 Key Takeaways

> "In hackathons, a working demo beats a perfect solution every time."

We learned that:
- ✅ **Iterate quickly**: Our 1-epoch model strategy saved hours
- ✅ **Document thoroughly**: Future us (and others) will thank us
- ✅ **Handle errors gracefully**: AWS quotas will surprise you
- ✅ **Optimize for demo**: Focus on end-to-end functionality first
- ✅ **Learn from failures**: Every error taught us something valuable

## 🙏 Acknowledgments

- **AWS SageMaker**: For providing powerful ML infrastructure
- **HuggingFace**: For wav2vec2 and the Transformers library
- **Open-Source Community**: For countless tutorials and documentation
- **Hackathon Organizers**: For creating this amazing learning opportunity

---

## 📁 Repository Structure

```
acmhack-backend/
├── data_pipeline/          # Audio generation & processing
│   ├── recipe_builder.py   # Synthetic audio recipes
│   ├── risk_event_synth.py # Event synthesis engine
│   └── weather_enricher.py # Context enrichment
├── training/               # ML training code
│   ├── train.py           # SageMaker training script
│   └── inference.py       # Custom inference handler
├── notebooks/              # Jupyter notebooks
│   ├── 01_build_training_data.ipynb
│   └── 02_train_and_deploy.ipynb
└── README.md              # Project documentation
```

## 🚀 Try It Yourself

1. Clone the repository
2. Set up AWS credentials
3. Open `notebooks/02_train_and_deploy.ipynb`
4. Follow the step-by-step guide
5. Deploy your own Acoustic Shield endpoint!

---

**Built with ❤️ during ACM Hack 2025**

*Time is money in hackathons—and every second saved could save a life.*
