# Updates Summary - File Generation Increased to 1000+

## Changes Made

### 1. **Updated `requirements.txt`** ✅
Added missing packages for training and inference:
- `torch>=2.3.0` - PyTorch for deep learning
- `transformers>=4.44.0` - HuggingFace transformers for wav2vec2
- `datasets>=2.14.0` - HuggingFace datasets for audiofolder loading
- `evaluate>=0.4.0` - Metrics computation (accuracy, F1)
- `soundfile>=0.12.0` - Audio file I/O
- `librosa>=0.10.0` - Audio processing utilities

### 2. **Updated `data_pipeline/recipe_builder.py`** ✅
**Added `recipes_per_event` parameter:**
- New parameter to generate multiple recipe variations per event
- Adds variation factor (5% increments) for diversity
- Adds random noise factor (±5%) for natural variation
- Each recipe variation gets unique filename: `evt_00001_v0_normal.wav`, `evt_00001_v1_normal.wav`, etc.

**Key changes:**
```python
def build_recipes(self, risk_events: List[Dict], recipes_per_event: int = 1) -> List[Dict]:
    # Generate multiple variations per event
    for variation_idx in range(recipes_per_event):
        recipe = self._create_recipe(event, variation_idx)
```

### 3. **Updated `notebooks/01_build_training_data.ipynb`** ✅
**New configuration parameters:**
```python
TOP_N_HOTSPOTS = 50        # Increased from 25
EVENTS_PER_HOTSPOT = 5     # Increased from 4
RECIPES_PER_EVENT = 4      # NEW: 4 variations per event
```

**Expected output calculation:**
```
50 hotspots × 5 events × 4 recipes = 1,000 audio files
```

### 4. **Created Test Script** ✅
Added `test_file_count.py` to verify configuration will generate target files.

## File Generation Breakdown

### Previous Configuration
- 25 hotspots × 3 points × 4 events = **300 files** ❌

### New Configuration
- 50 hotspots × 5 events × 4 recipes = **1,000 files** ✅

### Estimated Class Distribution
Based on typical severity distributions:
- **Normal**: ~300 files (30%)
- **TireSkid**: ~300 files (30%)
- **EmergencyBraking**: ~250 files (25%)
- **CollisionImminent**: ~150 files (15%)

## Recipe Variations Explained

Each event now generates 4 recipe variations with:
1. **Variation Index 0**: Base parameters
2. **Variation Index 1**: +5% intensity
3. **Variation Index 2**: +10% intensity
4. **Variation Index 3**: +15% intensity

Plus ±5% random noise on each parameter for natural diversity.

## How to Use

### Install All Dependencies
```bash
pip install -r requirements.txt
```

### Run the Pipeline
Open `notebooks/01_build_training_data.ipynb` and run all cells. The notebook will:
1. Extract 50 crash hotspots
2. Enrich with weather data
3. Generate 250 risk events (50 × 5)
4. Build 1,000 audio recipes (250 × 4)
5. Run SageMaker Processing job to generate WAV files
6. Output organized by class: `Normal/`, `TireSkid/`, `EmergencyBraking/`, `CollisionImminent/`

### Verify File Count
```bash
python test_file_count.py
```

## Next Steps

1. **Run the updated notebook** to generate 1,000 audio files
2. **Train the model** using `training/train.py` with the larger dataset
3. **Deploy endpoint** using `training/inference.py`
4. **Test inference** with real audio samples

## Files Modified
- ✅ `requirements.txt` - Added ML/audio packages
- ✅ `data_pipeline/recipe_builder.py` - Added recipes_per_event support
- ✅ `notebooks/01_build_training_data.ipynb` - Updated configuration
- ✅ `test_file_count.py` - Created verification script

All packages are now installed and the pipeline is configured to generate **1,000 audio files**! 🎉
