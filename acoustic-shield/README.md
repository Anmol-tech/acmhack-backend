# Acoustic Shield — Proactive Incident Prevention via Generative Audio AI

## Project Structure

```
acoustic-shield/
├── processing/
│   └── augment.py
├── training/
│   ├── train.py
│   └── inference.py
├── lambdas/
│   └── predict_proxy/
│       └── handler.py
├── frontend/
└── README.md
```

## 1. Generate Data (Processing)

- **Script:** `processing/augment.py`
- **Purpose:** Synthesizes 4 audio classes and outputs `.wav` files + `manifest.json`.
- **Run:**
  ```bash
  python processing/augment.py
  # Set COUNT_PER_CLASS env var to control number of samples
  ```
- **Output:** `/opt/ml/outputs/audio/<ClassName>/uuid.wav`, `/opt/ml/outputs/audio/manifest.json`

## 2. Train (SageMaker)

- **Script:** `training/train.py`
- **Purpose:** Fine-tunes `facebook/wav2vec2-base` for 4-class classification.
- **Run:**
  ```bash
  python training/train.py
  # Set AUDIO_DIR, MODEL_NAME, EPOCHS, BATCH_SIZE as needed
  ```
- **Output:** `/opt/ml/model/`

## 3. Deploy Endpoint

- **Script:** `training/inference.py`
- **Purpose:** Loads model + extractor, predicts class for input `.wav`.
- **Run:**
  ```bash
  python training/inference.py <audio.wav>
  # Set MODEL_DIR if needed
  ```

## 4. Test with Lambda + API Gateway + React Frontend

- **Lambda:** `lambdas/predict_proxy/handler.py`
- **Purpose:** Proxies `/predict` requests to SageMaker endpoint.
- **Env Vars:** `ENDPOINT` (SageMaker endpoint name)
- **Frontend:** `frontend/` (React + Vite)
- **Purpose:** Record/Upload audio, POST to `/predict`, display result.

## Deployment Notebooks / Commands

See `deployment.ipynb` for step-by-step SageMaker job launch, endpoint deploy, and test commands.

## Configuration

- **Buckets, roles, endpoints:** All configurable via env vars or parameters.
- **No hardcoded regions/account IDs.**

---

## Quickstart

1. **Generate Data:**
   - Run `augment.py` locally or via SageMaker Processing job.
2. **Train Model:**
   - Run `train.py` locally or via SageMaker Training job.
3. **Deploy Endpoint:**
   - Use `inference.py` for model serving; deploy as SageMaker endpoint.
4. **Test End-to-End:**
   - Use Lambda + API Gateway + React frontend for real-time predictions.

---

## AWS CLI Profile

- All deployment commands/notebooks use `--profile acm-hack`.
- Region is your default (not hardcoded).

---

## Manifest

- All generated files are summarized in `manifest.json`.

---

## Contact

For questions, reach out to the project owner.
