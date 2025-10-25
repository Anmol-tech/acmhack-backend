import os
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from scipy.io import wavfile
import json

MODEL_DIR = os.getenv("MODEL_DIR", "/opt/ml/model")
LABELS = ["Normal", "TireSkid", "EmergencyBraking", "CollisionImminent"]

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict(audio_path):
    sr, audio = wavfile.read(audio_path)
    if sr != 16000:
        raise ValueError("Audio must be 16 kHz mono")
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    label_idx = int(np.argmax(probs))
    return {
        "label": LABELS[label_idx],
        "confidence": float(probs[label_idx]),
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    import sys

    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))
