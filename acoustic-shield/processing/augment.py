import os
import uuid
import json
import numpy as np
from scipy.signal import chirp, sawtooth
from scipy.io import wavfile

AUDIO_CLASSES = ["Normal", "TireSkid", "EmergencyBraking", "CollisionImminent"]
COUNT_PER_CLASS = int(os.getenv("COUNT_PER_CLASS", 500))
OUTPUT_DIR = "/opt/ml/outputs/audio"
SAMPLE_RATE = 16000
DURATION = 2  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)
manifest = {}


def synthesize_normal():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.02 * np.random.randn(len(t))
    return audio


def synthesize_tireskid():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.1 * sawtooth(2 * np.pi * 800 * t) + 0.02 * np.random.randn(len(t))
    return audio


def synthesize_emergencybraking():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.1 * chirp(
        t, f0=1000, f1=200, t1=DURATION, method="linear"
    ) + 0.02 * np.random.randn(len(t))
    return audio


def synthesize_collisionimminent():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 1200 * t) * np.exp(
        -2 * t
    ) + 0.05 * np.random.randn(len(t))
    return audio


synth_map = {
    "Normal": synthesize_normal,
    "TireSkid": synthesize_tireskid,
    "EmergencyBraking": synthesize_emergencybraking,
    "CollisionImminent": synthesize_collisionimminent,
}

for cls in AUDIO_CLASSES:
    class_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(class_dir, exist_ok=True)
    manifest[cls] = []
    for _ in range(COUNT_PER_CLASS):
        audio = synth_map[cls]()
        fname = f"{uuid.uuid4()}.wav"
        fpath = os.path.join(class_dir, fname)
        wavfile.write(fpath, SAMPLE_RATE, audio.astype(np.float32))
        manifest[cls].append(fpath)

with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Generated {COUNT_PER_CLASS} samples per class. Manifest written.")
