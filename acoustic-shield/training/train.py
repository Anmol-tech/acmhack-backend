import os
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/wav2vec2-base")
OUTPUT_DIR = "/opt/ml/model"
AUDIO_DIR = os.getenv("AUDIO_DIR", "/opt/ml/outputs/audio")
LABELS = ["Normal", "TireSkid", "EmergencyBraking", "CollisionImminent"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    "audiofolder", data_dir=AUDIO_DIR, split="train", cache_dir="/tmp/hf_cache"
)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS)
)


def preprocess(batch):
    audio = batch["audio"]
    if audio["sampling_rate"] != 16000:
        batch["audio"] = Audio(sampling_rate=16000).cast(audio)
    batch["input_values"] = feature_extractor(
        audio["array"], sampling_rate=16000, return_tensors="pt"
    ).input_values[0]
    batch["label"] = LABELS.index(batch["label"])
    return batch


dataset = dataset.map(preprocess)


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    num_train_epochs=int(os.getenv("EPOCHS", 3)),
    per_device_train_batch_size=int(os.getenv("BATCH_SIZE", 8)),
    save_steps=50,
    logging_steps=10,
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
print("Model and feature extractor saved.")
