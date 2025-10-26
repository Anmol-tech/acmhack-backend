"""
SageMaker Training Script for Acoustic Shield Audio Classification
Fine-tunes facebook/wav2vec2-base on audiofolder dataset with 4 classes.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """
    Compute accuracy and macro-F1 score.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with accuracy and f1 metrics
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Acoustic Shield audio classifier")
    
    # SageMaker paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--val-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--max-audio-length', type=int, default=80000)  # ~5 seconds at 16kHz
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Acoustic Shield - Audio Classification Training")
    logger.info("=" * 80)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Training directory: {args.train_dir}")
    logger.info(f"Validation directory: {args.val_dir}")
    logger.info(f"Output directory: {args.output_data_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 80)
    
    # Load dataset using audiofolder
    logger.info("\n📂 Loading training dataset from audiofolder...")
    train_dataset = load_dataset(
        "audiofolder",
        data_dir=args.train_dir,
        split="train"
    )
    logger.info(f"✓ Loaded {len(train_dataset)} training samples")
    
    # Load validation dataset if provided
    if args.val_dir and os.path.exists(args.val_dir):
        logger.info(f"📂 Loading validation dataset from {args.val_dir}...")
        val_dataset = load_dataset(
            "audiofolder",
            data_dir=args.val_dir,
            split="train"
        )
        logger.info(f"✓ Loaded {len(val_dataset)} validation samples")
    else:
        logger.info("📊 No validation set provided, splitting train 90/10...")
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
        logger.info(f"✓ Split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Cast audio to 16kHz
    logger.info("\n🎵 Resampling audio to 16kHz...")
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
    logger.info("✓ Audio resampled to 16kHz")
    
    # Get label info
    labels = train_dataset.features["label"].names
    num_labels = len(labels)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    
    logger.info(f"\n🏷️  Labels detected: {labels}")
    logger.info(f"📊 Number of classes: {num_labels}")
    
    # Load feature extractor
    logger.info("\n🔧 Loading feature extractor from facebook/wav2vec2-base...")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    logger.info("✓ Feature extractor loaded")
    
    # Preprocess function
    def preprocess_function(examples):
        """Extract audio features for training."""
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            max_length=args.max_audio_length,
            truncation=True,
            padding=True
        )
        return inputs
    
    # Encode datasets
    logger.info("\n⚙️  Preprocessing audio features...")
    train_encoded = train_dataset.map(
        preprocess_function,
        remove_columns=["audio"],
        batched=True,
        batch_size=100,
        desc="Preprocessing train"
    )
    val_encoded = val_dataset.map(
        preprocess_function,
        remove_columns=["audio"],
        batched=True,
        batch_size=100,
        desc="Preprocessing validation"
    )
    logger.info("✓ Audio features extracted")
    
    # Load model
    logger.info("\n🤖 Loading wav2vec2-base model...")
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    logger.info("✓ Model loaded")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        evaluation_strategy="epoch",  # Changed from eval_strategy for transformers 4.28
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        report_to=[],
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    logger.info("\n🚀 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("🎓 Starting training...")
    logger.info("=" * 80)
    train_result = trainer.train()
    
    # Save model and feature extractor
    logger.info("\n💾 Saving model and feature extractor...")
    trainer.save_model(args.model_dir)
    feature_extractor.save_pretrained(args.model_dir)
    logger.info(f"✓ Model saved to {args.model_dir}")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    logger.info("\n📊 Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Save label mappings
    label_info = {
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": num_labels
    }
    label_path = os.path.join(args.model_dir, "label_info.json")
    with open(label_path, "w") as f:
        json.dump(label_info, f, indent=2)
    logger.info(f"✓ Label mappings saved to {label_path}")
    
    # Print final results
    logger.info("\n" + "=" * 80)
    logger.info("🎉 Training complete!")
    logger.info("=" * 80)
    logger.info(f"📈 Final Metrics:")
    logger.info(f"   Accuracy: {eval_metrics['eval_accuracy']:.4f}")
    logger.info(f"   F1 Score (macro): {eval_metrics['eval_f1']:.4f}")
    logger.info(f"   Loss: {eval_metrics['eval_loss']:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
