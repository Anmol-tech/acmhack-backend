"""
SageMaker Inference Handler for Acoustic Shield Audio Classification
Handles real-time audio classification requests with automatic resampling.
"""

import os
import json
import io
import logging

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global objects (loaded once at endpoint initialization)
model = None
feature_extractor = None
device = None
label_info = None


def model_fn(model_dir):
    """
    Load the model and feature extractor for inference.
    Called once when the endpoint is initialized.
    
    Args:
        model_dir: Path to the model artifacts directory
        
    Returns:
        The loaded model (feature extractor is stored in global variable)
    """
    global model, feature_extractor, device, label_info
    
    logger.info(f"Loading model from {model_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load feature extractor
    logger.info("Loading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForAudioClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Load label info if available
    label_info_path = os.path.join(model_dir, "label_info.json")
    if os.path.exists(label_info_path):
        with open(label_info_path, "r") as f:
            label_info = json.load(f)
        logger.info(f"Loaded label info: {label_info['labels']}")
    
    logger.info("✓ Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    Accepts audio/wav format and handles resampling if needed.
    
    Args:
        request_body: The request body bytes
        request_content_type: The content type of the request
        
    Returns:
        Dictionary with audio array and sampling rate
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == "audio/wav":
        # Read WAV file from bytes
        audio_data, sample_rate = sf.read(io.BytesIO(request_body))
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            logger.info(f"Converting stereo to mono (shape: {audio_data.shape})")
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Simple linear interpolation resampling
            duration = len(audio_data) / sample_rate
            target_length = int(duration * 16000)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
            sample_rate = 16000
        
        logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
        
        return {
            "array": audio_data,
            "sampling_rate": sample_rate
        }
    else:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Only 'audio/wav' is supported."
        )


def predict_fn(input_data, model):
    """
    Apply model to the incoming request.
    
    Args:
        input_data: Dictionary with audio array and sampling rate
        model: The loaded model
        
    Returns:
        Dictionary with prediction results
    """
    global feature_extractor, device
    
    logger.info("Running inference...")
    
    try:
        # Extract features
        inputs = feature_extractor(
            input_data["array"],
            sampling_rate=input_data["sampling_rate"],
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]
        
        # Get prediction
        predicted_class_idx = int(np.argmax(probs))
        predicted_label = model.config.id2label[predicted_class_idx]
        confidence = float(probs[predicted_class_idx])
        
        # Build probability dictionary for all classes
        prob_dict = {
            model.config.id2label[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")
        
        return {
            "label": predicted_label,
            "confidence": confidence,
            "probs": prob_dict
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise


def output_fn(prediction, accept):
    """
    Serialize the prediction result into the desired response format.
    
    Args:
        prediction: Dictionary with prediction results
        accept: The accept type requested by the client
        
    Returns:
        Tuple of (response_body, content_type)
    """
    logger.info(f"Formatting output with accept type: {accept}")
    
    if accept == "application/json":
        return json.dumps(prediction), accept
    
    raise ValueError(f"Unsupported accept type: {accept}")
