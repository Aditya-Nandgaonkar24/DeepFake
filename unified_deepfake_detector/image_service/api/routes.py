"""
Flask Backend API for AI Image Detection - Production Refactored
Handles image upload and analysis purely in-memory
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import json
import concurrent.futures
import hashlib
from functools import lru_cache
import io
import uuid

from flask import Flask, request, jsonify, Blueprint
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import analyzers from the new core structure
from core.analyzers.frequency_analyzer import FrequencyAnalyzer
from core.analyzers.noise_analyzer import NoiseAnalyzer
from core.analyzers.metadata_analyzer import MetadataAnalyzer
from core.analyzers.pixel_analyzer import PixelStatisticsAnalyzer
from core.analyzers.ela_analyzer import ErrorLevelAnalyzer
from core.ml.train_model import AIImageClassifier, ModelTrainer, get_transforms

# Security: Prevent Decompression Bomb attacks
Image.MAX_IMAGE_PIXELS = 100000000 # ~100 megapixels max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}

api_blueprint = Blueprint('api', __name__)

# Thread pool executor for parallel analysis
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Initialize analyzers
frequency_analyzer = FrequencyAnalyzer()
noise_analyzer = NoiseAnalyzer()
metadata_analyzer = MetadataAnalyzer()
pixel_analyzer = PixelStatisticsAnalyzer()
ela_analyzer = ErrorLevelAnalyzer()

# Basic LRU Cache Implementation
class AnalysisCache:
    def __init__(self, maxsize=100):
        self.cache = {}
        self.maxsize = maxsize
        self.keys = []
        
    def get(self, key):
        if key in self.cache:
            self.keys.remove(key)
            self.keys.append(key)
            return self.cache[key]
        return None
        
    def set(self, key, value):
        if key not in self.cache:
            if len(self.keys) >= self.maxsize:
                oldest = self.keys.pop(0)
                del self.cache[oldest]
            self.keys.append(key)
        self.cache[key] = value

analysis_cache = AnalysisCache(maxsize=100)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj

def combine_confidence_scores(frequency_result, noise_result, metadata_result, 
                            pixel_result, ela_result, model_prediction, image_resolution=None):
    # Extract scores
    freq_score = float(frequency_result['frequency_score'])
    noise_score = float(noise_result['noise_score'])
    metadata_score = float(metadata_result['metadata_score'])
    pixel_score = float(pixel_result['pixel_score'])
    ela_score = float(ela_result['ela_score'])
    
    # Model prediction confidence
    model_confidence = float(model_prediction['confidence'])
    predicted_class = model_prediction['class']
    
    # If model predicts "Real" with high confidence, that's a positive indicator
    if predicted_class.lower() == 'real':
        model_score = model_confidence
    else:
        # 'Fake' or any AI-class prediction -> inverse the score
        model_score = 100 - model_confidence
    
    # Weighted combination
    # Model trained on 224x224 high-res faces - primary signal
    final_score = (
        0.70 * model_score +      # Deep Learning Model: 70%
        0.06 * freq_score +       # Frequency (FFT): 6%
        0.06 * noise_score +      # Noise Pattern: 6%
        0.06 * pixel_score +      # Pixel Stats: 6%
        0.06 * metadata_score +   # Metadata (EXIF/ICC): 6%
        0.06 * ela_score          # Error Level Analysis: 6%
    )
    
    # Determine final classification
    if final_score >= 70:
        classification = "Real Image"
        confidence_level = "High"
    elif final_score >= 50:
        classification = "Uncertain"
        confidence_level = "Medium"
    else:
        classification = "AI-Generated"
        confidence_level = "Low"
    
    return {
        'final_score': round(float(final_score), 2),
        'classification': classification,
        'confidence_level': confidence_level,
        'component_scores': {
            'model': round(float(model_score), 2),
            'frequency': round(float(freq_score), 2),
            'noise': round(float(noise_score), 2),
            'pixel': round(float(pixel_score), 2),
            'metadata': round(float(metadata_score), 2),
            'ela': round(float(ela_score), 2)
        }
    }


def init_routes(app, model_trainer):
    """
    Register routes and attach rate limiter
    """
    
    # Setup Limiter to protect against DOS
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'components_active': 5
        })

    @app.route('/api/analyze', methods=['POST'])
    @limiter.limit("5 per minute") # strict rate limiting on heavy endpoints
    def analyze_image():
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        try:
            # 1. READ FILE STREAM INTO MEMORY (0 disk IO)
            img_bytes = file.read()
            
            # 2. Check Cache
            file_hash = hashlib.sha256(img_bytes).hexdigest()
            cached_result = analysis_cache.get(file_hash)
            if cached_result:
                print(f"✓ Returning cached result (Hash: {file_hash[:8]}...)")
                return jsonify(cached_result)
            
            # 3. Process image fully IN MEMORY
            try:
                # For PIL (Metadata/Details)
                img_pil = Image.open(io.BytesIO(img_bytes))
                
                # For OpenCV (Frequency/Noise/Pixels)
                np_img = np.frombuffer(img_bytes, np.uint8)
                image_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                
                if image_cv is None:
                    raise ValueError("Failed to decode image into OpenCV format")
                    
            except Image.DecompressionBombError:
                return jsonify({'error': 'Image is too large (Decompression Bomb Protection)'}), 400
            except Exception as e:
                return jsonify({'error': f'Could not decode image: {str(e)}'}), 400

            image_info = {
                'filename': getattr(file, 'filename', 'upload'),
                'format': img_pil.format,
                'mode': img_pil.mode,
                'size': img_pil.size,
                'width': img_pil.width,
                'height': img_pil.height,
                'file_size': len(img_bytes)
            }
            
            # TEMPORARILY write to disk ONLY for the ML model trainer since it 
            # currently depends on a filepath. UUID prevents overwrites.
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
            with open(temp_path, "wb") as f:
                f.write(img_bytes)
            
            def run_model_prediction(path):
                try:
                    return model_trainer.predict(path)
                except Exception as e:
                    print(f"Model prediction error: {e}")
                    return {
                        'class': 'Unknown', 'confidence': 50.0,
                        'probabilities': {'Real': 50.0, 'Stable_Diffusion': 12.5, 'Midjourney': 12.5, 'DALLE': 12.5, 'Unknown': 12.5}
                    }

            # 4. PARALLEL ANALYSIS
            future_freq = executor.submit(frequency_analyzer.analyze, image_cv)
            future_noise = executor.submit(noise_analyzer.analyze, image_cv)
            future_meta = executor.submit(metadata_analyzer.analyze, temp_path)
            future_pixel = executor.submit(pixel_analyzer.analyze, image_cv)
            future_ela = executor.submit(ela_analyzer.analyze, temp_path)
            future_model = executor.submit(run_model_prediction, temp_path)

            frequency_result = future_freq.result()
            noise_result = future_noise.result()
            metadata_result = future_meta.result()
            pixel_result = future_pixel.result()
            ela_result = future_ela.result()
            model_prediction = future_model.result()
            
            # Remove temp file
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

            # 5. Combine and Respond
            img_min_dim = min(image_info.get('width', 32), image_info.get('height', 32))
            final_result = combine_confidence_scores(
                frequency_result, noise_result, metadata_result, pixel_result, ela_result, model_prediction,
                image_resolution=img_min_dim
            )
            
            response = {
                'success': True,
                'image_info': image_info,
                'analysis': {
                    'final_result': final_result,
                    'model_prediction': model_prediction,
                    'frequency_analysis': frequency_result,
                    'noise_analysis': noise_result,
                    'metadata_analysis': metadata_result,
                    'pixel_analysis': pixel_result,
                    'ela_analysis': ela_result
                },
                'cached': False
            }
            response = convert_to_python_types(response)
            
            analysis_cache.set(file_hash, {**response, 'cached': True})
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        finally:
            # Prevent Memory Leaks
            if 'img_bytes' in locals():
                del img_bytes
            if 'image_cv' in locals():
                del image_cv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
