"""
Flask Backend API for AI Image Detection
Handles image upload and analysis
UPDATED: Higher weight for Deep Learning Model (70%)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
from PIL import Image
import json
import concurrent.futures
import hashlib
from functools import lru_cache

# Import analyzers
from frequency_analyzer import FrequencyAnalyzer
from noise_analyzer import NoiseAnalyzer
from metadata_analyzer import MetadataAnalyzer
from pixel_analyzer import PixelStatisticsAnalyzer
from train_model import AIImageClassifier, ModelTrainer, get_transforms

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '../static/uploads'
MODELS_FOLDER = '../models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Thread pool executor for parallel analysis
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Initialize analyzers
frequency_analyzer = FrequencyAnalyzer()
noise_analyzer = NoiseAnalyzer()
metadata_analyzer = MetadataAnalyzer()
pixel_analyzer = PixelStatisticsAnalyzer()

# Load trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AIImageClassifier(num_classes=5, pretrained=False)
model_trainer = ModelTrainer(model, device=device)

# Try to load trained model
model_path = os.path.join(MODELS_FOLDER, 'best_model.pth')
if os.path.exists(model_path):
    try:
        model_trainer.load_model(model_path)
        print(f"✓ Loaded trained model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Using untrained model - predictions will be random!")
else:
    print(f"Warning: Model not found at {model_path}")
    print("Please train the model first using train_model.py")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_python_types(obj):
    """Convert numpy/torch types to Python native types for JSON serialization"""
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

def get_image_details(image_path):
    """Extract basic image information"""
    img = Image.open(image_path)
    
    return {
        'filename': os.path.basename(image_path),
        'format': img.format,
        'mode': img.mode,
        'size': img.size,
        'width': img.width,
        'height': img.height,
        'file_size': os.path.getsize(image_path)
    }

def get_file_hash(filepath):
    """Calculate SHA-256 hash of a file for caching"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

@lru_cache(maxsize=100)
def get_cached_analysis(file_hash):
    """LRU Cache for analysis results. We just use the hash as key and return None if not computed."""
    return None

def set_cached_analysis(file_hash, result):
    """Hack to store in lru_cache"""
    get_cached_analysis.__wrapped__(file_hash) # bypass cache
    get_cached_analysis.cache_info() # just stats
    # Python's lru_cache is purely for args->return value. For explicit set, we maintain a separate dict:
    pass

# Simple dict-based LRU cache implementation for our specific needs
class AnalysisCache:
    def __init__(self, maxsize=100):
        self.cache = {}
        self.maxsize = maxsize
        self.keys = []
        
    def get(self, key):
        if key in self.cache:
            # Move to end to show it was recently used
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

def combine_confidence_scores(frequency_result, noise_result, metadata_result, 
                              pixel_result, model_prediction):
    """
    Combine all analysis results into final confidence score
    
    UPDATED WEIGHTS (Model is now dominant):
    - Deep Learning Model: 70% ← INCREASED from 40%
    - Frequency Analysis: 8%  ← DECREASED from 15%
    - Noise Analysis: 8%      ← DECREASED from 15%
    - Pixel Statistics: 7%    ← DECREASED from 15%
    - Metadata: 7%           ← DECREASED from 15%
    
    Total: 100%
    """
    
    # Extract scores
    freq_score = float(frequency_result['frequency_score'])
    noise_score = float(noise_result['noise_score'])
    metadata_score = float(metadata_result['metadata_score'])
    pixel_score = float(pixel_result['pixel_score'])
    
    # Model prediction confidence
    model_confidence = float(model_prediction['confidence'])
    predicted_class = model_prediction['class']
    
    # If model predicts "Real" with high confidence, that's a positive indicator
    if predicted_class == 'Real':
        model_score = model_confidence
    else:
        # If it predicts AI generation, inverse the score
        model_score = 100 - model_confidence
    
    # Weighted combination - MODEL NOW DOMINATES (70%)
    final_score = (
        0.70 * model_score +      # Deep Learning Model: 70%
        0.08 * freq_score +       # Frequency: 8%
        0.08 * noise_score +      # Noise: 8%
        0.07 * pixel_score +      # Pixels: 7%
        0.07 * metadata_score     # Metadata: 7%
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
            'metadata': round(float(metadata_score), 2)
        },
        'weights_used': {
            'model': '74%',
            'frequency': '7%',
            'noise': '7%',
            'pixel': '6%',
            'metadata': '6%'
        }
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists(model_path),
        'device': device
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Main endpoint for image analysis
    Accepts: multipart/form-data with 'image' field
    Returns: Complete analysis results
    """
    
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image for analysis
        image_cv = cv2.imread(filepath)
        
        if image_cv is None:
            return jsonify({'error': 'Could not read image file'}), 400
        
        # Check Cache
        file_hash = get_file_hash(filepath)
        cached_result = analysis_cache.get(file_hash)
        if cached_result:
            print(f"✓ Returning cached result for {filename} (Hash: {file_hash[:8]}...)")
            return jsonify(cached_result)

        # Get image details
        image_info = get_image_details(filepath)
        
        # Run all analyses in PARALLEL
        print(f"Analyzing image: {filename} (in parallel)")
        
        def run_model_prediction(path):
            try:
                return model_trainer.predict(path)
            except Exception as e:
                print(f"    Model prediction error: {e}")
                return {
                    'class': 'Unknown',
                    'confidence': 50.0,
                    'probabilities': {
                        'Real': 50.0, 'Stable_Diffusion': 12.5,
                        'Midjourney': 12.5, 'DALLE': 12.5, 'Unknown': 12.5
                    }
                }

        # Submit tasks to executor
        future_freq = executor.submit(frequency_analyzer.analyze, image_cv)
        future_noise = executor.submit(noise_analyzer.analyze, image_cv)
        future_meta = executor.submit(metadata_analyzer.analyze, filepath)
        future_pixel = executor.submit(pixel_analyzer.analyze, image_cv)
        future_model = executor.submit(run_model_prediction, filepath)

        # Wait for all to complete and get results
        frequency_result = future_freq.result()
        noise_result = future_noise.result()
        metadata_result = future_meta.result()
        pixel_result = future_pixel.result()
        model_prediction = future_model.result()
        
        # 6. Combine all results
        print("  - Combining results...")
        final_result = combine_confidence_scores(
            frequency_result,
            noise_result,
            metadata_result,
            pixel_result,
            model_prediction
        )
        
        # Convert all numpy types to Python types
        frequency_result = convert_to_python_types(frequency_result)
        noise_result = convert_to_python_types(noise_result)
        metadata_result = convert_to_python_types(metadata_result)
        pixel_result = convert_to_python_types(pixel_result)
        model_prediction = convert_to_python_types(model_prediction)
        final_result = convert_to_python_types(final_result)
        image_info = convert_to_python_types(image_info)
        
        # Prepare response
        response = {
            'success': True,
            'image_info': image_info,
            'analysis': {
                'final_result': final_result,
                'model_prediction': model_prediction,
                'frequency_analysis': frequency_result,
                'noise_analysis': noise_result,
                'metadata_analysis': metadata_result,
                'pixel_analysis': pixel_result
            },
            'cached': False
        }
        
        # Save to cache
        analysis_cache.set(file_hash, {**response, 'cached': True})
        
        print(f"✓ Analysis complete: {final_result['classification']} "
              f"({final_result['final_score']:.1f}%)")
        print(f"  Model contributed: {0.70 * float(final_result['component_scores']['model']):.1f}% to final score")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    model_exists = os.path.exists(model_path)
    
    return jsonify({
        'model_loaded': model_exists,
        'model_path': model_path if model_exists else None,
        'device': device,
        'classes': model_trainer.class_names,
        'model_weight': '70%',
        'note': 'Deep Learning Model now has dominant influence on final score'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("AI Image Detection Backend Server")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model loaded: {os.path.exists(model_path)}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("=" * 60)
    print("WEIGHT DISTRIBUTION:")
    print("  • Deep Learning Model: 70% (DOMINANT)")
    print("  • Frequency Analysis:   8%")
    print("  • Noise Analysis:       8%")
    print("  • Pixel Statistics:     7%")
    print("  • Metadata:             7%")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)







