"""
Configuration File
Centralized configuration for the AI Image Detector
"""

import os

class Config:
    """Application configuration"""
    
    # Paths - BASE_DIR is the project root (two levels up from core/config.py)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
    DATASETS_FOLDER = os.path.join(BASE_DIR, 'datasets')
    
    # File upload settings
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}
    
    # Model settings
    MODEL_NAME = 'efficientnet-b0'  # Options: b0, b1, b2, ... b7
    NUM_CLASSES = 5
    CLASS_NAMES = ['Real', 'Stable_Diffusion', 'Midjourney', 'DALLE', 'Unknown']
    
    # Training settings
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8  # 80% train, 20% validation
    
    # Image processing
    IMAGE_SIZE = (224, 224)  # EfficientNet input size
    
    # Analysis weights for final score
    WEIGHTS = {
        'model': 0.40,
        'frequency': 0.15,
        'noise': 0.15,
        'pixel': 0.15,
        'metadata': 0.15
    }
    
    # Score thresholds
    HIGH_CONFIDENCE_THRESHOLD = 70  # >= 70 = Real image
    LOW_CONFIDENCE_THRESHOLD = 50   # < 50 = AI generated
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # Device selection
    USE_GPU = True  # RTX 3050 Laptop GPU detected
    
    @classmethod
    def get_device(cls):
        """Get computation device"""
        if cls.USE_GPU:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cpu'
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.MODELS_FOLDER,
            os.path.join(cls.DATASETS_FOLDER, 'train', 'real'),
            os.path.join(cls.DATASETS_FOLDER, 'train', 'stable_diffusion'),
            os.path.join(cls.DATASETS_FOLDER, 'train', 'midjourney'),
            os.path.join(cls.DATASETS_FOLDER, 'train', 'dalle'),
            os.path.join(cls.DATASETS_FOLDER, 'train', 'unknown'),
            os.path.join(cls.DATASETS_FOLDER, 'test', 'real'),
            os.path.join(cls.DATASETS_FOLDER, 'test', 'stable_diffusion'),
            os.path.join(cls.DATASETS_FOLDER, 'test', 'midjourney'),
            os.path.join(cls.DATASETS_FOLDER, 'test', 'dalle'),
            os.path.join(cls.DATASETS_FOLDER, 'test', 'unknown'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Development configuration
class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    
# Production configuration
class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for production
    
# Testing configuration
class TestingConfig(Config):
    """Testing-specific configuration"""
    TESTING = True
    BATCH_SIZE = 8
    EPOCHS = 2  # Fewer epochs for quick testing

# Select configuration
config = Config  # Change to DevelopmentConfig or ProductionConfig as needed
