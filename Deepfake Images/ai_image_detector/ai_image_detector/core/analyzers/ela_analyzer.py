"""
Error Level Analysis (ELA) Analyzer
Detects image manipulations and AI creations by measuring differences
in JPEG compression levels across the image.
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
from core.config import Config

WEIGHTS = Config.WEIGHTS

class ErrorLevelAnalyzer:
    """Analyzes image compression artifacts to detect synthetic origin"""
    
    def __init__(self, quality=90):
        self.quality = quality
        
    def analyze(self, image_path):
        """
        Perform Error Level Analysis (ELA)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Analysis results with scores
        """
        try:
            # We must use Pillow for ELA, not OpenCV, to precisely control JPEG resave
            original = Image.open(image_path).convert('RGB')
            
            # Save temporary compressed version
            temp_path = image_path + ".ela_temp.jpg"
            original.save(temp_path, 'JPEG', quality=self.quality)
            
            # Load the compressed version
            compressed = Image.open(temp_path)
            
            # Subtract the two images
            ela_image = ImageChops.difference(original, compressed)
            
            # Enhance the difference to make it visible/measurable
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            
            # Handle edge case where images are perfectly identical
            if max_diff == 0:
                max_diff = 1
                
            scale = 255.0 / max_diff
            ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
            
            # Convert enhanced ELA image back to numpy for statistical analysis
            ela_np = np.array(ela_enhanced)
            
            # Calculate ELA metrics
            # High uniform error variance = Likely AI generated
            # Localized high error = Likely manually photoshopped
            # Low variance, predictable error = Unedited real photo
            
            mean_error = np.mean(ela_np)
            std_error = np.std(ela_np)
            max_error = np.max(ela_np)
            
            # Normalize to a 0-100 score where higher means more 'real'
            # AI images typically have unusually low or unnaturally uniform ELA.
            # Very high std_error implies manual tampering.
            
            score = 50.0 # Neutral
            
            if 5.0 < std_error < 25.0:
                score += 30 # Natural organic compression variation
            elif std_error < 3.0:
                score -= 30 # Dangerously uniform - likely heavily synthetic or totally flat AI
            elif std_error > 40.0:
                score -= 20 # High local tampering
                
            if 10.0 < mean_error < 40.0:
                score += 20
                
            final_score = max(0, min(100, score))
            
            # Cleanup temp file
            compressed.close()
            os.remove(temp_path)
            
            return {
                'ela_score': round(final_score, 2),
                'mean_error': round(float(mean_error), 2),
                'std_error_variance': round(float(std_error), 2),
                'max_error_spike': int(max_error)
            }
            
        except Exception as e:
            if os.path.exists(f"{image_path}.ela_temp.jpg"):
                try: os.remove(f"{image_path}.ela_temp.jpg")
                except: pass
                
            return {
                'ela_score': 50.0,
                'mean_error': 0.0,
                'std_error_variance': 0.0,
                'max_error_spike': 0,
                'error': str(e)
            }
