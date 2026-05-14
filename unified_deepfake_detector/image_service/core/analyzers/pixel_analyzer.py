"""
Pixel Statistics Analysis Module
Analyzes low-level pixel characteristics
"""

import numpy as np
import cv2
from scipy import stats
from core.config import Config
WEIGHTS = Config.WEIGHTS

class PixelStatisticsAnalyzer:
    """Analyzes pixel-level statistics"""
    
    def __init__(self):
        pass
        
    def analyze(self, image):
        """
        Perform comprehensive pixel statistics analysis
        
        Args:
            image: numpy array (H, W, C) or (H, W)
            
        Returns:
            dict: Analysis results with scores
        """
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Calculate various statistics
        color_distribution = self._analyze_color_distribution(image)
        saturation_stats = self._analyze_saturation(image)
        brightness_stats = self._analyze_brightness(image)
        contrast_ratio = self._analyze_contrast(image)
        histogram_analysis = self._analyze_histogram(image)
        
        # Calculate pixel statistics score (0-100)
        pixel_score = self._calculate_pixel_score(
            color_distribution,
            saturation_stats,
            brightness_stats,
            contrast_ratio,
            histogram_analysis
        )
        
        return {
            'pixel_score': round(pixel_score, 2),
            'color_variance': round(color_distribution['variance'], 4),
            'mean_saturation': round(saturation_stats['mean'], 4),
            'mean_brightness': round(brightness_stats['mean'], 4),
            'contrast_ratio': round(contrast_ratio, 4),
            'histogram_entropy': round(histogram_analysis['entropy'], 4),
            'confidence': self._interpret_score(pixel_score)
        }
    
    def _analyze_color_distribution(self, image):
        """Analyze color distribution across channels"""
        # Convert to float for calculations
        img_float = image.astype(np.float32)
        
        # Calculate mean and variance per channel
        means = np.mean(img_float, axis=(0, 1))
        variances = np.var(img_float, axis=(0, 1))
        
        # Calculate overall statistics
        overall_mean = np.mean(means)
        overall_variance = np.mean(variances)
        
        # Calculate color balance
        color_balance = np.std(means)
        
        return {
            'mean': overall_mean,
            'variance': overall_variance,
            'color_balance': color_balance
        }
    
    def _analyze_saturation(self, image):
        """Analyze color saturation"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        return {
            'mean': np.mean(saturation),
            'std': np.std(saturation),
            'max': np.max(saturation)
        }
    
    def _analyze_brightness(self, image):
        """Analyze brightness distribution"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        
        return {
            'mean': np.mean(value),
            'std': np.std(value),
            'dynamic_range': np.max(value) - np.min(value)
        }
    
    def _analyze_contrast(self, image):
        """Calculate contrast ratio"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Michelson contrast
        max_intensity = np.max(gray)
        min_intensity = np.min(gray)
        
        if max_intensity + min_intensity > 0:
            contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        else:
            contrast = 0
            
        return contrast
    
    def _analyze_histogram(self, image):
        """Analyze histogram characteristics"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-10)
        
        # Calculate entropy
        hist_entropy = stats.entropy(hist + 1e-10)
        
        # Check for suspicious peaks (AI images may have unnatural histograms)
        peak_count = np.sum(hist > np.mean(hist) + 2 * np.std(hist))
        
        return {
            'entropy': hist_entropy,
            'peak_count': peak_count
        }
    
    def _calculate_pixel_score(self, color_dist, saturation, brightness, 
                                contrast, histogram):
        """
        Calculate final pixel statistics score
        Higher score = more likely to be real
        """
        score = 0
        
        # Color variance contribution (0-20 points)
        # Real images have natural variance
        if 500 < color_dist['variance'] < 5000:
            score += 20
        elif color_dist['variance'] > 100:
            score += 10
        else:
            score += 5
            
        # Saturation contribution (0-20 points)
        # Real images have natural saturation
        if 50 < saturation['mean'] < 180:
            score += 20
        elif 20 < saturation['mean'] < 200:
            score += 10
        else:
            score += 5
            
        # Brightness contribution (0-20 points)
        if 60 < brightness['mean'] < 200:
            score += 20
        else:
            score += 10
            
        # Contrast contribution (0-20 points)
        if contrast > 0.5:
            score += 20
        elif contrast > 0.3:
            score += 15
        else:
            score += 10
            
        # Histogram entropy contribution (0-20 points)
        # Real images typically have higher entropy
        if histogram['entropy'] > 6.0:
            score += 20
        elif histogram['entropy'] > 4.5:
            score += 15
        else:
            score += 10
            
        return score
    
    def _interpret_score(self, score):
        """Interpret pixel statistics score"""
        if score >= 70:
            return "High confidence - Natural pixel distribution"
        elif score >= 50:
            return "Medium confidence - Acceptable pixel characteristics"
        else:
            return "Low confidence - Unusual pixel patterns"
