"""
Noise Pattern Analysis Module
Analyzes noise patterns to distinguish real camera noise from AI artifacts
"""

import numpy as np
import cv2
from core.config import Config
WEIGHTS = Config.WEIGHTS
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

class NoiseAnalyzer:
    """Analyzes noise patterns in images"""
    
    def __init__(self):
        self.kernel_sizes = [3, 5, 7]
        
    def analyze(self, image):
        """
        Perform comprehensive noise analysis
        
        Args:
            image: numpy array (H, W, C) or (H, W)
            
        Returns:
            dict: Analysis results with scores
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Extract noise residual
        noise_residual = self._extract_noise_residual(gray)
        
        # Analyze noise characteristics
        noise_std = self._compute_noise_std(noise_residual)
        noise_uniformity = self._compute_noise_uniformity(noise_residual)
        noise_correlation = self._compute_noise_correlation(noise_residual)
        grain_structure = self._analyze_grain_structure(noise_residual)
        
        # Calculate noise score (0-100)
        noise_score = self._calculate_noise_score(
            noise_std,
            noise_uniformity,
            noise_correlation,
            grain_structure
        )
        
        return {
            'noise_score': round(noise_score, 2),
            'noise_std': round(float(noise_std), 4),
            'noise_uniformity': round(noise_uniformity, 4),
            'noise_correlation': round(noise_correlation, 4),
            'grain_structure': round(grain_structure, 4),
            'confidence': self._interpret_score(noise_score)
        }
    
    def _extract_noise_residual(self, gray):
        """Extract noise residual using high-pass filtering"""
        # Apply Gaussian blur to get low-frequency content
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Subtract to get high-frequency (noise) residual
        noise_residual = gray.astype(np.float32) - blurred.astype(np.float32)
        
        return noise_residual
    
    def _compute_noise_std(self, noise_residual):
        """Compute standard deviation of noise"""
        return np.std(noise_residual)
    
    def _compute_noise_uniformity(self, noise_residual):
        """
        Measure uniformity of noise distribution
        Real camera noise is typically more uniform
        """
        # Divide image into blocks
        h, w = noise_residual.shape
        block_size = 32
        
        blocks_std = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = noise_residual[i:i+block_size, j:j+block_size]
                blocks_std.append(np.std(block))
        
        # Uniformity is inverse of variation in block STDs
        if len(blocks_std) > 0:
            uniformity = 1.0 / (1.0 + np.std(blocks_std))
        else:
            uniformity = 0.0
            
        return uniformity
    
    def _compute_noise_correlation(self, noise_residual):
        """
        Compute spatial correlation of noise
        Real camera noise has low correlation
        """
        # Compute autocorrelation at offset (1,0)
        shifted = np.roll(noise_residual, 1, axis=1)
        correlation = np.corrcoef(noise_residual.flatten(), shifted.flatten())[0, 1]
        
        # Return absolute correlation
        return abs(correlation)
    
    def _analyze_grain_structure(self, noise_residual):
        """
        Analyze grain-like structure in noise
        Real images have characteristic grain patterns
        """
        # Apply Laplacian to detect edges in noise
        laplacian = cv2.Laplacian(noise_residual.astype(np.float32), cv2.CV_32F)
        
        # Compute energy of Laplacian
        grain_energy = np.mean(np.abs(laplacian))
        
        return grain_energy
    
    def _calculate_noise_score(self, noise_std, noise_uniformity, 
                                noise_correlation, grain_structure):
        """
        Calculate final noise score
        Higher score = more likely to be real
        """
        score = 0
        
        # Noise STD contribution (0-30 points)
        # Real images typically have moderate noise (3-15)
        if 3 < noise_std < 15:
            score += 30
        elif 1 < noise_std < 20:
            score += 20
        else:
            score += 10
            
        # Uniformity contribution (0-30 points)
        # Real images have more uniform noise
        if noise_uniformity > 0.15:
            score += 30
        elif noise_uniformity > 0.10:
            score += 20
        else:
            score += 10
            
        # Correlation contribution (0-20 points)
        # Real noise has low correlation
        if noise_correlation < 0.3:
            score += 20
        elif noise_correlation < 0.5:
            score += 10
        else:
            score += 5
            
        # Grain structure contribution (0-20 points)
        if grain_structure > 1.0:
            score += 20
        elif grain_structure > 0.5:
            score += 10
        else:
            score += 5
            
        return score
    
    def _interpret_score(self, score):
        """Interpret noise score"""
        if score >= 70:
            return "High confidence - Real camera noise detected"
        elif score >= 50:
            return "Medium confidence - Ambiguous noise pattern"
        else:
            return "Low confidence - Synthetic noise pattern"
