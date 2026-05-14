"""
Frequency Analysis Module
Analyzes image frequency spectrum to detect AI artifacts
"""

import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import entropy
from core.config import Config
WEIGHTS = Config.WEIGHTS

class FrequencyAnalyzer:
    """Analyzes frequency domain characteristics of images"""
    
    def __init__(self):
        self.high_freq_threshold = 0.7
        
    def analyze(self, image):
        """
        Perform comprehensive frequency analysis
        
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
            
        # Compute 2D FFT
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Log scale for better visualization
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)
        
        # Extract features
        high_freq_energy = self._compute_high_freq_energy(magnitude_spectrum)
        radial_profile = self._compute_radial_profile(magnitude_spectrum)
        peak_anomalies = self._detect_peak_anomalies(magnitude_spectrum)
        spectral_entropy = self._compute_spectral_entropy(magnitude_spectrum)
        
        # Calculate frequency score (0-100)
        # Real images typically have more high-frequency content
        frequency_score = self._calculate_frequency_score(
            high_freq_energy, 
            peak_anomalies, 
            spectral_entropy
        )
        
        return {
            'frequency_score': round(frequency_score, 2),
            'high_freq_energy': round(high_freq_energy, 4),
            'peak_anomalies': round(peak_anomalies, 4),
            'spectral_entropy': round(spectral_entropy, 4),
            'radial_mean': round(float(np.mean(radial_profile)), 4),
            'confidence': self._interpret_score(frequency_score)
        }
    
    def _compute_high_freq_energy(self, magnitude_spectrum):
        """Calculate energy in high-frequency regions"""
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define high-frequency region (outer 30% of spectrum)
        radius = min(center_h, center_w)
        high_freq_radius = int(radius * self.high_freq_threshold)
        
        # Create mask for high frequencies
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > high_freq_radius**2
        
        total_energy = np.sum(magnitude_spectrum**2)
        high_freq_energy = np.sum((magnitude_spectrum * mask)**2)
        
        return high_freq_energy / (total_energy + 1e-10)
    
    def _compute_radial_profile(self, magnitude_spectrum):
        """Compute radial average of frequency spectrum"""
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)
        
        # Compute radial average
        radial_profile = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())
        
        return radial_profile
    
    def _detect_peak_anomalies(self, magnitude_spectrum):
        """Detect unusual peaks in frequency domain"""
        # Normalize spectrum
        normalized = magnitude_spectrum / (np.max(magnitude_spectrum) + 1e-10)
        
        # Count significant peaks (above 80th percentile)
        threshold = np.percentile(normalized, 80)
        peaks = np.sum(normalized > threshold)
        
        # AI images often have fewer distinct peaks
        total_pixels = magnitude_spectrum.size
        peak_ratio = peaks / total_pixels
        
        return peak_ratio
    
    def _compute_spectral_entropy(self, magnitude_spectrum):
        """Calculate entropy of frequency spectrum"""
        # Flatten and normalize
        flat_spectrum = magnitude_spectrum.flatten()
        flat_spectrum = flat_spectrum / (np.sum(flat_spectrum) + 1e-10)
        
        # Compute entropy
        spectral_entropy = entropy(flat_spectrum + 1e-10)
        
        return spectral_entropy
    
    def _calculate_frequency_score(self, high_freq_energy, peak_anomalies, spectral_entropy):
        """
        Calculate final frequency score
        Higher score = more likely to be real
        """
        # Real images typically have:
        # - More high-frequency energy
        # - More spectral entropy
        # - More peak anomalies
        
        score = 0
        
        # High frequency energy contribution (0-40 points)
        if high_freq_energy > 0.15:
            score += 40
        elif high_freq_energy > 0.10:
            score += 30
        elif high_freq_energy > 0.05:
            score += 20
        else:
            score += 10
            
        # Spectral entropy contribution (0-35 points)
        if spectral_entropy > 10:
            score += 35
        elif spectral_entropy > 8:
            score += 25
        elif spectral_entropy > 6:
            score += 15
        else:
            score += 5
            
        # Peak anomalies contribution (0-25 points)
        if peak_anomalies > 0.008:
            score += 25
        elif peak_anomalies > 0.005:
            score += 15
        else:
            score += 5
            
        return score
    
    def _interpret_score(self, score):
        """Interpret frequency score"""
        if score >= 70:
            return "High confidence - Real image characteristics"
        elif score >= 50:
            return "Medium confidence - Mixed characteristics"
        else:
            return "Low confidence - AI-generated characteristics"
