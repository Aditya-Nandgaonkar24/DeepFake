"""
Metadata Analysis Module
Examines EXIF and other metadata to detect AI generation indicators
"""

import piexif
from PIL import Image
from PIL.ExifTags import TAGS
import io
from core.config import Config
WEIGHTS = Config.WEIGHTS

class MetadataAnalyzer:
    """Analyzes image metadata for authenticity indicators"""
    
    def __init__(self):
        self.ai_software_keywords = [
            'stable diffusion', 'midjourney', 'dall-e', 'dalle',
            'ai', 'generated', 'artificial', 'synthetic',
            'openai', 'stability', 'adobe firefly', 'firefly'
        ]
        
        self.real_camera_manufacturers = [
            'canon', 'nikon', 'sony', 'fujifilm', 'olympus',
            'panasonic', 'leica', 'pentax', 'hasselblad',
            'phase one', 'apple', 'samsung', 'google', 'huawei'
        ]
        
    def analyze(self, image_path):
        """
        Perform comprehensive metadata analysis
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Analysis results with scores
        """
        try:
            img = Image.open(image_path)
            
            # Extract EXIF data
            exif_data = self._extract_exif(img)
            
            has_exif = len(exif_data) > 0
            camera_model = self._extract_camera_model(exif_data)
            software = self._extract_software(exif_data)
            has_camera_info = camera_model is not None
            has_icc_profile = self._check_icc_profile(img)
            
            # Check for AI indicators
            ai_indicators = self._check_ai_indicators(software)
            camera_authentic = self._check_camera_authenticity(camera_model)
            
            # Calculate metadata score (0-100)
            metadata_score = self._calculate_metadata_score(
                has_exif,
                has_camera_info,
                camera_authentic,
                ai_indicators,
                has_icc_profile
            )
            
            return {
                'metadata_score': round(metadata_score, 2),
                'has_exif': has_exif,
                'camera_model': camera_model if camera_model else 'Not found',
                'software': software if software else 'Not found',
                'exif_tags_count': len(exif_data),
                'ai_indicators_found': ai_indicators,
                'camera_authentic': camera_authentic,
                'has_icc_profile': has_icc_profile,
                'confidence': self._interpret_score(metadata_score)
            }
            
        except Exception as e:
            return {
                'metadata_score': 50.0, # Default to neutral on completely broken/stripped metadata
                'has_exif': False,
                'camera_model': 'Error reading metadata',
                'software': 'Error reading metadata',
                'exif_tags_count': 0,
                'ai_indicators_found': False,
                'camera_authentic': False,
                'has_icc_profile': False,
                'confidence': 'Unable to read metadata - Neutral State',
                'error': str(e)
            }
    
    def _extract_exif(self, img):
        """Extract EXIF data from image"""
        exif_data = {}
        
        try:
            # Try to get EXIF data
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
        except:
            pass
            
        return exif_data
    
    def _extract_camera_model(self, exif_data):
        """Extract camera model from EXIF"""
        model_keys = ['Model', 'CameraModel', 'Make']
        
        for key in model_keys:
            if key in exif_data:
                return str(exif_data[key])
                
        return None
    
    def _extract_software(self, exif_data):
        """Extract software information from EXIF"""
        software_keys = ['Software', 'ProcessingSoftware', 'CreatorTool']
        
        for key in software_keys:
            if key in exif_data:
                return str(exif_data[key])
                
        return None
    
    def _check_ai_indicators(self, software):
        """Check if software field contains AI generation indicators"""
        if not software:
            return False
            
        software_lower = software.lower()
        return any(keyword in software_lower for keyword in self.ai_software_keywords)
    
    def _check_camera_authenticity(self, camera_model):
        """Check if camera model is from a real manufacturer"""
        if not camera_model:
            return False
            
        camera_lower = camera_model.lower()
        return any(manufacturer in camera_lower for manufacturer in self.real_camera_manufacturers)
    
    def _check_icc_profile(self, img):
        """Check if image has an embedded color profile (common in real cameras, often missing in AI)"""
        return 'icc_profile' in img.info
    
    def _calculate_metadata_score(self, has_exif, has_camera_info, 
                                   camera_authentic, ai_indicators, has_icc_profile):
        """
        Calculate final metadata score
        Higher score = more likely to be real
        Handles social media (stripped metadata) by defaulting to neutral 50.
        """
        score = 50 # Start at neutral baseline (assume stripped metadata)
        
        # Immediate fatal penalties
        if ai_indicators:
            return 0.0 # Guaranteed AI
            
        # Social media case (completely stripped of EXIF and ICC)
        if not has_exif and not has_icc_profile:
            return 50.0 # Purely neutral, let the Deep Learning model decide
            
        # Additive indicators of real camera origins
        if has_exif:
            score += 15
        
        if has_icc_profile:
            score += 15 # Real cameras almost always embed ICC profiles
            
        if has_camera_info:
            score += 10
            if camera_authentic:
                score += 10
                
        return max(0, min(100, score))

    def _interpret_score(self, score):
        """Interpret metadata score"""
        if score >= 70:
            return "High confidence - Real camera metadata"
        elif score == 50:
            return "Neutral - Missing or stripped metadata (Social Media)"
        elif score >= 40:
            return "Medium confidence - Limited metadata"
        else:
            return "Low confidence - AI indicators found"
