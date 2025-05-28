"""
Image Preprocessing Pipeline
OpenCV and PIL-based image enhancement and normalization
"""

import logging
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Optional, Tuple, Dict, Any
from utils.logging import PerformanceLogger

class ImageProcessor:
    """Image preprocessing and enhancement pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_image_quality(self, image_bytes: bytes) -> bytes:
        """
        Enhance image quality for better CLIP processing
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Enhanced image bytes
            
        Raises:
            ValueError: If image processing fails
        """
        try:
            with PerformanceLogger("Image quality enhancement", self.logger):
                # Load image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Apply enhancement pipeline
                enhanced_image = self._apply_enhancement_pipeline(image)
                
                # Convert back to bytes
                output_buffer = io.BytesIO()
                enhanced_image.save(output_buffer, format='JPEG', quality=95)
                
                return output_buffer.getvalue()
                
        except Exception as e:
            self.logger.error(f"Image enhancement failed: {e}")
            raise ValueError(f"Image processing failed: {e}")
    
    def _apply_enhancement_pipeline(self, image: Image.Image) -> Image.Image:
        """
        Apply comprehensive image enhancement pipeline
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to OpenCV format for advanced processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply OpenCV enhancements
        cv_image = self._enhance_with_opencv(cv_image)
        
        # Convert back to PIL
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Apply PIL enhancements
        pil_image = self._enhance_with_pil(pil_image)
        
        return pil_image
    
    def _enhance_with_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        Apply OpenCV-based enhancements
        
        Args:
            image: OpenCV image array (BGR format)
            
        Returns:
            Enhanced OpenCV image array
        """
        # Noise reduction
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Adaptive histogram equalization for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sharpening kernel
        sharpening_kernel = np.array([[-1, -1, -1],
                                     [-1,  9, -1],
                                     [-1, -1, -1]])
        image = cv2.filter2D(image, -1, sharpening_kernel)
        
        # Ensure values are in valid range
        image = np.clip(image, 0, 255)
        
        return image
    
    def _enhance_with_pil(self, image: Image.Image) -> Image.Image:
        """
        Apply PIL-based enhancements
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        # Auto-adjust brightness and contrast
        image = self._auto_adjust_brightness_contrast(image)
        
        # Enhance color saturation slightly
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # Apply slight unsharp mask for better edge definition
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        return image
    
    def _auto_adjust_brightness_contrast(self, image: Image.Image) -> Image.Image:
        """
        Automatically adjust brightness and contrast based on image statistics
        
        Args:
            image: PIL Image object
            
        Returns:
            Adjusted PIL Image
        """
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        # Calculate image statistics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # Determine adjustment factors
        target_mean = 128  # Target middle brightness
        brightness_factor = target_mean / mean_brightness if mean_brightness > 0 else 1.0
        brightness_factor = np.clip(brightness_factor, 0.7, 1.5)  # Limit adjustment range
        
        # Contrast adjustment based on standard deviation
        target_std = 50  # Target contrast level
        contrast_factor = target_std / std_brightness if std_brightness > 0 else 1.0
        contrast_factor = np.clip(contrast_factor, 0.8, 1.3)  # Limit adjustment range
        
        # Apply adjustments
        if abs(brightness_factor - 1.0) > 0.1:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        if abs(contrast_factor - 1.0) > 0.1:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        return image
    
    def resize_image(self, image_bytes: bytes, max_size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image_bytes: Raw image bytes
            max_size: Maximum dimensions (width, height)
            
        Returns:
            Resized image bytes
        """
        try:
            with PerformanceLogger("Image resize", self.logger):
                image = Image.open(io.BytesIO(image_bytes))
                
                # Calculate new size maintaining aspect ratio
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                output_buffer = io.BytesIO()
                image.save(output_buffer, format='JPEG', quality=90)
                
                return output_buffer.getvalue()
                
        except Exception as e:
            self.logger.error(f"Image resize failed: {e}")
            raise ValueError(f"Image resize failed: {e}")
    
    def get_image_info(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Get information about an image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with image information
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "file_size": len(image_bytes),
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get image info: {e}")
            return {"error": str(e)}
    
    def validate_image(self, image_bytes: bytes, max_size_mb: int = 25) -> bool:
        """
        Validate image format and size
        
        Args:
            image_bytes: Raw image bytes
            max_size_mb: Maximum file size in MB
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check file size
            if len(image_bytes) > max_size_mb * 1024 * 1024:
                self.logger.warning(f"Image too large: {len(image_bytes)} bytes")
                return False
            
            # Try to open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check if it's a valid image format
            if image.format not in ['JPEG', 'PNG', 'GIF', 'WEBP', 'BMP']:
                self.logger.warning(f"Unsupported image format: {image.format}")
                return False
            
            # Check dimensions
            if image.width < 32 or image.height < 32:
                self.logger.warning(f"Image too small: {image.width}x{image.height}")
                return False
            
            if image.width > 4096 or image.height > 4096:
                self.logger.warning(f"Image too large: {image.width}x{image.height}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Image validation failed: {e}")
            return False
    
    def preprocess_for_clip(self, image_bytes: bytes) -> bytes:
        """
        Complete preprocessing pipeline for CLIP model
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed image bytes ready for CLIP
        """
        try:
            with PerformanceLogger("Complete image preprocessing", self.logger):
                # Validate image
                if not self.validate_image(image_bytes):
                    raise ValueError("Image validation failed")
                
                # Resize if too large
                image_info = self.get_image_info(image_bytes)
                if image_info.get("width", 0) > 1024 or image_info.get("height", 0) > 1024:
                    image_bytes = self.resize_image(image_bytes)
                
                # Enhance quality
                enhanced_bytes = self.enhance_image_quality(image_bytes)
                
                self.logger.debug("Image preprocessing completed successfully")
                return enhanced_bytes
                
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Preprocessing failed: {e}") 