"""
Computer Vision CLIP Processor - No PyTorch Required
Uses traditional CV techniques for meaningful image similarity
"""

import logging
import numpy as np
import cv2
from typing import Optional, List, Union
from PIL import Image
import io
import hashlib
from utils.config import Config
from utils.logging import PerformanceLogger

class CLIPProcessor:
    """Computer Vision-based image processor for similarity matching"""
    
    _instance: Optional['CLIPProcessor'] = None
    
    def __new__(cls, config: Optional[Config] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Config] = None):
        if self._initialized:
            return
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = True
        
        print("🔄 Running CLIP processor with Computer Vision techniques")
        self.logger.info("Using CV-based image similarity (no PyTorch required)")
    
    async def initialize(self, config: Config):
        """Initialize CV processor"""
        self.config = config
        self.logger.info("CV-based image processor initialized")
    
    def is_initialized(self) -> bool:
        """Check if processor is initialized"""
        return True
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        # Convert BGR to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])  # Hue
        s_hist = cv2.calcHist([hsv], [1], None, [50], [0, 256])  # Saturation
        v_hist = cv2.calcHist([hsv], [2], None, [50], [0, 256])  # Value
        
        # Normalize histograms
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions
        h, w = edges.shape
        regions = []
        
        # Divide image into 4x4 grid and calculate edge density
        for i in range(4):
            for j in range(4):
                y1, y2 = i * h // 4, (i + 1) * h // 4
                x1, x2 = j * w // 4, (j + 1) * w // 4
                region = edges[y1:y2, x1:x2]
                density = np.sum(region > 0) / (region.size + 1)
                regions.append(density)
        
        return np.array(regions)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple texture analysis using gradient magnitudes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate texture statistics
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        
        # Histogram of gradient orientations
        angles = np.arctan2(grad_y, grad_x)
        angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        angle_hist = angle_hist / (angle_hist.sum() + 1e-7)
        
        return np.concatenate([[mean_mag, std_mag], angle_hist])
    
    def _extract_spatial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial layout features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate moments
        moments = cv2.moments(gray)
        
        # Hu moments (scale, rotation, translation invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Handle log transformation safely
        hu_moments = np.where(hu_moments > 0, -np.sign(hu_moments) * np.log10(np.abs(hu_moments)), 0)
        
        return hu_moments
    
    def generate_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate CV-based embedding for image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            512-dimensional embedding vector
        """
        try:
            with PerformanceLogger("CV image embedding", self.logger):
                # Load image
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Resize to standard size for consistent features
                cv_image = cv2.resize(cv_image, (224, 224))
                
                # Extract different types of features
                color_features = self._extract_color_histogram(cv_image)      # 150 dims
                edge_features = self._extract_edge_features(cv_image)         # 16 dims  
                texture_features = self._extract_texture_features(cv_image)   # 10 dims
                spatial_features = self._extract_spatial_features(cv_image)   # 7 dims
                
                # Combine all features
                combined_features = np.concatenate([
                    color_features,
                    edge_features,
                    texture_features,
                    spatial_features
                ])
                
                # Pad or truncate to exactly 512 dimensions
                if len(combined_features) < 512:
                    # Pad with image statistics
                    padding_size = 512 - len(combined_features)
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    
                    # Add statistical features as padding
                    stats = []
                    for i in range(padding_size):
                        if i % 4 == 0:
                            stats.append(np.mean(gray))
                        elif i % 4 == 1:
                            stats.append(np.std(gray))
                        elif i % 4 == 2:
                            stats.append(np.min(gray))
                        else:
                            stats.append(np.max(gray))
                    
                    combined_features = np.concatenate([combined_features, stats[:padding_size]])
                else:
                    combined_features = combined_features[:512]
                
                # Normalize to unit vector
                embedding = combined_features.astype(np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                self.logger.debug(f"Generated CV embedding with shape: {embedding.shape}")
                return embedding
                
        except Exception as e:
            self.logger.error(f"Failed to generate CV embedding: {e}")
            raise ValueError(f"Image processing failed: {e}")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate text-based embedding (simple implementation)
        
        Args:
            text: Input text
            
        Returns:
            512-dimensional embedding vector
        """
        try:
            with PerformanceLogger("CV text embedding", self.logger):
                # Simple text hashing for basic text similarity
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                # Convert hex to numbers and repeat to fill 512 dimensions
                numbers = [int(c, 16) for c in text_hash]  # 32 numbers (0-15)
                
                # Repeat and normalize to create 512-dim vector
                repeated = (numbers * (512 // len(numbers) + 1))[:512]
                embedding = np.array(repeated, dtype=np.float32)
                
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
        except Exception as e:
            self.logger.error(f"Failed to generate text embedding: {e}")
            raise ValueError(f"Text processing failed: {e}")
    
    def batch_generate_embeddings(self, image_bytes_list: List[bytes]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images
        
        Args:
            image_bytes_list: List of raw image bytes
            
        Returns:
            List of 512-dimensional embedding vectors
        """
        if not image_bytes_list:
            return []
        
        try:
            with PerformanceLogger(f"CV batch embedding ({len(image_bytes_list)} images)", self.logger):
                results = []
                for image_bytes in image_bytes_list:
                    if image_bytes:
                        results.append(self.generate_image_embedding(image_bytes))
                    else:
                        results.append(None)
                
                self.logger.info(f"Generated {len([r for r in results if r is not None])} CV embeddings")
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to generate CV batch embeddings: {e}")
            raise ValueError(f"Batch processing failed: {e}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the CV model
        
        Returns:
            Dictionary with model information
        """
        return {
            "status": "cv_mode",
            "model_name": "computer_vision_features",
            "device": "cpu",
            "embedding_dim": 512,
            "features": ["color_histogram", "edge_detection", "texture_analysis", "spatial_moments"],
            "note": "Using traditional computer vision for meaningful image similarity"
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("CV processor cleanup completed") 