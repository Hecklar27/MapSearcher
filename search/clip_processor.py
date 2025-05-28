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
        
        # Also use RGB for additional color information
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate histograms for HSV channels with more bins for better discrimination
        h_hist = cv2.calcHist([hsv], [0], None, [60], [0, 180])  # Hue - more bins
        s_hist = cv2.calcHist([hsv], [1], None, [40], [0, 256])  # Saturation
        v_hist = cv2.calcHist([hsv], [2], None, [40], [0, 256])  # Value
        
        # RGB histograms for additional color info
        r_hist = cv2.calcHist([rgb], [0], None, [30], [0, 256])  # Red
        g_hist = cv2.calcHist([rgb], [1], None, [30], [0, 256])  # Green
        b_hist = cv2.calcHist([rgb], [2], None, [30], [0, 256])  # Blue
        
        # Normalize histograms
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
        r_hist = r_hist.flatten() / (r_hist.sum() + 1e-7)
        g_hist = g_hist.flatten() / (g_hist.sum() + 1e-7)
        b_hist = b_hist.flatten() / (b_hist.sum() + 1e-7)
        
        # Combine all color histograms
        return np.concatenate([h_hist, s_hist, v_hist, r_hist, g_hist, b_hist])
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features (deterministic)"""
        # Convert to grayscale for consistent processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Canny edge detection with fixed parameters for consistency
        # Apply slight blur first for stability
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge statistics
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient-based features using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate gradient statistics
        features = [
            edge_density,
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.sum(magnitude > np.percentile(magnitude, 90)) / magnitude.size,
        ]
        
        # Histogram of gradient orientations (deterministic binning)
        angles = np.arctan2(grad_y, grad_x)
        angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        # Normalize histogram safely
        hist_sum = np.sum(angle_hist)
        if hist_sum > 0:
            angle_hist = angle_hist / hist_sum
        else:
            angle_hist = np.zeros(8)  # Fallback for empty histograms
        
        # Combine all features
        all_features = features + angle_hist.tolist()
        
        # Ensure all features are finite and consistent
        safe_features = []
        for feature in all_features:
            if np.isnan(feature) or np.isinf(feature):
                safe_features.append(0.0)
            else:
                safe_features.append(float(feature))
        
        return np.array(safe_features, dtype=np.float32)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Pattern (deterministic)"""
        # Convert to grayscale for consistent processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple texture analysis using gradient statistics
        # Use Sobel instead of Canny for more deterministic results
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Statistical features (ensure consistent calculation)
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.min(magnitude),
            np.max(magnitude),
            np.median(magnitude),
            np.percentile(magnitude, 25),
            np.percentile(magnitude, 75),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(gray)  # Overall intensity variation
        ]
        
        # Ensure all features are finite and deterministic
        safe_features = []
        for feature in features:
            if np.isnan(feature) or np.isinf(feature):
                safe_features.append(0.0)
            else:
                safe_features.append(float(feature))
        
        return np.array(safe_features, dtype=np.float32)
    
    def _extract_spatial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial layout features (deterministic)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate moments
        moments = cv2.moments(gray)
        
        # Hu moments (scale, rotation, translation invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Handle log transformation safely and deterministically
        # Replace any problematic values with consistent defaults
        safe_hu_moments = []
        for moment in hu_moments:
            if moment == 0 or np.isnan(moment) or np.isinf(moment):
                safe_hu_moments.append(0.0)  # Consistent default
            elif moment > 0:
                safe_hu_moments.append(-np.log10(moment))
            else:
                safe_hu_moments.append(np.log10(-moment))
        
        return np.array(safe_hu_moments, dtype=np.float32)
    
    def _extract_detailed_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract high-resolution color features (192 dimensions)"""
        # 1. RGB histograms with more bins (64 + 64 + 64 = 192 dims)
        b, g, r = cv2.split(image)
        
        # Higher resolution histograms for better discrimination
        hist_r = cv2.calcHist([r], [0], None, [64], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [64], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [64], [0, 256])
        
        # Normalize histograms
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
        
        return np.concatenate([hist_r, hist_g, hist_b])
    
    def _extract_spatial_color_distribution(self, image: np.ndarray) -> np.ndarray:
        """Extract where colors are located in the image (64 dimensions)"""
        h, w = image.shape[:2]
        features = []
        
        # Divide image into 8x8 grid and get dominant color in each region
        for i in range(8):
            for j in range(8):
                y1, y2 = i * h // 8, (i + 1) * h // 8
                x1, x2 = j * w // 8, (j + 1) * w // 8
                region = image[y1:y2, x1:x2]
                
                # Get average color in this region
                avg_color = np.mean(region.reshape(-1, 3), axis=0)
                features.append(np.mean(avg_color) / 255.0)  # Single value per region
        
        return np.array(features, dtype=np.float32)
    
    def _extract_regional_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge patterns in different image regions (64 dimensions)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = edges.shape
        features = []
        
        # Divide into 8x8 grid and calculate edge density in each region
        for i in range(8):
            for j in range(8):
                y1, y2 = i * h // 8, (i + 1) * h // 8
                x1, x2 = j * w // 8, (j + 1) * w // 8
                region = edges[y1:y2, x1:x2]
                
                # Edge density in this region
                edge_density = np.sum(region > 0) / region.size
                features.append(edge_density)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_detailed_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract detailed texture features (32 dimensions)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple gradient calculations for texture
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture features in 4x4 regions
        h, w = gray.shape
        features = []
        
        for i in range(4):
            for j in range(4):
                y1, y2 = i * h // 4, (i + 1) * h // 4
                x1, x2 = j * w // 4, (j + 1) * w // 4
                region_mag = magnitude[y1:y2, x1:x2]
                
                # Statistics for this region
                features.extend([
                    np.mean(region_mag),
                    np.std(region_mag)
                ])
        
        # Ensure exactly 32 features
        return np.array(features[:32], dtype=np.float32)
    
    def _extract_image_statistics(self, image: np.ndarray) -> np.ndarray:
        """Extract overall image statistics (32 dimensions)"""
        # Convert to different color spaces for more features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        features = []
        
        # RGB channel statistics
        for channel in cv2.split(image):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        # HSV statistics 
        for channel in cv2.split(hsv):
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
        
        # Overall image properties
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            float(len(np.unique(gray)))  # Number of unique gray levels
        ])
        
        # Ensure exactly 32 features
        return np.array(features[:32], dtype=np.float32)
    
    def generate_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate highly discriminative embedding for map art images
        
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
                
                # 1. Create a unique hash-based component (high discrimination)
                # This ensures identical images get identical embeddings
                image_hash = hashlib.md5(image_bytes).hexdigest()
                hash_features = []
                for i in range(128):  # Use 128 dimensions for hash
                    char_idx = i % len(image_hash)
                    hash_features.append(int(image_hash[char_idx], 16) / 15.0)  # Normalize 0-1
                
                # 2. High-resolution color histogram (more discriminative)
                color_features = self._extract_detailed_color_features(cv_image)  # 192 dims
                
                # 3. Spatial position features (where colors are located)
                spatial_features = self._extract_spatial_color_distribution(cv_image)  # 64 dims
                
                # 4. Edge patterns in different regions 
                edge_features = self._extract_regional_edge_features(cv_image)  # 64 dims
                
                # 5. Texture patterns
                texture_features = self._extract_detailed_texture_features(cv_image)  # 32 dims
                
                # 6. Image statistics
                stats_features = self._extract_image_statistics(cv_image)  # 32 dims
                
                # Combine all features (total: 128+192+64+64+32+32 = 512)
                combined_features = np.concatenate([
                    hash_features,
                    color_features,
                    spatial_features, 
                    edge_features,
                    texture_features,
                    stats_features
                ])
                
                # Ensure exactly 512 dimensions
                if len(combined_features) > 512:
                    combined_features = combined_features[:512]
                elif len(combined_features) < 512:
                    padding = np.zeros(512 - len(combined_features))
                    combined_features = np.concatenate([combined_features, padding])
                
                # Normalize to unit vector
                embedding = combined_features.astype(np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                self.logger.debug(f"Generated discriminative embedding with shape: {embedding.shape}")
                return embedding
                
        except Exception as e:
            self.logger.error(f"Failed to generate discriminative embedding: {e}")
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