"""
CLIP Model Processor - Fallback Version
Simple fallback implementation without PyTorch dependencies
"""

import logging
import numpy as np
from typing import Optional, List, Union
from PIL import Image
import io
from utils.config import Config
from utils.logging import PerformanceLogger

class CLIPProcessor:
    """Fallback CLIP model processor using dummy embeddings"""
    
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
        self._fallback_mode = True
        
        print("🔄 Running CLIP processor in fallback mode (no PyTorch)")
    
    async def initialize(self, config: Config):
        """
        Initialize CLIP model (fallback mode)
        
        Args:
            config: Application configuration
        """
        self.logger.warning("Running in fallback mode - using dummy embeddings")
        self.config = config
    
    def is_initialized(self) -> bool:
        """Check if CLIP model is initialized"""
        return True
    
    def _generate_dummy_embedding(self, seed: int = None) -> np.ndarray:
        """Generate a dummy embedding for fallback mode"""
        if seed is not None:
            np.random.seed(seed)
        # Generate normalized random vector
        embedding = np.random.rand(512).astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def generate_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate dummy embedding for image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            512-dimensional embedding vector
        """
        try:
            with PerformanceLogger("Dummy image embedding", self.logger):
                # Generate deterministic dummy embedding based on image hash
                image_hash = hash(image_bytes) % 1000000
                embedding = self._generate_dummy_embedding(seed=image_hash)
                
                self.logger.debug(f"Generated dummy embedding with shape: {embedding.shape}")
                return embedding
                
        except Exception as e:
            self.logger.error(f"Failed to generate dummy embedding: {e}")
            raise ValueError(f"Image processing failed: {e}")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate dummy embedding for text
        
        Args:
            text: Input text
            
        Returns:
            512-dimensional embedding vector
        """
        try:
            with PerformanceLogger("Dummy text embedding", self.logger):
                # Generate deterministic dummy embedding based on text hash
                text_hash = hash(text) % 1000000
                embedding = self._generate_dummy_embedding(seed=text_hash)
                
                return embedding
                
        except Exception as e:
            self.logger.error(f"Failed to generate dummy text embedding: {e}")
            raise ValueError(f"Text processing failed: {e}")
    
    def batch_generate_embeddings(self, image_bytes_list: List[bytes]) -> List[np.ndarray]:
        """
        Generate dummy embeddings for multiple images
        
        Args:
            image_bytes_list: List of raw image bytes
            
        Returns:
            List of 512-dimensional embedding vectors
        """
        if not image_bytes_list:
            return []
        
        try:
            with PerformanceLogger(f"Dummy batch embedding ({len(image_bytes_list)} images)", self.logger):
                results = []
                for image_bytes in image_bytes_list:
                    if image_bytes:
                        image_hash = hash(image_bytes) % 1000000
                        results.append(self._generate_dummy_embedding(seed=image_hash))
                    else:
                        results.append(None)
                
                self.logger.info(f"Generated {len([r for r in results if r is not None])} dummy embeddings")
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to generate dummy batch embeddings: {e}")
            raise ValueError(f"Batch processing failed: {e}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the fallback model
        
        Returns:
            Dictionary with model information
        """
        return {
            "status": "fallback_mode",
            "model_name": "dummy_embeddings",
            "device": "cpu",
            "embedding_dim": 512,
            "cuda_available": False,
            "gpu_memory_allocated": 0,
            "gpu_memory_cached": 0,
            "note": "Using dummy embeddings - PyTorch/CLIP not available"
        }
    
    def cleanup(self):
        """Clean up resources (no-op in fallback mode)"""
        self.logger.info("Cleanup completed (fallback mode)") 