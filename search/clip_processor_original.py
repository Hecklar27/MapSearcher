"""
CLIP Model Processor
Singleton pattern for efficient CLIP model loading and embedding generation
"""

import logging
import numpy as np
from typing import Optional, List, Union
from PIL import Image
import io
from utils.config import Config
from utils.logging import PerformanceLogger

# Try to import torch and open_clip, but handle gracefully if not available
try:
    import torch
    import open_clip
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    open_clip = None
    print(f"⚠️  PyTorch/CLIP not available: {e}")
    print("🔄 Running in fallback mode with dummy embeddings")

class CLIPProcessor:
    """Singleton CLIP model processor for image embeddings"""
    
    _instance: Optional['CLIPProcessor'] = None
    _model = None
    _preprocess = None
    _tokenizer = None
    _device = None
    
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
        self._fallback_mode = not TORCH_AVAILABLE
    
    async def initialize(self, config: Config):
        """
        Initialize CLIP model (async to avoid blocking)
        
        Args:
            config: Application configuration
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available - running in fallback mode")
            self._fallback_mode = True
            return
        
        if self._model is not None:
            self.logger.info("CLIP model already initialized")
            return
        
        try:
            with PerformanceLogger("CLIP model initialization", self.logger):
                self.logger.info(f"Loading CLIP model: {config.clip_model}")
                
                # Set device
                self._device = torch.device('cpu')  # Force CPU for now
                self.logger.info(f"Using device: {self._device}")
                
                # Load CLIP model
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    config.clip_model,
                    pretrained='openai',
                    device=self._device
                )
                
                # Set model to evaluation mode
                self._model.eval()
                
                # Get tokenizer
                self._tokenizer = open_clip.get_tokenizer(config.clip_model)
                
                self.logger.info("CLIP model initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize CLIP model: {e}")
            self.logger.warning("Falling back to dummy embeddings")
            self._fallback_mode = True
    
    def is_initialized(self) -> bool:
        """Check if CLIP model is initialized"""
        return self._model is not None or self._fallback_mode
    
    def _generate_dummy_embedding(self, seed: int = None) -> np.ndarray:
        """Generate a dummy embedding for fallback mode"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.rand(512).astype(np.float32)
    
    def generate_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        Generate CLIP embedding for image
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            512-dimensional embedding vector
            
        Raises:
            RuntimeError: If model not initialized
            ValueError: If image processing fails
        """
        if not self.is_initialized():
            raise RuntimeError("CLIP model not initialized. Call initialize() first.")
        
        if self._fallback_mode:
            # Generate deterministic dummy embedding based on image hash
            image_hash = hash(image_bytes) % 1000000
            return self._generate_dummy_embedding(seed=image_hash)
        
        try:
            with PerformanceLogger("CLIP image embedding", self.logger):
                # Load image from bytes
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess image
                image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
                
                # Generate embedding
                with torch.no_grad():
                    image_features = self._model.encode_image(image_tensor)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy
                    embedding = image_features.cpu().numpy().flatten()
                
                self.logger.debug(f"Generated embedding with shape: {embedding.shape}")
                return embedding.astype(np.float32)
                
        except Exception as e:
            self.logger.error(f"Failed to generate image embedding: {e}")
            raise ValueError(f"Image processing failed: {e}")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text
        
        Args:
            text: Input text
            
        Returns:
            512-dimensional embedding vector
            
        Raises:
            RuntimeError: If model not initialized
        """
        if not self.is_initialized():
            raise RuntimeError("CLIP model not initialized. Call initialize() first.")
        
        if self._fallback_mode:
            # Generate deterministic dummy embedding based on text hash
            text_hash = hash(text) % 1000000
            return self._generate_dummy_embedding(seed=text_hash)
        
        try:
            with PerformanceLogger("CLIP text embedding", self.logger):
                # Tokenize text
                text_tokens = self._tokenizer([text]).to(self._device)
                
                # Generate embedding
                with torch.no_grad():
                    text_features = self._model.encode_text(text_tokens)
                    
                    # Normalize features
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy
                    embedding = text_features.cpu().numpy().flatten()
                
                return embedding.astype(np.float32)
                
        except Exception as e:
            self.logger.error(f"Failed to generate text embedding: {e}")
            raise ValueError(f"Text processing failed: {e}")
    
    def batch_generate_embeddings(self, image_bytes_list: List[bytes]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images in batch
        
        Args:
            image_bytes_list: List of raw image bytes
            
        Returns:
            List of 512-dimensional embedding vectors
            
        Raises:
            RuntimeError: If model not initialized
        """
        if not self.is_initialized():
            raise RuntimeError("CLIP model not initialized. Call initialize() first.")
        
        if not image_bytes_list:
            return []
        
        if self._fallback_mode:
            # Generate dummy embeddings for each image
            results = []
            for image_bytes in image_bytes_list:
                if image_bytes:
                    image_hash = hash(image_bytes) % 1000000
                    results.append(self._generate_dummy_embedding(seed=image_hash))
                else:
                    results.append(None)
            return results
        
        try:
            with PerformanceLogger(f"CLIP batch embedding ({len(image_bytes_list)} images)", self.logger):
                images = []
                
                # Load and preprocess all images
                for image_bytes in image_bytes_list:
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        images.append(self._preprocess(image))
                    except Exception as e:
                        self.logger.warning(f"Failed to process image in batch: {e}")
                        # Add None placeholder for failed images
                        images.append(None)
                
                # Filter out failed images and create batch tensor
                valid_images = [img for img in images if img is not None]
                if not valid_images:
                    return []
                
                batch_tensor = torch.stack(valid_images).to(self._device)
                
                # Generate embeddings
                with torch.no_grad():
                    image_features = self._model.encode_image(batch_tensor)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy
                    embeddings = image_features.cpu().numpy()
                
                # Create result list with None for failed images
                results = []
                valid_idx = 0
                
                for img in images:
                    if img is not None:
                        results.append(embeddings[valid_idx].astype(np.float32))
                        valid_idx += 1
                    else:
                        results.append(None)
                
                self.logger.info(f"Generated {valid_idx} embeddings from {len(image_bytes_list)} images")
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            raise ValueError(f"Batch processing failed: {e}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self._fallback_mode:
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
        
        if not self.is_initialized():
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.config.clip_model if self.config else "unknown",
            "device": str(self._device),
            "embedding_dim": 512,
            "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "gpu_memory_allocated": torch.cuda.memory_allocated() if TORCH_AVAILABLE and torch.cuda.is_available() else 0,
            "gpu_memory_cached": torch.cuda.memory_reserved() if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        }
    
    def cleanup(self):
        """Clean up GPU memory"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared") 