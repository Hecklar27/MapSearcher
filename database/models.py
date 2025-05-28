"""
Database Models and Schemas
Data structures for image storage and metadata
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class ImageMetadata:
    """Metadata for stored images"""
    upload_timestamp: datetime
    image_dimensions: Dict[str, int]  # {"width": int, "height": int}
    file_size: int
    channel_id: int
    user_id: int
    filename: Optional[str] = None
    content_type: Optional[str] = None

@dataclass
class ImageDocument:
    """Complete image document for MongoDB storage"""
    discord_message_id: int
    message_link: str
    image_url: str
    clip_embedding: List[float]  # 512-dimensional CLIP vector
    metadata: ImageMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion"""
        doc = asdict(self)
        # Convert metadata to dict
        doc['metadata'] = asdict(self.metadata)
        return doc
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageDocument':
        """Create ImageDocument from MongoDB document"""
        metadata_dict = data.pop('metadata')
        metadata = ImageMetadata(**metadata_dict)
        return cls(metadata=metadata, **data)

@dataclass
class SearchResult:
    """Search result with similarity score"""
    image_document: ImageDocument
    similarity_score: float
    
    @property
    def message_link(self) -> str:
        """Get Discord message link"""
        return self.image_document.message_link
    
    @property
    def image_url(self) -> str:
        """Get image URL"""
        return self.image_document.image_url
    
    @property
    def upload_date(self) -> datetime:
        """Get upload timestamp"""
        return self.image_document.metadata.upload_timestamp

class EmbeddingUtils:
    """Utilities for handling CLIP embeddings"""
    
    @staticmethod
    def numpy_to_list(embedding: np.ndarray) -> List[float]:
        """
        Convert numpy array to list for MongoDB storage
        
        Args:
            embedding: NumPy array of CLIP embedding
            
        Returns:
            List of floats for database storage
        """
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        return embedding.tolist()
    
    @staticmethod
    def list_to_numpy(embedding: List[float]) -> np.ndarray:
        """
        Convert list to numpy array for computation
        
        Args:
            embedding: List of floats from database
            
        Returns:
            NumPy array for similarity computation
        """
        return np.array(embedding, dtype=np.float32)
    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector for cosine similarity
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, float(similarity)))

# MongoDB collection names
IMAGES_COLLECTION = "images"

# Vector search index configuration for MongoDB Atlas
VECTOR_SEARCH_INDEX_CONFIG = {
    "mappings": {
        "dynamic": True,
        "fields": {
            "clip_embedding": {
                "dimensions": 512,
                "similarity": "cosine",
                "type": "knnVector"
            }
        }
    }
} 