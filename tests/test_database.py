"""
Database Layer Tests
Unit tests for MongoDB operations and models
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from database.models import ImageDocument, ImageMetadata, EmbeddingUtils
from database.repository import ImageRepository
import numpy as np

class TestImageModels:
    """Test database models and utilities"""
    
    def test_image_metadata_creation(self):
        """Test ImageMetadata creation"""
        metadata = ImageMetadata(
            upload_timestamp=datetime.now(),
            image_dimensions={"width": 1920, "height": 1080},
            file_size=1024000,
            channel_id=123456789,
            user_id=987654321,
            filename="test.png",
            content_type="image/png"
        )
        
        assert metadata.channel_id == 123456789
        assert metadata.user_id == 987654321
        assert metadata.filename == "test.png"
    
    def test_image_document_serialization(self):
        """Test ImageDocument to/from dict conversion"""
        metadata = ImageMetadata(
            upload_timestamp=datetime.now(),
            image_dimensions={"width": 1920, "height": 1080},
            file_size=1024000,
            channel_id=123456789,
            user_id=987654321
        )
        
        doc = ImageDocument(
            discord_message_id=555666777,
            message_link="https://discord.com/channels/123/456/555666777",
            image_url="https://cdn.discordapp.com/attachments/123/456/test.png",
            clip_embedding=[0.1, 0.2, 0.3] * 170 + [0.1, 0.2],  # 512 dimensions
            metadata=metadata
        )
        
        # Test to_dict
        doc_dict = doc.to_dict()
        assert doc_dict["discord_message_id"] == 555666777
        assert "metadata" in doc_dict
        assert doc_dict["metadata"]["channel_id"] == 123456789
        
        # Test from_dict
        reconstructed = ImageDocument.from_dict(doc_dict)
        assert reconstructed.discord_message_id == doc.discord_message_id
        assert reconstructed.metadata.channel_id == metadata.channel_id

class TestEmbeddingUtils:
    """Test embedding utility functions"""
    
    def test_numpy_list_conversion(self):
        """Test numpy array to list conversion"""
        # Create test embedding
        embedding = np.random.rand(512).astype(np.float32)
        
        # Convert to list
        embedding_list = EmbeddingUtils.numpy_to_list(embedding)
        assert len(embedding_list) == 512
        assert isinstance(embedding_list[0], float)
        
        # Convert back to numpy
        embedding_np = EmbeddingUtils.list_to_numpy(embedding_list)
        assert embedding_np.shape == (512,)
        assert embedding_np.dtype == np.float32
        
        # Check values are preserved
        np.testing.assert_array_almost_equal(embedding, embedding_np)
    
    def test_normalize_embedding(self):
        """Test embedding normalization"""
        # Test normal vector
        embedding = np.array([3.0, 4.0], dtype=np.float32)
        normalized = EmbeddingUtils.normalize_embedding(embedding)
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected)
        
        # Test zero vector
        zero_embedding = np.zeros(5, dtype=np.float32)
        normalized_zero = EmbeddingUtils.normalize_embedding(zero_embedding)
        np.testing.assert_array_equal(normalized_zero, zero_embedding)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        similarity = EmbeddingUtils.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec4 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = EmbeddingUtils.cosine_similarity(vec3, vec4)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test opposite vectors
        vec5 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec6 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        similarity = EmbeddingUtils.cosine_similarity(vec5, vec6)
        assert abs(similarity - 0.0) < 1e-6  # Clamped to 0

@pytest.mark.asyncio
class TestImageRepository:
    """Test image repository operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_database = MagicMock()
        self.mock_collection = AsyncMock()
        self.mock_database.__getitem__.return_value = self.mock_collection
        self.repository = ImageRepository(self.mock_database)
    
    async def test_store_image_success(self):
        """Test successful image storage"""
        # Mock successful insertion
        self.mock_collection.insert_one = AsyncMock()
        
        # Create test image document
        metadata = ImageMetadata(
            upload_timestamp=datetime.now(),
            image_dimensions={"width": 1920, "height": 1080},
            file_size=1024000,
            channel_id=123456789,
            user_id=987654321
        )
        
        doc = ImageDocument(
            discord_message_id=555666777,
            message_link="https://discord.com/channels/123/456/555666777",
            image_url="https://cdn.discordapp.com/attachments/123/456/test.png",
            clip_embedding=[0.1] * 512,
            metadata=metadata
        )
        
        # Test storage
        result = await self.repository.store_image(doc)
        assert result is True
        self.mock_collection.insert_one.assert_called_once()
    
    async def test_get_image_count(self):
        """Test getting image count"""
        # Mock count result
        self.mock_collection.count_documents = AsyncMock(return_value=42)
        
        count = await self.repository.get_image_count()
        assert count == 42
        self.mock_collection.count_documents.assert_called_once_with({})

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 