"""
Search Engine Tests
Unit tests for CLIP processor, image processor, and search engine
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from search.clip_processor import CLIPProcessor
from search.image_processor import ImageProcessor
from search.search_engine import SearchEngine
from database.models import ImageDocument, ImageMetadata, SearchResult
from utils.config import Config
import io
from PIL import Image

class TestImageProcessor:
    """Test image preprocessing pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()
    
    def create_test_image(self, width: int = 256, height: int = 256, format: str = 'JPEG') -> bytes:
        """Create a test image"""
        image = Image.new('RGB', (width, height), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def test_validate_image_valid(self):
        """Test image validation with valid image"""
        image_bytes = self.create_test_image()
        assert self.processor.validate_image(image_bytes) is True
    
    def test_validate_image_too_large(self):
        """Test image validation with oversized image"""
        # Create large image data
        large_data = b'fake_image_data' * 1000000  # ~15MB of fake data
        assert self.processor.validate_image(large_data, max_size_mb=10) is False
    
    def test_get_image_info(self):
        """Test getting image information"""
        image_bytes = self.create_test_image(512, 256)
        info = self.processor.get_image_info(image_bytes)
        
        assert info['width'] == 512
        assert info['height'] == 256
        assert info['format'] == 'JPEG'
        assert info['mode'] == 'RGB'
    
    def test_resize_image(self):
        """Test image resizing"""
        image_bytes = self.create_test_image(2048, 1024)
        resized_bytes = self.processor.resize_image(image_bytes, max_size=(512, 512))
        
        # Check that image was resized
        resized_info = self.processor.get_image_info(resized_bytes)
        assert resized_info['width'] <= 512
        assert resized_info['height'] <= 512
    
    def test_enhance_image_quality(self):
        """Test image quality enhancement"""
        image_bytes = self.create_test_image()
        enhanced_bytes = self.processor.enhance_image_quality(image_bytes)
        
        # Should return valid image bytes
        assert len(enhanced_bytes) > 0
        enhanced_info = self.processor.get_image_info(enhanced_bytes)
        assert 'error' not in enhanced_info

class TestCLIPProcessor:
    """Test CLIP model processor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MagicMock()
        self.config.clip_model = "ViT-B-32"
        self.config.device = "cpu"
        
        self.processor = CLIPProcessor(self.config)
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that CLIPProcessor follows singleton pattern"""
        processor1 = CLIPProcessor()
        processor2 = CLIPProcessor()
        assert processor1 is processor2
    
    def test_not_initialized(self):
        """Test behavior when model not initialized"""
        assert not self.processor.is_initialized()
        
        with pytest.raises(RuntimeError):
            self.processor.generate_image_embedding(b"fake_image_data")
    
    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.get_tokenizer')
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_tokenizer, mock_create_model):
        """Test successful model initialization"""
        # Mock CLIP model components
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = MagicMock()
        
        await self.processor.initialize(self.config)
        
        assert self.processor.is_initialized()
        mock_create_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    def test_get_model_info_not_initialized(self):
        """Test getting model info when not initialized"""
        info = self.processor.get_model_info()
        assert info["status"] == "not_initialized"

@pytest.mark.asyncio
class TestSearchEngine:
    """Test search engine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MagicMock()
        self.config.clip_model = "ViT-B-32"
        self.config.device = "cpu"
        self.config.max_concurrent_downloads = 5
        self.config.search_results_limit = 3
        self.config.max_image_size_bytes = 25 * 1024 * 1024
        self.config.max_image_size_mb = 25
        self.config.processing_timeout_seconds = 30
        
        self.mock_repository = AsyncMock()
        self.search_engine = SearchEngine(self.config, self.mock_repository)
    
    @patch('search.search_engine.CLIPProcessor')
    async def test_initialize_success(self, mock_clip_class):
        """Test successful search engine initialization"""
        mock_clip_instance = AsyncMock()
        mock_clip_instance.get_model_info.return_value = {"status": "initialized"}
        mock_clip_class.return_value = mock_clip_instance
        
        await self.search_engine.initialize()
        
        mock_clip_instance.initialize.assert_called_once_with(self.config)
    
    def create_test_image_bytes(self) -> bytes:
        """Create test image bytes"""
        image = Image.new('RGB', (256, 256), color='blue')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    @patch('search.search_engine.CLIPProcessor')
    async def test_reverse_search(self, mock_clip_class):
        """Test reverse image search"""
        # Mock CLIP processor
        mock_clip_instance = AsyncMock()
        mock_embedding = np.random.rand(512).astype(np.float32)
        mock_clip_instance.generate_image_embedding.return_value = mock_embedding
        mock_clip_class.return_value = mock_clip_instance
        
        # Mock search results
        mock_results = [
            SearchResult(
                image_document=ImageDocument(
                    discord_message_id=123,
                    message_link="https://discord.com/channels/1/2/123",
                    image_url="https://example.com/image.jpg",
                    clip_embedding=[0.1] * 512,
                    metadata=ImageMetadata(
                        upload_timestamp=datetime.now(),
                        image_dimensions={"width": 256, "height": 256},
                        file_size=1024,
                        channel_id=2,
                        user_id=456
                    )
                ),
                similarity_score=0.95
            )
        ]
        self.mock_repository.vector_search.return_value = mock_results
        
        # Test search
        image_bytes = self.create_test_image_bytes()
        results = await self.search_engine.reverse_search(image_bytes)
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
        self.mock_repository.vector_search.assert_called_once()
    
    @patch('utils.helpers.download_image_bytes')
    async def test_index_discord_message_success(self, mock_download):
        """Test successful message indexing"""
        # Mock image download
        image_bytes = self.create_test_image_bytes()
        mock_download.return_value = image_bytes
        
        # Mock repository
        self.mock_repository.get_image_by_message_id.return_value = None
        self.mock_repository.store_image.return_value = True
        
        # Mock CLIP processor
        with patch.object(self.search_engine.clip_processor, 'generate_image_embedding') as mock_embedding:
            mock_embedding.return_value = np.random.rand(512).astype(np.float32)
            
            result = await self.search_engine.index_discord_message(
                message_id=123,
                channel_id=456,
                user_id=789,
                image_url="https://example.com/image.jpg",
                guild_id=111
            )
        
        assert result is True
        mock_download.assert_called_once()
        self.mock_repository.store_image.assert_called_once()
    
    async def test_index_discord_message_already_exists(self):
        """Test indexing message that already exists"""
        # Mock existing document
        existing_doc = ImageDocument(
            discord_message_id=123,
            message_link="https://discord.com/channels/1/2/123",
            image_url="https://example.com/image.jpg",
            clip_embedding=[0.1] * 512,
            metadata=ImageMetadata(
                upload_timestamp=datetime.now(),
                image_dimensions={"width": 256, "height": 256},
                file_size=1024,
                channel_id=2,
                user_id=456
            )
        )
        self.mock_repository.get_image_by_message_id.return_value = existing_doc
        
        result = await self.search_engine.index_discord_message(
            message_id=123,
            channel_id=456,
            user_id=789,
            image_url="https://example.com/image.jpg",
            guild_id=111
        )
        
        assert result is False
    
    async def test_get_search_statistics(self):
        """Test getting search statistics"""
        # Mock repository responses
        self.mock_repository.get_image_count.return_value = 42
        self.mock_repository.get_recent_images.return_value = []
        
        # Mock CLIP processor
        with patch.object(self.search_engine.clip_processor, 'get_model_info') as mock_info:
            mock_info.return_value = {"status": "initialized", "device": "cpu"}
            
            stats = await self.search_engine.get_search_statistics()
        
        assert stats["total_indexed_images"] == 42
        assert "clip_model_info" in stats
    
    async def test_batch_index_messages(self):
        """Test batch message indexing"""
        message_data = [
            {
                'message_id': 123,
                'channel_id': 456,
                'user_id': 789,
                'image_url': 'https://example.com/image1.jpg',
                'guild_id': 111
            },
            {
                'message_id': 124,
                'channel_id': 456,
                'user_id': 789,
                'image_url': 'https://example.com/image2.jpg',
                'guild_id': 111
            }
        ]
        
        # Mock successful indexing
        with patch.object(self.search_engine, 'index_discord_message') as mock_index:
            mock_index.return_value = True
            
            stats = await self.search_engine.batch_index_messages(message_data)
        
        assert stats["total"] == 2
        assert stats["successful"] == 2
        assert stats["failed"] == 0
        assert stats["errors"] == 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 