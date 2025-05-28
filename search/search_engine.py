"""
Search Engine
Combines image processing, CLIP embeddings, and database operations
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from search.clip_processor import CLIPProcessor
from search.image_processor import ImageProcessor
from database.repository import ImageRepository
from database.models import ImageDocument, ImageMetadata, SearchResult, EmbeddingUtils
from utils.config import Config
from utils.helpers import download_image_bytes, format_discord_message_link
from utils.logging import PerformanceLogger

class SearchEngine:
    """Main search engine for map art reverse search"""
    
    def __init__(self, config: Config, image_repository: ImageRepository):
        self.config = config
        self.image_repository = image_repository
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.clip_processor = CLIPProcessor(config)
        self.image_processor = ImageProcessor()
        
        # Processing semaphore to limit concurrent operations
        self.processing_semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
    
    async def initialize(self):
        """Initialize the search engine components"""
        try:
            self.logger.info("Initializing search engine...")
            
            # Initialize CLIP model
            await self.clip_processor.initialize(self.config)
            
            # Log model info
            model_info = self.clip_processor.get_model_info()
            self.logger.info(f"Search engine initialized: {model_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize search engine: {e}")
            raise
    
    async def reverse_search(self, image_bytes: bytes) -> List[SearchResult]:
        """
        Perform reverse image search
        
        Args:
            image_bytes: Query image bytes
            
        Returns:
            List of search results with similarity scores
        """
        try:
            with PerformanceLogger("Reverse image search", self.logger):
                # Preprocess image
                processed_bytes = self.image_processor.preprocess_for_clip(image_bytes)
                
                # Generate embedding
                embedding = self.clip_processor.generate_image_embedding(processed_bytes)
                
                # Convert to list for database search
                embedding_list = EmbeddingUtils.numpy_to_list(embedding)
                
                # Search database
                results = await self.image_repository.vector_search(
                    embedding_list, 
                    limit=self.config.search_results_limit
                )
                
                self.logger.info(f"Found {len(results)} similar images")
                return results
                
        except Exception as e:
            self.logger.error(f"Reverse search failed: {e}")
            raise
    
    async def index_discord_message(self, message_id: int, channel_id: int, user_id: int, 
                                  image_url: str, guild_id: int) -> bool:
        """
        Index a Discord message with image attachment
        
        Args:
            message_id: Discord message ID
            channel_id: Discord channel ID
            user_id: Discord user ID
            image_url: URL of the image attachment
            guild_id: Discord guild ID
            
        Returns:
            True if indexed successfully, False otherwise
        """
        async with self.processing_semaphore:
            try:
                with PerformanceLogger(f"Index message {message_id}", self.logger):
                    # Check if already indexed
                    existing = await self.image_repository.get_image_by_message_id(message_id)
                    if existing:
                        self.logger.debug(f"Message {message_id} already indexed")
                        return False
                    
                    # Download image
                    image_bytes = await download_image_bytes(
                        image_url, 
                        self.config.max_image_size_bytes,
                        timeout=self.config.processing_timeout_seconds
                    )
                    
                    if not image_bytes:
                        self.logger.warning(f"Failed to download image from {image_url}")
                        return False
                    
                    # Validate and preprocess image
                    if not self.image_processor.validate_image(image_bytes, self.config.max_image_size_mb):
                        self.logger.warning(f"Image validation failed for {image_url}")
                        return False
                    
                    processed_bytes = self.image_processor.preprocess_for_clip(image_bytes)
                    
                    # Generate embedding
                    embedding = self.clip_processor.generate_image_embedding(processed_bytes)
                    embedding_list = EmbeddingUtils.numpy_to_list(embedding)
                    
                    # Get image info
                    image_info = self.image_processor.get_image_info(image_bytes)
                    
                    # Create metadata
                    metadata = ImageMetadata(
                        upload_timestamp=datetime.now(),
                        image_dimensions={
                            "width": image_info.get("width", 0),
                            "height": image_info.get("height", 0)
                        },
                        file_size=len(image_bytes),
                        channel_id=channel_id,
                        user_id=user_id,
                        filename=image_url.split('/')[-1] if '/' in image_url else None,
                        content_type=f"image/{image_info.get('format', 'unknown').lower()}"
                    )
                    
                    # Create document
                    document = ImageDocument(
                        discord_message_id=message_id,
                        message_link=format_discord_message_link(guild_id, channel_id, message_id),
                        image_url=image_url,
                        clip_embedding=embedding_list,
                        metadata=metadata
                    )
                    
                    # Store in database
                    success = await self.image_repository.store_image(document)
                    
                    if success:
                        self.logger.info(f"Successfully indexed message {message_id}")
                    
                    return success
                    
            except Exception as e:
                self.logger.error(f"Failed to index message {message_id}: {e}")
                return False
    
    async def batch_index_messages(self, message_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index multiple Discord messages in batch
        
        Args:
            message_data: List of message data dictionaries
            
        Returns:
            Dictionary with indexing statistics
        """
        try:
            with PerformanceLogger(f"Batch index {len(message_data)} messages", self.logger):
                # Process messages concurrently
                tasks = []
                for data in message_data:
                    task = self.index_discord_message(
                        data['message_id'],
                        data['channel_id'],
                        data['user_id'],
                        data['image_url'],
                        data['guild_id']
                    )
                    tasks.append(task)
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count results
                successful = sum(1 for r in results if r is True)
                failed = sum(1 for r in results if r is False)
                errors = sum(1 for r in results if isinstance(r, Exception))
                
                stats = {
                    "total": len(message_data),
                    "successful": successful,
                    "failed": failed,
                    "errors": errors
                }
                
                self.logger.info(f"Batch indexing completed: {stats}")
                return stats
                
        except Exception as e:
            self.logger.error(f"Batch indexing failed: {e}")
            return {"total": len(message_data), "successful": 0, "failed": 0, "errors": len(message_data)}
    
    async def search_by_text(self, text_query: str) -> List[SearchResult]:
        """
        Search for images using text description
        
        Args:
            text_query: Text description to search for
            
        Returns:
            List of search results
        """
        try:
            with PerformanceLogger("Text-based search", self.logger):
                # Generate text embedding
                embedding = self.clip_processor.generate_text_embedding(text_query)
                embedding_list = EmbeddingUtils.numpy_to_list(embedding)
                
                # Search database
                results = await self.image_repository.vector_search(
                    embedding_list,
                    limit=self.config.search_results_limit
                )
                
                self.logger.info(f"Text search for '{text_query}' found {len(results)} results")
                return results
                
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            raise
    
    async def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            total_images = await self.image_repository.get_image_count()
            recent_images = await self.image_repository.get_recent_images(limit=5)
            model_info = self.clip_processor.get_model_info()
            
            return {
                "total_indexed_images": total_images,
                "recent_images_count": len(recent_images),
                "clip_model_info": model_info,
                "last_indexed": recent_images[0].metadata.upload_timestamp if recent_images else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    async def reindex_channel(self, channel_id: int, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Reindex all images from a specific channel
        
        Args:
            channel_id: Discord channel ID to reindex
            limit: Maximum number of images to process
            
        Returns:
            Reindexing statistics
        """
        try:
            self.logger.info(f"Starting reindex of channel {channel_id}")
            
            # Get existing images from channel
            existing_images = await self.image_repository.get_images_by_channel(
                channel_id, 
                limit=limit or 1000
            )
            
            # This would need Discord API integration to fetch message history
            # For now, return statistics of existing images
            stats = {
                "channel_id": channel_id,
                "existing_images": len(existing_images),
                "reindexed": 0,
                "errors": 0
            }
            
            self.logger.info(f"Channel reindex completed: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Channel reindex failed: {e}")
            return {"channel_id": channel_id, "existing_images": 0, "reindexed": 0, "errors": 1}
    
    def cleanup(self):
        """Clean up search engine resources"""
        self.logger.info("Cleaning up search engine...")
        if self.clip_processor:
            self.clip_processor.cleanup() 