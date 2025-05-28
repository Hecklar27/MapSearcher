"""
Image Repository
Database operations for image storage and vector search
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
from database.models import (
    ImageDocument, ImageMetadata, SearchResult, 
    EmbeddingUtils, IMAGES_COLLECTION
)
from utils.logging import PerformanceLogger

class ImageRepository:
    """Repository for image database operations"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.database = database
        self.collection = database[IMAGES_COLLECTION]
        self.logger = logging.getLogger(__name__)
    
    async def store_image(self, image_doc: ImageDocument) -> bool:
        """
        Store image document in database
        
        Args:
            image_doc: Image document to store
            
        Returns:
            True if stored successfully, False if already exists
        """
        try:
            with PerformanceLogger("store_image", self.logger):
                doc_dict = image_doc.to_dict()
                await self.collection.insert_one(doc_dict)
                
                self.logger.info(f"Stored image document for message {image_doc.discord_message_id}")
                return True
                
        except DuplicateKeyError:
            self.logger.warning(f"Image already exists for message {image_doc.discord_message_id}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to store image document: {e}")
            raise
    
    async def get_image_by_message_id(self, message_id: int) -> Optional[ImageDocument]:
        """
        Retrieve image document by Discord message ID
        
        Args:
            message_id: Discord message ID
            
        Returns:
            Image document or None if not found
        """
        try:
            doc = await self.collection.find_one({"discord_message_id": message_id})
            if doc:
                # Remove MongoDB _id field
                doc.pop('_id', None)
                return ImageDocument.from_dict(doc)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve image by message ID {message_id}: {e}")
            raise
    
    async def vector_search(self, query_embedding: List[float], limit: int = 3) -> List[SearchResult]:
        """
        Perform vector similarity search using MongoDB Atlas Vector Search
        
        Args:
            query_embedding: CLIP embedding to search for
            limit: Maximum number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        try:
            with PerformanceLogger("vector_search", self.logger):
                # MongoDB Atlas Vector Search aggregation pipeline
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_search",  # Updated to match actual index name
                            "path": "clip_embedding",
                            "queryVector": query_embedding,
                            "numCandidates": limit * 10,  # Search more candidates for better results
                            "limit": limit
                        }
                    },
                    {
                        "$addFields": {
                            "similarity_score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                results = []
                async for doc in self.collection.aggregate(pipeline):
                    # Remove MongoDB _id field
                    similarity_score = doc.pop('similarity_score')
                    doc.pop('_id', None)
                    
                    # Create ImageDocument and SearchResult
                    image_doc = ImageDocument.from_dict(doc)
                    search_result = SearchResult(
                        image_document=image_doc,
                        similarity_score=similarity_score
                    )
                    results.append(search_result)
                
                self.logger.info(f"Vector search returned {len(results)} results")
                
                # If vector search returns no results but we have images in database,
                # fallback to manual similarity calculation
                if len(results) == 0:
                    total_images = await self.get_image_count()
                    if total_images > 0:
                        self.logger.warning(f"Vector search returned 0 results but database has {total_images} images. Falling back to manual similarity.")
                        return await self._fallback_similarity_search(query_embedding, limit)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            # Fallback to manual similarity calculation if vector search fails
            return await self._fallback_similarity_search(query_embedding, limit)
    
    async def _fallback_similarity_search(self, query_embedding: List[float], limit: int) -> List[SearchResult]:
        """
        Fallback similarity search using manual cosine similarity calculation
        
        Args:
            query_embedding: CLIP embedding to search for
            limit: Maximum number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        try:
            self.logger.warning("Using fallback similarity search - vector search index not available")
            
            # Get all documents (for better accuracy, process all images, not just first 1000)
            cursor = self.collection.find({})
            
            results = []
            query_np = EmbeddingUtils.list_to_numpy(query_embedding)
            
            async for doc in cursor:
                doc.pop('_id', None)
                image_doc = ImageDocument.from_dict(doc)
                
                # Calculate similarity
                stored_np = EmbeddingUtils.list_to_numpy(image_doc.clip_embedding)
                similarity = EmbeddingUtils.cosine_similarity(query_np, stored_np)
                
                results.append(SearchResult(
                    image_document=image_doc,
                    similarity_score=similarity
                ))
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Only return results above a certain threshold for better quality
            min_threshold = 0.3  # Only return results with at least 30% similarity
            filtered_results = [r for r in results if r.similarity_score >= min_threshold]
            
            # If no results above threshold, return top results anyway (but limit to reasonable similarity)
            if not filtered_results and results:
                # Return top results but log that they may be poor matches
                self.logger.warning(f"No high-quality matches found. Best similarity: {results[0].similarity_score:.3f}")
                filtered_results = results[:limit]
            
            final_results = filtered_results[:limit]
            
            if final_results:
                best_score = final_results[0].similarity_score
                self.logger.info(f"Fallback search: found {len(final_results)} results, best similarity: {best_score:.3f}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Fallback similarity search failed: {e}")
            return []
    
    async def get_recent_images(self, limit: int = 10) -> List[ImageDocument]:
        """
        Get recently uploaded images
        
        Args:
            limit: Maximum number of images to return
            
        Returns:
            List of recent image documents
        """
        try:
            cursor = self.collection.find({}).sort("metadata.upload_timestamp", -1).limit(limit)
            
            images = []
            async for doc in cursor:
                doc.pop('_id', None)
                images.append(ImageDocument.from_dict(doc))
            
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to get recent images: {e}")
            return []
    
    async def get_image_count(self) -> int:
        """
        Get total number of images in database
        
        Returns:
            Total image count
        """
        try:
            count = await self.collection.count_documents({})
            return count
        except Exception as e:
            self.logger.error(f"Failed to get image count: {e}")
            return 0
    
    async def delete_image(self, message_id: int) -> bool:
        """
        Delete image by Discord message ID
        
        Args:
            message_id: Discord message ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            result = await self.collection.delete_one({"discord_message_id": message_id})
            
            if result.deleted_count > 0:
                self.logger.info(f"Deleted image for message {message_id}")
                return True
            else:
                self.logger.warning(f"No image found for message {message_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete image for message {message_id}: {e}")
            raise
    
    async def get_images_by_channel(self, channel_id: int, limit: int = 100) -> List[ImageDocument]:
        """
        Get images from specific channel
        
        Args:
            channel_id: Discord channel ID
            limit: Maximum number of images to return
            
        Returns:
            List of image documents from the channel
        """
        try:
            cursor = self.collection.find(
                {"metadata.channel_id": channel_id}
            ).sort("metadata.upload_timestamp", -1).limit(limit)
            
            images = []
            async for doc in cursor:
                doc.pop('_id', None)
                images.append(ImageDocument.from_dict(doc))
            
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to get images from channel {channel_id}: {e}")
            return []

class MockImageRepository:
    """Mock repository for demo mode when database is not available"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mock_images = []  # In-memory storage for demo
        self.logger.info("MockImageRepository initialized for demo mode")
    
    async def store_image(self, image_document: ImageDocument) -> bool:
        """Store image in mock repository"""
        try:
            # Add to mock storage
            self.mock_images.append(image_document)
            self.logger.info(f"Stored image {image_document.discord_message_id} in mock repository")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store image in mock repository: {e}")
            return False
    
    async def get_image_by_message_id(self, message_id: int) -> Optional[ImageDocument]:
        """Get image by message ID from mock repository"""
        for img in self.mock_images:
            if img.discord_message_id == message_id:
                return img
        return None
    
    async def vector_search(self, embedding: List[float], limit: int = 10) -> List[SearchResult]:
        """Perform mock vector search"""
        import random
        
        # Create mock results with random similarity scores
        results = []
        for i, img in enumerate(self.mock_images[:limit]):
            # Generate random but consistent similarity based on message ID
            random.seed(img.discord_message_id)
            similarity = random.uniform(0.6, 0.95)
            
            result = SearchResult(
                image_document=img,
                similarity_score=similarity
            )
            results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        self.logger.info(f"Mock vector search returned {len(results)} results")
        return results
    
    async def get_image_count(self) -> int:
        """Get total number of images in mock repository"""
        return len(self.mock_images)
    
    async def get_recent_images(self, limit: int = 10) -> List[ImageDocument]:
        """Get recent images from mock repository"""
        # Return most recently added images
        return self.mock_images[-limit:] if self.mock_images else []
    
    async def get_images_by_channel(self, channel_id: int, limit: int = 100) -> List[ImageDocument]:
        """Get images by channel ID from mock repository"""
        channel_images = [img for img in self.mock_images if img.metadata.channel_id == channel_id]
        return channel_images[:limit]
    
    async def delete_image(self, message_id: int) -> bool:
        """Delete image from mock repository"""
        for i, img in enumerate(self.mock_images):
            if img.discord_message_id == message_id:
                del self.mock_images[i]
                self.logger.info(f"Deleted image {message_id} from mock repository")
                return True
        return False 