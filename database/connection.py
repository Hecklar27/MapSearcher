"""
MongoDB Connection Management
Async connection handling with Motor driver
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from utils.config import Config

class DatabaseConnection:
    """MongoDB connection manager with async Motor driver"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Connecting to MongoDB Atlas...")
            
            # Create Motor client with connection options
            self._client = AsyncIOMotorClient(
                self.config.mongodb_uri,
                maxPoolSize=20,  # Maximum connections in pool
                minPoolSize=5,   # Minimum connections in pool
                maxIdleTimeMS=30000,  # Close connections after 30s idle
                serverSelectionTimeoutMS=5000,  # 5s timeout for server selection
                connectTimeoutMS=10000,  # 10s timeout for initial connection
                socketTimeoutMS=20000,   # 20s timeout for socket operations
            )
            
            # Get database reference
            self._database = self._client[self.config.mongodb_database]
            
            # Test connection
            await self._client.admin.command('ping')
            
            self.logger.info(f"Successfully connected to MongoDB database: {self.config.mongodb_database}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self._client:
            self.logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
            self._database = None
    
    @property
    def database(self) -> AsyncIOMotorDatabase:
        """
        Get database instance
        
        Returns:
            MongoDB database instance
            
        Raises:
            RuntimeError: If not connected to database
        """
        if self._database is None:
            raise RuntimeError("Not connected to database. Call connect() first.")
        return self._database
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to database"""
        return self._client is not None and self._database is not None
    
    async def health_check(self) -> bool:
        """
        Perform database health check
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            if not self.is_connected:
                return False
            
            # Ping database
            await self._client.admin.command('ping')
            return True
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    async def create_indexes(self):
        """Create necessary database indexes"""
        try:
            if not self.is_connected:
                raise RuntimeError("Not connected to database")
            
            images_collection = self._database.images
            
            # Create index on discord_message_id for fast lookups
            await images_collection.create_index("discord_message_id", unique=True)
            
            # Create index on channel_id for filtering
            await images_collection.create_index("metadata.channel_id")
            
            # Create index on upload_timestamp for sorting
            await images_collection.create_index("metadata.upload_timestamp")
            
            self.logger.info("Database indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create database indexes: {e}")
            raise 