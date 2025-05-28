"""
Configuration Management
Environment variable handling and validation
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

class Config:
    """Application configuration from environment variables"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Discord configuration
        self.discord_token = self._get_required_env("DISCORD_TOKEN")
        self.map_channel_id = int(self._get_required_env("MAP_CHANNEL_ID"))
        
        # MongoDB configuration
        self.mongodb_uri = self._get_required_env("MONGODB_URI")
        self.mongodb_database = self._get_env("MONGODB_DATABASE", "mapart_search")
        
        # CLIP model configuration
        self.clip_model = self._get_env("CLIP_MODEL", "ViT-B-32")
        self.device = self._get_env("DEVICE", "cpu")  # Force CPU for now
        
        # Image processing configuration
        self.max_image_size_mb = int(self._get_env("MAX_IMAGE_SIZE_MB", "25"))
        self.max_image_size_bytes = self.max_image_size_mb * 1024 * 1024
        self.processing_timeout_seconds = int(self._get_env("PROCESSING_TIMEOUT_SECONDS", "30"))
        
        # Search configuration
        self.search_results_limit = int(self._get_env("SEARCH_RESULTS_LIMIT", "3"))
        self.similarity_threshold = float(self._get_env("SIMILARITY_THRESHOLD", "0.7"))
        
        # Performance configuration
        self.max_concurrent_downloads = int(self._get_env("MAX_CONCURRENT_DOWNLOADS", "5"))
        self.batch_size = int(self._get_env("BATCH_SIZE", "10"))
        
        # Logging configuration
        self.log_level = self._get_env("LOG_LEVEL", "INFO")
        self.log_file = self._get_env("LOG_FILE", "bot.log")
        
        # Development configuration
        self.debug_mode = self._get_env("DEBUG_MODE", "false").lower() == "true"
        
        # Validate configuration
        self._validate_config()
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _get_env(self, key: str, default: str) -> str:
        """Get environment variable with default value"""
        return os.getenv(key, default)
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate Discord token format - discord.py needs token WITHOUT "Bot " prefix
        if self.discord_token.startswith(('Bot ', 'Bearer ')):
            # Remove Bot/Bearer prefix for discord.py
            self.discord_token = self.discord_token.replace("Bot ", "").replace("Bearer ", "").strip()
        
        # Validate MongoDB URI (more lenient for testing)
        if not self.mongodb_uri.startswith(('mongodb://', 'mongodb+srv://')):
            # For testing, allow placeholder URIs
            if 'username:password@cluster' in self.mongodb_uri:
                logging.warning("Using placeholder MongoDB URI - replace with real credentials")
            else:
                raise ValueError("Invalid MongoDB URI format")
        
        # Validate CLIP model
        valid_clip_models = [
            "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336",
            "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"
        ]
        if self.clip_model not in valid_clip_models:
            logging.warning(f"Unknown CLIP model: {self.clip_model}. Using ViT-B-32")
            self.clip_model = "ViT-B-32"
        
        # Validate device
        if self.device not in ["cuda", "cpu", "auto"]:
            logging.warning(f"Invalid device: {self.device}. Using auto")
            self.device = "auto"
        
        # Validate numeric ranges
        if self.max_image_size_mb <= 0 or self.max_image_size_mb > 100:
            raise ValueError("MAX_IMAGE_SIZE_MB must be between 1 and 100")
        
        if self.processing_timeout_seconds <= 0 or self.processing_timeout_seconds > 300:
            raise ValueError("PROCESSING_TIMEOUT_SECONDS must be between 1 and 300")
        
        if self.search_results_limit <= 0 or self.search_results_limit > 10:
            raise ValueError("SEARCH_RESULTS_LIMIT must be between 1 and 10")
        
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        
        if self.max_concurrent_downloads <= 0 or self.max_concurrent_downloads > 20:
            raise ValueError("MAX_CONCURRENT_DOWNLOADS must be between 1 and 20")
        
        if self.batch_size <= 0 or self.batch_size > 100:
            raise ValueError("BATCH_SIZE must be between 1 and 100")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            logging.warning(f"Invalid log level: {self.log_level}. Using INFO")
            self.log_level = "INFO"
    
    def get_summary(self) -> dict:
        """Get configuration summary (excluding sensitive data)"""
        return {
            "mongodb_database": self.mongodb_database,
            "clip_model": self.clip_model,
            "device": self.device,
            "max_image_size_mb": self.max_image_size_mb,
            "processing_timeout_seconds": self.processing_timeout_seconds,
            "search_results_limit": self.search_results_limit,
            "similarity_threshold": self.similarity_threshold,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "batch_size": self.batch_size,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode
        } 