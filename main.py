#!/usr/bin/env python3
"""
Discord Map Art Reverse Search Bot
Main application entry point
"""

import asyncio
import logging
from bot.main import MapArtBot
from utils.config import Config
from utils.logging import setup_logging

async def main():
    """Main application entry point"""
    # Load configuration
    config = Config()
    
    # Setup logging
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Discord Map Art Reverse Search Bot...")
    
    try:
        # Initialize and run the bot
        bot = MapArtBot(config)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 