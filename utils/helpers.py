"""
Utility Helper Functions
Common utility functions for the application
"""

import re
import aiohttp
import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def is_image_url(url: str) -> bool:
    """
    Check if URL points to an image file
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to be an image
    """
    if not url:
        return False
    
    # Check file extension
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg'}
    parsed_url = urlparse(url.lower())
    path = parsed_url.path
    
    # Check if path ends with image extension
    for ext in image_extensions:
        if path.endswith(ext):
            return True
    
    # Check Discord CDN URLs (they don't always have extensions)
    if 'cdn.discordapp.com' in url or 'media.discordapp.net' in url:
        return True
    
    return False

def format_discord_message_link(guild_id: int, channel_id: int, message_id: int) -> str:
    """
    Format Discord message link
    
    Args:
        guild_id: Discord guild ID
        channel_id: Discord channel ID
        message_id: Discord message ID
        
    Returns:
        Formatted Discord message URL
    """
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

def extract_message_ids_from_link(link: str) -> Optional[Dict[str, int]]:
    """
    Extract guild, channel, and message IDs from Discord link
    
    Args:
        link: Discord message link
        
    Returns:
        Dictionary with IDs or None if invalid
    """
    pattern = r'https://discord\.com/channels/(\d+)/(\d+)/(\d+)'
    match = re.match(pattern, link)
    
    if match:
        return {
            'guild_id': int(match.group(1)),
            'channel_id': int(match.group(2)),
            'message_id': int(match.group(3))
        }
    
    return None

async def download_image_bytes(url: str, max_size_bytes: int = 25 * 1024 * 1024, 
                              timeout: int = 30) -> Optional[bytes]:
    """
    Download image from URL with size and timeout limits
    
    Args:
        url: Image URL to download
        max_size_bytes: Maximum file size in bytes
        timeout: Timeout in seconds
        
    Returns:
        Image bytes or None if download failed
    """
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(url) as response:
                # Check response status
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} when downloading {url}")
                    return None
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size_bytes:
                    logger.warning(f"Image too large: {content_length} bytes from {url}")
                    return None
                
                # Download with size limit
                data = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    data.extend(chunk)
                    if len(data) > max_size_bytes:
                        logger.warning(f"Image exceeded size limit during download: {url}")
                        return None
                
                logger.debug(f"Downloaded {len(data)} bytes from {url}")
                return bytes(data)
                
    except asyncio.TimeoutError:
        logger.warning(f"Timeout downloading image from {url}")
        return None
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None

def validate_discord_snowflake(snowflake: str) -> bool:
    """
    Validate Discord snowflake ID format
    
    Args:
        snowflake: String to validate
        
    Returns:
        True if valid snowflake format
    """
    if not snowflake or not snowflake.isdigit():
        return False
    
    # Discord snowflakes are 64-bit integers
    try:
        snowflake_int = int(snowflake)
        # Discord epoch started 2015-01-01, so valid snowflakes should be large
        return snowflake_int > 4194304  # Minimum valid Discord snowflake
    except ValueError:
        return False

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def parse_similarity_score(score: float) -> Dict[str, Any]:
    """
    Parse similarity score into categories and descriptions
    
    Args:
        score: Similarity score (0.0 to 1.0)
        
    Returns:
        Dictionary with score analysis
    """
    if score >= 0.95:
        category = "Identical"
        description = "Nearly identical images"
        color = 0x00FF00  # Green
    elif score >= 0.85:
        category = "Very Similar"
        description = "Very similar with minor differences"
        color = 0x7FFF00  # Light green
    elif score >= 0.75:
        category = "Similar"
        description = "Similar images with some differences"
        color = 0xFFFF00  # Yellow
    elif score >= 0.60:
        category = "Somewhat Similar"
        description = "Some similarities but noticeable differences"
        color = 0xFF7F00  # Orange
    else:
        category = "Different"
        description = "Different images with few similarities"
        color = 0xFF0000  # Red
    
    return {
        "score": score,
        "percentage": f"{score * 100:.1f}%",
        "category": category,
        "description": description,
        "color": color
    }

async def batch_download_images(urls: List[str], max_size_bytes: int = 25 * 1024 * 1024,
                               timeout: int = 30, max_concurrent: int = 5) -> List[Optional[bytes]]:
    """
    Download multiple images concurrently
    
    Args:
        urls: List of image URLs
        max_size_bytes: Maximum file size per image
        timeout: Timeout per download
        max_concurrent: Maximum concurrent downloads
        
    Returns:
        List of image bytes (None for failed downloads)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_with_semaphore(url: str) -> Optional[bytes]:
        async with semaphore:
            return await download_image_bytes(url, max_size_bytes, timeout)
    
    tasks = [download_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to None
    return [result if not isinstance(result, Exception) else None for result in results]

def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """
    Create a text progress bar
    
    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = f"{progress * 100:.1f}%"
    
    return f"[{bar}] {percentage} ({current}/{total})" 