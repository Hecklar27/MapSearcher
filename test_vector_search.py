#!/usr/bin/env python3
"""
Test script to verify vector search behavior
"""

import asyncio
import numpy as np
import logging
from search.clip_processor import CLIPProcessor
from search.search_engine import SearchEngine
from database.connection import DatabaseConnection
from database.repository import ImageRepository
from database.models import ImageDocument, ImageMetadata, EmbeddingUtils
from utils.config import Config
from datetime import datetime
from PIL import Image
import io

async def test_vector_search():
    """Test vector search behavior"""
    print("Testing vector search behavior...")
    
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    config = Config()
    
    # Connect to database
    db_conn = DatabaseConnection(config)
    connected = await db_conn.connect()
    if not connected:
        print("❌ Cannot connect to database")
        return
    
    # Create repository and search engine
    repository = ImageRepository(db_conn.database)
    search_engine = SearchEngine(config, repository)
    await search_engine.initialize()
    
    # Create test images
    print("\n1. Creating test images...")
    
    # Black image
    black_img = Image.new('RGB', (100, 100), color='black')
    black_bytes = io.BytesIO()
    black_img.save(black_bytes, format='PNG')
    black_image_bytes = black_bytes.getvalue()
    
    # White image  
    white_img = Image.new('RGB', (100, 100), color='white')
    white_bytes = io.BytesIO()
    white_img.save(white_bytes, format='PNG')
    white_image_bytes = white_bytes.getvalue()
    
    # Generate embeddings
    processor = CLIPProcessor(config)
    await processor.initialize(config)
    
    black_embedding = processor.generate_image_embedding(black_image_bytes)
    white_embedding = processor.generate_image_embedding(white_image_bytes)
    
    print(f"Black embedding shape: {black_embedding.shape}")
    print(f"White embedding shape: {white_embedding.shape}")
    
    # Test self-similarity
    self_similarity = EmbeddingUtils.cosine_similarity(black_embedding, black_embedding)
    cross_similarity = EmbeddingUtils.cosine_similarity(black_embedding, white_embedding)
    print(f"Black vs Black similarity: {self_similarity:.6f}")
    print(f"Black vs White similarity: {cross_similarity:.6f}")
    
    # Store black image in database
    print("\n2. Storing black image in database...")
    
    black_doc = ImageDocument(
        discord_message_id=12345,
        message_link="https://discord.com/channels/123/456/12345",
        image_url="https://example.com/black.png",
        clip_embedding=EmbeddingUtils.numpy_to_list(black_embedding),
        metadata=ImageMetadata(
            upload_timestamp=datetime.now(),
            image_dimensions={"width": 100, "height": 100},
            file_size=len(black_image_bytes),
            channel_id=456,
            user_id=789,
            filename="black.png",
            content_type="image/png"
        )
    )
    
    stored = await repository.store_image(black_doc)
    print(f"Stored black image: {stored}")
    
    # Test vector search with same image (should be perfect match)
    print("\n3. Testing vector search with exact same image...")
    
    try:
        # Search for exact same black image
        black_embedding_list = EmbeddingUtils.numpy_to_list(black_embedding)
        results = await repository.vector_search(black_embedding_list, limit=3)
        
        print(f"Found {len(results)} results")
        if results:
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result.similarity_score:.6f} similarity")
                if abs(result.similarity_score - 1.0) < 0.0001:
                    print("✅ Perfect match found!")
                else:
                    print(f"❌ Expected perfect match (1.0), got {result.similarity_score:.6f}")
        else:
            print("❌ No results found - vector search may be failing")
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        print("This suggests vector search index is not working")
    
    # Test with different image
    print("\n4. Testing vector search with different image...")
    
    try:
        white_embedding_list = EmbeddingUtils.numpy_to_list(white_embedding)
        results = await repository.vector_search(white_embedding_list, limit=3)
        
        print(f"Found {len(results)} results")
        if results:
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result.similarity_score:.6f} similarity")
                print(f"Expected: Low similarity (around {cross_similarity:.6f})")
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
    
    # Test multiple searches of same image to check consistency
    print("\n5. Testing consistency - searching same image multiple times...")
    
    similarities = []
    for i in range(5):
        try:
            results = await repository.vector_search(black_embedding_list, limit=1)
            if results:
                similarity = results[0].similarity_score
                similarities.append(similarity)
                print(f"Search {i+1}: {similarity:.6f}")
            else:
                print(f"Search {i+1}: No results")
        except Exception as e:
            print(f"Search {i+1}: Failed - {e}")
    
    if similarities:
        if all(abs(s - similarities[0]) < 1e-6 for s in similarities):
            print("✅ All searches returned consistent results!")
        else:
            print("❌ Searches returned different results!")
            print(f"Range: {min(similarities):.6f} to {max(similarities):.6f}")
    
    # Clean up
    print("\n6. Cleaning up...")
    await repository.delete_image(12345)
    await db_conn.disconnect()
    
    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(test_vector_search()) 