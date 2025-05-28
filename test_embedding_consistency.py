#!/usr/bin/env python3
"""
Test script to verify embedding consistency
Checks if the same image produces identical embeddings
"""

import asyncio
import numpy as np
from search.clip_processor import CLIPProcessor
from utils.config import Config

async def test_embedding_consistency():
    """Test that the same image produces identical embeddings"""
    print("Testing embedding consistency...")
    
    # Initialize processor
    config = Config()
    processor = CLIPProcessor(config)
    await processor.initialize(config)
    
    # Create a simple test image (black square)
    from PIL import Image
    import io
    
    # Create a simple 100x100 black image
    test_image = Image.new('RGB', (100, 100), color='black')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    image_bytes = img_bytes.getvalue()
    
    print(f"Test image size: {len(image_bytes)} bytes")
    
    # Generate embeddings multiple times
    embeddings = []
    for i in range(5):
        embedding = processor.generate_image_embedding(image_bytes)
        embeddings.append(embedding)
        print(f"Embedding {i+1} shape: {embedding.shape}")
        print(f"Embedding {i+1} first 5 values: {embedding[:5]}")
        print(f"Embedding {i+1} sum: {np.sum(embedding):.6f}")
        print()
    
    # Check if all embeddings are identical
    all_identical = True
    for i in range(1, len(embeddings)):
        if not np.array_equal(embeddings[0], embeddings[i]):
            all_identical = False
            diff = np.abs(embeddings[0] - embeddings[i])
            max_diff = np.max(diff)
            print(f"❌ Embedding {i+1} differs from first embedding!")
            print(f"   Max difference: {max_diff}")
            print(f"   Different elements: {np.sum(diff > 1e-10)}")
            break
    
    if all_identical:
        print("✅ All embeddings are identical - consistent!")
    else:
        print("❌ Embeddings are not consistent!")
    
    # Test with a different but similar image
    print("\n" + "="*50)
    print("Testing with slightly different image...")
    
    # Create a 100x100 white image
    test_image2 = Image.new('RGB', (100, 100), color='white')
    img_bytes2 = io.BytesIO()
    test_image2.save(img_bytes2, format='PNG')
    image_bytes2 = img_bytes2.getvalue()
    
    embedding1 = processor.generate_image_embedding(image_bytes)
    embedding2 = processor.generate_image_embedding(image_bytes2)
    
    # Calculate similarity
    from database.models import EmbeddingUtils
    similarity = EmbeddingUtils.cosine_similarity(embedding1, embedding2)
    
    print(f"Black vs White image similarity: {similarity:.4f}")
    print(f"Expected: Should be low (different images)")
    
    # Test same image again to verify consistency
    print("\n" + "="*50)
    print("Re-testing original image for consistency...")
    
    embedding3 = processor.generate_image_embedding(image_bytes)
    same_image_similarity = EmbeddingUtils.cosine_similarity(embedding1, embedding3)
    
    print(f"Same image similarity: {same_image_similarity:.10f}")
    print(f"Expected: Should be 1.0 (identical images)")
    
    if abs(same_image_similarity - 1.0) < 1e-6:
        print("✅ Same image produces consistent embeddings!")
    else:
        print("❌ Same image produces different embeddings!")

if __name__ == "__main__":
    asyncio.run(test_embedding_consistency()) 