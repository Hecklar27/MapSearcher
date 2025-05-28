#!/usr/bin/env python3
"""
Demo Test Script - Test Bot Functionality Without Discord
"""

import asyncio
import logging
from utils.config import Config
from search.search_engine import SearchEngine
from database.connection import DatabaseConnection
from database.repository import ImageRepository
from search.image_processor import ImageProcessor
from search.clip_processor import CLIPProcessor

async def demo_test():
    """Test bot functionality without Discord connection"""
    print("🔄 Starting Demo Test - No Discord Required!")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test 1: Configuration
        print("\n📋 Test 1: Configuration Loading")
        config = Config()
        print(f"✅ Config loaded - Device: {config.device}, Model: {config.clip_model}")
        
        # Test 2: CLIP Processor
        print("\n🧠 Test 2: CLIP Processor (Fallback Mode)")
        clip_processor = CLIPProcessor(config)
        await clip_processor.initialize(config)
        
        # Test dummy embedding generation
        test_image_data = b"fake_image_data_for_testing"
        embedding = clip_processor.generate_image_embedding(test_image_data)
        print(f"✅ Generated embedding shape: {embedding.shape}")
        
        model_info = clip_processor.get_model_info()
        print(f"✅ Model status: {model_info['status']}")
        
        # Test 3: Image Processor
        print("\n🖼️ Test 3: Image Processor")
        image_processor = ImageProcessor()
        
        # Create a simple test image
        from PIL import Image
        import io
        test_img = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='JPEG')
        test_img_bytes = img_buffer.getvalue()
        
        # Test image validation
        is_valid = image_processor.validate_image(test_img_bytes)
        print(f"✅ Image validation: {is_valid}")
        
        # Test image info
        img_info = image_processor.get_image_info(test_img_bytes)
        print(f"✅ Image info: {img_info['width']}x{img_info['height']}")
        
        # Test 4: Database Connection (will fail with placeholder URI, but that's OK)
        print("\n🗄️ Test 4: Database Connection")
        try:
            db_connection = DatabaseConnection(config)
            print("✅ Database connection object created")
            print("⚠️ Actual connection will fail with placeholder URI (expected)")
        except Exception as e:
            print(f"⚠️ Database connection failed (expected): {str(e)[:50]}...")
        
        # Test 5: Search Engine Components
        print("\n🔍 Test 5: Search Engine Components")
        
        # Create mock repository for testing
        class MockRepository:
            async def vector_search(self, embedding, limit=3):
                return []
            async def get_image_count(self):
                return 0
            async def get_recent_images(self, limit=5):
                return []
        
        mock_repo = MockRepository()
        search_engine = SearchEngine(config, mock_repo)
        await search_engine.initialize()
        
        # Test search functionality
        stats = await search_engine.get_search_statistics()
        print(f"✅ Search engine stats: {stats}")
        
        # Test reverse search with dummy data
        results = await search_engine.reverse_search(test_img_bytes)
        print(f"✅ Reverse search completed: {len(results)} results")
        
        print("\n🎉 Demo Test Complete!")
        print("\n📊 Summary:")
        print("✅ Configuration: Working")
        print("✅ CLIP Processor: Working (fallback mode)")
        print("✅ Image Processor: Working")
        print("✅ Search Engine: Working")
        print("⚠️ Database: Needs real MongoDB URI")
        print("⚠️ Discord: Needs real bot token")
        
        print("\n🚀 Next Steps:")
        print("1. Follow QUICK_START.md to get Discord bot token")
        print("2. Set up MongoDB Atlas database")
        print("3. Update .env file with real credentials")
        print("4. Run: python main.py")
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_test()) 