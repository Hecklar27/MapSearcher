# Technical Context - Discord Map Art Reverse Search Bot

## Technology Stack

### Core Framework
- **Python 3.9+** - Main development language
- **discord.py 2.3.0+** - Discord bot framework with slash commands support
- **asyncio** - Asynchronous programming for non-blocking operations

### AI/ML Components
- **PyTorch 2.0.0+** - Deep learning framework
- **CLIP (ViT-B/32)** - OpenAI's vision-language model for image embeddings
- **OpenCV 4.8.0+** - Computer vision and image preprocessing
- **Pillow 10.0.0+** - Python Imaging Library for image manipulation
- **NumPy 1.24.0+** - Numerical computing for vector operations

### Database & Storage
- **MongoDB Atlas** - Cloud-hosted document database with vector search
- **Motor 3.3.0+** - Async MongoDB driver for Python
- **GridFS** - MongoDB's file storage system for large images

### Development Environment
- **Local Development** - Primary development environment
- **GPU Support** - NVIDIA CUDA for CLIP inference acceleration
- **Environment Variables** - python-dotenv for configuration management

## Architecture Decisions

### Single Server Focus
- Bot designed for single Discord server deployment
- Configuration tied to specific server and channel IDs
- Simplified permission model and data isolation

### Vector Search Strategy
- 512-dimensional CLIP embeddings for semantic similarity
- MongoDB Atlas Vector Search for efficient similarity queries
- Cosine similarity for image matching

### Image Processing Pipeline
1. **Download** - Fetch images from Discord CDN
2. **Preprocess** - Normalize, resize, enhance quality
3. **Extract** - Generate CLIP embeddings
4. **Store** - Save vectors and metadata to MongoDB

### Performance Considerations
- Async/await pattern for non-blocking Discord operations
- Background task queue for image processing
- Caching of CLIP model in memory
- Batch processing for historical channel indexing

## Development Setup Requirements

### Hardware
- **GPU**: NVIDIA RTX 3060+ recommended (CUDA support)
- **RAM**: 16GB+ for model loading and vector operations
- **Storage**: 10GB+ for dependencies and model cache

### Software Dependencies
```bash
# Core bot framework
discord.py>=2.3.0

# AI/ML stack
torch>=2.0.0
clip-by-openai>=1.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# Database
motor>=3.3.0
pymongo>=4.5.0

# Utilities
python-dotenv>=1.0.0
aiohttp>=3.8.0
```

### Environment Configuration
```bash
DISCORD_TOKEN=your_bot_token
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
TARGET_GUILD_ID=your_server_id
MAP_CHANNEL_ID=your_channel_id
CLIP_MODEL=ViT-B/32
DEVICE=cuda  # or cpu for development
```

## Deployment Architecture
- Single-server deployment model
- Local development with cloud database
- Environment-based configuration switching
- Graceful error handling and logging 