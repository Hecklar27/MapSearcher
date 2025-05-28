# Discord Map Art Reverse Search Bot

A Discord bot that archives Minecraft map art images and provides reverse image search functionality. Users can upload photos of maps (including poor quality phone photos) and the bot finds matching images from Discord archives with links to original posts.

## Features

- **🔍 Reverse Image Search**: Upload any map art image and find similar maps in the archive
- **📚 Automatic Archiving**: Automatically indexes new map art as it's posted to monitored channels
- **🤖 AI-Powered**: Uses OpenAI's CLIP model for semantic image understanding
- **📱 Quality Handling**: Works with poor quality photos, different angles, and lighting conditions
- **⚡ Fast Search**: Sub-2 second response times with GPU acceleration
- **🎯 Accurate Results**: Returns top 3 most similar images with similarity scores

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3060+ recommended (CUDA support)
- **RAM**: 16GB+ for model loading and vector operations
- **Storage**: 10GB+ for dependencies and model cache

### Software
- Python 3.9+
- CUDA-compatible GPU drivers (for GPU acceleration)
- MongoDB Atlas account (free tier sufficient for testing)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MapSearcher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Configure MongoDB Atlas**
   - Create a MongoDB Atlas cluster
   - Set up vector search index (see Database Setup section)
   - Add connection string to `.env`

5. **Create Discord Bot**
   - Go to Discord Developer Portal
   - Create new application and bot
   - Copy bot token to `.env`
   - Invite bot to your server with appropriate permissions

## Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Discord Bot Configuration
DISCORD_TOKEN=your_bot_token_here
TARGET_GUILD_ID=your_server_id_here
MAP_CHANNEL_ID=your_channel_id_here

# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=map_art_bot

# AI/ML Configuration
CLIP_MODEL=ViT-B/32
DEVICE=cuda  # or cpu for development without GPU
```

### Discord Permissions

The bot requires the following permissions:
- Read Messages
- Send Messages
- Use Slash Commands
- Read Message History
- Attach Files

## Usage

### Commands

- `/reverse_search` - Upload an image to find similar map art in the archive

### Automatic Indexing

The bot automatically processes and indexes all images posted to the configured map art channel. No manual intervention required.

## Database Setup

### MongoDB Atlas Vector Search Index

Create a vector search index in MongoDB Atlas with the following configuration:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "clip_embedding": {
        "dimensions": 512,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

## Development

### Running the Bot

```bash
python main.py
```

### Testing

```bash
pytest tests/
```

### Project Structure

```
MapSearcher/
├── bot/                    # Discord bot implementation
├── search/                 # Image processing and search engine
├── database/              # MongoDB operations
├── utils/                 # Configuration and utilities
├── tests/                 # Test suite
├── memory-bank/           # Project documentation
├── requirements.txt       # Python dependencies
├── env.example           # Environment configuration template
└── main.py               # Application entry point
```

## Performance

- **Search Response**: < 2 seconds
- **Accuracy**: 90%+ for good images, 75%+ for poor photos
- **Indexing Rate**: 100+ images/hour
- **Memory Usage**: ~4GB GPU memory for CLIP model

## Architecture

The bot uses an event-driven architecture with the following components:

1. **Discord Bot Layer**: Handles user interactions and commands
2. **Image Processing**: CLIP-based embedding generation with preprocessing
3. **Vector Database**: MongoDB Atlas for similarity search
4. **Background Tasks**: Async processing for non-blocking operations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU mode
2. **Slow Search**: Ensure vector search index is properly configured
3. **Bot Not Responding**: Check Discord token and permissions
4. **Database Connection**: Verify MongoDB URI and network access

### Logs

The bot provides detailed logging. Set `LOG_LEVEL=DEBUG` for verbose output.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs for error details
- Open an issue on GitHub 