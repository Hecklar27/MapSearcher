# System Patterns - Discord Map Art Reverse Search Bot

## Core Architecture Pattern

### Event-Driven Processing
```
Discord Message → Image Detection → Background Processing → Database Storage
                                ↓
User Search Request → Vector Query → Similarity Ranking → Discord Response
```

### Component Separation
- **Bot Layer**: Discord interaction and command handling
- **Processing Layer**: Image analysis and CLIP embedding generation
- **Storage Layer**: MongoDB operations and vector search
- **Search Layer**: Similarity computation and result ranking

## Key Design Patterns

### 1. Async/Await Pattern
All Discord and database operations use async/await to prevent blocking:
```python
async def process_image(message):
    image_data = await download_image(message.attachments[0].url)
    embedding = await generate_embedding(image_data)
    await store_in_database(embedding, message)
```

### 2. Background Task Queue
Heavy processing (CLIP inference) runs in background tasks:
```python
@bot.event
async def on_message(message):
    if has_image(message):
        bot.loop.create_task(process_image_background(message))
```

### 3. Singleton Pattern for CLIP Model
CLIP model loaded once and reused across requests:
```python
class CLIPProcessor:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = clip.load("ViT-B/32")
        return cls._instance
```

### 4. Repository Pattern for Database
Abstract database operations behind clean interfaces:
```python
class ImageRepository:
    async def store_image(self, embedding, metadata):
        # MongoDB operations
    
    async def search_similar(self, query_embedding, limit=3):
        # Vector search operations
```

## Data Flow Architecture

### Indexing Flow
1. **Message Event** → Discord message with image attachment
2. **Image Download** → Fetch image from Discord CDN
3. **Preprocessing** → Enhance image quality, normalize format
4. **CLIP Processing** → Generate 512-dim embedding vector
5. **Database Storage** → Store vector + metadata in MongoDB

### Search Flow
1. **User Upload** → Image attached to reverse search command
2. **Image Processing** → Same preprocessing + CLIP pipeline
3. **Vector Search** → MongoDB Atlas vector similarity query
4. **Result Ranking** → Sort by cosine similarity score
5. **Discord Response** → Format and send top 3 matches

## Database Schema Design

### Images Collection
```javascript
{
  _id: ObjectId,
  discord_message_id: String,
  message_link: String,
  image_url: String,
  clip_embedding: [Number], // 512 dimensions
  metadata: {
    upload_timestamp: Date,
    image_dimensions: {width: Number, height: Number},
    file_size: Number,
    channel_id: String,
    user_id: String
  }
}
```

### Vector Search Index
```javascript
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

## Error Handling Patterns

### Graceful Degradation
- If CLIP model fails → log error, continue with other images
- If database unavailable → queue operations for retry
- If image download fails → skip and continue processing

### Retry Logic
```python
@retry(max_attempts=3, backoff_factor=2)
async def download_image(url):
    # Download with exponential backoff
```

### Logging Strategy
- Structured logging with correlation IDs
- Performance metrics for search response times
- Error tracking for failed image processing

## Performance Optimization Patterns

### Batch Processing
- Process multiple historical images in batches
- Batch database insertions for efficiency
- Parallel CLIP inference when possible

### Caching Strategy
- In-memory cache for frequently accessed embeddings
- Discord message cache to avoid re-processing
- Model weight caching to speed startup

### Resource Management
- Connection pooling for MongoDB
- GPU memory management for CLIP
- Async context managers for cleanup 