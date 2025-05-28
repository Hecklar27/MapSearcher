# Discord Map Art Reverse Search Bot - Project Brief

## Project Overview
A Discord bot that archives Minecraft map art images and provides reverse image search functionality. Users can upload photos of maps (including poor quality phone photos) and the bot finds matching images from Discord archives with links to original posts.

## Core Requirements
1. **Archive System**: Automatically scan Discord channels for map art and build a searchable database
2. **Reverse Search**: Process uploaded images to find similar maps and return Discord message links
3. **Real-time Indexing**: Auto-process new images as they're posted to monitored channels
4. **Quality Handling**: Handle poor lighting, angles, and blurry photos using AI preprocessing

## Success Criteria
- Search response time: < 2 seconds
- Accuracy: 90%+ for good images, 75%+ for poor photos
- Handle 100+ images/hour indexing
- Robust handling of various image qualities and formats

## Technical Foundation
- **Language**: Python 3.9+
- **AI/ML**: PyTorch + CLIP for image embeddings
- **Database**: MongoDB with vector search capabilities
- **Bot Framework**: discord.py
- **Image Processing**: OpenCV + PIL

## Key Constraints
- Must work with existing Discord channels
- Handle real-time processing without blocking
- Scalable vector search for large image collections
- GPU acceleration recommended for CLIP inference 