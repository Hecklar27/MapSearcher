"""
Discord Bot Main Class
Core bot implementation and initialization
"""

import logging
import discord
from discord.ext import commands
from utils.config import Config
from database.connection import DatabaseConnection
from database.repository import ImageRepository
from search.search_engine import SearchEngine
from utils.helpers import is_image_url
import asyncio

class MapArtBot:
    """Main Discord bot class for Map Art Reverse Search"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure Discord intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        
        # Initialize bot with intents
        self.bot = commands.Bot(
            command_prefix='!',  # Fallback prefix, mainly using slash commands
            intents=intents,
            help_command=None
        )
        
        # Initialize database components
        self.database_connection = DatabaseConnection(config)
        self.image_repository = None
        self.search_engine = None
        
        # Set up event handlers
        self._setup_events()
        self._setup_commands()
    
    def _setup_events(self):
        """Set up Discord event handlers"""
        
        @self.bot.event
        async def on_ready():
            self.logger.info(f"Bot logged in as {self.bot.user} (ID: {self.bot.user.id})")
            self.logger.info(f"Connected to {len(self.bot.guilds)} guilds")
            
            # Initialize database connection
            await self._initialize_database()
            
            # Initialize search engine
            await self._initialize_search_engine()
            
            # Index existing messages if database is empty
            await self._index_existing_messages()
            
            # Sync slash commands
            try:
                synced = await self.bot.tree.sync()
                self.logger.info(f"Synced {len(synced)} slash commands")
            except Exception as e:
                self.logger.error(f"Failed to sync commands: {e}")
        
        @self.bot.event
        async def on_message(message):
            # Ignore bot messages
            if message.author.bot:
                return
            
            # Only process messages from the target channel
            if message.channel.id != self.config.map_channel_id:
                return
            
            # Check if message has image attachments
            if message.attachments:
                await self._process_new_image(message)
            
            # Process commands
            await self.bot.process_commands(message)
        
        @self.bot.event
        async def on_error(event, *args, **kwargs):
            self.logger.error(f"Discord event error in {event}", exc_info=True)
    
    def _setup_commands(self):
        """Set up Discord slash commands"""
        
        @self.bot.tree.command(name="reverse_search", description="Find similar map art images")
        async def reverse_search(interaction: discord.Interaction, image: discord.Attachment):
            """Reverse search command"""
            try:
                # Quick defer to acknowledge the interaction
                await interaction.response.defer()
                
                # Validate attachment
                if not is_image_url(image.url):
                    await interaction.followup.send("❌ Please upload a valid image file.")
                    return
                
                if image.size > self.config.max_image_size_bytes:
                    await interaction.followup.send(f"❌ Image too large. Maximum size: {self.config.max_image_size_mb}MB")
                    return
                
                # Check if search engine is ready
                if not self.search_engine:
                    await interaction.followup.send("❌ Search engine not initialized.")
                    return
                
                # Send processing message
                await interaction.followup.send("🔍 Processing your image and searching for similar map art...")
                
                # Download image
                try:
                    image_bytes = await image.read()
                except Exception as e:
                    self.logger.error(f"Failed to download image: {e}")
                    await interaction.followup.send("❌ Failed to download image. Please try again.")
                    return
                
                # Perform search with timeout protection
                try:
                    results = await asyncio.wait_for(
                        self.search_engine.reverse_search(image_bytes), 
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    await interaction.followup.send("❌ Search timed out. Please try with a smaller image.")
                    return
                except Exception as e:
                    self.logger.error(f"Search failed: {e}")
                    await interaction.followup.send("❌ Search failed. Please try again.")
                    return
                
                # Format and send results
                await self._send_search_results(interaction, results)
                
            except discord.NotFound:
                # Interaction already timed out
                self.logger.warning("Discord interaction timed out")
            except Exception as e:
                self.logger.error(f"Reverse search command failed: {e}")
                try:
                    await interaction.followup.send("❌ An unexpected error occurred. Please try again.")
                except:
                    pass  # Interaction may be expired
        
        @self.bot.tree.command(name="stats", description="Show bot statistics")
        async def stats(interaction: discord.Interaction):
            """Statistics command"""
            await interaction.response.defer()
            
            try:
                if not self.search_engine:
                    await interaction.followup.send("❌ Search engine not initialized.")
                    return
                
                stats = await self.search_engine.get_search_statistics()
                
                embed = discord.Embed(
                    title="🔍 Map Art Search Bot Statistics",
                    color=discord.Color.blue()
                )
                
                embed.add_field(
                    name="📊 Database",
                    value=f"Total Images: {stats.get('total_indexed_images', 0)}",
                    inline=True
                )
                
                model_info = stats.get('clip_model_info', {})
                embed.add_field(
                    name="🤖 AI Model",
                    value=f"Status: {model_info.get('status', 'Unknown')}\nDevice: {model_info.get('device', 'Unknown')}",
                    inline=True
                )
                
                if stats.get('last_indexed'):
                    embed.add_field(
                        name="⏰ Last Indexed",
                        value=f"<t:{int(stats['last_indexed'].timestamp())}:R>",
                        inline=True
                    )
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                self.logger.error(f"Stats command failed: {e}")
                await interaction.followup.send("❌ Failed to get statistics.")
    
    async def _process_new_image(self, message: discord.Message):
        """Process new image attachment for indexing"""
        try:
            for attachment in message.attachments:
                if is_image_url(attachment.url):
                    self.logger.info(f"Processing new image from message {message.id}")
                    
                    # Index the image in background
                    if self.search_engine:
                        success = await self.search_engine.index_discord_message(
                            message.id,
                            message.channel.id,
                            message.author.id,
                            attachment.url,
                            message.guild.id
                        )
                        
                        if success:
                            self.logger.info(f"Successfully indexed image from message {message.id}")
                        else:
                            self.logger.warning(f"Failed to index image from message {message.id}")
                    
        except Exception as e:
            self.logger.error(f"Failed to process new image: {e}")
    
    async def _send_search_results(self, interaction: discord.Interaction, results: list):
        """Send formatted search results"""
        if not results:
            embed = discord.Embed(
                title="🔍 No Similar Images Found",
                description="No similar map art found in the database.",
                color=discord.Color.orange()
            )
            await interaction.followup.send(embed=embed)
            return
        
        embed = discord.Embed(
            title="🔍 Similar Map Art Found",
            description=f"Found {len(results)} similar images:",
            color=discord.Color.green()
        )
        
        for i, result in enumerate(results, 1):
            similarity_percent = f"{result.similarity_score * 100:.1f}%"
            
            # Create Discord message link format
            message_link = result.message_link if hasattr(result, 'message_link') else result.image_document.message_link
            upload_date = result.upload_date if hasattr(result, 'upload_date') else result.image_document.metadata.upload_timestamp
            image_url = result.image_url if hasattr(result, 'image_url') else result.image_document.image_url
            
            embed.add_field(
                name=f"#{i} - {similarity_percent} Match",
                value=f"[View Original Message]({message_link})\n"
                      f"Uploaded: <t:{int(upload_date.timestamp())}:R>",
                inline=False
            )
            
            # Set thumbnail to first result
            if i == 1:
                embed.set_thumbnail(url=image_url)
        
        embed.set_footer(text="Click the links to view the original Discord messages")
        await interaction.followup.send(embed=embed)
    
    async def _initialize_database(self):
        """Initialize database connection and repository"""
        try:
            self.logger.info("Initializing database connection...")
            
            # Check if we're using placeholder URI
            if 'username:password@cluster' in self.config.mongodb_uri:
                self.logger.warning("Using placeholder MongoDB URI - running in demo mode")
                self.logger.warning("Database functionality will be limited")
                # Create a mock repository for demo purposes
                from database.repository import MockImageRepository
                self.image_repository = MockImageRepository()
                self.logger.info("Demo mode: Using mock repository")
                return
            
            # Connect to MongoDB
            connected = await self.database_connection.connect()
            if not connected:
                self.logger.error("Failed to connect to database - falling back to demo mode")
                from database.repository import MockImageRepository
                self.image_repository = MockImageRepository()
                return
            
            # Create repository with the real database
            self.image_repository = ImageRepository(self.database_connection.database)
            self.logger.info("Real database repository created successfully")
            
            # Create indexes
            await self.database_connection.create_indexes()
            
            # Log database stats and confirm we're using real database
            image_count = await self.image_repository.get_image_count()
            self.logger.info(f"✅ REAL DATABASE initialized. Current image count: {image_count}")
            
            # Index existing messages from the target channel if database is empty
            if image_count == 0:
                self.logger.info("Database is empty - will index existing messages when bot starts")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.logger.warning("Falling back to demo mode")
            from database.repository import MockImageRepository
            self.image_repository = MockImageRepository()
    
    async def _initialize_search_engine(self):
        """Initialize search engine"""
        try:
            if not self.image_repository:
                self.logger.error("Cannot initialize search engine: database not ready")
                return
            
            self.logger.info("Initializing search engine...")
            
            # Create search engine
            self.search_engine = SearchEngine(self.config, self.image_repository)
            
            # Initialize components
            await self.search_engine.initialize()
            
            self.logger.info("Search engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Search engine initialization failed: {e}")
    
    async def _index_existing_messages(self):
        """Index existing messages from the target Discord channel if database is empty"""
        try:
            if not self.image_repository or not self.search_engine:
                self.logger.warning("Cannot index existing messages: components not ready")
                return
            
            # Check if database already has images
            image_count = await self.image_repository.get_image_count()
            if image_count > 0:
                self.logger.info(f"Database already has {image_count} images - skipping historical indexing")
                return
            
            self.logger.info("Database is empty - indexing existing messages from target Discord channel...")
            
            # Get the target channel
            channel = self.bot.get_channel(self.config.map_channel_id)
            if not channel:
                self.logger.error(f"Cannot find Discord channel with ID {self.config.map_channel_id}")
                return
            
            self.logger.info(f"Found target channel: #{channel.name}")
            
            # Fetch historical messages with images (limit to last 1000 to avoid overwhelming)
            processed_count = 0
            failed_count = 0
            
            async for message in channel.history(limit=1000):
                if message.attachments:
                    for attachment in message.attachments:
                        if is_image_url(attachment.url):
                            try:
                                success = await self.search_engine.index_discord_message(
                                    message.id,
                                    message.channel.id,
                                    message.author.id,
                                    attachment.url,
                                    message.guild.id
                                )
                                
                                if success:
                                    processed_count += 1
                                    self.logger.info(f"Indexed historical message {message.id} ({processed_count} total)")
                                else:
                                    failed_count += 1
                                    
                            except Exception as e:
                                failed_count += 1
                                self.logger.error(f"Failed to index message {message.id}: {e}")
            
            final_count = await self.image_repository.get_image_count()
            self.logger.info(f"✅ Historical indexing complete: {processed_count} processed, {failed_count} failed, {final_count} total in database")
            
        except Exception as e:
            self.logger.error(f"Failed to index existing messages: {e}")
    
    async def start(self):
        """Start the Discord bot"""
        try:
            self.logger.info("Starting Discord bot...")
            await self.bot.start(self.config.discord_token)
        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up bot resources...")
        
        # Clean up search engine
        if self.search_engine:
            self.search_engine.cleanup()
        
        # Close database connection
        if self.database_connection:
            await self.database_connection.disconnect()
        
        if not self.bot.is_closed():
            await self.bot.close() 