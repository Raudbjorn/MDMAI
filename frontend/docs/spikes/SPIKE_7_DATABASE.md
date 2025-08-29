# Spike 7: Database & Storage Strategy

## Overview
This spike defines the database architecture and storage strategies for the TTRPG MCP Server, focusing on ChromaDB for vector storage, SQLite for structured data, and efficient caching layers.

## 1. Database Architecture

### 1.1 Multi-Database Strategy
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import chromadb
import aiosqlite
from redis import asyncio as aioredis

@dataclass
class DatabaseConfig:
    chroma_path: str = "./data/chroma"
    sqlite_path: str = "./data/ttrpg.db"
    redis_url: str = "redis://localhost:6379"
    
class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.chroma_client: Optional[chromadb.Client] = None
        self.sqlite_conn: Optional[aiosqlite.Connection] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialize all database connections"""
        # ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.chroma_path
        )
        
        # SQLite for structured data
        self.sqlite_conn = await aiosqlite.connect(
            self.config.sqlite_path
        )
        await self._initialize_schema()
        
        # Redis for caching
        self.redis_client = await aioredis.from_url(
            self.config.redis_url,
            decode_responses=True
        )
```

### 1.2 ChromaDB Collections Design
```python
class ChromaCollections:
    """ChromaDB collection definitions for TTRPG content"""
    
    COLLECTIONS = {
        'rulebooks': {
            'name': 'rulebooks',
            'metadata': {
                'description': 'Game system rules and mechanics',
                'embedding_function': 'sentence-transformers',
                'distance_metric': 'cosine'
            },
            'schema': {
                'page_number': int,
                'chapter': str,
                'section': str,
                'game_system': str,
                'book_title': str,
                'content_type': str  # 'rule', 'table', 'example', 'sidebar'
            }
        },
        'monsters': {
            'name': 'monsters',
            'metadata': {
                'description': 'Monster stat blocks and descriptions',
                'embedding_function': 'sentence-transformers'
            },
            'schema': {
                'name': str,
                'cr': float,
                'type': str,
                'size': str,
                'alignment': str,
                'environment': str,
                'source': str
            }
        },
        'spells': {
            'name': 'spells',
            'metadata': {
                'description': 'Spell descriptions and mechanics'
            },
            'schema': {
                'name': str,
                'level': int,
                'school': str,
                'classes': list,
                'components': list,
                'range': str,
                'duration': str
            }
        },
        'campaign_notes': {
            'name': 'campaign_notes',
            'metadata': {
                'description': 'Campaign-specific content'
            },
            'schema': {
                'campaign_id': str,
                'session_number': int,
                'note_type': str,  # 'npc', 'location', 'plot', 'lore'
                'tags': list,
                'created_by': str,
                'created_at': str
            }
        }
    }
    
    def create_collections(self, client: chromadb.Client):
        """Create all ChromaDB collections"""
        for collection_config in self.COLLECTIONS.values():
            client.get_or_create_collection(
                name=collection_config['name'],
                metadata=collection_config['metadata']
            )
```

### 1.3 SQLite Schema
```sql
-- campaigns table
CREATE TABLE campaigns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gm_user_id TEXT NOT NULL,
    settings JSON,
    status TEXT DEFAULT 'active'
);

-- sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    session_number INTEGER NOT NULL,
    date TIMESTAMP NOT NULL,
    duration_minutes INTEGER,
    summary TEXT,
    notes JSON,
    participants JSON,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
);

-- characters table
CREATE TABLE characters (
    id TEXT PRIMARY KEY,
    campaign_id TEXT,
    player_id TEXT,
    name TEXT NOT NULL,
    system TEXT NOT NULL,
    level INTEGER,
    class TEXT,
    race TEXT,
    stats JSON,
    inventory JSON,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
);

-- initiative_tracker table
CREATE TABLE initiative_tracker (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    round_number INTEGER DEFAULT 1,
    turn_order JSON NOT NULL,
    current_turn INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- dice_rolls table
CREATE TABLE dice_rolls (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    player_id TEXT NOT NULL,
    expression TEXT NOT NULL,
    result INTEGER NOT NULL,
    breakdown JSON,
    purpose TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Create indexes
CREATE INDEX idx_campaigns_gm ON campaigns(gm_user_id);
CREATE INDEX idx_sessions_campaign ON sessions(campaign_id);
CREATE INDEX idx_characters_campaign ON characters(campaign_id);
CREATE INDEX idx_characters_player ON characters(player_id);
CREATE INDEX idx_dice_rolls_session ON dice_rolls(session_id);
CREATE INDEX idx_dice_rolls_player ON dice_rolls(player_id);
```

## 2. Storage Optimization

### 2.1 PDF Processing Pipeline
```python
import hashlib
from typing import List, Dict
import pypdf
from PIL import Image
import io

class PDFStorageOptimizer:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.chunk_size = 1000  # characters per chunk
        self.overlap = 200  # character overlap between chunks
        
    async def process_pdf(self, pdf_path: str, metadata: Dict) -> Dict:
        """Process and store PDF with deduplication"""
        # Calculate file hash for deduplication
        file_hash = self._calculate_hash(pdf_path)
        
        # Check if already processed
        existing = await self.db.redis_client.get(f"pdf:{file_hash}")
        if existing:
            return {"status": "already_processed", "hash": file_hash}
        
        # Extract content
        chunks = await self._extract_chunks(pdf_path)
        
        # Store in ChromaDB with deduplication
        collection = self.db.chroma_client.get_collection("rulebooks")
        
        stored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
            
            # Check for duplicate chunk
            existing_chunk = collection.get(
                where={"chunk_hash": chunk_hash}
            )
            
            if not existing_chunk['ids']:
                # Store new chunk
                collection.add(
                    documents=[chunk['text']],
                    metadatas=[{
                        **metadata,
                        **chunk['metadata'],
                        'chunk_hash': chunk_hash,
                        'file_hash': file_hash
                    }],
                    ids=[f"{file_hash}_{i}"]
                )
                stored_chunks.append(chunk_hash)
        
        # Cache processing result
        await self.db.redis_client.setex(
            f"pdf:{file_hash}",
            86400,  # 24 hour TTL
            json.dumps({
                "chunks": len(stored_chunks),
                "metadata": metadata
            })
        )
        
        return {
            "status": "processed",
            "hash": file_hash,
            "chunks": len(stored_chunks)
        }
    
    async def _extract_chunks(self, pdf_path: str) -> List[Dict]:
        """Extract text chunks with metadata"""
        chunks = []
        
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Extract tables separately
                tables = self._extract_tables(page)
                
                # Smart chunking with overlap
                page_chunks = self._smart_chunk(
                    text, 
                    page_num + 1,
                    tables
                )
                chunks.extend(page_chunks)
        
        return chunks
    
    def _smart_chunk(self, text: str, page_num: int, tables: List) -> List[Dict]:
        """Intelligent chunking that preserves context"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'page_number': page_num,
                            'has_tables': len(tables) > 0
                        }
                    })
                current_chunk = para + "\n\n"
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'page_number': page_num,
                    'has_tables': len(tables) > 0
                }
            })
        
        return chunks
```

### 2.2 Image Storage Strategy
```python
class ImageStorageManager:
    def __init__(self, storage_path: str = "./data/images"):
        self.storage_path = storage_path
        self.thumbnails_path = f"{storage_path}/thumbnails"
        
    async def store_map(self, image_data: bytes, metadata: Dict) -> str:
        """Store map image with multiple resolutions"""
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Open image
        img = Image.open(io.BytesIO(image_data))
        
        # Store original
        original_path = f"{self.storage_path}/maps/{image_hash}.webp"
        img.save(original_path, 'WEBP', quality=85)
        
        # Generate resolutions for different uses
        resolutions = {
            'thumbnail': (256, 256),
            'preview': (800, 800),
            'display': (1920, 1920),
            'full': None  # Keep original
        }
        
        paths = {'original': original_path}
        
        for name, size in resolutions.items():
            if size:
                resized = img.copy()
                resized.thumbnail(size, Image.Resampling.LANCZOS)
                path = f"{self.storage_path}/maps/{image_hash}_{name}.webp"
                resized.save(path, 'WEBP', quality=80)
                paths[name] = path
        
        # Store metadata in database
        await self._store_metadata(image_hash, metadata, paths)
        
        return image_hash
    
    async def get_map_url(self, image_hash: str, resolution: str = 'display') -> str:
        """Get URL for specific resolution"""
        # In production, this would return CDN URL
        return f"/api/images/maps/{image_hash}/{resolution}"
```

## 3. Caching Strategy

### 3.1 Multi-Level Cache
```python
class CacheManager:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.memory_cache = {}  # L1 cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    async def get(self, key: str, cache_level: int = 2) -> Optional[Any]:
        """Get from cache with multiple levels"""
        # L1: Memory cache
        if cache_level >= 1 and key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        # L2: Redis cache
        if cache_level >= 2:
            value = await self.redis.get(key)
            if value:
                self.cache_stats['hits'] += 1
                # Promote to L1
                self.memory_cache[key] = json.loads(value)
                return self.memory_cache[key]
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in cache with TTL"""
        # Store in L1 (memory)
        self.memory_cache[key] = value
        
        # Store in L2 (Redis)
        await self.redis.setex(
            key,
            ttl,
            json.dumps(value)
        )
        
        # Manage memory cache size
        if len(self.memory_cache) > 1000:
            # Simple LRU eviction
            oldest = next(iter(self.memory_cache))
            del self.memory_cache[oldest]
            self.cache_stats['evictions'] += 1
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        # Clear from Redis
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, 
                match=pattern,
                count=100
            )
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break
        
        # Clear from memory cache
        keys_to_delete = [
            k for k in self.memory_cache.keys() 
            if self._matches_pattern(k, pattern)
        ]
        for key in keys_to_delete:
            del self.memory_cache[key]
```

### 3.2 Query Result Caching
```python
class QueryCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        
    def cache_key(self, query: str, params: Dict) -> str:
        """Generate cache key for query"""
        # Normalize query
        normalized = query.lower().strip()
        
        # Include relevant params
        param_str = json.dumps(params, sort_keys=True)
        
        # Generate hash
        return f"query:{hashlib.md5(f'{normalized}:{param_str}'.encode()).hexdigest()}"
    
    async def get_or_compute(
        self, 
        query: str, 
        params: Dict,
        compute_fn: Callable,
        ttl: int = 300
    ) -> Any:
        """Get from cache or compute"""
        key = self.cache_key(query, params)
        
        # Try cache first
        cached = await self.cache.get(key)
        if cached:
            return cached
        
        # Compute result
        result = await compute_fn(query, params)
        
        # Cache result
        await self.cache.set(key, result, ttl)
        
        return result
```

## 4. Data Migration & Versioning

### 4.1 Schema Migration System
```python
class MigrationManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations_table = "schema_migrations"
        
    async def initialize(self):
        """Create migrations table"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.migrations_table} (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            await db.commit()
    
    async def migrate(self):
        """Run pending migrations"""
        current_version = await self._get_current_version()
        
        migrations = self._get_migrations()
        
        for version, migration in migrations.items():
            if version > current_version:
                await self._apply_migration(version, migration)
    
    def _get_migrations(self) -> Dict[int, Dict]:
        """Get all migrations"""
        return {
            1: {
                'description': 'Initial schema',
                'up': """
                    CREATE TABLE campaigns (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL
                    );
                """,
                'down': "DROP TABLE campaigns;"
            },
            2: {
                'description': 'Add sessions table',
                'up': """
                    CREATE TABLE sessions (
                        id TEXT PRIMARY KEY,
                        campaign_id TEXT NOT NULL,
                        FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
                    );
                """,
                'down': "DROP TABLE sessions;"
            }
            # Add more migrations as needed
        }
```

### 4.2 Backup Strategy
```python
import shutil
from datetime import datetime

class BackupManager:
    def __init__(self, data_dir: str, backup_dir: str):
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        
    async def backup(self, backup_type: str = 'full') -> str:
        """Create backup of all databases"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{self.backup_dir}/{backup_type}_{timestamp}"
        
        os.makedirs(backup_path, exist_ok=True)
        
        # Backup SQLite
        shutil.copy2(
            f"{self.data_dir}/ttrpg.db",
            f"{backup_path}/ttrpg.db"
        )
        
        # Backup ChromaDB
        shutil.copytree(
            f"{self.data_dir}/chroma",
            f"{backup_path}/chroma"
        )
        
        # Create backup manifest
        manifest = {
            'timestamp': timestamp,
            'type': backup_type,
            'databases': ['sqlite', 'chroma'],
            'size': self._get_dir_size(backup_path)
        }
        
        with open(f"{backup_path}/manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Compress backup
        shutil.make_archive(backup_path, 'zip', backup_path)
        
        # Clean up uncompressed backup
        shutil.rmtree(backup_path)
        
        return f"{backup_path}.zip"
    
    async def restore(self, backup_file: str) -> bool:
        """Restore from backup"""
        # Extract backup
        temp_dir = f"/tmp/restore_{datetime.utcnow().timestamp()}"
        shutil.unpack_archive(backup_file, temp_dir)
        
        # Verify manifest
        with open(f"{temp_dir}/manifest.json") as f:
            manifest = json.load(f)
        
        # Backup current data
        await self.backup('pre_restore')
        
        # Restore databases
        shutil.copy2(f"{temp_dir}/ttrpg.db", f"{self.data_dir}/ttrpg.db")
        shutil.rmtree(f"{self.data_dir}/chroma")
        shutil.copytree(f"{temp_dir}/chroma", f"{self.data_dir}/chroma")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return True
```

## 5. Performance Optimization

### 5.1 Connection Pooling
```python
class ConnectionPool:
    def __init__(self, db_path: str, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = asyncio.Queue(maxsize=pool_size)
        self.semaphore = asyncio.Semaphore(pool_size)
        
    async def initialize(self):
        """Create connection pool"""
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await self.connections.put(conn)
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        async with self.semaphore:
            conn = await self.connections.get()
            try:
                yield conn
            finally:
                await self.connections.put(conn)
```

### 5.2 Query Optimization
```python
class QueryOptimizer:
    def __init__(self):
        self.query_plans = {}
        
    async def analyze_query(self, conn: aiosqlite.Connection, query: str):
        """Analyze query performance"""
        # Get query plan
        plan = await conn.execute(f"EXPLAIN QUERY PLAN {query}")
        plan_text = await plan.fetchall()
        
        # Check for missing indexes
        if any('SCAN' in str(row) for row in plan_text):
            return {
                'optimization': 'missing_index',
                'suggestion': 'Consider adding an index'
            }
        
        return {'optimization': None}
    
    async def optimize_search(self, query: str, filters: Dict) -> str:
        """Optimize search query"""
        # Use full-text search for text queries
        if 'text' in filters:
            return f"""
                SELECT * FROM documents
                WHERE id IN (
                    SELECT docid FROM documents_fts
                    WHERE documents_fts MATCH ?
                )
            """
        
        # Use covering index for common queries
        if set(filters.keys()) == {'campaign_id', 'session_id'}:
            return """
                SELECT * FROM dice_rolls
                WHERE campaign_id = ? AND session_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
            """
        
        return query
```

## Implementation Timeline

### Week 1: Core Database Setup
- [ ] ChromaDB collections
- [ ] SQLite schema
- [ ] Migration system

### Week 2: Storage Optimization
- [ ] PDF processing pipeline
- [ ] Image storage
- [ ] Deduplication

### Week 3: Caching Layer
- [ ] Redis integration
- [ ] Multi-level cache
- [ ] Cache invalidation

### Week 4: Performance & Monitoring
- [ ] Connection pooling
- [ ] Query optimization
- [ ] Backup system
- [ ] Monitoring dashboard

## Monitoring & Maintenance

### Database Health Metrics
```python
class DatabaseMonitor:
    async def get_health_metrics(self) -> Dict:
        return {
            'sqlite': {
                'size_mb': os.path.getsize(self.sqlite_path) / 1024 / 1024,
                'connections': self.pool.connections.qsize(),
                'wal_size': self._get_wal_size()
            },
            'chroma': {
                'collections': len(self.chroma_client.list_collections()),
                'total_embeddings': self._count_embeddings()
            },
            'redis': {
                'memory_mb': await self.redis.memory_usage(),
                'keys': await self.redis.dbsize(),
                'hit_rate': self.cache.cache_stats['hits'] / 
                           (self.cache.cache_stats['hits'] + 
                            self.cache.cache_stats['misses'])
            }
        }
```

## Conclusion

This database strategy provides a robust foundation for the TTRPG MCP Server with efficient storage, fast retrieval, and proper scaling capabilities. The multi-database approach leverages the strengths of each technology while maintaining data consistency and performance.