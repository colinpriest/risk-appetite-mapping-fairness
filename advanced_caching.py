#!/usr/bin/env python3
"""
Advanced Semantic Caching System for LLM Risk Fairness Experiments

This module provides intelligent caching capabilities including:
- Semantic similarity caching to avoid near-duplicate calls
- Distributed caching across multiple machines
- Cache compression and archiving
- Automatic cache warming for common patterns
- Cache analytics and optimization recommendations
"""

import os
import json
import hashlib
import pickle
import gzip
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
import sqlite3

# ML libraries for semantic similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

# Optional distributed caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcached
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False


@dataclass
class CacheConfig:
    """Configuration for advanced caching system."""
    # Basic settings
    cache_dir: str = "advanced_llm_cache"
    max_cache_size_gb: float = 10.0
    compression_enabled: bool = True
    semantic_similarity_threshold: float = 0.85
    
    # Semantic search settings
    embedding_model: str = "all-MiniLM-L6-v2"
    max_semantic_cache_entries: int = 10000
    semantic_index_rebuild_interval: int = 1000  # entries
    
    # Distributed settings
    distributed_backend: str = "none"  # none, redis, memcached
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    
    # Performance settings
    cache_warmup_enabled: bool = True
    precompute_common_patterns: bool = True
    async_cache_operations: bool = True
    
    # Maintenance settings
    cleanup_interval_hours: int = 24
    archive_old_entries_days: int = 30
    max_memory_usage_mb: int = 1000


@dataclass
class CacheEntry:
    """Advanced cache entry with metadata."""
    key: str
    prompt_hash: str
    prompt_text: str
    response_data: Dict[str, Any]
    model: str
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    response_time: float
    token_count: int
    cost: float
    
    # Semantic features
    embedding: Optional[np.ndarray] = None
    similarity_cluster: Optional[int] = None
    
    # Quality metrics
    confidence_score: float = 0.0
    validation_passed: bool = True


class SemanticCacheIndex:
    """Semantic similarity index for intelligent cache matching."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        try:
            self.encoder = SentenceTransformer(config.embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}, falling back to TF-IDF")
            self.encoder = None
            self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            self.embedding_dim = 1000
        
        # Initialize FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.entry_keys = []  # Maps index positions to cache keys
        self.embeddings_cache = {}
        
        self._lock = threading.RLock()
    
    def add_entry(self, cache_key: str, prompt_text: str):
        """Add entry to semantic index."""
        
        with self._lock:
            try:
                # Generate embedding
                if self.encoder:
                    embedding = self.encoder.encode([prompt_text])[0]
                else:
                    # Fallback to TF-IDF
                    if not hasattr(self, '_tfidf_fitted'):
                        # Initialize TF-IDF with some common text
                        self.tfidf.fit([prompt_text])
                        self._tfidf_fitted = True
                    
                    embedding = self.tfidf.transform([prompt_text]).toarray()[0]
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                # Add to FAISS index
                self.index.add(embedding.reshape(1, -1).astype('float32'))
                self.entry_keys.append(cache_key)
                self.embeddings_cache[cache_key] = embedding
                
                self.logger.debug(f"Added semantic index entry for key: {cache_key}")
                
            except Exception as e:
                self.logger.error(f"Failed to add semantic index entry: {e}")
    
    def find_similar(self, prompt_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find similar cached entries."""
        
        with self._lock:
            try:
                if len(self.entry_keys) == 0:
                    return []
                
                # Generate query embedding
                if self.encoder:
                    query_embedding = self.encoder.encode([prompt_text])[0]
                else:
                    if hasattr(self, '_tfidf_fitted'):
                        query_embedding = self.tfidf.transform([prompt_text]).toarray()[0]
                    else:
                        return []
                
                # Normalize
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Search in FAISS index
                k = min(k, len(self.entry_keys))
                similarities, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), k
                )
                
                # Return results with similarity scores
                results = []
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx < len(self.entry_keys):
                        cache_key = self.entry_keys[idx]
                        results.append((cache_key, float(similarity)))
                
                return results
                
            except Exception as e:
                self.logger.error(f"Failed to find similar entries: {e}")
                return []
    
    def remove_entry(self, cache_key: str):
        """Remove entry from semantic index."""
        
        with self._lock:
            try:
                if cache_key in self.embeddings_cache:
                    # Find index position
                    if cache_key in self.entry_keys:
                        idx = self.entry_keys.index(cache_key)
                        
                        # Remove from tracking
                        del self.entry_keys[idx]
                        del self.embeddings_cache[cache_key]
                        
                        # Note: FAISS doesn't support removal, so we rebuild periodically
                        if len(self.entry_keys) % self.config.semantic_index_rebuild_interval == 0:
                            self._rebuild_index()
                        
                        self.logger.debug(f"Removed semantic index entry for key: {cache_key}")
                
            except Exception as e:
                self.logger.error(f"Failed to remove semantic index entry: {e}")
    
    def _rebuild_index(self):
        """Rebuild FAISS index from scratch."""
        
        try:
            self.logger.info("Rebuilding semantic index...")
            
            # Create new index
            new_index = faiss.IndexFlatIP(self.embedding_dim)
            new_keys = []
            
            # Re-add all embeddings
            embeddings_to_add = []
            for key in self.entry_keys:
                if key in self.embeddings_cache:
                    embeddings_to_add.append(self.embeddings_cache[key])
                    new_keys.append(key)
            
            if embeddings_to_add:
                embeddings_matrix = np.vstack(embeddings_to_add).astype('float32')
                new_index.add(embeddings_matrix)
            
            # Replace old index
            self.index = new_index
            self.entry_keys = new_keys
            
            self.logger.info(f"Semantic index rebuilt with {len(new_keys)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild semantic index: {e}")


class DistributedCacheBackend:
    """Distributed caching backend supporting Redis and Memcached."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backend = None
        
        if config.distributed_backend == "redis" and REDIS_AVAILABLE:
            self._init_redis()
        elif config.distributed_backend == "memcached" and MEMCACHED_AVAILABLE:
            self._init_memcached()
    
    def _init_redis(self):
        """Initialize Redis backend."""
        try:
            self.backend = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False  # We handle binary data
            )
            # Test connection
            self.backend.ping()
            self.logger.info("Redis distributed cache initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis: {e}")
            self.backend = None
    
    def _init_memcached(self):
        """Initialize Memcached backend."""
        try:
            self.backend = memcached.Client([f'{self.config.redis_host}:11211'])
            self.logger.info("Memcached distributed cache initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Memcached: {e}")
            self.backend = None
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value from distributed cache."""
        if not self.backend:
            return None
        
        try:
            if self.config.distributed_backend == "redis":
                return self.backend.get(key)
            elif self.config.distributed_backend == "memcached":
                return self.backend.get(key)
        except Exception as e:
            self.logger.error(f"Distributed cache get failed: {e}")
        
        return None
    
    def set(self, key: str, value: bytes, ttl: int = 86400) -> bool:
        """Set value in distributed cache."""
        if not self.backend:
            return False
        
        try:
            if self.config.distributed_backend == "redis":
                return self.backend.setex(key, ttl, value)
            elif self.config.distributed_backend == "memcached":
                return self.backend.set(key, value, time=ttl)
        except Exception as e:
            self.logger.error(f"Distributed cache set failed: {e}")
        
        return False
    
    def delete(self, key: str) -> bool:
        """Delete value from distributed cache."""
        if not self.backend:
            return False
        
        try:
            if self.config.distributed_backend == "redis":
                return bool(self.backend.delete(key))
            elif self.config.distributed_backend == "memcached":
                return self.backend.delete(key)
        except Exception as e:
            self.logger.error(f"Distributed cache delete failed: {e}")
        
        return False


class AdvancedLLMCache:
    """Advanced LLM caching system with semantic similarity and distributed support."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize local database
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Initialize semantic index
        self.semantic_index = SemanticCacheIndex(self.config)
        
        # Initialize distributed backend
        self.distributed_backend = DistributedCacheBackend(self.config)
        
        # Statistics tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'semantic_hits': 0,
            'distributed_hits': 0,
            'total_entries': 0,
            'cache_size_mb': 0.0
        }
        
        # Load existing cache entries into semantic index
        self._warmup_semantic_index()
        
        # Start background maintenance
        self._start_maintenance_thread()
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    prompt_text TEXT,
                    model TEXT,
                    timestamp TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    response_time REAL,
                    token_count INTEGER,
                    cost REAL,
                    file_path TEXT,
                    file_size INTEGER,
                    confidence_score REAL,
                    validation_passed INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON cache_entries(prompt_hash)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model ON cache_entries(model)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            ''')
    
    def _warmup_semantic_index(self):
        """Load existing cache entries into semantic index."""
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    'SELECT key, prompt_text FROM cache_entries ORDER BY last_accessed DESC LIMIT ?',
                    (self.config.max_semantic_cache_entries,)
                )
                
                for key, prompt_text in cursor:
                    self.semantic_index.add_entry(key, prompt_text)
                    
            self.logger.info("Semantic index warmed up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to warm up semantic index: {e}")
    
    def _start_maintenance_thread(self):
        """Start background maintenance thread."""
        
        def maintenance_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval_hours * 3600)
                    self._perform_maintenance()
                except Exception as e:
                    self.logger.error(f"Maintenance thread error: {e}")
        
        thread = threading.Thread(target=maintenance_worker, daemon=True)
        thread.start()
        self.logger.info("Background maintenance thread started")
    
    def get(self, prompt: str, model: str, temperature: float = 0.0, 
           **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response with semantic similarity matching."""
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model, temperature, **kwargs)
        
        # Try exact match first
        exact_result = self._get_exact(cache_key)
        if exact_result:
            self.stats['hits'] += 1
            self._update_access_stats(cache_key)
            return exact_result
        
        # Try semantic similarity match
        if self.config.semantic_similarity_threshold > 0:
            semantic_result = self._get_semantic_similar(prompt, model, temperature, **kwargs)
            if semantic_result:
                self.stats['semantic_hits'] += 1
                return semantic_result
        
        # Try distributed cache
        if self.distributed_backend.backend:
            distributed_result = self._get_distributed(cache_key)
            if distributed_result:
                self.stats['distributed_hits'] += 1
                # Store locally for future access
                self._store_local(cache_key, distributed_result, prompt, model, temperature, **kwargs)
                return distributed_result
        
        self.stats['misses'] += 1
        return None
    
    def set(self, prompt: str, model: str, response: Dict[str, Any], 
           temperature: float = 0.0, response_time: float = 0.0, 
           token_count: int = 0, cost: float = 0.0, **kwargs) -> str:
        """Store response in cache with full metadata."""
        
        cache_key = self._generate_cache_key(prompt, model, temperature, **kwargs)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            prompt_hash=self._hash_prompt(prompt, model, temperature, **kwargs),
            prompt_text=prompt,
            response_data=response,
            model=model,
            timestamp=datetime.now(timezone.utc),
            access_count=1,
            last_accessed=datetime.now(timezone.utc),
            response_time=response_time,
            token_count=token_count,
            cost=cost
        )
        
        # Store locally
        self._store_local_entry(entry)
        
        # Store in distributed cache
        if self.distributed_backend.backend:
            self._store_distributed(cache_key, entry)
        
        # Add to semantic index
        self.semantic_index.add_entry(cache_key, prompt)
        
        self.stats['total_entries'] += 1
        
        return cache_key
    
    def _get_exact(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get exact cache match."""
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    'SELECT file_path FROM cache_entries WHERE key = ?',
                    (cache_key,)
                )
                result = cursor.fetchone()
                
                if result:
                    file_path = Path(result[0])
                    if file_path.exists():
                        return self._load_cache_file(file_path)
                    else:
                        # Clean up stale database entry
                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (cache_key,))
        
        except Exception as e:
            self.logger.error(f"Failed to get exact cache match: {e}")
        
        return None
    
    def _get_semantic_similar(self, prompt: str, model: str, temperature: float, 
                            **kwargs) -> Optional[Dict[str, Any]]:
        """Get semantically similar cached response."""
        
        try:
            similar_entries = self.semantic_index.find_similar(prompt, k=5)
            
            for cache_key, similarity in similar_entries:
                if similarity >= self.config.semantic_similarity_threshold:
                    # Verify model compatibility
                    with sqlite3.connect(str(self.db_path)) as conn:
                        cursor = conn.execute(
                            'SELECT model, file_path FROM cache_entries WHERE key = ?',
                            (cache_key,)
                        )
                        result = cursor.fetchone()
                        
                        if result and result[0] == model:
                            file_path = Path(result[1])
                            if file_path.exists():
                                cached_response = self._load_cache_file(file_path)
                                if cached_response:
                                    self.logger.debug(f"Semantic cache hit (similarity: {similarity:.3f})")
                                    self._update_access_stats(cache_key)
                                    return cached_response
        
        except Exception as e:
            self.logger.error(f"Failed to get semantic similar match: {e}")
        
        return None
    
    def _get_distributed(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from distributed cache."""
        
        try:
            cached_data = self.distributed_backend.get(cache_key)
            if cached_data:
                # Decompress if needed
                if self.config.compression_enabled:
                    cached_data = gzip.decompress(cached_data)
                
                # Deserialize
                return pickle.loads(cached_data)
        
        except Exception as e:
            self.logger.error(f"Failed to get distributed cache match: {e}")
        
        return None
    
    def _store_local_entry(self, entry: CacheEntry):
        """Store cache entry locally."""
        
        try:
            # Create cache file
            cache_file = self.cache_dir / f"{entry.key}.cache"
            
            data_to_store = {
                'response_data': entry.response_data,
                'metadata': asdict(entry)
            }
            
            # Compress if enabled
            if self.config.compression_enabled:
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(data_to_store, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_to_store, f)
            
            # Store metadata in database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, prompt_hash, prompt_text, model, timestamp, access_count, 
                     last_accessed, response_time, token_count, cost, file_path, file_size,
                     confidence_score, validation_passed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.key,
                    entry.prompt_hash,
                    entry.prompt_text,
                    entry.model,
                    entry.timestamp.isoformat(),
                    entry.access_count,
                    entry.last_accessed.isoformat(),
                    entry.response_time,
                    entry.token_count,
                    entry.cost,
                    str(cache_file),
                    cache_file.stat().st_size,
                    entry.confidence_score,
                    int(entry.validation_passed)
                ))
            
            self.logger.debug(f"Stored cache entry: {entry.key}")
            
        except Exception as e:
            self.logger.error(f"Failed to store local cache entry: {e}")
    
    def _store_local(self, cache_key: str, response_data: Dict[str, Any], 
                    prompt: str, model: str, temperature: float, **kwargs):
        """Store response data locally (simplified)."""
        
        entry = CacheEntry(
            key=cache_key,
            prompt_hash=self._hash_prompt(prompt, model, temperature, **kwargs),
            prompt_text=prompt,
            response_data=response_data,
            model=model,
            timestamp=datetime.now(timezone.utc),
            access_count=1,
            last_accessed=datetime.now(timezone.utc),
            response_time=0.0,
            token_count=0,
            cost=0.0
        )
        
        self._store_local_entry(entry)
    
    def _store_distributed(self, cache_key: str, entry: CacheEntry):
        """Store entry in distributed cache."""
        
        try:
            data_to_store = {
                'response_data': entry.response_data,
                'metadata': asdict(entry)
            }
            
            # Serialize
            serialized_data = pickle.dumps(data_to_store)
            
            # Compress if enabled
            if self.config.compression_enabled:
                serialized_data = gzip.compress(serialized_data)
            
            # Store with TTL
            ttl = 7 * 24 * 3600  # 7 days
            success = self.distributed_backend.set(cache_key, serialized_data, ttl)
            
            if success:
                self.logger.debug(f"Stored in distributed cache: {cache_key}")
        
        except Exception as e:
            self.logger.error(f"Failed to store in distributed cache: {e}")
    
    def _load_cache_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load cached response from file."""
        
        try:
            if self.config.compression_enabled and file_path.suffix == '.cache':
                with gzip.open(file_path, 'rb') as f:
                    cached_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    cached_data = pickle.load(f)
            
            return cached_data.get('response_data')
            
        except Exception as e:
            self.logger.error(f"Failed to load cache file {file_path}: {e}")
            return None
    
    def _update_access_stats(self, cache_key: str):
        """Update access statistics for cache entry."""
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, 
                        last_accessed = ? 
                    WHERE key = ?
                ''', (datetime.now(timezone.utc).isoformat(), cache_key))
        
        except Exception as e:
            self.logger.error(f"Failed to update access stats: {e}")
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float, **kwargs) -> str:
        """Generate unique cache key."""
        
        # Create deterministic key from all parameters
        key_data = {
            'prompt': prompt,
            'model': model,
            'temperature': temperature,
            **kwargs
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _hash_prompt(self, prompt: str, model: str, temperature: float, **kwargs) -> str:
        """Generate hash for prompt content."""
        
        content = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _perform_maintenance(self):
        """Perform cache maintenance tasks."""
        
        self.logger.info("Starting cache maintenance...")
        
        try:
            # Clean up old entries
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.archive_old_entries_days)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Find old entries
                cursor = conn.execute(
                    'SELECT key, file_path FROM cache_entries WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                
                old_entries = cursor.fetchall()
                
                # Remove old files and database entries
                for key, file_path in old_entries:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                        self.semantic_index.remove_entry(key)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to remove old cache entry {key}: {e}")
                
                self.logger.info(f"Cleaned up {len(old_entries)} old cache entries")
            
            # Update statistics
            self._update_cache_statistics()
            
        except Exception as e:
            self.logger.error(f"Cache maintenance failed: {e}")
    
    def _update_cache_statistics(self):
        """Update cache statistics."""
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('SELECT COUNT(*), SUM(file_size) FROM cache_entries')
                count, total_size = cursor.fetchone()
                
                self.stats['total_entries'] = count or 0
                self.stats['cache_size_mb'] = (total_size or 0) / (1024 * 1024)
        
        except Exception as e:
            self.logger.error(f"Failed to update cache statistics: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        self._update_cache_statistics()
        
        hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        semantic_rate = self.stats['semantic_hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        distributed_rate = self.stats['distributed_hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'semantic_hit_rate': semantic_rate,
            'distributed_hit_rate': distributed_rate,
            'total_requests': self.stats['hits'] + self.stats['misses'],
            'cache_efficiency': hit_rate,
            'storage_efficiency': self.stats['cache_size_mb'] / max(1, self.stats['total_entries'])
        }
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Analyze cache usage and provide optimization recommendations."""
        
        try:
            recommendations = []
            
            with sqlite3.connect(str(self.db_path)) as conn:
                # Analyze access patterns
                cursor = conn.execute('''
                    SELECT access_count, COUNT(*) as entries, AVG(file_size) as avg_size
                    FROM cache_entries 
                    GROUP BY access_count 
                    ORDER BY access_count DESC
                ''')
                
                access_patterns = cursor.fetchall()
                
                # Find rarely accessed entries
                rarely_accessed = sum(count for acc, count, _ in access_patterns if acc <= 1)
                total_entries = sum(count for _, count, _ in access_patterns)
                
                if rarely_accessed / max(1, total_entries) > 0.5:
                    recommendations.append("Consider increasing semantic similarity threshold to reduce rarely used entries")
                
                # Analyze model distribution
                cursor = conn.execute('''
                    SELECT model, COUNT(*), AVG(cost), SUM(file_size)
                    FROM cache_entries 
                    GROUP BY model
                ''')
                
                model_stats = cursor.fetchall()
                
                # Find most expensive/large models
                for model, count, avg_cost, total_size in model_stats:
                    if avg_cost > 0.01:  # High cost models
                        recommendations.append(f"Model {model} has high average cost (${avg_cost:.4f}) - prioritize caching")
                    
                    if total_size > 50 * 1024 * 1024:  # Large storage models
                        recommendations.append(f"Model {model} uses significant storage - consider compression")
            
            stats = self.get_statistics()
            
            # General recommendations
            if stats['hit_rate'] < 0.3:
                recommendations.append("Low cache hit rate - consider enabling semantic similarity matching")
            
            if stats['cache_size_mb'] > self.config.max_cache_size_gb * 1024:
                recommendations.append("Cache size exceeds configured limit - enable automatic cleanup")
            
            return {
                'cache_stats': stats,
                'access_patterns': access_patterns,
                'model_stats': model_stats,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize cache: {e}")
            return {'error': str(e)}


# Integration with existing experiment system
def create_advanced_cache_client(config: CacheConfig = None) -> AdvancedLLMCache:
    """Factory function to create advanced cache client."""
    
    if config is None:
        config = CacheConfig()
    
    return AdvancedLLMCache(config)


def main():
    """CLI interface for cache management."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced LLM Cache Management")
    parser.add_argument('command', choices=['stats', 'optimize', 'cleanup', 'warmup'])
    parser.add_argument('--cache-dir', default='advanced_llm_cache', 
                       help='Cache directory path')
    parser.add_argument('--semantic-threshold', type=float, default=0.85,
                       help='Semantic similarity threshold')
    parser.add_argument('--distributed-backend', choices=['none', 'redis', 'memcached'],
                       default='none', help='Distributed cache backend')
    
    args = parser.parse_args()
    
    # Create cache configuration
    config = CacheConfig(
        cache_dir=args.cache_dir,
        semantic_similarity_threshold=args.semantic_threshold,
        distributed_backend=args.distributed_backend
    )
    
    # Create cache instance
    cache = AdvancedLLMCache(config)
    
    if args.command == 'stats':
        stats = cache.get_statistics()
        print("Cache Statistics:")
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'optimize':
        optimization = cache.optimize_cache()
        print("Cache Optimization Analysis:")
        print(json.dumps(optimization, indent=2, default=str))
    
    elif args.command == 'cleanup':
        cache._perform_maintenance()
        print("Cache cleanup completed")
    
    elif args.command == 'warmup':
        cache._warmup_semantic_index()
        print("Semantic index warmed up")


if __name__ == "__main__":
    main()