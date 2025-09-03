"""
Token Optimization Strategies with Context Compression and Caching.

This module provides advanced token optimization techniques:
- Intelligent context compression and truncation
- Semantic-aware message pruning
- Request caching and deduplication
- Token usage analytics and recommendations
- Context sliding window management
- Message importance scoring
- Template-based optimization
"""

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from structlog import get_logger

logger = get_logger(__name__)


class MessageType:
    """Classification of message types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class CompressionStrategy:
    """Compression strategies for different scenarios."""
    PRESERVE_RECENT = "preserve_recent"        # Keep most recent messages
    PRESERVE_IMPORTANT = "preserve_important"  # Keep semantically important messages
    SLIDING_WINDOW = "sliding_window"          # Maintain sliding window
    HIERARCHICAL = "hierarchical"              # Compress older messages more aggressively
    SEMANTIC_CLUSTERING = "semantic_clustering" # Group similar messages
    TEMPLATE_BASED = "template_based"          # Use templates for common patterns


class TokenEstimator:
    """Estimate token counts for different providers."""
    
    # Rough token estimation ratios (chars per token)
    PROVIDER_RATIOS = {
        'openai': 4.0,
        'anthropic': 3.5,
        'google': 4.2,
        'default': 4.0
    }
    
    # Overhead tokens for message structure
    MESSAGE_OVERHEAD = {
        'system': 10,
        'user': 8,
        'assistant': 8,
        'tool_call': 15,
        'tool_result': 12
    }
    
    @classmethod
    def estimate_tokens(
        cls,
        text: str,
        provider: str = 'default',
        message_role: Optional[str] = None
    ) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        
        # Base estimation from character count
        ratio = cls.PROVIDER_RATIOS.get(provider, cls.PROVIDER_RATIOS['default'])
        base_tokens = len(text) / ratio
        
        # Add message overhead
        overhead = 0
        if message_role:
            overhead = cls.MESSAGE_OVERHEAD.get(message_role, 5)
        
        return int(base_tokens + overhead)
    
    @classmethod
    def estimate_message_tokens(cls, message: Dict[str, Any], provider: str = 'default') -> int:
        """Estimate tokens for a complete message."""
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        # Handle different content types
        if isinstance(content, str):
            return cls.estimate_tokens(content, provider, role)
        elif isinstance(content, list):
            # Multimodal content
            total_tokens = 0
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        total_tokens += cls.estimate_tokens(item.get('text', ''), provider)
                    elif item.get('type') == 'image':
                        # Rough estimate for image tokens
                        total_tokens += 765  # Standard image token cost
                    elif item.get('type') == 'tool_use':
                        # Tool use overhead
                        total_tokens += cls.estimate_tokens(
                            json.dumps(item.get('input', {})), provider
                        ) + 20
            
            # Add role overhead
            total_tokens += cls.MESSAGE_OVERHEAD.get(role, 5)
            return total_tokens
        
        return 0
    
    @classmethod
    def estimate_conversation_tokens(
        cls,
        messages: List[Dict[str, Any]],
        provider: str = 'default'
    ) -> int:
        """Estimate total tokens for a conversation."""
        total_tokens = 0
        
        for message in messages:
            total_tokens += cls.estimate_message_tokens(message, provider)
        
        # Add conversation overhead (varies by provider)
        conversation_overhead = 10
        return total_tokens + conversation_overhead


class MessageImportanceScorer:
    """Score message importance for compression decisions."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.importance_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def score_messages(self, messages: List[Dict[str, Any]]) -> List[float]:
        """Score importance of each message in the conversation."""
        
        # Extract text content from messages
        texts = []
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                # Extract text from multimodal content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                texts.append(' '.join(text_parts))
            else:
                texts.append('')
        
        if len(texts) < 2:
            return [1.0] * len(messages)
        
        # Calculate TF-IDF vectors
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate importance scores
            scores = []
            
            for i, message in enumerate(messages):
                score = self._calculate_message_score(message, tfidf_matrix[i], i, len(messages))
                scores.append(score)
            
            # Normalize scores
            if scores:
                max_score = max(scores)
                if max_score > 0:
                    scores = [s / max_score for s in scores]
            
            return scores
            
        except Exception as e:
            logger.warning(f"Error scoring message importance: {e}")
            # Fallback to position-based scoring
            return self._fallback_position_scoring(messages)
    
    def _calculate_message_score(
        self,
        message: Dict[str, Any],
        tfidf_vector,
        position: int,
        total_messages: int
    ) -> float:
        """Calculate importance score for a single message."""
        
        base_score = 0.0
        role = message.get('role', 'user')
        
        # Role-based scoring
        role_weights = {
            'system': 1.0,      # System messages are always important
            'user': 0.8,        # User messages are generally important
            'assistant': 0.7,   # Assistant messages vary in importance
            'tool_call': 0.6,   # Tool calls are moderately important
            'tool_result': 0.4  # Tool results are less important
        }
        
        base_score = role_weights.get(role, 0.5)
        
        # Content-based scoring (TF-IDF density)
        if hasattr(tfidf_vector, 'data') and len(tfidf_vector.data) > 0:
            content_score = np.mean(tfidf_vector.data)
            base_score += content_score * 0.5
        
        # Position-based scoring (recent messages more important)
        recency_factor = (total_messages - position) / total_messages
        base_score += recency_factor * 0.3
        
        # Length penalty (very short messages less important)
        content_length = len(str(message.get('content', '')))
        if content_length < 20:
            base_score *= 0.7
        elif content_length > 1000:
            base_score *= 1.2  # Long messages might be more important
        
        # Special patterns that indicate importance
        content_str = str(message.get('content', '')).lower()
        
        # Error messages are important
        if any(keyword in content_str for keyword in ['error', 'exception', 'failed', 'problem']):
            base_score *= 1.3
        
        # Questions are important
        if '?' in content_str:
            base_score *= 1.1
        
        # Instructions/commands are important
        if any(keyword in content_str for keyword in ['please', 'can you', 'i need', 'help me']):
            base_score *= 1.2
        
        return base_score
    
    def _fallback_position_scoring(self, messages: List[Dict[str, Any]]) -> List[float]:
        """Fallback scoring based on position and role."""
        scores = []
        
        for i, message in enumerate(messages):
            role = message.get('role', 'user')
            
            # Base score by role
            if role == 'system':
                score = 1.0
            elif role == 'user':
                score = 0.8
            elif role == 'assistant':
                score = 0.6
            else:
                score = 0.4
            
            # Recency boost
            recency_factor = (len(messages) - i) / len(messages)
            score += recency_factor * 0.5
            
            scores.append(score)
        
        return scores


class SemanticCache:
    """Semantic caching for similar requests."""
    
    def __init__(self, max_cache_size: int = 10000, similarity_threshold: float = 0.85):
        self.cache = {}  # hash -> (response, timestamp, usage_count)
        self.embeddings = {}  # hash -> embedding
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Semantic cache initialized with max size {max_cache_size}")
    
    def _create_request_signature(self, messages: List[Dict[str, Any]], model: str = '') -> str:
        """Create a signature for caching purposes."""
        # Extract key content for hashing
        content_parts = []
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if isinstance(content, str):
                content_parts.append(f"{role}:{content}")
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content_parts.append(f"{role}:{''.join(text_parts)}")
        
        # Create hash
        signature_text = '|'.join(content_parts) + f"|model:{model}"
        return hashlib.sha256(signature_text.encode()).hexdigest()[:16]
    
    def _get_content_embedding(self, messages: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Get content embedding for similarity matching."""
        # Extract text content
        all_text = []
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                all_text.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        all_text.append(item.get('text', ''))
        
        combined_text = ' '.join(all_text)
        
        if not combined_text.strip():
            return None
        
        try:
            # Use TF-IDF for embedding (simple but effective)
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                # Vectorizer is already fitted
                embedding = self.tfidf_vectorizer.transform([combined_text])
            else:
                # Fit vectorizer with current text (not ideal but fallback)
                embedding = self.tfidf_vectorizer.fit_transform([combined_text])
            
            return embedding.toarray()[0]
            
        except Exception as e:
            logger.warning(f"Error creating embedding: {e}")
            return None
    
    def get_cached_response(
        self,
        messages: List[Dict[str, Any]],
        model: str = '',
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available and similar enough."""
        
        request_signature = self._create_request_signature(messages, model)
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # Check exact match first
        if request_signature in self.cache:
            response, timestamp, usage_count = self.cache[request_signature]
            if timestamp >= cutoff_time:
                # Update usage count
                self.cache[request_signature] = (response, timestamp, usage_count + 1)
                self.cache_hits += 1
                logger.debug(f"Cache hit (exact): {request_signature}")
                return response
        
        # Check semantic similarity
        current_embedding = self._get_content_embedding(messages)
        if current_embedding is None:
            self.cache_misses += 1
            return None
        
        best_similarity = 0.0
        best_match = None
        
        # Compare with cached embeddings
        for cached_signature, cached_embedding in self.embeddings.items():
            if cached_signature not in self.cache:
                continue
            
            response, timestamp, usage_count = self.cache[cached_signature]
            if timestamp < cutoff_time:
                continue
            
            # Calculate similarity
            try:
                similarity = cosine_similarity(
                    current_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = (cached_signature, response)
                    
            except Exception as e:
                logger.debug(f"Error calculating similarity: {e}")
                continue
        
        if best_match:
            cached_signature, response = best_match
            # Update usage count
            _, timestamp, usage_count = self.cache[cached_signature]
            self.cache[cached_signature] = (response, timestamp, usage_count + 1)
            self.cache_hits += 1
            logger.debug(f"Cache hit (semantic): {cached_signature}, similarity: {best_similarity:.3f}")
            return response
        
        self.cache_misses += 1
        return None
    
    def cache_response(
        self,
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
        model: str = ''
    ) -> None:
        """Cache a response."""
        
        request_signature = self._create_request_signature(messages, model)
        current_time = time.time()
        
        # Get embedding for semantic matching
        embedding = self._get_content_embedding(messages)
        
        # Store in cache
        self.cache[request_signature] = (response, current_time, 0)
        
        if embedding is not None:
            self.embeddings[request_signature] = embedding
        
        # Cleanup if cache is too large
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()
        
        logger.debug(f"Cached response: {request_signature}")
    
    def _cleanup_cache(self) -> None:
        """Remove old and unused cache entries."""
        current_time = time.time()
        cutoff_time = current_time - (48 * 3600)  # 48 hours
        
        # Remove old entries
        expired_keys = [
            key for key, (_, timestamp, _) in self.cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.embeddings.pop(key, None)
        
        # If still too large, remove least used entries
        if len(self.cache) > self.max_cache_size:
            # Sort by usage count and age
            cache_items = [(key, usage_count, timestamp) 
                          for key, (_, timestamp, usage_count) in self.cache.items()]
            cache_items.sort(key=lambda x: (x[1], x[2]))  # Sort by usage, then timestamp
            
            # Remove least used entries
            entries_to_remove = len(self.cache) - int(self.max_cache_size * 0.8)
            for key, _, _ in cache_items[:entries_to_remove]:
                del self.cache[key]
                self.embeddings.pop(key, None)
        
        logger.debug(f"Cache cleanup completed. Size: {len(self.cache)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'embeddings_stored': len(self.embeddings)
        }


class TokenOptimizer:
    """Main token optimization system."""
    
    def __init__(self):
        self.token_estimator = TokenEstimator()
        self.importance_scorer = MessageImportanceScorer()
        self.semantic_cache = SemanticCache()
        
        # Optimization statistics
        self.optimization_stats = {
            'tokens_saved': 0,
            'compressions_performed': 0,
            'cache_utilization': 0
        }
        
        logger.info("Token Optimizer initialized")
    
    def optimize_conversation(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        strategy: str = CompressionStrategy.PRESERVE_RECENT,
        provider: str = 'default',
        preserve_system: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Optimize conversation to fit within token limits."""
        
        # Estimate current token usage
        current_tokens = self.token_estimator.estimate_conversation_tokens(messages, provider)
        
        optimization_info = {
            'original_tokens': current_tokens,
            'target_tokens': max_tokens,
            'strategy_used': strategy,
            'messages_removed': 0,
            'tokens_saved': 0,
            'compression_ratio': 1.0
        }
        
        # If already within limits, return as-is
        if current_tokens <= max_tokens:
            optimization_info['final_tokens'] = current_tokens
            return messages, optimization_info
        
        # Apply compression strategy
        if strategy == CompressionStrategy.PRESERVE_RECENT:
            optimized_messages = self._compress_preserve_recent(
                messages, max_tokens, provider, preserve_system
            )
        elif strategy == CompressionStrategy.PRESERVE_IMPORTANT:
            optimized_messages = self._compress_preserve_important(
                messages, max_tokens, provider, preserve_system
            )
        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            optimized_messages = self._compress_sliding_window(
                messages, max_tokens, provider, preserve_system
            )
        elif strategy == CompressionStrategy.HIERARCHICAL:
            optimized_messages = self._compress_hierarchical(
                messages, max_tokens, provider, preserve_system
            )
        elif strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
            optimized_messages = self._compress_semantic_clustering(
                messages, max_tokens, provider, preserve_system
            )
        else:
            # Default to preserve_recent
            optimized_messages = self._compress_preserve_recent(
                messages, max_tokens, provider, preserve_system
            )
        
        # Calculate final statistics
        final_tokens = self.token_estimator.estimate_conversation_tokens(optimized_messages, provider)
        
        optimization_info.update({
            'final_tokens': final_tokens,
            'messages_removed': len(messages) - len(optimized_messages),
            'tokens_saved': current_tokens - final_tokens,
            'compression_ratio': final_tokens / current_tokens if current_tokens > 0 else 1.0
        })
        
        # Update global statistics
        self.optimization_stats['tokens_saved'] += optimization_info['tokens_saved']
        self.optimization_stats['compressions_performed'] += 1
        
        logger.info(f"Optimized conversation: {current_tokens} -> {final_tokens} tokens ({strategy})")
        
        return optimized_messages, optimization_info
    
    def _compress_preserve_recent(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        provider: str,
        preserve_system: bool
    ) -> List[Dict[str, Any]]:
        """Compress by keeping most recent messages."""
        
        system_messages = []
        other_messages = []
        
        # Separate system messages if preserving
        if preserve_system:
            for message in messages:
                if message.get('role') == 'system':
                    system_messages.append(message)
                else:
                    other_messages.append(message)
        else:
            other_messages = messages
        
        # Calculate system message tokens
        system_tokens = sum(
            self.token_estimator.estimate_message_tokens(msg, provider)
            for msg in system_messages
        )
        
        # Available tokens for other messages
        available_tokens = max_tokens - system_tokens
        
        if available_tokens <= 0:
            # System messages already exceed limit
            return system_messages
        
        # Add recent messages until we hit the limit
        selected_messages = []
        current_tokens = 0
        
        for message in reversed(other_messages):
            message_tokens = self.token_estimator.estimate_message_tokens(message, provider)
            
            if current_tokens + message_tokens <= available_tokens:
                selected_messages.insert(0, message)
                current_tokens += message_tokens
            else:
                break
        
        return system_messages + selected_messages
    
    def _compress_preserve_important(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        provider: str,
        preserve_system: bool
    ) -> List[Dict[str, Any]]:
        """Compress by keeping most important messages."""
        
        system_messages = []
        scoreable_messages = []
        
        # Separate system messages
        if preserve_system:
            for message in messages:
                if message.get('role') == 'system':
                    system_messages.append(message)
                else:
                    scoreable_messages.append(message)
        else:
            scoreable_messages = messages
        
        if not scoreable_messages:
            return system_messages
        
        # Score messages by importance
        importance_scores = self.importance_scorer.score_messages(scoreable_messages)
        
        # Create list of (message, score, tokens) tuples
        message_data = []
        for message, score in zip(scoreable_messages, importance_scores):
            tokens = self.token_estimator.estimate_message_tokens(message, provider)
            message_data.append((message, score, tokens))
        
        # Sort by importance score (descending)
        message_data.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate system message tokens
        system_tokens = sum(
            self.token_estimator.estimate_message_tokens(msg, provider)
            for msg in system_messages
        )
        
        available_tokens = max_tokens - system_tokens
        
        # Select messages by importance until token limit
        selected_data = []
        current_tokens = 0
        
        for message, score, tokens in message_data:
            if current_tokens + tokens <= available_tokens:
                selected_data.append((message, score, tokens))
                current_tokens += tokens
            else:
                # Try to truncate this message if it's important enough
                if score > 0.7 and current_tokens < available_tokens * 0.9:
                    truncated_message = self._truncate_message(
                        message, available_tokens - current_tokens, provider
                    )
                    if truncated_message:
                        selected_data.append((truncated_message, score, available_tokens - current_tokens))
                        break
        
        # Sort selected messages back to chronological order
        # We need to find original positions
        selected_messages = [data[0] for data in selected_data]
        ordered_messages = []
        
        for original_message in scoreable_messages:
            if any(msg is original_message for msg in selected_messages):
                ordered_messages.append(original_message)
        
        return system_messages + ordered_messages
    
    def _compress_sliding_window(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        provider: str,
        preserve_system: bool
    ) -> List[Dict[str, Any]]:
        """Compress using sliding window approach."""
        
        # For sliding window, we keep some old context and recent messages
        system_messages = []
        other_messages = []
        
        if preserve_system:
            for message in messages:
                if message.get('role') == 'system':
                    system_messages.append(message)
                else:
                    other_messages.append(message)
        else:
            other_messages = messages
        
        if len(other_messages) <= 2:
            return messages  # Too few messages to compress effectively
        
        # Calculate tokens
        system_tokens = sum(
            self.token_estimator.estimate_message_tokens(msg, provider)
            for msg in system_messages
        )
        
        available_tokens = max_tokens - system_tokens
        
        # Reserve 70% for recent messages, 30% for early context
        recent_tokens_budget = int(available_tokens * 0.7)
        context_tokens_budget = available_tokens - recent_tokens_budget
        
        # Get recent messages
        recent_messages = []
        recent_tokens = 0
        
        for message in reversed(other_messages):
            message_tokens = self.token_estimator.estimate_message_tokens(message, provider)
            if recent_tokens + message_tokens <= recent_tokens_budget:
                recent_messages.insert(0, message)
                recent_tokens += message_tokens
            else:
                break
        
        # Get early context (first few messages)
        context_messages = []
        context_tokens = 0
        early_messages = other_messages[:len(other_messages) - len(recent_messages)]
        
        for message in early_messages:
            message_tokens = self.token_estimator.estimate_message_tokens(message, provider)
            if context_tokens + message_tokens <= context_tokens_budget:
                context_messages.append(message)
                context_tokens += message_tokens
            else:
                break
        
        # Combine all parts
        result = system_messages + context_messages + recent_messages
        
        # Add separator if there's a gap
        if context_messages and recent_messages and len(context_messages) + len(recent_messages) < len(other_messages):
            separator = {
                'role': 'system',
                'content': f'[... {len(other_messages) - len(context_messages) - len(recent_messages)} messages omitted ...]'
            }
            result = system_messages + context_messages + [separator] + recent_messages
        
        return result
    
    def _compress_hierarchical(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        provider: str,
        preserve_system: bool
    ) -> List[Dict[str, Any]]:
        """Compress using hierarchical strategy (compress older messages more)."""
        
        # Similar to preserve_important but with age-based compression levels
        system_messages = []
        other_messages = []
        
        if preserve_system:
            for message in messages:
                if message.get('role') == 'system':
                    system_messages.append(message)
                else:
                    other_messages.append(message)
        else:
            other_messages = messages
        
        if not other_messages:
            return system_messages
        
        # Divide messages into age groups
        num_messages = len(other_messages)
        
        # Recent (last 25%), semi-recent (25-50%), older (50-75%), oldest (75-100%)
        recent_start = int(num_messages * 0.75)
        semi_recent_start = int(num_messages * 0.5)
        older_start = int(num_messages * 0.25)
        
        recent_messages = other_messages[recent_start:]
        semi_recent_messages = other_messages[semi_recent_start:recent_start]
        older_messages = other_messages[older_start:semi_recent_start]
        oldest_messages = other_messages[:older_start]
        
        # Calculate system tokens
        system_tokens = sum(
            self.token_estimator.estimate_message_tokens(msg, provider)
            for msg in system_messages
        )
        
        available_tokens = max_tokens - system_tokens
        
        # Allocate tokens: Recent(50%), Semi-recent(30%), Older(15%), Oldest(5%)
        token_budgets = {
            'recent': int(available_tokens * 0.5),
            'semi_recent': int(available_tokens * 0.3),
            'older': int(available_tokens * 0.15),
            'oldest': int(available_tokens * 0.05)
        }
        
        # Select messages from each group
        selected_groups = {}
        
        for group_name, group_messages in [
            ('recent', recent_messages),
            ('semi_recent', semi_recent_messages),
            ('older', older_messages),
            ('oldest', oldest_messages)
        ]:
            selected_groups[group_name] = self._select_messages_by_budget(
                group_messages, token_budgets[group_name], provider
            )
        
        # Combine groups in chronological order
        result = system_messages
        for group_messages in [selected_groups['oldest'], selected_groups['older'], 
                             selected_groups['semi_recent'], selected_groups['recent']]:
            result.extend(group_messages)
        
        return result
    
    def _compress_semantic_clustering(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        provider: str,
        preserve_system: bool
    ) -> List[Dict[str, Any]]:
        """Compress using semantic clustering to group similar messages."""
        
        # This is a simplified version - full implementation would use proper clustering
        # For now, use importance-based selection with some clustering heuristics
        
        system_messages = []
        other_messages = []
        
        if preserve_system:
            for message in messages:
                if message.get('role') == 'system':
                    system_messages.append(message)
                else:
                    other_messages.append(message)
        else:
            other_messages = messages
        
        # Group messages by similarity (simplified approach)
        message_clusters = self._simple_cluster_messages(other_messages)
        
        # Select representative messages from each cluster
        system_tokens = sum(
            self.token_estimator.estimate_message_tokens(msg, provider)
            for msg in system_messages
        )
        
        available_tokens = max_tokens - system_tokens
        tokens_per_cluster = available_tokens // max(1, len(message_clusters))
        
        selected_messages = []
        for cluster in message_clusters:
            cluster_selected = self._select_messages_by_budget(
                cluster, tokens_per_cluster, provider
            )
            selected_messages.extend(cluster_selected)
        
        # Sort back to chronological order
        chronological_selected = []
        for original_message in other_messages:
            if original_message in selected_messages:
                chronological_selected.append(original_message)
        
        return system_messages + chronological_selected
    
    def _simple_cluster_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Simple clustering based on message roles and keywords."""
        
        clusters = []
        current_cluster = []
        last_role = None
        
        for message in messages:
            role = message.get('role', 'user')
            
            # Start new cluster if role changes or cluster gets too big
            if role != last_role or len(current_cluster) >= 5:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [message]
                last_role = role
            else:
                current_cluster.append(message)
        
        # Add final cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _select_messages_by_budget(
        self,
        messages: List[Dict[str, Any]],
        token_budget: int,
        provider: str
    ) -> List[Dict[str, Any]]:
        """Select messages that fit within token budget."""
        
        if not messages:
            return []
        
        # If budget is very small, take most recent
        if token_budget <= 100:
            for message in reversed(messages):
                tokens = self.token_estimator.estimate_message_tokens(message, provider)
                if tokens <= token_budget:
                    return [message]
            return []
        
        # Score messages and select best ones that fit
        importance_scores = self.importance_scorer.score_messages(messages)
        
        message_data = []
        for message, score in zip(messages, importance_scores):
            tokens = self.token_estimator.estimate_message_tokens(message, provider)
            message_data.append((message, score, tokens))
        
        # Sort by importance
        message_data.sort(key=lambda x: x[1], reverse=True)
        
        # Select messages that fit
        selected = []
        current_tokens = 0
        
        for message, score, tokens in message_data:
            if current_tokens + tokens <= token_budget:
                selected.append(message)
                current_tokens += tokens
            else:
                # Try to truncate if important and budget allows
                if score > 0.6 and current_tokens < token_budget * 0.8:
                    truncated = self._truncate_message(
                        message, token_budget - current_tokens, provider
                    )
                    if truncated:
                        selected.append(truncated)
                        break
        
        return selected
    
    def _truncate_message(
        self,
        message: Dict[str, Any],
        max_tokens: int,
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """Truncate a message to fit within token limit."""
        
        if max_tokens <= 20:  # Too small to truncate meaningfully
            return None
        
        content = message.get('content', '')
        if not isinstance(content, str):
            return None  # Can't truncate non-string content easily
        
        # Reserve tokens for role overhead and truncation marker
        available_tokens = max_tokens - 15
        
        # Estimate how many characters we can keep
        ratio = self.token_estimator.PROVIDER_RATIOS.get(provider, 4.0)
        max_chars = int(available_tokens * ratio)
        
        if len(content) <= max_chars:
            return message  # Already fits
        
        # Truncate and add marker
        truncated_content = content[:max_chars - 20] + "... [truncated]"
        
        truncated_message = message.copy()
        truncated_message['content'] = truncated_content
        
        return truncated_message
    
    def get_optimization_recommendations(
        self,
        messages: List[Dict[str, Any]],
        provider: str = 'default'
    ) -> Dict[str, Any]:
        """Get recommendations for optimizing token usage."""
        
        current_tokens = self.token_estimator.estimate_conversation_tokens(messages, provider)
        
        recommendations = {
            'current_tokens': current_tokens,
            'analysis': {
                'message_count': len(messages),
                'average_tokens_per_message': current_tokens / max(1, len(messages)),
                'role_distribution': defaultdict(int),
                'long_messages': []
            },
            'optimization_opportunities': [],
            'estimated_savings': {}
        }
        
        # Analyze message distribution
        for i, message in enumerate(messages):
            role = message.get('role', 'unknown')
            recommendations['analysis']['role_distribution'][role] += 1
            
            message_tokens = self.token_estimator.estimate_message_tokens(message, provider)
            
            # Flag long messages
            if message_tokens > 500:
                recommendations['analysis']['long_messages'].append({
                    'index': i,
                    'role': role,
                    'tokens': message_tokens,
                    'content_length': len(str(message.get('content', '')))
                })
        
        # Generate optimization opportunities
        if len(messages) > 10:
            recommendations['optimization_opportunities'].append({
                'type': 'conversation_length',
                'description': 'Consider using sliding window or importance-based compression',
                'potential_savings': 'Up to 40% token reduction'
            })
        
        if recommendations['analysis']['long_messages']:
            avg_long_tokens = sum(m['tokens'] for m in recommendations['analysis']['long_messages'])
            recommendations['optimization_opportunities'].append({
                'type': 'message_truncation',
                'description': f"Truncate {len(recommendations['analysis']['long_messages'])} long messages",
                'potential_savings': f'{avg_long_tokens} tokens'
            })
        
        # Check for repetitive content
        if self._has_repetitive_content(messages):
            recommendations['optimization_opportunities'].append({
                'type': 'deduplication',
                'description': 'Remove or merge similar messages',
                'potential_savings': 'Up to 20% token reduction'
            })
        
        # Cache recommendations
        cache_stats = self.semantic_cache.get_cache_stats()
        if cache_stats['hit_rate'] < 0.5:
            recommendations['optimization_opportunities'].append({
                'type': 'caching',
                'description': 'Enable semantic caching for similar requests',
                'potential_savings': f"Current hit rate: {cache_stats['hit_rate']:.1%}"
            })
        
        return recommendations
    
    def _has_repetitive_content(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if conversation has repetitive content."""
        
        if len(messages) < 4:
            return False
        
        # Simple check for repeated phrases
        content_parts = []
        for message in messages:
            content = str(message.get('content', ''))
            if len(content) > 50:  # Only check substantial messages
                content_parts.append(content.lower())
        
        if len(content_parts) < 3:
            return False
        
        # Check for similar content
        similar_count = 0
        for i in range(len(content_parts)):
            for j in range(i + 1, len(content_parts)):
                # Simple similarity check
                common_words = set(content_parts[i].split()) & set(content_parts[j].split())
                if len(common_words) > 10:  # Many common words
                    similar_count += 1
        
        return similar_count > len(content_parts) * 0.3  # More than 30% similar pairs
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        
        cache_stats = self.semantic_cache.get_cache_stats()
        
        return {
            'total_tokens_saved': self.optimization_stats['tokens_saved'],
            'compressions_performed': self.optimization_stats['compressions_performed'],
            'average_tokens_saved_per_compression': (
                self.optimization_stats['tokens_saved'] / max(1, self.optimization_stats['compressions_performed'])
            ),
            'cache_statistics': cache_stats,
            'supported_strategies': [
                CompressionStrategy.PRESERVE_RECENT,
                CompressionStrategy.PRESERVE_IMPORTANT,
                CompressionStrategy.SLIDING_WINDOW,
                CompressionStrategy.HIERARCHICAL,
                CompressionStrategy.SEMANTIC_CLUSTERING
            ]
        }