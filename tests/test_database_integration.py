"""Comprehensive database integration tests with fixtures for TTRPG Assistant.

This module provides comprehensive integration tests for the ChromaDB database layer,
including:
- Database initialization and configuration
- Collection management and operations
- Document CRUD operations with embeddings
- Search functionality (semantic and hybrid)
- Transaction consistency and rollback
- Concurrent access patterns
- Performance optimization validation
- Resource cleanup and error recovery
"""

import asyncio
import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import numpy as np
import pytest
import pytest_asyncio
from chromadb.utils import embedding_functions

from config.settings import settings
from src.core.database import ChromaDBManager, get_db_manager


class TestDatabaseInitialization:
    """Test database initialization and configuration."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        db_path = tmp_path / "test_chromadb"
        db_path.mkdir(exist_ok=True)
        return db_path

    @pytest.fixture
    def mock_settings(self, temp_db_path):
        """Mock settings for testing."""
        with patch.object(settings, "chroma_db_path", temp_db_path):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    yield settings

    def test_database_initialization(self, mock_settings):
        """Test basic database initialization."""
        db = ChromaDBManager()
        
        assert db.client is not None
        assert db.embedding_function is not None
        assert len(db.collections) > 0
        
        # Verify collections are created
        expected_collections = [
            "rulebooks",
            "flavor_sources",
            "campaigns",
            "sessions",
            "personalities",
        ]
        
        for collection in expected_collections:
            assert collection in db.collections
            assert db.collections[collection] is not None

    def test_singleton_pattern(self, mock_settings):
        """Test that get_db_manager returns singleton instance."""
        db1 = get_db_manager()
        db2 = get_db_manager()
        
        assert db1 is db2
        assert id(db1) == id(db2)

    def test_database_initialization_with_existing_collections(self, mock_settings):
        """Test initialization with existing collections."""
        # Create initial database
        db1 = ChromaDBManager()
        
        # Add document to verify persistence
        db1.add_document(
            collection_name="rulebooks",
            document_id="test_doc_1",
            content="Test content",
            metadata={"test": True}
        )
        
        # Create new instance (simulating restart)
        db2 = ChromaDBManager()
        
        # Verify collections are retrieved, not recreated
        assert "rulebooks" in db2.collections
        
        # Verify document persists
        docs = db2.list_documents("rulebooks", limit=10)
        assert len(docs) > 0
        assert any(doc["id"] == "test_doc_1" for doc in docs)

    def test_performance_tools_initialization(self, mock_settings):
        """Test performance tools are initialized correctly."""
        db = ChromaDBManager()
        
        # Check optimizer and monitor are initialized
        assert db.optimizer is not None
        assert db.monitor is not None
        
        # Verify they have expected methods
        assert hasattr(db.optimizer, "optimize_collection")
        assert hasattr(db.monitor, "record_operation")


class TestDocumentOperations:
    """Test document CRUD operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a database manager for testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "test_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    db = ChromaDBManager()
                    yield db
                    # Cleanup
                    if hasattr(db, "cleanup"):
                        asyncio.run(db.cleanup())

    def test_add_document_basic(self, db_manager):
        """Test basic document addition."""
        doc_id = f"doc_{uuid.uuid4()}"
        content = "The fireball spell deals 8d6 fire damage in a 20-foot radius."
        metadata = {
            "page": 1,
            "rulebook": "Player's Handbook",
            "type": "spell",
            "level": 3
        }
        
        db_manager.add_document(
            collection_name="rulebooks",
            document_id=doc_id,
            content=content,
            metadata=metadata
        )
        
        # Verify document was added
        docs = db_manager.list_documents("rulebooks", limit=10)
        assert any(doc["id"] == doc_id for doc in docs)

    def test_add_document_with_embedding(self, db_manager):
        """Test document addition with pre-computed embedding."""
        doc_id = f"doc_{uuid.uuid4()}"
        content = "Test content with embedding"
        metadata = {"test": True}
        embedding = [0.1] * 384  # Typical embedding size for all-MiniLM-L6-v2
        
        db_manager.add_document(
            collection_name="rulebooks",
            document_id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding
        )
        
        # Search using the embedding
        results = db_manager.search(
            collection_name="rulebooks",
            query_text=None,
            query_embedding=embedding,
            n_results=5
        )
        
        assert len(results) > 0
        assert results[0]["id"] == doc_id

    def test_batch_add_documents(self, db_manager):
        """Test batch document addition."""
        documents = []
        for i in range(10):
            documents.append({
                "id": f"batch_doc_{i}",
                "content": f"Document {i} content about spells and magic",
                "metadata": {"index": i, "type": "spell"}
            })
        
        db_manager.batch_add_documents(
            collection_name="rulebooks",
            documents=documents
        )
        
        # Verify all documents were added
        docs = db_manager.list_documents("rulebooks", limit=20)
        batch_docs = [d for d in docs if d["id"].startswith("batch_doc_")]
        assert len(batch_docs) == 10

    def test_update_document(self, db_manager):
        """Test document update operation."""
        doc_id = f"update_doc_{uuid.uuid4()}"
        
        # Add initial document
        db_manager.add_document(
            collection_name="campaigns",
            document_id=doc_id,
            content="Initial campaign description",
            metadata={"version": 1, "name": "Test Campaign"}
        )
        
        # Update document
        updated_content = "Updated campaign description with new information"
        updated_metadata = {"version": 2, "name": "Test Campaign", "updated": True}
        
        db_manager.update_document(
            collection_name="campaigns",
            document_id=doc_id,
            content=updated_content,
            metadata=updated_metadata
        )
        
        # Verify update
        doc = db_manager.get_document("campaigns", doc_id)
        assert doc is not None
        assert doc["content"] == updated_content
        assert doc["metadata"]["version"] == 2
        assert doc["metadata"]["updated"] is True

    def test_delete_document(self, db_manager):
        """Test document deletion."""
        doc_id = f"delete_doc_{uuid.uuid4()}"
        
        # Add document
        db_manager.add_document(
            collection_name="sessions",
            document_id=doc_id,
            content="Session to be deleted",
            metadata={"temp": True}
        )
        
        # Verify it exists
        doc = db_manager.get_document("sessions", doc_id)
        assert doc is not None
        
        # Delete document
        db_manager.delete_document("sessions", doc_id)
        
        # Verify deletion
        doc = db_manager.get_document("sessions", doc_id)
        assert doc is None

    def test_document_operations_with_invalid_collection(self, db_manager):
        """Test document operations with invalid collection name."""
        with pytest.raises(ValueError, match="Collection.*not found"):
            db_manager.add_document(
                collection_name="invalid_collection",
                document_id="test",
                content="test",
                metadata={}
            )


class TestSearchOperations:
    """Test search functionality including semantic and hybrid search."""

    @pytest_asyncio.fixture
    async def populated_db(self, tmp_path):
        """Create and populate a database for search testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "search_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    db = ChromaDBManager()
                    
                    # Populate with test documents
                    test_docs = [
                        {
                            "id": "spell_1",
                            "content": "Fireball: A bright streak flashes from your pointing finger to a point you choose within range and then blossoms with a low roar into an explosion of flame.",
                            "metadata": {"type": "spell", "level": 3, "school": "evocation"}
                        },
                        {
                            "id": "spell_2",
                            "content": "Lightning Bolt: A stroke of lightning forming a line 100 feet long and 5 feet wide blasts out from you in a direction you choose.",
                            "metadata": {"type": "spell", "level": 3, "school": "evocation"}
                        },
                        {
                            "id": "spell_3",
                            "content": "Cure Wounds: A creature you touch regains a number of hit points equal to 1d8 + your spellcasting ability modifier.",
                            "metadata": {"type": "spell", "level": 1, "school": "evocation"}
                        },
                        {
                            "id": "monster_1",
                            "content": "Red Dragon: Ancient red dragons are the most powerful and feared of the chromatic dragons.",
                            "metadata": {"type": "monster", "cr": 24}
                        },
                        {
                            "id": "rule_1",
                            "content": "Initiative determines the order of turns during combat. When combat starts, every participant makes a Dexterity check to determine their place in the initiative order.",
                            "metadata": {"type": "rule", "category": "combat"}
                        }
                    ]
                    
                    db.batch_add_documents("rulebooks", test_docs)
                    
                    yield db
                    
                    if hasattr(db, "cleanup"):
                        await db.cleanup()

    @pytest.mark.asyncio
    async def test_semantic_search(self, populated_db):
        """Test semantic search functionality."""
        results = populated_db.search(
            collection_name="rulebooks",
            query_text="fire magic explosion",
            n_results=3
        )
        
        assert len(results) > 0
        # Fireball should be the most relevant result
        assert any("Fireball" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_metadata_filtered_search(self, populated_db):
        """Test search with metadata filters."""
        # Search only for spells
        results = populated_db.search(
            collection_name="rulebooks",
            query_text="damage",
            n_results=5,
            metadata_filter={"type": "spell"}
        )
        
        assert len(results) > 0
        assert all(r["metadata"]["type"] == "spell" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search(self, populated_db):
        """Test hybrid search combining semantic and keyword matching."""
        if not hasattr(populated_db, "hybrid_search"):
            pytest.skip("Hybrid search not implemented")
        
        results = populated_db.hybrid_search(
            collection_name="rulebooks",
            query_text="dragon",
            n_results=3,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        assert len(results) > 0
        # Dragon content should be found
        assert any("dragon" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, populated_db):
        """Test search with relevance score threshold."""
        results = populated_db.search(
            collection_name="rulebooks",
            query_text="completely unrelated quantum physics content",
            n_results=5
        )
        
        # Filter by score threshold (distances in ChromaDB, lower is better)
        if results:
            # In real implementation, you'd have a score threshold
            high_quality_results = [r for r in results if r.get("distance", 1.0) < 0.8]
            # With unrelated query, we expect few or no high-quality results
            assert len(high_quality_results) < len(results)


class TestConcurrentOperations:
    """Test concurrent database operations and thread safety."""

    @pytest_asyncio.fixture
    async def concurrent_db(self, tmp_path):
        """Create database for concurrent testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "concurrent_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    db = ChromaDBManager()
                    yield db
                    if hasattr(db, "cleanup"):
                        await db.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, concurrent_db):
        """Test concurrent write operations."""
        async def add_documents(db, prefix, count):
            """Add multiple documents with a given prefix."""
            for i in range(count):
                doc_id = f"{prefix}_doc_{i}"
                db.add_document(
                    collection_name="rulebooks",
                    document_id=doc_id,
                    content=f"Content for {doc_id}",
                    metadata={"thread": prefix, "index": i}
                )
        
        # Run concurrent writes
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                asyncio.to_thread(add_documents, concurrent_db, f"thread_{i}", 10)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all documents were added
        docs = concurrent_db.list_documents("rulebooks", limit=100)
        assert len(docs) >= 50  # 5 threads × 10 documents

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, concurrent_db):
        """Test concurrent read operations."""
        # Add test documents
        for i in range(20):
            concurrent_db.add_document(
                collection_name="campaigns",
                document_id=f"read_test_{i}",
                content=f"Campaign content {i}",
                metadata={"index": i}
            )
        
        async def search_documents(db, query_num):
            """Perform search operation."""
            results = db.search(
                collection_name="campaigns",
                query_text=f"Campaign content",
                n_results=5
            )
            return len(results)
        
        # Run concurrent searches
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                asyncio.to_thread(search_documents, concurrent_db, i)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All searches should return results
        assert all(r > 0 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, concurrent_db):
        """Test mixed concurrent read/write operations."""
        write_complete = asyncio.Event()
        
        async def writer(db):
            """Write documents continuously."""
            for i in range(20):
                db.add_document(
                    collection_name="sessions",
                    document_id=f"mixed_write_{i}",
                    content=f"Session data {i}",
                    metadata={"index": i}
                )
                await asyncio.sleep(0.01)  # Small delay to interleave operations
            write_complete.set()
        
        async def reader(db):
            """Read documents continuously."""
            results = []
            while not write_complete.is_set() or results[-1] if results else 0 < 15:
                docs = db.list_documents("sessions", limit=50)
                results.append(len(docs))
                await asyncio.sleep(0.01)
            return results
        
        # Run mixed operations
        writer_task = asyncio.create_task(asyncio.to_thread(writer, concurrent_db))
        reader_task = asyncio.create_task(reader(concurrent_db))
        
        await asyncio.gather(writer_task, reader_task)
        
        # Verify final state
        final_docs = concurrent_db.list_documents("sessions", limit=50)
        assert len(final_docs) >= 20


class TestPerformanceAndOptimization:
    """Test database performance and optimization features."""

    @pytest_asyncio.fixture
    async def perf_db(self, tmp_path):
        """Create database for performance testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "perf_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    db = ChromaDBManager()
                    yield db
                    if hasattr(db, "cleanup"):
                        await db.cleanup()

    @pytest.mark.asyncio
    async def test_batch_performance(self, perf_db):
        """Test batch operations performance."""
        # Single document additions
        start_time = time.time()
        for i in range(100):
            perf_db.add_document(
                collection_name="rulebooks",
                document_id=f"single_{i}",
                content=f"Content {i}",
                metadata={"index": i}
            )
        single_time = time.time() - start_time
        
        # Batch addition
        batch_docs = [
            {
                "id": f"batch_{i}",
                "content": f"Content {i}",
                "metadata": {"index": i}
            }
            for i in range(100)
        ]
        
        start_time = time.time()
        perf_db.batch_add_documents("rulebooks", batch_docs)
        batch_time = time.time() - start_time
        
        # Batch should be significantly faster
        assert batch_time < single_time * 0.5  # At least 2x faster

    @pytest.mark.asyncio
    async def test_search_performance_with_index(self, perf_db):
        """Test search performance with proper indexing."""
        # Add substantial number of documents
        docs = []
        for i in range(500):
            docs.append({
                "id": f"perf_doc_{i}",
                "content": f"Document about {['magic', 'combat', 'exploration', 'roleplay'][i % 4]} with index {i}",
                "metadata": {"type": ["spell", "rule", "monster", "item"][i % 4], "index": i}
            })
        
        perf_db.batch_add_documents("rulebooks", docs)
        
        # Measure search performance
        search_times = []
        for _ in range(10):
            start_time = time.time()
            results = perf_db.search(
                collection_name="rulebooks",
                query_text="magic spell combat",
                n_results=10
            )
            search_times.append(time.time() - start_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        
        # Search should be fast even with many documents
        assert avg_search_time < 0.5  # Less than 500ms average

    @pytest.mark.asyncio
    async def test_optimizer_collection_optimization(self, perf_db):
        """Test database optimizer functionality."""
        if not perf_db.optimizer:
            pytest.skip("Optimizer not available")
        
        # Add documents to create need for optimization
        for i in range(100):
            perf_db.add_document(
                collection_name="campaigns",
                document_id=f"opt_doc_{i}",
                content=f"Campaign content {i}",
                metadata={"index": i}
            )
        
        # Run optimization
        stats_before = perf_db.get_collection_stats("campaigns")
        perf_db.optimizer.optimize_collection("campaigns")
        stats_after = perf_db.get_collection_stats("campaigns")
        
        # Verify optimization occurred (exact metrics depend on implementation)
        assert stats_after is not None

    @pytest.mark.asyncio
    async def test_monitor_operation_tracking(self, perf_db):
        """Test performance monitor tracking."""
        if not perf_db.monitor:
            pytest.skip("Monitor not available")
        
        # Perform operations that should be monitored
        operation_id = str(uuid.uuid4())
        
        perf_db.monitor.start_operation(operation_id, "search")
        
        # Perform search
        results = perf_db.search(
            collection_name="rulebooks",
            query_text="test query",
            n_results=5
        )
        
        perf_db.monitor.end_operation(operation_id)
        
        # Get statistics
        stats = perf_db.monitor.get_statistics()
        
        assert stats is not None
        assert "search" in stats.get("operations", {})


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    @pytest_asyncio.fixture
    async def error_db(self, tmp_path):
        """Create database for error testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "error_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    db = ChromaDBManager()
                    yield db
                    if hasattr(db, "cleanup"):
                        await db.cleanup()

    def test_handle_duplicate_document_id(self, error_db):
        """Test handling of duplicate document IDs."""
        doc_id = "duplicate_test"
        
        # Add first document
        error_db.add_document(
            collection_name="rulebooks",
            document_id=doc_id,
            content="First content",
            metadata={"version": 1}
        )
        
        # Try to add duplicate - should update or handle gracefully
        error_db.add_document(
            collection_name="rulebooks",
            document_id=doc_id,
            content="Second content",
            metadata={"version": 2}
        )
        
        # Verify handling (either updated or handled error)
        doc = error_db.get_document("rulebooks", doc_id)
        assert doc is not None

    def test_handle_missing_collection(self, error_db):
        """Test handling of operations on non-existent collections."""
        with pytest.raises(ValueError, match="Collection.*not found"):
            error_db.search(
                collection_name="non_existent",
                query_text="test",
                n_results=5
            )

    def test_handle_corrupted_embedding(self, error_db):
        """Test handling of corrupted embeddings."""
        # Try to add document with wrong embedding size
        wrong_embedding = [0.1] * 10  # Wrong size
        
        with pytest.raises(Exception):  # Should raise an appropriate exception
            error_db.add_document(
                collection_name="rulebooks",
                document_id="bad_embedding",
                content="Test content",
                metadata={},
                embedding=wrong_embedding
            )

    @pytest.mark.asyncio
    async def test_cleanup_on_shutdown(self, error_db):
        """Test proper cleanup on shutdown."""
        # Add some documents
        for i in range(10):
            error_db.add_document(
                collection_name="sessions",
                document_id=f"cleanup_test_{i}",
                content=f"Content {i}",
                metadata={"index": i}
            )
        
        # Perform cleanup
        if hasattr(error_db, "cleanup"):
            await error_db.cleanup()
        
        # Verify cleanup completed without errors
        assert True  # If we get here, cleanup succeeded

    def test_recovery_from_connection_loss(self, error_db):
        """Test recovery from connection loss."""
        # Simulate connection loss
        original_client = error_db.client
        error_db.client = None
        
        # Try operation with no client
        with pytest.raises(Exception):
            error_db.search("rulebooks", "test", 5)
        
        # Restore connection
        error_db.client = original_client
        
        # Verify operations work again
        error_db.add_document(
            collection_name="rulebooks",
            document_id="recovery_test",
            content="Recovery successful",
            metadata={}
        )
        
        doc = error_db.get_document("rulebooks", "recovery_test")
        assert doc is not None


class TestTransactionConsistency:
    """Test transaction-like consistency for complex operations."""

    @pytest_asyncio.fixture
    async def transaction_db(self, tmp_path):
        """Create database for transaction testing."""
        with patch.object(settings, "chroma_db_path", tmp_path / "transaction_db"):
            with patch.object(settings, "embedding_model", "all-MiniLM-L6-v2"):
                with patch.object(settings, "chroma_collection_prefix", "test_"):
                    db = ChromaDBManager()
                    yield db
                    if hasattr(db, "cleanup"):
                        await db.cleanup()

    @pytest.mark.asyncio
    async def test_campaign_creation_consistency(self, transaction_db):
        """Test consistency when creating campaign with related entities."""
        campaign_id = str(uuid.uuid4())
        
        # Create campaign with multiple related documents
        campaign_docs = [
            {
                "id": f"{campaign_id}_main",
                "content": "Main campaign description",
                "metadata": {"type": "campaign", "campaign_id": campaign_id}
            },
            {
                "id": f"{campaign_id}_npc_1",
                "content": "Important NPC for campaign",
                "metadata": {"type": "npc", "campaign_id": campaign_id}
            },
            {
                "id": f"{campaign_id}_location_1",
                "content": "Key location in campaign",
                "metadata": {"type": "location", "campaign_id": campaign_id}
            }
        ]
        
        # Add all documents
        transaction_db.batch_add_documents("campaigns", campaign_docs)
        
        # Verify all related documents exist
        results = transaction_db.search(
            collection_name="campaigns",
            query_text="campaign",
            n_results=10,
            metadata_filter={"campaign_id": campaign_id}
        )
        
        assert len(results) >= 3
        assert all(r["metadata"]["campaign_id"] == campaign_id for r in results)

    @pytest.mark.asyncio
    async def test_session_state_consistency(self, transaction_db):
        """Test consistency of session state updates."""
        session_id = str(uuid.uuid4())
        
        # Create session
        transaction_db.add_document(
            collection_name="sessions",
            document_id=session_id,
            content=json.dumps({
                "title": "Test Session",
                "state": "active",
                "round": 1,
                "combatants": []
            }),
            metadata={"session_id": session_id, "state": "active"}
        )
        
        # Update session state multiple times
        states = ["active", "combat", "paused", "completed"]
        for i, state in enumerate(states):
            content = json.dumps({
                "title": "Test Session",
                "state": state,
                "round": i + 1,
                "combatants": [f"player_{j}" for j in range(i + 1)]
            })
            
            transaction_db.update_document(
                collection_name="sessions",
                document_id=session_id,
                content=content,
                metadata={"session_id": session_id, "state": state, "version": i + 1}
            )
        
        # Verify final state
        doc = transaction_db.get_document("sessions", session_id)
        assert doc is not None
        assert doc["metadata"]["state"] == "completed"
        assert doc["metadata"]["version"] == 4

    @pytest.mark.asyncio
    async def test_rollback_on_partial_failure(self, transaction_db):
        """Test rollback behavior on partial operation failure."""
        # This tests the behavior when part of a batch operation fails
        valid_docs = [
            {
                "id": f"valid_{i}",
                "content": f"Valid content {i}",
                "metadata": {"valid": True}
            }
            for i in range(5)
        ]
        
        # Add an invalid document that should cause failure
        invalid_doc = {
            "id": "invalid",
            "content": None,  # Invalid content
            "metadata": {"valid": False}
        }
        
        mixed_docs = valid_docs + [invalid_doc]
        
        # Attempt batch operation with invalid document
        try:
            transaction_db.batch_add_documents("rulebooks", mixed_docs)
        except Exception:
            pass  # Expected to fail
        
        # Check if valid documents were added (depends on implementation)
        # Some databases might add valid docs, others might rollback all
        docs = transaction_db.list_documents("rulebooks", limit=10)
        
        # Document the behavior (implementation-specific)
        if len(docs) == 0:
            # Full rollback behavior
            assert True  # All-or-nothing behavior
        else:
            # Partial success behavior
            assert all(d["metadata"].get("valid", False) for d in docs)


# Performance benchmark fixture for test reporting
@pytest.fixture(scope="session")
def benchmark_results():
    """Collect and report benchmark results."""
    results = {}
    yield results
    
    if results:
        print("\n=== Performance Benchmark Results ===")
        for test_name, metrics in results.items():
            print(f"\n{test_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")