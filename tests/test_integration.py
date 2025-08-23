"""Integration tests for end-to-end workflows."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from pathlib import Path
from datetime import datetime
import tempfile
import json

# Integration test scenarios that test multiple components working together


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_game_session_workflow(self):
        """Test a complete game session from start to finish."""
        # This tests the integration of:
        # - Campaign creation
        # - Session management
        # - Combat tracking
        # - Note taking
        # - Session completion
        
        with patch('src.core.database.ChromaDBManager') as MockDB:
            mock_db = MockDB.return_value
            mock_db.add_document = AsyncMock()
            mock_db.search = AsyncMock(return_value=[])
            mock_db.update_document = AsyncMock()
            mock_db.get_document = AsyncMock()
            
            # Import components
            from src.campaign.campaign_manager import CampaignManager
            from src.session.session_manager import SessionManager
            
            # Initialize managers
            campaign_mgr = CampaignManager(mock_db)
            session_mgr = SessionManager(mock_db)
            
            # Step 1: Create a campaign
            campaign_id = await campaign_mgr.create_campaign(
                name="Test Campaign",
                system="D&D 5e",
                description="Integration test campaign"
            )
            assert campaign_id is not None
            
            # Step 2: Add NPCs to campaign
            npc_id = await campaign_mgr.add_npc(campaign_id, {
                "name": "Dragon",
                "role": "boss",
                "hp": 200
            })
            assert npc_id is not None
            
            # Step 3: Create a session
            session_id = await session_mgr.create_session(
                campaign_id=campaign_id,
                title="Boss Battle",
                session_number=10
            )
            assert session_id is not None
            
            # Step 4: Add combatants to initiative
            await session_mgr.add_to_initiative(
                session_id, "Fighter", 15, True, 50
            )
            await session_mgr.add_monster(
                session_id, "Dragon", 200, 18, 10
            )
            
            # Step 5: Run combat rounds
            await session_mgr.start_combat(session_id)
            await session_mgr.advance_round(
                session_id,
                actions=["Fighter attacks Dragon"]
            )
            
            # Step 6: Add session notes
            await session_mgr.add_note(
                session_id,
                "Epic battle with the dragon begins!"
            )
            
            # Step 7: End session
            await session_mgr.end_session(
                session_id,
                "Dragon defeated, party victorious!"
            )
            
            # Verify the workflow
            assert mock_db.add_document.call_count >= 3  # Campaign, NPC, Session
            assert mock_db.update_document.called  # Session updates
    
    @pytest.mark.asyncio
    async def test_pdf_to_search_workflow(self):
        """Test PDF processing to search workflow."""
        # This tests:
        # - PDF processing
        # - Content chunking
        # - Embedding generation
        # - Search functionality
        
        with patch('src.core.database.ChromaDBManager') as MockDB:
            mock_db = MockDB.return_value
            mock_db.add_document = AsyncMock()
            mock_db.search = AsyncMock()
            
            from src.pdf_processing.pipeline import PDFProcessingPipeline
            from src.search.search_service import SearchService
            
            # Create temp PDF file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(b'%PDF-1.4\nTest PDF content')
            
            try:
                # Process PDF
                pipeline = PDFProcessingPipeline()
                
                with patch.object(pipeline.parser, 'extract_text_from_pdf') as mock_extract:
                    mock_extract.return_value = {
                        "text": "Fireball spell deals 8d6 fire damage",
                        "total_pages": 1,
                        "file_hash": "test_hash",
                        "file_name": "test.pdf",
                        "tables": []
                    }
                    
                    with patch.object(pipeline, '_is_duplicate', return_value=False):
                        with patch.object(pipeline.chunker, 'chunk_document') as mock_chunk:
                            mock_chunk.return_value = [{
                                "id": "chunk_1",
                                "content": "Fireball spell deals 8d6 fire damage",
                                "metadata": {"page": 1}
                            }]
                            
                            with patch.object(pipeline.embedding_generator, 'generate_embeddings') as mock_embed:
                                mock_embed.return_value = [[0.1, 0.2, 0.3]]
                                
                                result = await pipeline.process_pdf(
                                    pdf_path=tmp_path,
                                    rulebook_name="Test Book",
                                    system="D&D 5e"
                                )
                                
                                assert result["status"] == "success"
                
                # Search for content
                search_service = SearchService(mock_db)
                mock_db.search.return_value = [{
                    "id": "chunk_1",
                    "content": "Fireball spell deals 8d6 fire damage",
                    "metadata": {"page": 1}
                }]
                
                search_results = await search_service.search("fireball", "rulebooks")
                assert len(search_results) > 0
                assert "Fireball" in search_results[0]["content"]
                
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_character_generation_with_rulebook_integration(self):
        """Test character generation with rulebook lookups."""
        # This tests:
        # - Character generation
        # - Rulebook linking
        # - Backstory generation with personality
        
        with patch('src.core.database.ChromaDBManager') as MockDB:
            mock_db = MockDB.return_value
            mock_db.search = AsyncMock()
            mock_db.add_document = AsyncMock()
            
            from src.character_generation.character_generator import CharacterGenerator
            from src.campaign.rulebook_linker import RulebookLinker
            
            # Mock rulebook data
            mock_db.search.return_value = [{
                "id": "rule_1",
                "content": "Fighter class: d10 hit die, proficient with all armor",
                "metadata": {"type": "class", "name": "fighter"}
            }]
            
            # Generate character
            generator = CharacterGenerator(mock_db)
            character = generator.generate_character(
                name="Test Hero",
                race="human",
                character_class="fighter",
                level=5
            )
            
            assert character["name"] == "Test Hero"
            assert character["class"] == "fighter"
            assert character["level"] == 5
            
            # Link to rulebook
            linker = RulebookLinker(mock_db)
            links = await linker.link_npc_to_rules(character)
            
            assert len(links) > 0
            mock_db.search.assert_called()
    
    @pytest.mark.asyncio
    async def test_campaign_with_personality_system(self):
        """Test campaign with personality-aware responses."""
        # This tests:
        # - Personality extraction
        # - Campaign context
        # - Personality-aware generation
        
        with patch('src.personality.personality_manager.PersonalityManager') as MockPM:
            mock_pm = MockPM.return_value
            mock_pm.get_personality = Mock(return_value={
                "tone": "mysterious",
                "vocabulary": ["shadows", "whispers", "darkness"],
                "style": "Victorian gothic"
            })
            
            from src.character_generation.backstory_generator import BackstoryGenerator
            
            generator = BackstoryGenerator(personality_manager=mock_pm)
            
            # Generate backstory with personality
            backstory = generator.generate_backstory(
                character_class="rogue",
                race="halfling",
                system="Blades in the Dark"
            )
            
            assert backstory is not None
            # Would check for personality-influenced content
            mock_pm.get_personality.assert_called_with("Blades in the Dark")


class TestCrossComponentIntegration:
    """Test integration between different components."""
    
    @pytest.mark.asyncio
    async def test_search_with_campaign_context(self):
        """Test search that considers campaign context."""
        with patch('src.core.database.ChromaDBManager') as MockDB:
            mock_db = MockDB.return_value
            
            # Mock campaign data
            mock_db.search = AsyncMock()
            mock_db.search.side_effect = [
                # First call - campaign NPCs
                [{"id": "npc_1", "content": '{"name": "Evil Wizard", "class": "wizard"}'}],
                # Second call - rulebook search
                [{"id": "rule_1", "content": "Wizard spells"}]
            ]
            
            from src.search.search_service import SearchService
            
            service = SearchService(mock_db)
            
            # Search with campaign context
            results = await service.search_with_context(
                query="wizard spells",
                campaign_id="campaign_1"
            )
            
            # Should have both rulebook and campaign results
            assert mock_db.search.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_session_with_monster_lookup(self):
        """Test session management with monster stat lookups."""
        with patch('src.core.database.ChromaDBManager') as MockDB:
            mock_db = MockDB.return_value
            
            # Mock monster stats from rulebook
            mock_db.search = AsyncMock(return_value=[{
                "id": "monster_1",
                "content": '{"name": "Goblin", "hp": 7, "ac": 15, "attacks": ["scimitar"]}',
                "metadata": {"type": "monster"}
            }])
            
            from src.session.session_manager import SessionManager
            
            manager = SessionManager(mock_db)
            
            # Add monster using rulebook lookup
            monster = await manager.add_monster_from_rulebook(
                session_id="session_1",
                monster_name="Goblin"
            )
            
            assert monster is not None
            mock_db.search.assert_called()
    
    @pytest.mark.asyncio
    async def test_parallel_processing_integration(self):
        """Test parallel processing with multiple components."""
        from src.performance.parallel_processor import ParallelProcessor, ResourceLimits
        
        processor = ParallelProcessor(ResourceLimits(max_workers=2))
        
        await processor.initialize()
        
        try:
            # Submit multiple different task types
            tasks = []
            
            # Mock PDF task
            with patch.object(processor, '_process_pdf_task') as mock_pdf:
                mock_pdf.return_value = {"status": "success"}
                
                pdf_task = await processor.submit_task(
                    "pdf_processing",
                    {"pdf_path": "test.pdf"}
                )
                tasks.append(pdf_task)
            
            # Mock search task
            with patch.object(processor, '_process_search_task') as mock_search:
                mock_search.return_value = {"results": []}
                
                search_task = await processor.submit_task(
                    "search",
                    {"queries": [{"query": "test"}]}
                )
                tasks.append(search_task)
            
            # Wait for completion
            completed = await processor.wait_for_all([t.id for t in tasks])
            
            assert len(completed) == 2
            assert all(t.status.value in ["completed", "failed"] for t in completed)
            
        finally:
            await processor.shutdown()


class TestPerformanceIntegration:
    """Test performance and caching integration."""
    
    @pytest.mark.asyncio
    async def test_cache_with_search(self):
        """Test caching integration with search."""
        from src.performance.cache_system import CacheSystem, CachePolicy
        from src.search.search_service import SearchService
        
        # Create cache
        cache = CacheSystem(
            name="test_cache",
            max_size=10,
            ttl_seconds=60,
            policy=CachePolicy.LRU
        )
        
        mock_db = Mock()
        mock_db.search = AsyncMock(return_value=[
            {"id": "1", "content": "Cached result"}
        ])
        
        service = SearchService(mock_db)
        service.cache = cache
        service.enable_cache = True
        
        # First search - hits database
        result1 = await service.search("test query", "rulebooks")
        assert mock_db.search.call_count == 1
        
        # Second search - hits cache
        result2 = await service.search("test query", "rulebooks")
        assert mock_db.search.call_count == 1  # Not called again
        
        assert result1 == result2
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats["statistics"]["total_hits"] == 1
        assert stats["statistics"]["total_misses"] == 1
        
        cache.shutdown()
    
    @pytest.mark.asyncio
    async def test_database_optimization(self):
        """Test database optimization integration."""
        from src.performance.database_optimizer import DatabaseOptimizer
        
        mock_db = Mock()
        mock_db.collections = {"test": Mock()}
        
        optimizer = DatabaseOptimizer(mock_db)
        
        # Track queries
        optimizer.track_query_performance(
            query="test",
            collection="test",
            execution_time=0.1,
            result_count=5
        )
        
        # Get optimization suggestions
        suggestions = optimizer.analyze_performance()
        
        assert "queries" in suggestions
        assert suggestions["total_queries"] >= 1
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test performance monitoring integration."""
        from src.performance.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Record operations
        monitor.record_operation("search", 0.1, True)
        monitor.record_operation("pdf_process", 5.0, True)
        monitor.record_operation("search", 0.2, False)  # Failed
        
        # Generate report
        report = monitor.generate_report()
        
        assert report["total_operations"] == 3
        assert report["success_rate"] < 1.0  # One failure
        assert "search" in report["operations"]
        assert "pdf_process" in report["operations"]


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    @pytest.mark.asyncio
    async def test_partial_pdf_processing_recovery(self):
        """Test recovery from partial PDF processing failure."""
        from src.pdf_processing.pipeline import PDFProcessingPipeline
        
        pipeline = PDFProcessingPipeline()
        
        # Mock partial failure
        with patch.object(pipeline.parser, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                "text": "Partial content",
                "total_pages": 10,
                "file_hash": "test_hash",
                "file_name": "test.pdf",
                "errors": ["Page 5 failed to parse"]
            }
            
            with patch.object(pipeline, '_is_duplicate', return_value=False):
                with patch.object(pipeline.chunker, 'chunk_document') as mock_chunk:
                    mock_chunk.return_value = []  # Some chunks
                    
                    with patch.object(pipeline.embedding_generator, 'generate_embeddings'):
                        result = await pipeline.process_pdf(
                            pdf_path="test.pdf",
                            rulebook_name="Test",
                            system="D&D 5e"
                        )
                        
                        # Should still process partial content
                        assert "error" in result or result["status"] == "partial"
    
    @pytest.mark.asyncio
    async def test_database_connection_recovery(self):
        """Test recovery from database connection issues."""
        mock_db = Mock()
        mock_db.search = AsyncMock()
        
        # Simulate connection failure then recovery
        mock_db.search.side_effect = [
            Exception("Connection lost"),
            [],  # Recovered
        ]
        
        from src.search.search_service import SearchService
        
        service = SearchService(mock_db)
        
        # First attempt fails
        with pytest.raises(Exception):
            await service.search("test", "rulebooks")
        
        # Second attempt succeeds
        results = await service.search("test", "rulebooks")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_concurrent_resource_cleanup(self):
        """Test proper cleanup of resources in concurrent scenarios."""
        from src.performance.parallel_processor import ParallelProcessor, ResourceLimits
        
        processors = []
        
        # Create multiple processors
        for i in range(3):
            processor = ParallelProcessor(ResourceLimits(max_workers=2))
            await processor.initialize()
            processors.append(processor)
        
        # Submit tasks to each
        for processor in processors:
            with patch.object(processor, '_process_search_task') as mock_task:
                mock_task.return_value = {"results": []}
                await processor.submit_task("search", {"queries": []})
        
        # Clean up all
        for processor in processors:
            await processor.shutdown()
            assert processor._shutdown is True


class TestDataConsistency:
    """Test data consistency across components."""
    
    @pytest.mark.asyncio
    async def test_campaign_version_consistency(self):
        """Test campaign versioning maintains consistency."""
        mock_db = Mock()
        mock_db.add_document = AsyncMock()
        mock_db.get_document = AsyncMock()
        mock_db.update_document = AsyncMock()
        
        from src.campaign.campaign_manager import CampaignManager
        
        manager = CampaignManager(mock_db)
        
        # Create version snapshot
        version_id = await manager.create_version_snapshot(
            "campaign_1",
            "Before major event"
        )
        
        # Modify campaign
        await manager.update_campaign("campaign_1", {"status": "modified"})
        
        # Rollback
        mock_db.get_document.return_value = {
            "id": version_id,
            "content": '{"snapshot": {"status": "original"}}',
            "metadata": {"type": "version"}
        }
        
        await manager.rollback_to_version("campaign_1", version_id)
        
        # Verify rollback
        mock_db.update_document.assert_called()
    
    @pytest.mark.asyncio
    async def test_cross_reference_integrity(self):
        """Test integrity of cross-references between components."""
        mock_db = Mock()
        mock_db.search = AsyncMock()
        mock_db.add_document = AsyncMock()
        
        from src.campaign.campaign_manager import CampaignManager
        from src.session.session_manager import SessionManager
        
        campaign_mgr = CampaignManager(mock_db)
        session_mgr = SessionManager(mock_db)
        
        # Create campaign
        campaign_id = "test_campaign"
        
        # Create session referencing campaign
        session_id = await session_mgr.create_session(
            campaign_id=campaign_id,
            title="Test Session"
        )
        
        # Verify reference integrity
        call_args = mock_db.add_document.call_args
        assert call_args[1]["metadata"]["campaign_id"] == campaign_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])