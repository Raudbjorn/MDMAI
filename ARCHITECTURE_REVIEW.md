# TTRPG Assistant MCP Server - Comprehensive Architecture Review

## Executive Summary

This document presents a comprehensive review of the MDMAI project's architecture changes conducted by specialized AI agents (mcp-protocol-expert, llm-architect, and context-manager). The review analyzed the diff between the `main` and `feature/ui-provider-integration-planning` branches, as well as the port-free IPC proof-of-concept work.

## Review Participants

- **MCP Protocol Expert**: Evaluated MCP compliance, protocol implementation, and bridge architecture
- **LLM Architect**: Assessed AI provider integration, cost optimization, and model selection strategies
- **Context Manager**: Reviewed state persistence, collaborative features, and performance requirements

## Key Findings and Recommendations

### 1. MCP Protocol Architecture (mcp-protocol-expert)

#### Strengths
- ✅ Correct use of FastMCP framework with proper tool registration patterns
- ✅ Already configured for stdio mode (`mcp_stdio_mode: true`)
- ✅ Excellent understanding of port-free requirements with Unix domain sockets and Apache Arrow
- ✅ Modular tool organization by domain (campaign, session, character_generation)

#### Critical Issues to Address

1. **JSON-RPC 2.0 Compliance**
   - Current POC uses custom Protocol Buffers instead of MCP-required JSON-RPC
   - **Recommendation**: Keep protobuf for optimization but wrap in JSON-RPC envelope
   ```python
   # Required format for MCP compliance
   {
       "jsonrpc": "2.0",
       "id": "unique-id",
       "method": "tools/call",
       "params": {...}
   }
   ```

2. **Missing MCP Protocol Methods**
   - Not implementing required methods: `initialize`, `tools/list`, `resources/list`
   - **Recommendation**: Extend FastMCP to expose these methods through the bridge

3. **Process Lifecycle Management**
   - No health monitoring or automatic recovery for MCP processes
   - **Recommendation**: Implement robust process manager with health checks
   ```python
   class MCPProcessManager:
       async def spawn_server(self, session_id: str):
           # Add resource limits and health monitoring
           process = await self._create_process_with_limits()
           asyncio.create_task(self._monitor_health(process, session_id))
   ```

#### Migration Strategy
- **Phase 1**: Create MCPProtocolAdapter wrapping FastMCP with JSON-RPC 2.0
- **Phase 2**: Integrate Apache Arrow selectively for large data (>100KB)
- **Phase 3**: Add process sandboxing with resource limits
- **Phase 4**: Implement connection pooling and intelligent data routing

### 2. LLM Integration Architecture (llm-architect)

#### Key Architecture Recommendations

1. **Unified Provider Abstraction**
   ```python
   class TTRPGAIProvider(ABC):
       @abstractmethod
       async def generate_character(self, params: CharacterParams) -> AsyncIterator[CharacterData]:
           pass
       
       @abstractmethod
       async def apply_personality(self, content: str, profile: PersonalityProfile) -> str:
           pass
   ```

2. **Cost-Aware Provider Routing**
   - Character generation → Anthropic (best for creative/narrative)
   - Rules lookup → Gemini (cost-effective for factual)
   - Structured data → OpenAI (best for function calling)

3. **Token Optimization Strategies**
   - Implement semantic chunking with 30% overlap
   - Use tiktoken for accurate token counting across providers
   - Apply dynamic context window allocation

4. **Personality System Integration**
   ```python
   class PersonalityEnhancer:
       def __init__(self, personality_manager: PersonalityManager):
           self.active_profiles = {
               "dnd5e": WiseSageProfile(),
               "blades": ShadowyInformantProfile(),
               "delta_green": ClassifiedHandlerProfile()
           }
   ```

#### Implementation Priorities
1. **High Priority**: Provider abstraction, token optimization, personality integration
2. **Medium Priority**: Cost routing, response caching, fallback mechanisms
3. **Low Priority**: Fine-tuning preparation, advanced RAG patterns

### 3. Context Management Architecture (context-manager)

#### Critical Design Patterns

1. **Hybrid Storage Architecture**
   ```python
   class ContextStorageSystem:
       def __init__(self):
           self.hot_storage = RedisCache()      # <10ms access
           self.warm_storage = SQLiteDB()       # <50ms access
           self.cold_storage = S3Compatible()   # Archival
   ```

2. **CRDT-Based Conflict Resolution**
   - **Yjs** for collaborative canvas (complex shared state)
   - **Automerge** for campaign documents (structured data)
   - **LWW-Element-Set** for simple properties (HP, status)

3. **Performance Requirements**
   - Context retrieval: <50ms for 95th percentile ✅
   - State sync: <100ms cross-component
   - Memory efficiency: <100MB per active session

4. **Session Recovery Strategy**
   ```python
   class SessionRecoveryManager:
       async def recover_session(self, session_id: str):
           # 1. Restore from hot cache
           # 2. Rebuild from event store if needed
           # 3. Reconstruct from audit log as fallback
   ```

#### Collaborative Features Assessment
- ✅ WebSocket standardization correct for real-time sync
- ✅ CRDT approach appropriate for conflict resolution
- ⚠️ Need to add event sourcing for audit trail
- ⚠️ Missing distributed lock mechanism for critical sections

### 4. Integration Points and Dependencies

#### File Structure Updates Needed
```
./MDMAI/
├── src/
│   ├── bridge/                 # New: Bridge service
│   │   ├── mcp_adapter.py      # JSON-RPC wrapper
│   │   ├── process_manager.py  # Process lifecycle
│   │   └── transport.py        # WebSocket handler
│   ├── llm/                    # New: LLM integration
│   │   ├── providers/          # Provider implementations
│   │   ├── router.py           # Cost-aware routing
│   │   └── personality.py      # Personality enhancement
│   ├── context/                # New: Context management
│   │   ├── storage.py          # Hybrid storage
│   │   ├── crdt.py            # Conflict resolution
│   │   └── recovery.py        # Session recovery
│   └── [existing modules]
```

### 5. Security Considerations

#### Critical Security Requirements
1. **Process Isolation**: Each session in separate process with cgroups limits
2. **Credential Encryption**: Use AWS KMS or HashiCorp Vault patterns
3. **Audit Logging**: Structured logs with correlation IDs for compliance
4. **Rate Limiting**: Token bucket algorithm per user/tool combination

### 6. Performance Optimization Strategy

#### Caching Architecture
```python
class IntelligentCache:
    def __init__(self):
        self.cache_layers = [
            BrowserCache(ttl=300),      # 5 min
            CDNCache(ttl=3600),         # 1 hour
            RedisCache(ttl=86400),      # 1 day
            DiskCache(ttl=604800)       # 1 week
        ]
```

#### Data Routing Optimization
- Small data (<10KB): Inline JSON-RPC
- Medium data (10-100KB): Compressed JSON
- Large data (>100KB): Apache Arrow shared memory

### 7. Risk Assessment and Mitigation

#### High-Risk Areas
1. **MCP Protocol Compliance**: Breaking changes could disconnect from ecosystem
   - **Mitigation**: Strict adherence to JSON-RPC 2.0 spec
   
2. **Provider API Stability**: Changes could break integrations
   - **Mitigation**: Version pinning, adapter pattern, fallback providers
   
3. **Context Size Growth**: Could exceed memory limits
   - **Mitigation**: Intelligent pruning, compression, archival

4. **Real-time Sync Complexity**: Race conditions in collaborative editing
   - **Mitigation**: CRDT adoption, event sourcing, optimistic locking

### 8. Implementation Roadmap

#### Week 1-2: Foundation
- [ ] Implement MCPProtocolAdapter with JSON-RPC compliance
- [ ] Create process manager with health monitoring
- [ ] Set up WebSocket transport layer

#### Week 3-4: Core Integration
- [ ] Build unified AI provider abstraction
- [ ] Implement cost-aware routing
- [ ] Create context storage system

#### Week 5-6: Advanced Features
- [ ] Integrate CRDT conflict resolution
- [ ] Add personality system enhancement
- [ ] Implement session recovery

#### Week 7-8: Optimization & Testing
- [ ] Performance optimization
- [ ] Security hardening
- [ ] Load testing and benchmarking

### 9. Success Metrics

#### Technical Metrics
- MCP protocol compliance: 100%
- Context retrieval latency: <50ms (p95)
- Cache hit rate: >90%
- Process recovery time: <5s
- Memory per session: <100MB

#### Business Metrics
- AI provider cost reduction: 30-40%
- User session continuity: >99.9%
- Concurrent user capacity: 100+ per instance
- Tool execution success rate: >99%

## Updated Architecture - Technology Stack Migration

### Frontend Migration: React to SvelteKit
A comprehensive migration plan has been documented in `SVELTEKIT_MIGRATION.md` that covers:
- SSR-first architecture leveraging SvelteKit's built-in capabilities
- Simplified state management with native Svelte stores
- API routes integration for MCP tool exposure
- WebSocket/SSE setup for real-time updates
- Removal of separate mobile app in favor of responsive web design

### Backend Modernization: Python with Result Pattern
The Python backend modernization is detailed in `PYTHON_MODERNIZATION.md` including:
- Migration to error-as-values pattern using Result types
- Modern Python dependencies (Python 3.11+)
- Comprehensive error handling without exceptions
- Type-safe error propagation throughout the stack
- Alignment with SvelteKit's Result pattern for consistency

## Conclusion

The architecture demonstrates strong understanding of requirements and appropriate technology choices. The key areas requiring attention are:

1. **MCP Protocol Compliance**: Must align with JSON-RPC 2.0 standard
2. **Process Management**: Need robust lifecycle management
3. **Conflict Resolution**: CRDT implementation critical for collaboration
4. **Cost Optimization**: Provider routing essential for sustainability
5. **Technology Stack**: Migration to SvelteKit + modern Python with Result pattern

With these adjustments, the architecture will provide a robust, scalable, and cost-effective solution for web-based TTRPG assistance with multi-provider AI integration. The migration to SvelteKit and Result pattern provides better developer experience, improved error handling, and consistent patterns across the full stack.

## Appendix: Code Review Issues Resolution

The four code review issues have been successfully addressed:

1. **✅ Real-time Communication**: Standardized on WebSocket
2. **✅ Testing Priority**: Elevated to high priority
3. **✅ Conflict Resolution**: Enhanced with OT/CRDT hybrid
4. **✅ Offline Capabilities**: Scoped appropriately for v1.0

These fixes have been committed to the `feature/ui-provider-integration-planning` branch with detailed explanations in commit c70b4b3.