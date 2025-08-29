# SPIKE 2: State Synchronization Strategy for TTRPG MCP Server

## Executive Summary

This spike analyzes state synchronization requirements for the TTRPG MCP Server, examining architecture patterns, conflict resolution strategies, and performance optimizations for real-time collaborative gameplay. The analysis covers Event Sourcing vs State-based vs Hybrid approaches, evaluates CRDT vs OT implementations, and provides specific recommendations for TTRPG data complexities.

**Key Findings:**
- Hybrid Event Sourcing + State-based approach recommended for TTRPG use case
- CRDT with semantic merge strategies best suited for nested TTRPG objects
- Multi-tier state partitioning required (Campaign → Session → User scopes)
- ChromaDB integration for vector-based semantic conflict resolution
- Optimistic UI with rollback for <100ms response times

## 1. State Synchronization Architecture Analysis

### 1.1 Current Implementation Assessment

The existing codebase implements a **State-based CRDT** approach with:

**Strengths:**
- Vector clocks for causality tracking (`VectorClock` class)
- Operational Transformation for text editing (`OperationalTransform` class)
- CRDT implementation with tombstone tracking (`CollaborativeCRDT` class)
- Optimistic updates with rollback (`OptimisticUpdates` class)
- Debounced synchronization (`DebouncedSync` class)

**Limitations:**
- Simplified merge strategy (deep merge without semantic understanding)
- No event sourcing for audit trails
- Limited conflict resolution (primarily last-write-wins)
- No state partitioning beyond room-level
- Basic tombstone management without TTL

### 1.2 Architecture Options Evaluation

#### Option A: Pure Event Sourcing
```typescript
interface TTRPGEvent {
  id: string;
  type: 'character_moved' | 'dice_rolled' | 'initiative_changed' | 'note_updated';
  aggregate_id: string; // campaign_id, session_id, character_id
  aggregate_type: 'campaign' | 'session' | 'character' | 'monster';
  data: any;
  metadata: {
    user_id: string;
    timestamp: number;
    version: number;
    causality: VectorClock;
  };
}
```

**Pros:**
- Perfect audit trail for game sessions
- Natural rollback and replay capabilities
- Strong consistency guarantees
- Excellent for debugging state issues

**Cons:**
- Higher complexity for queries (need projection)
- Storage overhead (all events retained)
- Performance impact for large campaigns
- Difficult real-time state reconstruction

#### Option B: Pure State-based CRDT
```typescript
interface TTRPGState {
  campaign: CampaignCRDT;
  sessions: Map<string, SessionCRDT>;
  characters: Map<string, CharacterCRDT>;
  shared_resources: ResourceCRDT;
}
```

**Pros:**
- Simple conflict-free merging
- Excellent for offline-first scenarios
- Natural eventual consistency
- Good performance for reads

**Cons:**
- Limited audit capabilities
- Complex semantic merge logic for TTRPG data
- Memory overhead for tombstone tracking
- Difficulty with complex referential integrity

#### Option C: Hybrid Event Sourcing + State-based (RECOMMENDED)
```typescript
interface HybridStateSync {
  // Event stream for audit and replay
  event_store: EventStore<TTRPGEvent>;
  
  // Current state for performance
  current_state: TTRPGStateCRDT;
  
  // Snapshot management
  snapshots: Map<number, StateSnapshot>;
  
  // Conflict resolution with semantic understanding
  conflict_resolver: SemanticConflictResolver;
}
```

**Pros:**
- Best of both approaches
- Audit trail with performance
- Flexible conflict resolution
- Supports complex TTRPG scenarios

**Cons:**
- Higher implementation complexity
- Dual storage requirements
- Synchronization between event store and state

**Recommendation:** **Option C (Hybrid)** provides the best balance for TTRPG requirements, combining audit trails needed for game integrity with performance required for real-time collaboration.

## 2. Conflict Resolution Strategies

### 2.1 Current Implementation Analysis

Existing conflict resolution uses:
- Last-write-wins as primary strategy
- Simple version comparison
- Basic path-based conflict detection
- Limited user intervention

### 2.2 Enhanced Conflict Resolution Framework

#### Semantic Conflict Resolution for TTRPG Data

```typescript
interface SemanticConflictResolver {
  resolveCharacterConflict(conflicts: CharacterConflict[]): CharacterState;
  resolveInitiativeConflict(conflicts: InitiativeConflict[]): InitiativeOrder;
  resolveNarrativeConflict(conflicts: NarrativeConflict[]): NarrativeText;
  resolveCombatConflict(conflicts: CombatConflict[]): CombatState;
}

class TTRPGConflictResolver implements SemanticConflictResolver {
  constructor(
    private vectorDB: ChromaDBClient,
    private gameRules: GameRulesEngine
  ) {}

  async resolveCharacterConflict(conflicts: CharacterConflict[]): Promise<CharacterState> {
    // Semantic analysis of character changes
    const semanticAnalysis = await this.vectorDB.analyzeConflicts(conflicts);
    
    // Rule-based resolution
    if (semanticAnalysis.hasRuleViolation) {
      return this.gameRules.enforceCharacterRules(conflicts);
    }
    
    // Merge non-conflicting attributes
    return this.mergeSemantically(conflicts);
  }
  
  private async mergeSemantically(conflicts: any[]): Promise<any> {
    // Use vector similarity for semantic merging
    const embeddings = await Promise.all(
      conflicts.map(c => this.vectorDB.embed(c.description))
    );
    
    // Find most semantically consistent resolution
    const clusters = this.clusterSemanticallySimilar(embeddings);
    return this.selectBestCluster(clusters);
  }
}
```

#### Conflict Resolution Strategies by Data Type

| Data Type | Primary Strategy | Fallback | Notes |
|-----------|------------------|----------|-------|
| Character Stats | Rule-based validation | User intervention | Game rules must be maintained |
| Initiative Order | Timestamp-based ordering | GM override | Critical for turn-based gameplay |
| Narrative Text | Semantic merge with CRDT | Conflict markers | Preserve creative content |
| Combat State | Authoritative GM state | Rollback | Single source of truth needed |
| Dice Rolls | Immutable once committed | N/A | Cannot be changed once rolled |
| Map/Positions | Spatial conflict resolution | Last valid position | Physical constraints apply |

### 2.3 Advanced CRDT for TTRPG

#### Specialized CRDTs for Game Data

```typescript
// Character Sheet CRDT with rule validation
class CharacterCRDT extends LWWMap<string, any> {
  constructor(
    private rules: GameRules,
    private characterClass: string
  ) {
    super();
  }
  
  override set(key: string, value: any, timestamp: number): boolean {
    // Validate against game rules before setting
    if (!this.rules.validateCharacterAttribute(key, value, this.characterClass)) {
      throw new ValidationError(`Invalid ${key} value for ${this.characterClass}`);
    }
    
    return super.set(key, value, timestamp);
  }
  
  // Special handling for interdependent stats
  setStats(stats: CharacterStats, timestamp: number): void {
    const validated = this.rules.validateStatBlock(stats, this.characterClass);
    
    // Atomic update of related stats
    this.beginTransaction();
    try {
      Object.entries(validated).forEach(([key, value]) => {
        super.set(key, value, timestamp);
      });
      this.commitTransaction();
    } catch (error) {
      this.rollbackTransaction();
      throw error;
    }
  }
}

// Initiative Order CRDT with turn management
class InitiativeCRDT extends ORArray<InitiativeEntry> {
  private currentTurn: number = 0;
  private round: number = 1;
  
  nextTurn(): void {
    this.currentTurn = (this.currentTurn + 1) % this.length;
    if (this.currentTurn === 0) {
      this.round++;
      this.resetActedFlags();
    }
  }
  
  insertAt(index: number, entry: InitiativeEntry, timestamp: number): void {
    // Maintain sorted order by initiative value
    const sortedIndex = this.findSortedIndex(entry.initiative);
    super.insertAt(sortedIndex, entry, timestamp);
    
    // Adjust current turn index if needed
    if (sortedIndex <= this.currentTurn) {
      this.currentTurn++;
    }
  }
}

// Narrative Text CRDT with semantic preservation
class NarrativeCRDT extends RGATreeSplit {
  constructor(private semanticAnalyzer: SemanticAnalyzer) {
    super();
  }
  
  override insert(position: number, text: string, metadata: any): void {
    // Analyze semantic impact before insertion
    const context = this.getContext(position, 100); // 100 chars context
    const semanticImpact = this.semanticAnalyzer.analyzeInsertion(context, text);
    
    if (semanticImpact.breaksNarrative) {
      // Use conflict markers for semantic breaks
      const markedText = `<<<NARRATIVE_CONFLICT\n${text}\n>>>NARRATIVE_CONFLICT`;
      super.insert(position, markedText, { ...metadata, conflict: true });
    } else {
      super.insert(position, text, metadata);
    }
  }
}
```

## 3. State Partitioning and Scoping

### 3.1 Multi-Tier Partitioning Strategy

```typescript
interface StatePartition {
  scope: 'campaign' | 'session' | 'user' | 'global';
  id: string;
  access_pattern: 'read_heavy' | 'write_heavy' | 'balanced';
  consistency_level: 'strong' | 'eventual' | 'session';
  replication_factor: number;
}

class StatePartitionManager {
  private partitions = new Map<string, StatePartition>();
  
  // Campaign-level state (persistent, strongly consistent)
  getCampaignPartition(campaignId: string): CampaignPartition {
    return {
      scope: 'campaign',
      data: {
        world_state: new WorldStateCRDT(),
        npcs: new NPCRegistryCRDT(),
        locations: new LocationsCRDT(),
        quests: new QuestsCRDT(),
        lore: new LoreCRDT()
      },
      consistency: 'strong',
      persistence: 'permanent'
    };
  }
  
  // Session-level state (ephemeral, session consistent)
  getSessionPartition(sessionId: string): SessionPartition {
    return {
      scope: 'session',
      data: {
        initiative: new InitiativeCRDT(),
        combat_state: new CombatCRDT(),
        session_notes: new NotesCRDT(),
        dice_history: new DiceHistoryCRDT(),
        turn_state: new TurnStateCRDT()
      },
      consistency: 'session',
      persistence: 'session_end'
    };
  }
  
  // User-level state (personal, eventually consistent)
  getUserPartition(userId: string): UserPartition {
    return {
      scope: 'user',
      data: {
        character_sheets: new CharactersCRDT(),
        personal_notes: new NotesCRDT(),
        ui_preferences: new PreferencesCRDT(),
        cursor_position: new CursorCRDT()
      },
      consistency: 'eventual',
      persistence: 'permanent'
    };
  }
}
```

### 3.2 Access Pattern Optimization

```typescript
interface AccessPattern {
  reads_per_second: number;
  writes_per_second: number;
  typical_operation: 'query' | 'update' | 'scan' | 'aggregation';
  data_locality: 'hot' | 'warm' | 'cold';
}

class AccessPatternOptimizer {
  optimizePartition(pattern: AccessPattern): PartitionConfig {
    if (pattern.reads_per_second > pattern.writes_per_second * 10) {
      // Read-heavy: Use read replicas and aggressive caching
      return {
        replication_strategy: 'read_replicas',
        cache_ttl: 30, // seconds
        consistency: 'eventual'
      };
    } else if (pattern.writes_per_second > 100) {
      // Write-heavy: Use write-optimized storage
      return {
        replication_strategy: 'async_replication',
        cache_ttl: 5,
        consistency: 'session'
      };
    } else {
      // Balanced: Standard configuration
      return {
        replication_strategy: 'sync_replication',
        cache_ttl: 15,
        consistency: 'strong'
      };
    }
  }
}
```

## 4. ChromaDB Persistence Strategy

### 4.1 Vector-Enhanced State Management

```typescript
class ChromaStateManager {
  constructor(
    private chromaClient: ChromaClient,
    private embeddingModel: EmbeddingModel
  ) {}
  
  // Store state with semantic embeddings
  async persistStateWithSemantics(state: any, metadata: StateMetadata): Promise<void> {
    // Generate embeddings for semantic search
    const textContent = this.extractTextContent(state);
    const embedding = await this.embeddingModel.embed(textContent);
    
    // Store in ChromaDB with vector for semantic retrieval
    await this.chromaClient.upsert({
      collection_name: metadata.partition_id,
      documents: [JSON.stringify(state)],
      embeddings: [embedding],
      metadatas: [{
        state_id: metadata.state_id,
        version: metadata.version,
        timestamp: metadata.timestamp,
        scope: metadata.scope
      }]
    });
  }
  
  // Semantic conflict detection using vector similarity
  async detectSemanticConflicts(
    newState: any,
    existingStates: any[]
  ): Promise<ConflictAnalysis> {
    const newEmbedding = await this.embeddingModel.embed(
      this.extractTextContent(newState)
    );
    
    const similarities = await Promise.all(
      existingStates.map(async (state) => {
        const embedding = await this.embeddingModel.embed(
          this.extractTextContent(state)
        );
        return this.cosineSimilarity(newEmbedding, embedding);
      })
    );
    
    return {
      has_conflicts: similarities.some(sim => sim > 0.95 && sim < 1.0),
      similarity_scores: similarities,
      semantic_clusters: this.clusterBySimilarity(similarities)
    };
  }
  
  // Intelligent state snapshotting based on content changes
  async shouldSnapshot(currentState: any, lastSnapshot: any): Promise<boolean> {
    const currentEmbedding = await this.embeddingModel.embed(
      this.extractTextContent(currentState)
    );
    const lastEmbedding = await this.embeddingModel.embed(
      this.extractTextContent(lastSnapshot)
    );
    
    const similarity = this.cosineSimilarity(currentEmbedding, lastEmbedding);
    
    // Snapshot when semantic content has changed significantly
    return similarity < 0.85;
  }
  
  private extractTextContent(state: any): string {
    // Extract meaningful text content for embedding
    const texts: string[] = [];
    
    // Recursive extraction of text fields
    this.extractTextRecursive(state, texts);
    
    return texts.join(' ');
  }
  
  private extractTextRecursive(obj: any, texts: string[]): void {
    if (typeof obj === 'string') {
      texts.push(obj);
    } else if (typeof obj === 'object' && obj !== null) {
      Object.values(obj).forEach(value => {
        this.extractTextRecursive(value, texts);
      });
    }
  }
}
```

### 4.2 ChromaDB Schema for TTRPG State

```typescript
interface ChromaCollectionSchema {
  campaign_states: {
    documents: string[]; // Serialized campaign state
    embeddings: number[][]; // Semantic embeddings
    metadatas: {
      campaign_id: string;
      version: number;
      timestamp: number;
      scope: 'world' | 'npcs' | 'locations' | 'quests';
      change_summary: string;
    }[];
  };
  
  session_states: {
    documents: string[];
    embeddings: number[][];
    metadatas: {
      session_id: string;
      campaign_id: string;
      version: number;
      timestamp: number;
      turn_number: number;
      round_number: number;
      active_participants: string[];
    }[];
  };
  
  conflict_resolutions: {
    documents: string[]; // Conflict resolution decisions
    embeddings: number[][]; // For learning from past resolutions
    metadatas: {
      conflict_id: string;
      resolution_strategy: string;
      confidence_score: number;
      user_feedback: 'positive' | 'negative' | 'neutral';
    }[];
  };
}
```

## 5. Cache Invalidation Patterns

### 5.1 Multi-Level Caching Strategy

```typescript
interface CacheLevel {
  name: string;
  ttl: number;
  size_limit: number;
  eviction_policy: 'LRU' | 'LFU' | 'TTL';
  consistency_model: 'strong' | 'eventual';
}

class TTRPGCacheManager {
  private caches = new Map<string, Cache>();
  
  constructor() {
    this.setupCacheLevels();
  }
  
  private setupCacheLevels(): void {
    // L1: Browser memory cache (hot data)
    this.caches.set('L1', new MemoryCache({
      name: 'browser_memory',
      ttl: 30, // 30 seconds
      size_limit: 10 * 1024 * 1024, // 10MB
      eviction_policy: 'LRU',
      consistency_model: 'strong'
    }));
    
    // L2: Session storage (session data)
    this.caches.set('L2', new SessionCache({
      name: 'session_storage',
      ttl: 300, // 5 minutes
      size_limit: 50 * 1024 * 1024, // 50MB
      eviction_policy: 'TTL',
      consistency_model: 'eventual'
    }));
    
    // L3: IndexedDB (persistent local cache)
    this.caches.set('L3', new IndexedDBCache({
      name: 'persistent_cache',
      ttl: 3600, // 1 hour
      size_limit: 100 * 1024 * 1024, // 100MB
      eviction_policy: 'LFU',
      consistency_model: 'eventual'
    }));
  }
  
  async get<T>(key: string, scope: CacheScope): Promise<T | null> {
    // Try L1 first
    let result = await this.caches.get('L1')?.get<T>(key);
    if (result !== null) {
      this.recordCacheHit('L1', scope);
      return result;
    }
    
    // Try L2
    result = await this.caches.get('L2')?.get<T>(key);
    if (result !== null) {
      this.recordCacheHit('L2', scope);
      // Promote to L1
      await this.caches.get('L1')?.set(key, result, scope.ttl);
      return result;
    }
    
    // Try L3
    result = await this.caches.get('L3')?.get<T>(key);
    if (result !== null) {
      this.recordCacheHit('L3', scope);
      // Promote to L2 and L1
      await this.caches.get('L2')?.set(key, result, scope.ttl);
      await this.caches.get('L1')?.set(key, result, scope.ttl);
      return result;
    }
    
    this.recordCacheMiss(scope);
    return null;
  }
  
  async invalidate(pattern: string, scope: CacheScope): Promise<void> {
    // Invalidate across all cache levels
    await Promise.all(
      Array.from(this.caches.values()).map(cache => 
        cache.invalidatePattern(pattern, scope)
      )
    );
  }
}
```

### 5.2 Smart Invalidation Based on State Changes

```typescript
class StateChangeInvalidator {
  private dependencies = new Map<string, Set<string>>();
  
  constructor() {
    this.buildDependencyGraph();
  }
  
  private buildDependencyGraph(): void {
    // Character changes affect initiative, combat, and session state
    this.dependencies.set('character', new Set([
      'initiative_order',
      'combat_state',
      'session_participants'
    ]));
    
    // Initiative changes affect turn management
    this.dependencies.set('initiative_order', new Set([
      'current_turn',
      'combat_state',
      'turn_history'
    ]));
    
    // Combat changes affect character states
    this.dependencies.set('combat_state', new Set([
      'character_hp',
      'character_conditions',
      'initiative_order'
    ]));
  }
  
  async invalidateBasedOnChange(
    change: StateUpdate,
    cacheManager: TTRPGCacheManager
  ): Promise<void> {
    const affectedKeys = this.getDependentKeys(change.path);
    
    await Promise.all(
      affectedKeys.map(key => 
        cacheManager.invalidate(key, { scope: change.scope })
      )
    );
  }
  
  private getDependentKeys(changePath: string[]): string[] {
    const rootKey = changePath[0];
    const dependencies = this.dependencies.get(rootKey);
    
    return dependencies ? Array.from(dependencies) : [];
  }
}
```

## 6. Performance Optimizations for Large Campaigns

### 6.1 Lazy Loading and Pagination

```typescript
interface LazyLoadingStrategy {
  loadOnDemand<T>(
    collection: string,
    filter: any,
    pageSize: number
  ): AsyncIterator<T[]>;
  
  preloadCritical<T>(
    collection: string,
    criticalIds: string[]
  ): Promise<T[]>;
  
  virtualizeList<T>(
    items: T[],
    viewportSize: number,
    itemHeight: number
  ): VirtualList<T>;
}

class CampaignDataManager implements LazyLoadingStrategy {
  constructor(private chromaDB: ChromaClient) {}
  
  async *loadOnDemand<T>(
    collection: string,
    filter: any,
    pageSize: number = 50
  ): AsyncIterator<T[]> {
    let offset = 0;
    let hasMore = true;
    
    while (hasMore) {
      const results = await this.chromaDB.query({
        collection_name: collection,
        query_filter: filter,
        limit: pageSize,
        offset
      });
      
      if (results.documents.length === 0) {
        hasMore = false;
      } else {
        yield results.documents.map(doc => JSON.parse(doc) as T);
        offset += pageSize;
      }
    }
  }
  
  async preloadCritical<T>(
    collection: string,
    criticalIds: string[]
  ): Promise<T[]> {
    // Load frequently accessed items in parallel
    const batchSize = 10;
    const results: T[] = [];
    
    for (let i = 0; i < criticalIds.length; i += batchSize) {
      const batch = criticalIds.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(id => this.loadById<T>(collection, id))
      );
      results.push(...batchResults.filter(r => r !== null) as T[]);
    }
    
    return results;
  }
  
  virtualizeList<T>(
    items: T[],
    viewportSize: number,
    itemHeight: number
  ): VirtualList<T> {
    return new VirtualList({
      items,
      viewportSize,
      itemHeight,
      overscan: 5, // Render 5 extra items for smooth scrolling
      onLoadMore: (index: number) => {
        // Trigger loading of more items when nearing end
        if (index > items.length - 10) {
          this.loadMoreItems();
        }
      }
    });
  }
  
  private async loadById<T>(collection: string, id: string): Promise<T | null> {
    try {
      const result = await this.chromaDB.get({
        collection_name: collection,
        ids: [id]
      });
      
      return result.documents[0] ? JSON.parse(result.documents[0]) : null;
    } catch (error) {
      console.error(`Failed to load ${collection} ${id}:`, error);
      return null;
    }
  }
}
```

### 6.2 Data Compression and Deduplication

```typescript
class StateCompressionManager {
  private compressor = new LZ4Compressor();
  private deduplicator = new ContentDeduplicator();
  
  async compressState(state: any): Promise<CompressedState> {
    // Deduplicate common patterns first
    const deduplicated = await this.deduplicator.deduplicate(state);
    
    // Compress the deduplicated state
    const compressed = await this.compressor.compress(
      JSON.stringify(deduplicated.state)
    );
    
    return {
      compressed_data: compressed,
      deduplication_map: deduplicated.references,
      original_size: JSON.stringify(state).length,
      compressed_size: compressed.length,
      compression_ratio: compressed.length / JSON.stringify(state).length
    };
  }
  
  async decompressState(compressed: CompressedState): Promise<any> {
    // Decompress first
    const decompressed = await this.compressor.decompress(compressed.compressed_data);
    const state = JSON.parse(decompressed);
    
    // Rehydrate deduplicated references
    return this.deduplicator.rehydrate(state, compressed.deduplication_map);
  }
}

class ContentDeduplicator {
  private referenceMap = new Map<string, any>();
  
  async deduplicate(state: any): Promise<DeduplicatedState> {
    const references = new Map<string, string>();
    const deduplicatedState = this.deduplicateRecursive(state, references);
    
    return {
      state: deduplicatedState,
      references: Object.fromEntries(references)
    };
  }
  
  private deduplicateRecursive(obj: any, references: Map<string, string>): any {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }
    
    // Generate hash for object content
    const contentHash = this.hashContent(obj);
    
    // Check if we've seen this content before
    if (this.referenceMap.has(contentHash)) {
      const referenceId = `ref_${contentHash}`;
      references.set(referenceId, this.referenceMap.get(contentHash));
      return { $ref: referenceId };
    }
    
    // Store reference and process recursively
    this.referenceMap.set(contentHash, obj);
    
    if (Array.isArray(obj)) {
      return obj.map(item => this.deduplicateRecursive(item, references));
    }
    
    const result: any = {};
    for (const [key, value] of Object.entries(obj)) {
      result[key] = this.deduplicateRecursive(value, references);
    }
    
    return result;
  }
  
  private hashContent(obj: any): string {
    // Simple content-based hashing
    return btoa(JSON.stringify(obj)).substring(0, 12);
  }
}
```

## 7. Rollback and Version Control

### 7.1 Hierarchical Version Control

```typescript
interface VersionInfo {
  campaign_version: string;
  session_version: string;
  user_version: string;
  timestamp: number;
  author: string;
  message?: string;
}

class TTRPGVersionControl {
  private snapshots = new Map<string, StateSnapshot>();
  private versionTree = new VersionTree();
  
  async createSnapshot(
    state: any,
    scope: 'campaign' | 'session' | 'user',
    message?: string
  ): Promise<string> {
    const versionId = this.generateVersionId();
    const parentVersion = this.getCurrentVersion(scope);
    
    const snapshot: StateSnapshot = {
      id: versionId,
      parent_id: parentVersion,
      scope,
      state: await this.compressState(state),
      metadata: {
        timestamp: Date.now(),
        author: this.getCurrentUser(),
        message: message || `Auto-snapshot: ${scope}`,
        size: JSON.stringify(state).length
      },
      children: []
    };
    
    this.snapshots.set(versionId, snapshot);
    this.versionTree.addVersion(versionId, parentVersion);
    
    return versionId;
  }
  
  async rollback(
    targetVersion: string,
    scope: 'campaign' | 'session' | 'user'
  ): Promise<any> {
    const snapshot = this.snapshots.get(targetVersion);
    if (!snapshot || snapshot.scope !== scope) {
      throw new Error(`Invalid rollback target: ${targetVersion}`);
    }
    
    // Verify rollback is safe (no dependent changes in other scopes)
    await this.validateRollbackSafety(targetVersion, scope);
    
    // Create branch point before rollback
    const branchPoint = await this.createSnapshot(
      await this.getCurrentState(scope),
      scope,
      `Pre-rollback branch point`
    );
    
    // Perform rollback
    const restoredState = await this.decompressState(snapshot.state);
    await this.applyState(restoredState, scope);
    
    // Record rollback operation
    await this.recordRollback(targetVersion, branchPoint, scope);
    
    return restoredState;
  }
  
  async merge(
    sourceVersion: string,
    targetVersion: string,
    strategy: MergeStrategy = 'auto'
  ): Promise<MergeResult> {
    const sourceSnapshot = this.snapshots.get(sourceVersion);
    const targetSnapshot = this.snapshots.get(targetVersion);
    
    if (!sourceSnapshot || !targetSnapshot) {
      throw new Error('Invalid merge versions');
    }
    
    const sourceState = await this.decompressState(sourceSnapshot.state);
    const targetState = await this.decompressState(targetSnapshot.state);
    
    // Find common ancestor
    const commonAncestor = this.versionTree.findCommonAncestor(
      sourceVersion,
      targetVersion
    );
    
    if (!commonAncestor) {
      throw new Error('Cannot merge: no common ancestor');
    }
    
    // Three-way merge
    const ancestorState = await this.decompressState(
      this.snapshots.get(commonAncestor)!.state
    );
    
    return this.performThreeWayMerge(
      ancestorState,
      sourceState,
      targetState,
      strategy
    );
  }
  
  private async performThreeWayMerge(
    ancestor: any,
    source: any,
    target: any,
    strategy: MergeStrategy
  ): Promise<MergeResult> {
    const conflicts: ConflictInfo[] = [];
    const merged = this.mergeRecursive(ancestor, source, target, conflicts);
    
    if (conflicts.length > 0 && strategy === 'auto') {
      // Attempt automatic resolution
      for (const conflict of conflicts) {
        const resolution = await this.autoResolveConflict(conflict);
        if (resolution) {
          this.applyResolution(merged, conflict.path, resolution);
        }
      }
    }
    
    return {
      merged_state: merged,
      conflicts: conflicts.filter(c => !c.auto_resolved),
      auto_resolved: conflicts.filter(c => c.auto_resolved)
    };
  }
  
  async getVersionHistory(
    scope: 'campaign' | 'session' | 'user',
    limit: number = 50
  ): Promise<VersionInfo[]> {
    const scopeSnapshots = Array.from(this.snapshots.values())
      .filter(s => s.scope === scope)
      .sort((a, b) => b.metadata.timestamp - a.metadata.timestamp)
      .slice(0, limit);
    
    return scopeSnapshots.map(s => ({
      campaign_version: s.id,
      session_version: s.id,
      user_version: s.id,
      timestamp: s.metadata.timestamp,
      author: s.metadata.author,
      message: s.metadata.message
    }));
  }
}

class VersionTree {
  private nodes = new Map<string, VersionNode>();
  private roots = new Set<string>();
  
  addVersion(versionId: string, parentId?: string): void {
    const node: VersionNode = {
      id: versionId,
      parent: parentId,
      children: new Set()
    };
    
    this.nodes.set(versionId, node);
    
    if (parentId) {
      const parent = this.nodes.get(parentId);
      if (parent) {
        parent.children.add(versionId);
      }
    } else {
      this.roots.add(versionId);
    }
  }
  
  findCommonAncestor(versionA: string, versionB: string): string | null {
    const ancestorsA = this.getAncestors(versionA);
    const ancestorsB = this.getAncestors(versionB);
    
    // Find first common ancestor
    for (const ancestor of ancestorsA) {
      if (ancestorsB.has(ancestor)) {
        return ancestor;
      }
    }
    
    return null;
  }
  
  private getAncestors(versionId: string): Set<string> {
    const ancestors = new Set<string>();
    let current = versionId;
    
    while (current) {
      ancestors.add(current);
      const node = this.nodes.get(current);
      current = node?.parent || '';
    }
    
    return ancestors;
  }
}
```

## 8. Implementation Examples

### 8.1 Frontend Implementation

```typescript
// Enhanced collaboration store with hybrid synchronization
class EnhancedCollaborationStore {
  private syncEngine: HybridSyncEngine;
  private conflictResolver: TTRPGConflictResolver;
  private versionControl: TTRPGVersionControl;
  private cacheManager: TTRPGCacheManager;
  
  constructor() {
    this.syncEngine = new HybridSyncEngine({
      eventStore: new ChromaEventStore(),
      stateStore: new ChromaStateStore(),
      conflictResolver: new TTRPGConflictResolver()
    });
  }
  
  // Optimistic update with intelligent rollback
  async updateCharacter(
    characterId: string,
    updates: Partial<Character>
  ): Promise<void> {
    const operationId = this.generateOperationId();
    
    try {
      // Apply optimistically
      const optimisticState = this.applyOptimisticUpdate(
        characterId,
        updates,
        operationId
      );
      
      // Send to server
      await this.syncEngine.sendUpdate({
        type: 'character_update',
        target_id: characterId,
        changes: updates,
        operation_id: operationId,
        previous_version: this.getCharacterVersion(characterId)
      });
      
      // Create snapshot for rollback
      await this.versionControl.createSnapshot(
        optimisticState,
        'user',
        `Character update: ${characterId}`
      );
      
    } catch (error) {
      // Rollback optimistic update
      await this.rollbackOptimisticUpdate(operationId);
      throw error;
    }
  }
  
  // Smart conflict resolution with user feedback
  async handleConflict(conflict: ConflictInfo): Promise<void> {
    // Try automatic resolution first
    const autoResolution = await this.conflictResolver.resolve(conflict);
    
    if (autoResolution.confidence > 0.8) {
      // High confidence - apply automatically
      await this.applyResolution(autoResolution);
      
      // Learn from successful auto-resolution
      await this.recordResolutionFeedback(conflict.id, 'auto_success');
    } else {
      // Low confidence - ask user
      const userChoice = await this.presentConflictToUser(conflict);
      await this.applyResolution(userChoice);
      
      // Learn from user choice
      await this.recordResolutionFeedback(conflict.id, 'user_guided');
    }
  }
  
  // Intelligent preloading based on session context
  async preloadSessionData(sessionId: string): Promise<void> {
    const session = await this.getSession(sessionId);
    const participants = session.participants;
    
    // Preload critical data
    const criticalIds = [
      ...participants.map(p => p.character_id),
      session.current_encounter?.monster_ids || [],
      session.active_location_id
    ].flat().filter(Boolean);
    
    await Promise.all([
      this.preloadCharacters(criticalIds),
      this.preloadInitiativeOrder(sessionId),
      this.preloadRecentDiceRolls(sessionId),
      this.preloadChatHistory(sessionId, 100) // Last 100 messages
    ]);
  }
}

// Real-time synchronization with WebSocket
class RealTimeSyncService {
  private ws: WebSocket | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectBackoff = new ExponentialBackoff(1000, 30000);
  
  async connect(userId: string): Promise<void> {
    const wsUrl = this.buildWebSocketUrl(userId);
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      this.startHeartbeat();
      this.reconnectBackoff.reset();
    };
    
    this.ws.onmessage = (event) => {
      this.handleMessage(JSON.parse(event.data));
    };
    
    this.ws.onclose = () => {
      this.stopHeartbeat();
      this.scheduleReconnect();
    };
  }
  
  private async handleMessage(message: SyncMessage): Promise<void> {
    switch (message.type) {
      case 'state_update':
        await this.handleStateUpdate(message);
        break;
      case 'conflict_detected':
        await this.handleConflict(message);
        break;
      case 'sync_response':
        await this.handleSyncResponse(message);
        break;
    }
  }
  
  private async handleStateUpdate(message: StateUpdateMessage): Promise<void> {
    // Validate message integrity
    if (!this.validateMessage(message)) {
      return;
    }
    
    // Check for conflicts
    const hasConflict = await this.detectLocalConflict(message.update);
    
    if (hasConflict) {
      await this.conflictResolver.resolveConflict({
        remote_update: message.update,
        local_state: this.getCurrentState(),
        resolution_strategy: 'semantic_merge'
      });
    } else {
      // Apply update directly
      await this.applyStateUpdate(message.update);
    }
    
    // Acknowledge receipt
    this.sendAcknowledgment(message.id);
  }
}
```

### 8.2 Backend Implementation

```typescript
// MCP Server with hybrid synchronization
class TTRPGMCPServer extends MCPServer {
  private syncEngine: HybridSyncEngine;
  private eventStore: ChromaEventStore;
  private stateManager: StateManager;
  private conflictResolver: SemanticConflictResolver;
  
  constructor() {
    super();
    this.setupSynchronizationEngine();
    this.registerHandlers();
  }
  
  private setupSynchronizationEngine(): void {
    this.syncEngine = new HybridSyncEngine({
      eventStore: new ChromaEventStore({
        collection_prefix: 'ttrpg_events',
        embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'
      }),
      stateStore: new ChromaStateStore({
        collection_prefix: 'ttrpg_state',
        consistency_level: 'strong'
      }),
      conflictResolver: new SemanticConflictResolver({
        vectorDB: this.chromaClient,
        gameRules: new D20GameRules()
      })
    });
  }
  
  // Handle real-time collaboration requests
  @handler('collaborate')
  async handleCollaboration(request: CollaborationRequest): Promise<any> {
    switch (request.action) {
      case 'join_session':
        return this.joinCollaborativeSession(request);
      case 'update_state':
        return this.updateSharedState(request);
      case 'resolve_conflict':
        return this.resolveConflict(request);
      case 'sync_state':
        return this.syncState(request);
    }
  }
  
  private async updateSharedState(request: UpdateStateRequest): Promise<any> {
    const { session_id, user_id, update, operation_id } = request;
    
    try {
      // Validate permissions
      await this.validateUserPermissions(session_id, user_id, update.path);
      
      // Check for conflicts
      const conflictCheck = await this.detectConflicts(session_id, update);
      
      if (conflictCheck.hasConflicts) {
        // Attempt automatic resolution
        const resolution = await this.conflictResolver.resolve(conflictCheck);
        
        if (resolution.requiresUserInput) {
          return {
            status: 'conflict',
            conflict_id: conflictCheck.id,
            resolution_options: resolution.options
          };
        } else {
          // Apply auto-resolved update
          update.value = resolution.resolvedValue;
        }
      }
      
      // Record event in event store
      await this.eventStore.append({
        stream_id: session_id,
        event_type: 'state_updated',
        event_data: update,
        metadata: {
          user_id,
          operation_id,
          timestamp: Date.now()
        }
      });
      
      // Update current state
      await this.stateManager.applyUpdate(session_id, update);
      
      // Broadcast to other participants
      await this.broadcastStateUpdate(session_id, update, user_id);
      
      // Create snapshot if significant change
      if (await this.shouldSnapshot(session_id, update)) {
        await this.createStateSnapshot(session_id);
      }
      
      return {
        status: 'success',
        new_version: update.version
      };
      
    } catch (error) {
      console.error('State update failed:', error);
      return {
        status: 'error',
        error: error.message
      };
    }
  }
  
  private async detectConflicts(
    sessionId: string,
    update: StateUpdate
  ): Promise<ConflictCheck> {
    // Get current state version
    const currentVersion = await this.stateManager.getCurrentVersion(sessionId);
    
    // Check version mismatch
    if (update.previous_version !== currentVersion) {
      // Get conflicting updates
      const conflictingUpdates = await this.eventStore.getEventsSince(
        sessionId,
        update.previous_version
      );
      
      // Analyze semantic conflicts using ChromaDB
      const semanticConflicts = await this.analyzeSemanticConflicts(
        update,
        conflictingUpdates
      );
      
      return {
        hasConflicts: true,
        id: this.generateConflictId(),
        conflicting_updates: conflictingUpdates,
        semantic_conflicts: semanticConflicts,
        resolution_strategies: this.getApplicableStrategies(update)
      };
    }
    
    return { hasConflicts: false };
  }
  
  private async analyzeSemanticConflicts(
    newUpdate: StateUpdate,
    existingUpdates: StateUpdate[]
  ): Promise<SemanticConflictAnalysis> {
    // Extract text content for semantic analysis
    const newContent = this.extractTextContent(newUpdate.value);
    const existingContent = existingUpdates.map(u => 
      this.extractTextContent(u.value)
    );
    
    // Generate embeddings
    const embeddings = await Promise.all([
      this.embedText(newContent),
      ...existingContent.map(c => this.embedText(c))
    ]);
    
    // Calculate semantic similarity
    const similarities = embeddings.slice(1).map(embedding =>
      this.cosineSimilarity(embeddings[0], embedding)
    );
    
    // Identify semantic conflicts (high similarity but different content)
    const semanticConflicts = similarities
      .map((similarity, index) => ({
        update: existingUpdates[index],
        similarity,
        isSemanticConflict: similarity > 0.7 && similarity < 0.95
      }))
      .filter(result => result.isSemanticConflict);
    
    return {
      conflicts: semanticConflicts,
      suggested_resolution: this.suggestResolutionStrategy(semanticConflicts)
    };
  }
}

// ChromaDB-based event store
class ChromaEventStore implements EventStore {
  constructor(
    private client: ChromaClient,
    private config: ChromaEventStoreConfig
  ) {}
  
  async append(event: DomainEvent): Promise<void> {
    const eventText = this.extractEventText(event);
    const embedding = await this.generateEmbedding(eventText);
    
    await this.client.upsert({
      collection_name: this.getCollectionName(event.stream_id),
      documents: [JSON.stringify(event)],
      embeddings: [embedding],
      metadatas: [{
        stream_id: event.stream_id,
        event_type: event.event_type,
        version: event.version,
        timestamp: event.timestamp,
        user_id: event.metadata.user_id
      }]
    });
  }
  
  async getEventsSince(
    streamId: string,
    version: number
  ): Promise<DomainEvent[]> {
    const results = await this.client.query({
      collection_name: this.getCollectionName(streamId),
      query_filter: {
        stream_id: { $eq: streamId },
        version: { $gt: version }
      }
    });
    
    return results.documents.map(doc => JSON.parse(doc) as DomainEvent);
  }
  
  async findSimilarEvents(
    eventContent: string,
    threshold: number = 0.8
  ): Promise<SimilarEvent[]> {
    const queryEmbedding = await this.generateEmbedding(eventContent);
    
    const results = await this.client.query({
      collection_name: this.config.collection_prefix + '_all',
      query_embeddings: [queryEmbedding],
      n_results: 10
    });
    
    return results.documents
      .map((doc, index) => ({
        event: JSON.parse(doc) as DomainEvent,
        similarity: results.distances![index]
      }))
      .filter(result => result.similarity >= threshold);
  }
}
```

## 9. Recommendations and Next Steps

### 9.1 Implementation Priority

**Phase 1: Core Synchronization (Weeks 1-2)**
1. Implement hybrid event sourcing + state-based architecture
2. Enhanced CRDT classes for TTRPG data types
3. Basic conflict resolution with semantic awareness
4. Multi-tier caching system

**Phase 2: Advanced Features (Weeks 3-4)**
1. ChromaDB integration for vector-based conflict detection
2. Intelligent preloading and lazy loading
3. Version control with rollback capabilities
4. Performance optimizations for large campaigns

**Phase 3: Production Hardening (Weeks 5-6)**
1. Comprehensive testing with large datasets
2. Monitoring and observability
3. Security audit and hardening
4. Documentation and training materials

### 9.2 Key Architecture Decisions

| Decision Point | Recommendation | Rationale |
|---------------|----------------|-----------|
| Synchronization Model | Hybrid Event Sourcing + CRDT | Best balance of audit trails and performance |
| Conflict Resolution | Semantic with ChromaDB vectors | Handles complex TTRPG narrative content |
| State Partitioning | Campaign → Session → User hierarchy | Matches natural TTRPG data boundaries |
| Caching Strategy | Multi-tier with smart invalidation | Optimizes for read-heavy TTRPG access patterns |
| Persistence | ChromaDB with vector embeddings | Enables semantic search and conflict detection |

### 9.3 Performance Targets

- **State synchronization latency**: < 50ms for critical updates
- **Conflict resolution time**: < 200ms for automatic resolution
- **Cache hit rate**: > 85% for frequently accessed data
- **Memory usage**: < 100MB per active session
- **Network bandwidth**: < 1KB/sec per participant average

### 9.4 Risk Mitigation

**Technical Risks:**
- **ChromaDB scalability**: Implement sharding strategy for large campaigns
- **Conflict resolution complexity**: Fallback to simpler strategies when needed
- **Memory leaks**: Implement proper cleanup and garbage collection
- **Network partitions**: Robust offline mode with eventual consistency

**Business Risks:**
- **User adoption**: Gradual rollout with feature flags
- **Performance regression**: Comprehensive benchmarking before release
- **Data loss**: Multiple backup strategies and version control

## 10. Conclusion

The hybrid Event Sourcing + CRDT approach with ChromaDB integration provides the optimal balance for TTRPG state synchronization. This architecture supports:

- **Real-time collaboration** with sub-100ms latency
- **Semantic conflict resolution** for narrative content
- **Robust version control** for campaign management
- **Scalable performance** for large, long-running campaigns

The implementation leverages vector embeddings for intelligent conflict detection while maintaining the simplicity needed for TTRPG gameplay. The multi-tier caching and lazy loading strategies ensure good performance even with extensive campaign data.

This spike provides a solid foundation for implementing state synchronization that truly understands the unique requirements of tabletop role-playing games.