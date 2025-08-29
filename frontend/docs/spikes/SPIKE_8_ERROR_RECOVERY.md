# Spike 8: Error Recovery & Resilience

## Overview
This spike defines comprehensive error recovery strategies for the TTRPG MCP Server to ensure reliability during sessions.

## 1. Error Classification

### 1.1 Error Taxonomy
```typescript
enum ErrorSeverity {
  CRITICAL = 'critical',  // System failure, session cannot continue
  HIGH = 'high',         // Major feature broken, degraded experience
  MEDIUM = 'medium',     // Minor feature issue, workaround available
  LOW = 'low'           // Cosmetic issue, no impact on functionality
}

interface TTRPGError {
  code: string;
  severity: ErrorSeverity;
  category: 'network' | 'data' | 'auth' | 'integration' | 'system';
  message: string;
  recoverable: boolean;
  retryStrategy?: RetryStrategy;
  fallback?: () => Promise<any>;
  context?: Record<string, any>;
}
```

### 1.2 Error Recovery Matrix
```typescript
const ERROR_RECOVERY_STRATEGIES: Record<string, RecoveryStrategy> = {
  'WEBSOCKET_DISCONNECTED': {
    severity: 'high',
    recoverable: true,
    strategy: 'exponential_backoff',
    maxRetries: 5,
    fallback: 'polling_mode'
  },
  'MCP_TIMEOUT': {
    severity: 'medium',
    recoverable: true,
    strategy: 'immediate_retry',
    maxRetries: 3,
    fallback: 'cached_response'
  },
  'DATABASE_UNAVAILABLE': {
    severity: 'critical',
    recoverable: false,
    strategy: 'circuit_breaker',
    fallback: 'readonly_mode'
  },
  'AI_PROVIDER_ERROR': {
    severity: 'medium',
    recoverable: true,
    strategy: 'provider_fallback',
    fallback: 'alternative_provider'
  }
};
```

## 2. Recovery Strategies

### 2.1 Circuit Breaker Pattern
```python
from enum import Enum
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = 'closed'  # Normal operation
    OPEN = 'open'      # Failing, reject calls
    HALF_OPEN = 'half_open'  # Testing recovery

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > 
            timedelta(seconds=self.recovery_timeout)
        )
```

### 2.2 Exponential Backoff with Jitter
```typescript
export class ExponentialBackoff {
  private attempt = 0;
  private readonly maxAttempts: number;
  private readonly baseDelay: number;
  private readonly maxDelay: number;
  
  constructor(options: {
    maxAttempts?: number;
    baseDelay?: number;
    maxDelay?: number;
  } = {}) {
    this.maxAttempts = options.maxAttempts ?? 5;
    this.baseDelay = options.baseDelay ?? 1000;
    this.maxDelay = options.maxDelay ?? 30000;
  }
  
  async execute<T>(
    fn: () => Promise<T>,
    onRetry?: (attempt: number, error: Error) => void
  ): Promise<T> {
    while (this.attempt < this.maxAttempts) {
      try {
        return await fn();
      } catch (error) {
        this.attempt++;
        
        if (this.attempt >= this.maxAttempts) {
          throw error;
        }
        
        const delay = this.calculateDelay();
        onRetry?.(this.attempt, error as Error);
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw new Error('Max attempts reached');
  }
  
  private calculateDelay(): number {
    // Exponential backoff with jitter
    const exponentialDelay = Math.min(
      this.baseDelay * Math.pow(2, this.attempt - 1),
      this.maxDelay
    );
    
    // Add jitter (±25%)
    const jitter = exponentialDelay * 0.25 * (Math.random() * 2 - 1);
    
    return Math.round(exponentialDelay + jitter);
  }
}
```

### 2.3 Saga Pattern for Distributed Transactions
```python
from typing import List, Callable, Any
import asyncio

class Saga:
    """Saga pattern for distributed transactions in TTRPG operations"""
    
    def __init__(self, name: str):
        self.name = name
        self.steps: List[SagaStep] = []
        self.completed_steps: List[str] = []
        
    def add_step(
        self,
        name: str,
        action: Callable,
        compensate: Callable
    ):
        self.steps.append(SagaStep(name, action, compensate))
        
    async def execute(self, context: dict) -> dict:
        """Execute saga with automatic compensation on failure"""
        try:
            for step in self.steps:
                result = await step.action(context)
                context[f'{step.name}_result'] = result
                self.completed_steps.append(step.name)
                
            return context
            
        except Exception as e:
            # Compensate in reverse order
            await self._compensate(context)
            raise SagaException(f"Saga {self.name} failed: {e}")
    
    async def _compensate(self, context: dict):
        """Run compensation for completed steps"""
        for step_name in reversed(self.completed_steps):
            step = next(s for s in self.steps if s.name == step_name)
            try:
                await step.compensate(context)
            except Exception as e:
                # Log compensation failure but continue
                logger.error(f"Compensation failed for {step_name}: {e}")

# Example: Character creation saga
character_creation_saga = Saga("character_creation")

character_creation_saga.add_step(
    "validate_stats",
    action=lambda ctx: validate_character_stats(ctx['stats']),
    compensate=lambda ctx: None  # No compensation needed
)

character_creation_saga.add_step(
    "save_to_db",
    action=lambda ctx: save_character_to_db(ctx['character']),
    compensate=lambda ctx: delete_character_from_db(ctx['character_id'])
)

character_creation_saga.add_step(
    "update_campaign",
    action=lambda ctx: add_character_to_campaign(ctx['campaign_id'], ctx['character_id']),
    compensate=lambda ctx: remove_character_from_campaign(ctx['campaign_id'], ctx['character_id'])
)

character_creation_saga.add_step(
    "notify_players",
    action=lambda ctx: broadcast_new_character(ctx['character']),
    compensate=lambda ctx: broadcast_character_removal(ctx['character_id'])
)
```

## 3. Session Recovery

### 3.1 Session State Persistence
```typescript
export class SessionRecovery {
  private readonly CHECKPOINT_INTERVAL = 30000; // 30 seconds
  private checkpointTimer: NodeJS.Timer | null = null;
  
  constructor(
    private storage: IndexedDBStorage,
    private sessionId: string
  ) {}
  
  startCheckpointing() {
    this.checkpointTimer = setInterval(
      () => this.saveCheckpoint(),
      this.CHECKPOINT_INTERVAL
    );
  }
  
  async saveCheckpoint() {
    const state = {
      sessionId: this.sessionId,
      timestamp: Date.now(),
      initiative: await this.getInitiativeState(),
      characters: await this.getCharacterStates(),
      notes: await this.getSessionNotes(),
      dice_history: await this.getDiceHistory(),
      map_state: await this.getMapState()
    };
    
    await this.storage.saveCheckpoint(state);
  }
  
  async recoverSession(): Promise<SessionState | null> {
    const checkpoints = await this.storage.getCheckpoints(this.sessionId);
    
    if (checkpoints.length === 0) {
      return null;
    }
    
    // Get most recent valid checkpoint
    for (const checkpoint of checkpoints) {
      if (await this.validateCheckpoint(checkpoint)) {
        return this.restoreFromCheckpoint(checkpoint);
      }
    }
    
    return null;
  }
  
  private async validateCheckpoint(checkpoint: any): Promise<boolean> {
    // Verify data integrity
    if (!checkpoint.sessionId || !checkpoint.timestamp) {
      return false;
    }
    
    // Check if checkpoint is not too old (max 24 hours)
    const age = Date.now() - checkpoint.timestamp;
    if (age > 24 * 60 * 60 * 1000) {
      return false;
    }
    
    // Verify critical data exists
    return !!checkpoint.initiative && !!checkpoint.characters;
  }
}
```

### 3.2 Reconnection Manager
```typescript
export class ReconnectionManager {
  private reconnectAttempts = 0;
  private isReconnecting = false;
  
  constructor(
    private wsClient: WebSocketClient,
    private sseClient: SSEClient,
    private onReconnect: () => void
  ) {
    this.setupListeners();
  }
  
  private setupListeners() {
    // WebSocket disconnection
    this.wsClient.on('disconnect', () => {
      this.handleDisconnection('websocket');
    });
    
    // SSE disconnection
    this.sseClient.on('error', () => {
      this.handleDisconnection('sse');
    });
  }
  
  private async handleDisconnection(source: string) {
    if (this.isReconnecting) return;
    
    this.isReconnecting = true;
    
    // Save current state
    await this.saveLocalState();
    
    // Attempt reconnection
    while (this.reconnectAttempts < 5) {
      try {
        await this.reconnect(source);
        await this.resyncState();
        this.onReconnect();
        break;
      } catch (error) {
        this.reconnectAttempts++;
        await this.waitWithBackoff();
      }
    }
    
    if (this.reconnectAttempts >= 5) {
      this.enterOfflineMode();
    }
    
    this.isReconnecting = false;
  }
  
  private async resyncState() {
    // Get server state
    const serverState = await this.fetchServerState();
    
    // Get local state
    const localState = await this.getLocalState();
    
    // Merge states
    const mergedState = this.mergeStates(serverState, localState);
    
    // Apply merged state
    await this.applyState(mergedState);
  }
}
```

## 4. Data Recovery

### 4.1 Conflict Resolution
```python
class ConflictResolver:
    """Resolve conflicts in TTRPG data"""
    
    def resolve_character_conflict(
        self,
        local: dict,
        remote: dict
    ) -> dict:
        """Resolve character data conflicts"""
        # Use timestamp-based resolution for most fields
        resolved = {}
        
        # HP: Take the lower value (conservative approach)
        resolved['hp'] = min(local.get('hp', 0), remote.get('hp', 0))
        
        # XP: Take the higher value (never lose progress)
        resolved['xp'] = max(local.get('xp', 0), remote.get('xp', 0))
        
        # Inventory: Merge with deduplication
        local_items = set(local.get('inventory', []))
        remote_items = set(remote.get('inventory', []))
        resolved['inventory'] = list(local_items | remote_items)
        
        # Status effects: Union of both
        resolved['conditions'] = list(set(
            local.get('conditions', []) + 
            remote.get('conditions', [])
        ))
        
        # Notes: Concatenate with timestamps
        resolved['notes'] = self.merge_notes(
            local.get('notes', ''),
            remote.get('notes', '')
        )
        
        return resolved
    
    def resolve_dice_roll_conflict(
        self,
        local: dict,
        remote: dict
    ) -> dict:
        """Dice rolls should never conflict - keep both"""
        # Return both rolls with conflict marker
        return {
            'conflict': True,
            'local_roll': local,
            'remote_roll': remote,
            'resolution': 'manual_required'
        }
```

### 4.2 Data Validation & Repair
```python
class DataValidator:
    def validate_and_repair(self, data: dict, schema: dict) -> tuple[bool, dict]:
        """Validate data and attempt repairs"""
        errors = []
        repaired = data.copy()
        
        for field, rules in schema.items():
            if field not in repaired:
                if rules.get('required'):
                    # Attempt to provide default
                    if 'default' in rules:
                        repaired[field] = rules['default']
                        errors.append(f"Added default for {field}")
                    else:
                        errors.append(f"Missing required field: {field}")
            else:
                # Validate type
                expected_type = rules.get('type')
                if expected_type and not isinstance(repaired[field], expected_type):
                    # Attempt type coercion
                    try:
                        repaired[field] = expected_type(repaired[field])
                        errors.append(f"Coerced {field} to {expected_type}")
                    except:
                        errors.append(f"Invalid type for {field}")
                
                # Validate range
                if 'min' in rules and repaired[field] < rules['min']:
                    repaired[field] = rules['min']
                    errors.append(f"Clamped {field} to minimum")
                
                if 'max' in rules and repaired[field] > rules['max']:
                    repaired[field] = rules['max']
                    errors.append(f"Clamped {field} to maximum")
        
        is_valid = len(errors) == 0
        return is_valid, repaired
```

## 5. User Experience During Errors

### 5.1 Error Notification System
```svelte
<!-- ErrorNotification.svelte -->
<script lang="ts">
  import { errorStore } from '$lib/stores/errors';
  import { fly, fade } from 'svelte/transition';
  
  let errors = $state<TTRPGError[]>([]);
  
  $effect(() => {
    errors = errorStore.errors;
  });
  
  function dismissError(id: string) {
    errorStore.dismiss(id);
  }
  
  function getErrorIcon(severity: ErrorSeverity) {
    switch(severity) {
      case 'critical': return '🔴';
      case 'high': return '🟠';
      case 'medium': return '🟡';
      case 'low': return '🔵';
    }
  }
</script>

<div class="error-container">
  {#each errors as error (error.id)}
    <div 
      class="error-notification {error.severity}"
      transition:fly={{ y: 20, duration: 300 }}
    >
      <span class="error-icon">{getErrorIcon(error.severity)}</span>
      <div class="error-content">
        <div class="error-message">{error.message}</div>
        {#if error.recoverable}
          <div class="error-status">Attempting recovery...</div>
        {/if}
        {#if error.fallback}
          <button onclick={() => error.fallback?.()}>
            Use Alternative
          </button>
        {/if}
      </div>
      <button 
        class="dismiss-btn"
        onclick={() => dismissError(error.id)}
      >
        ×
      </button>
    </div>
  {/each}
</div>
```

### 5.2 Graceful Degradation
```typescript
export class GracefulDegradation {
  private features = new Map<string, FeatureStatus>();
  
  registerFeature(name: string, config: FeatureConfig) {
    this.features.set(name, {
      name,
      status: 'available',
      dependencies: config.dependencies || [],
      fallback: config.fallback,
      priority: config.priority || 'medium'
    });
  }
  
  async checkFeatureAvailability() {
    for (const [name, feature] of this.features) {
      try {
        // Check dependencies
        for (const dep of feature.dependencies) {
          if (!await this.isDependencyAvailable(dep)) {
            this.degradeFeature(name);
            break;
          }
        }
      } catch (error) {
        this.degradeFeature(name);
      }
    }
  }
  
  private degradeFeature(name: string) {
    const feature = this.features.get(name);
    if (!feature) return;
    
    feature.status = 'degraded';
    
    // Notify user
    this.notifyDegradation(feature);
    
    // Activate fallback
    if (feature.fallback) {
      feature.fallback();
    }
  }
}
```

## 6. Monitoring & Alerting

### 6.1 Error Tracking
```python
class ErrorTracker:
    def __init__(self):
        self.errors = []
        self.error_rates = {}
        
    def track_error(self, error: dict):
        """Track error for analysis"""
        self.errors.append({
            **error,
            'timestamp': datetime.utcnow(),
            'session_id': self.get_session_id(),
            'user_id': self.get_user_id()
        })
        
        # Update error rates
        error_type = error.get('type')
        if error_type not in self.error_rates:
            self.error_rates[error_type] = ErrorRate()
        self.error_rates[error_type].increment()
        
        # Check for error patterns
        self.detect_error_patterns()
    
    def detect_error_patterns(self):
        """Detect concerning error patterns"""
        # High error rate
        for error_type, rate in self.error_rates.items():
            if rate.get_rate() > 10:  # 10 errors per minute
                self.alert_high_error_rate(error_type)
        
        # Error cascade
        recent_errors = self.get_recent_errors(60)  # Last minute
        if len(recent_errors) > 50:
            self.alert_error_cascade()
        
        # Repeated failures
        if self.has_repeated_failures():
            self.alert_repeated_failures()
```

## Implementation Timeline

### Week 1: Core Error Handling
- [ ] Error classification system
- [ ] Circuit breaker implementation
- [ ] Exponential backoff

### Week 2: Recovery Mechanisms
- [ ] Session recovery
- [ ] Reconnection manager
- [ ] Data conflict resolution

### Week 3: User Experience
- [ ] Error notifications
- [ ] Graceful degradation
- [ ] Offline mode transitions

### Week 4: Monitoring
- [ ] Error tracking
- [ ] Pattern detection
- [ ] Alerting system

## Conclusion

This comprehensive error recovery strategy ensures the TTRPG MCP Server can handle failures gracefully, recover automatically when possible, and provide clear feedback to users during issues.