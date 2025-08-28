# Provider Management UI

Complete provider management system for the TTRPG Assistant frontend, integrating with the AI provider backend infrastructure.

## Features

### 1. Provider Configuration (`ProviderConfig.svelte`)
- Configure Anthropic, OpenAI, and Google AI providers
- Set API keys with secure encryption
- Adjust rate limits, timeouts, and retry settings
- Priority-based provider selection
- Real-time health status monitoring

### 2. Credential Management (`CredentialManager.svelte`)
- Secure API key input with validation
- Client-side encryption before transmission
- Visual feedback for validation status
- Direct links to provider API key pages
- Masked display of sensitive data

### 3. Provider Switching (`ProviderSwitcher.svelte`)
- Hot-swap between configured providers
- Real-time status indicators
- Latency and rate limit display
- Priority-based provider ordering
- Dropdown interface with quick access

### 4. Cost Dashboard (`CostDashboard.svelte`)
- Real-time cost tracking by provider
- Budget configuration and alerts
- Daily, weekly, and monthly views
- Cost breakdown visualizations
- Budget usage warnings

### 5. Usage Analytics (`UsageAnalytics.svelte`)
- Request volume tracking
- Token usage statistics
- Success rate monitoring
- Latency performance metrics
- Provider comparison tables

## Security Features

- **Client-side encryption**: API keys are encrypted using Web Crypto API before sending to backend
- **Secure storage**: Encrypted credentials stored with AES-GCM encryption
- **Key validation**: Format validation before API calls
- **HTTPS enforcement**: Security warnings for non-secure contexts
- **Token masking**: Sensitive data masked in UI displays

## Store Architecture

The provider management uses Svelte 5 runes for state management:

```typescript
// Provider store with reactive state
class ProviderStore {
  private state = $state<ProviderState>({...});
  
  // Derived values
  enabledProviders = $derived(...);
  totalCost = $derived(...);
  overallHealth = $derived(...);
}
```

## API Integration

The system communicates with the backend through:
- `/api/providers/configure` - Provider configuration
- `/api/providers/credentials` - Secure credential storage
- `/api/providers/health` - Health status monitoring
- `/api/providers/stats` - Usage statistics
- `/api/providers/budgets` - Cost budget management

## Usage

### Basic Setup

```svelte
<script lang="ts">
  import { ProviderConfig } from '$lib/components/providers';
  import { ProviderType } from '$lib/types/providers';
</script>

<ProviderConfig providerType={ProviderType.ANTHROPIC} />
```

### Provider Switching

```svelte
<script lang="ts">
  import { ProviderSwitcher } from '$lib/components/providers';
</script>

<!-- Add to navigation or toolbar -->
<ProviderSwitcher />
```

### Cost Monitoring

```svelte
<script lang="ts">
  import { CostDashboard } from '$lib/components/providers';
</script>

<CostDashboard />
```

## Type Safety

All components use TypeScript with comprehensive type definitions:

```typescript
interface ProviderConfig {
  provider_type: ProviderType;
  api_key: string;
  enabled: boolean;
  priority: number;
  // ... more configuration
}
```

## Performance Optimizations

- **Lazy loading**: Components loaded on demand
- **Debounced API calls**: Prevents excessive requests
- **Cached health checks**: 30-second cache for health status
- **Optimistic UI**: Immediate feedback before server confirmation

## Accessibility

- ARIA labels for all interactive elements
- Keyboard navigation support
- Focus management for modals
- Screen reader announcements for status changes

## Error Handling

Follows error-as-values pattern:

```typescript
type Result<T, E = Error> = 
  | { ok: true; value: T }
  | { ok: false; error: E };
```

## Testing

Components are designed for testability:
- Isolated business logic in stores
- Mockable API client
- Component prop interfaces for easy testing
- Error boundary support

## Future Enhancements

- [ ] Model selection per provider
- [ ] Advanced cost forecasting
- [ ] Provider fallback chains
- [ ] Usage quota notifications
- [ ] Export usage reports
- [ ] Team-based budgets