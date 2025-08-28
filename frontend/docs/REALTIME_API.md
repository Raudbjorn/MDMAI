# Real-time Features API Documentation

## Overview
The real-time features provide WebSocket and Server-Sent Events (SSE) support for collaborative functionality in the TTRPG Assistant.

## WebSocket API

### Connection
```typescript
const client = new WebSocketClient('ws://localhost:8080', {
  reconnect: true,
  maxReconnectAttempts: 5,
  reconnectDelay: 1000,
  heartbeatInterval: 30000
});

await client.connect();
```

### Message Format
```typescript
interface WSMessage {
  type: 'request' | 'response' | 'event' | 'error';
  id?: string;
  method?: string;
  params?: any;
  data?: any;
  error?: {
    code: number;
    message: string;
  };
  timestamp: number;
}
```

### Events

#### Connection Events
- `connected`: WebSocket connection established
- `disconnected`: Connection closed
- `reconnecting`: Attempting to reconnect
- `error`: Connection error occurred

#### Message Events
- `message`: Incoming message received
- `heartbeat`: Heartbeat received
- `latency`: Latency measurement updated

### Methods

#### send(message: any): void
Send a message through the WebSocket connection.

#### request(method: string, params?: any, timeout?: number): Promise<any>
Send a request and wait for response.

#### disconnect(): void
Close the WebSocket connection.

## SSE API

### Connection
```typescript
const client = new SSEClient('http://localhost:8080/sse', {
  reconnect: true,
  reconnectDelay: 1000,
  eventFilter: (event) => event.type !== 'heartbeat'
});

await client.connect();
```

### Event Format
```typescript
interface SSEEvent {
  id?: string;
  type: string;
  data: any;
  timestamp: number;
}
```

### Methods

#### on(event: string, handler: Function): void
Register an event handler.

#### off(event: string, handler: Function): void
Remove an event handler.

## Collaborative Canvas API

### Drawing Operations
```typescript
interface DrawOperation {
  type: 'draw' | 'erase' | 'shape';
  tool: 'pen' | 'eraser' | 'rectangle' | 'circle';
  points: Point[];
  color: string;
  size: number;
  userId: string;
}
```

### Canvas Events
- `draw`: Drawing operation performed
- `clear`: Canvas cleared
- `undo`: Undo operation
- `redo`: Redo operation

## Presence System

### Presence Data
```typescript
interface PresenceData {
  userId: string;
  status: 'online' | 'away' | 'offline';
  activity?: {
    type: 'typing' | 'drawing' | 'viewing';
    location?: string;
  };
  cursor?: {
    x: number;
    y: number;
  };
  lastSeen: number;
}
```

## Activity Feed

### Activity Types
```typescript
type ActivityType = 
  | 'user_joined'
  | 'user_left'
  | 'chat_message'
  | 'dice_roll'
  | 'turn_changed'
  | 'map_updated'
  | 'note_edited';
```

### Activity Format
```typescript
interface Activity {
  id: string;
  type: ActivityType;
  userId: string;
  username: string;
  data: any;
  timestamp: number;
}
```

## Error Handling

### Error Codes
- `1000`: Normal closure
- `1001`: Going away
- `1002`: Protocol error
- `1003`: Unsupported data
- `1006`: Abnormal closure
- `1007`: Invalid data
- `1008`: Policy violation
- `1009`: Message too big
- `1011`: Internal error

### Retry Strategy
- Exponential backoff with jitter
- Maximum retry attempts: 5
- Max delay: 30 seconds

## Security Considerations

### Authentication
- JWT tokens in connection headers
- Session validation on connect
- Automatic token refresh

### Rate Limiting
- Maximum 100 messages per minute
- Throttled cursor updates (60Hz max)
- Debounced drawing operations

### Data Validation
- Input sanitization
- Message size limits (64KB)
- Type validation on all messages

## Performance Guidelines

### Optimization Tips
1. Batch multiple operations
2. Use throttling for frequent updates
3. Implement message deduplication
4. Use binary formats for large data
5. Enable compression when available

### Monitoring
- Track connection latency
- Monitor message queue size
- Log reconnection attempts
- Measure bandwidth usage

## Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Fallback Mechanism
1. Attempt WebSocket connection
2. Fall back to SSE if WebSocket fails
3. Fall back to polling if SSE fails
4. Show offline mode if all fail