# Spike 9: Testing Strategy

## Overview
Comprehensive testing strategy for the TTRPG MCP Server covering unit, integration, end-to-end, and performance testing.

## 1. Testing Architecture

### 1.1 Test Pyramid
```
         /\
        /E2E\        (5%) - Critical user journeys
       /------\
      /Integration\  (25%) - API, WebSocket, MCP tools  
     /-------------\
    /     Unit      \ (70%) - Components, utilities, stores
   /------------------\
```

### 1.2 Testing Stack
```yaml
# Frontend Testing
unit:
  framework: vitest
  coverage: c8
  mocking: vitest-mock

integration:
  framework: playwright
  api: msw (Mock Service Worker)
  
e2e:
  framework: playwright
  environment: docker-compose
  
# Backend Testing  
unit:
  framework: pytest
  coverage: pytest-cov
  mocking: pytest-mock
  
integration:
  framework: pytest-asyncio
  database: pytest-postgresql
  
performance:
  framework: locust
  monitoring: prometheus
```

## 2. Unit Testing

### 2.1 Frontend Component Testing
```typescript
// SharedDiceRoller.test.ts
import { render, fireEvent, waitFor } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import SharedDiceRoller from '$lib/components/collaboration/SharedDiceRoller.svelte';
import { collaborationStore } from '$lib/stores/collaboration.svelte';

describe('SharedDiceRoller', () => {
  it('should parse dice expressions correctly', () => {
    const { getByRole, getByText } = render(SharedDiceRoller, {
      props: { roomId: 'test-room' }
    });
    
    const input = getByRole('textbox');
    const rollButton = getByText('Roll');
    
    fireEvent.input(input, { target: { value: '3d6+2' } });
    fireEvent.click(rollButton);
    
    expect(collaborationStore.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'dice_roll',
        data: expect.objectContaining({
          expression: '3d6+2',
          dice: [
            { count: 3, sides: 6 }
          ],
          modifier: 2
        })
      })
    );
  });
  
  it('should handle advantage rolls', async () => {
    const { getByRole, getByLabelText } = render(SharedDiceRoller, {
      props: { roomId: 'test-room' }
    });
    
    const advantageCheckbox = getByLabelText('Advantage');
    fireEvent.click(advantageCheckbox);
    
    const input = getByRole('textbox');
    fireEvent.input(input, { target: { value: '1d20+5' } });
    
    const rollButton = getByRole('button', { name: /roll/i });
    fireEvent.click(rollButton);
    
    await waitFor(() => {
      expect(collaborationStore.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            advantage: true,
            rolls: expect.arrayContaining([
              expect.any(Number),
              expect.any(Number)
            ])
          })
        })
      );
    });
  });
});
```

### 2.2 Backend MCP Tool Testing
```python
# test_mcp_tools.py
import pytest
from unittest.mock import Mock, patch
from src.tools import search_rules, roll_dice, get_monster

@pytest.mark.asyncio
async def test_search_rules():
    """Test rule searching with ChromaDB"""
    with patch('chromadb.Client') as mock_client:
        # Setup mock
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['Advantage: Roll twice, take higher']],
            'metadatas': [{'page': 173, 'book': 'PHB'}],
            'distances': [[0.2]]
        }
        mock_client.get_collection.return_value = mock_collection
        
        # Test
        result = await search_rules(
            query="advantage",
            game_system="dnd5e",
            limit=1
        )
        
        # Assert
        assert len(result['results']) == 1
        assert 'Advantage' in result['results'][0]['content']
        assert result['results'][0]['relevance'] > 0.8

@pytest.mark.asyncio  
async def test_roll_dice_with_modifiers():
    """Test dice rolling with various modifiers"""
    test_cases = [
        ("3d6+2", 5, 20),  # Min 5 (3*1+2), Max 20 (3*6+2)
        ("1d20", 1, 20),
        ("2d10+5", 7, 25),
        ("4d6kh3", 3, 18),  # Keep highest 3
    ]
    
    for expression, min_val, max_val in test_cases:
        result = await roll_dice(expression)
        
        assert min_val <= result['total'] <= max_val
        assert result['expression'] == expression
        assert 'breakdown' in result
        assert 'rolls' in result

@pytest.mark.asyncio
async def test_get_monster_with_scaling():
    """Test monster retrieval with party scaling"""
    with patch('chromadb.Client') as mock_client:
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'documents': [json.dumps({
                'name': 'Goblin',
                'hp': 7,
                'ac': 15,
                'cr': 0.25
            })]
        }
        mock_client.get_collection.return_value = mock_collection
        
        result = await get_monster(
            name="Goblin",
            party_level=5,
            party_size=4
        )
        
        # Should scale up for higher level party
        assert result['hp'] > 7  # Scaled HP
        assert result['quantity'] > 1  # Multiple goblins
        assert 'tactical_notes' in result
```

## 3. Integration Testing

### 3.1 WebSocket Integration Tests
```typescript
// websocket-integration.test.ts
import { WebSocketServer } from 'ws';
import { EnhancedWebSocketClient } from '$lib/realtime/websocket-client';

describe('WebSocket Integration', () => {
  let server: WebSocketServer;
  let client: EnhancedWebSocketClient;
  
  beforeEach(() => {
    server = new WebSocketServer({ port: 8080 });
    client = new EnhancedWebSocketClient({
      url: 'ws://localhost:8080',
      reconnectDelay: 100
    });
  });
  
  afterEach(() => {
    client.destroy();
    server.close();
  });
  
  it('should handle reconnection on disconnect', async () => {
    let connectionCount = 0;
    
    server.on('connection', (ws) => {
      connectionCount++;
      if (connectionCount === 1) {
        // Force disconnect after 100ms
        setTimeout(() => ws.close(), 100);
      }
    });
    
    await client.connect();
    
    // Wait for reconnection
    await new Promise(resolve => setTimeout(resolve, 500));
    
    expect(connectionCount).toBe(2);
    expect(client.getState().status).toBe('connected');
  });
  
  it('should queue messages while disconnected', async () => {
    const receivedMessages: any[] = [];
    
    server.on('connection', (ws) => {
      ws.on('message', (data) => {
        receivedMessages.push(JSON.parse(data.toString()));
      });
    });
    
    // Send while disconnected
    client.send({ type: 'test1' });
    
    // Connect
    await client.connect();
    
    // Send while connected  
    client.send({ type: 'test2' });
    
    await new Promise(resolve => setTimeout(resolve, 100));
    
    expect(receivedMessages).toHaveLength(2);
    expect(receivedMessages[0].type).toBe('test1');
    expect(receivedMessages[1].type).toBe('test2');
  });
});
```

### 3.2 MCP Bridge Integration
```python
# test_mcp_bridge.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from src.bridge.bridge_server import app, MCPBridge

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mcp_bridge():
    bridge = MCPBridge()
    await bridge.initialize()
    yield bridge
    await bridge.cleanup()

@pytest.mark.asyncio
async def test_mcp_tool_execution(mcp_bridge):
    """Test executing MCP tool through bridge"""
    result = await mcp_bridge.execute_tool(
        "search_rules",
        {
            "query": "fireball",
            "game_system": "dnd5e"
        }
    )
    
    assert result['success'] is True
    assert 'results' in result['data']
    assert len(result['data']['results']) > 0

@pytest.mark.asyncio
async def test_websocket_to_mcp_flow(client):
    """Test full flow from WebSocket to MCP"""
    with client.websocket_connect("/ws") as websocket:
        # Send MCP tool request
        websocket.send_json({
            "id": "123",
            "type": "tool",
            "tool": "roll_dice",
            "params": {
                "expression": "1d20+5"
            }
        })
        
        # Receive response
        response = websocket.receive_json()
        
        assert response['id'] == "123"
        assert response['type'] == "tool_result"
        assert 'total' in response['data']
        assert 6 <= response['data']['total'] <= 25
```

## 4. End-to-End Testing

### 4.1 Critical User Journeys
```typescript
// e2e/campaign-session.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Campaign Session Flow', () => {
  test('complete session workflow', async ({ page, context }) => {
    // 1. Login as GM
    await page.goto('/login');
    await page.fill('[name=email]', 'gm@example.com');
    await page.fill('[name=password]', 'test123');
    await page.click('button[type=submit]');
    
    // 2. Create campaign
    await page.goto('/campaigns/new');
    await page.fill('[name=name]', 'Test Campaign');
    await page.selectOption('[name=system]', 'dnd5e');
    await page.click('button:has-text("Create")');
    
    // 3. Start session
    await page.click('button:has-text("Start Session")');
    await expect(page.locator('.session-status')).toContainText('Live');
    
    // 4. Open in new tab as player
    const playerPage = await context.newPage();
    await playerPage.goto(page.url());
    await playerPage.fill('[name=email]', 'player@example.com');
    await playerPage.fill('[name=password]', 'test123');
    await playerPage.click('button[type=submit]');
    
    // 5. GM adds monster
    await page.click('button:has-text("Add Monster")');
    await page.fill('[name=search]', 'Goblin');
    await page.click('.monster-result:first-child');
    await page.click('button:has-text("Add to Combat")');
    
    // 6. Player should see update
    await expect(playerPage.locator('.initiative-tracker')).toContainText('Goblin');
    
    // 7. Player rolls dice
    await playerPage.fill('.dice-expression', '1d20+3');
    await playerPage.click('button:has-text("Roll")');
    
    // 8. GM should see roll
    await expect(page.locator('.activity-feed')).toContainText('rolled 1d20+3');
    
    // 9. End session
    await page.click('button:has-text("End Session")');
    await expect(page.locator('.session-status')).toContainText('Ended');
  });
});
```

### 4.2 Error Recovery E2E
```typescript
test('handles network disconnection gracefully', async ({ page, context }) => {
  await page.goto('/session/active');
  
  // Verify connected
  await expect(page.locator('.connection-status')).toHaveClass(/connected/);
  
  // Simulate offline
  await context.setOffline(true);
  
  // Should show offline status
  await expect(page.locator('.connection-status')).toHaveClass(/offline/);
  
  // Should queue actions
  await page.fill('.dice-expression', '2d6');
  await page.click('button:has-text("Roll")');
  await expect(page.locator('.sync-queue')).toContainText('1 pending');
  
  // Go back online
  await context.setOffline(false);
  
  // Should reconnect and sync
  await expect(page.locator('.connection-status')).toHaveClass(/connected/);
  await expect(page.locator('.sync-queue')).toContainText('0 pending');
  await expect(page.locator('.activity-feed')).toContainText('2d6');
});
```

## 5. Performance Testing

### 5.1 Load Testing with Locust
```python
# locustfile.py
from locust import HttpUser, task, between
import random
import json

class TTRPGUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        response = self.client.post("/api/auth/login", json={
            "email": f"user{random.randint(1,100)}@example.com",
            "password": "test123"
        })
        self.token = response.json()["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
        # Join session
        self.session_id = "test_session_001"
        self.ws = self.client.websocket(
            f"/ws?token={self.token}&session={self.session_id}"
        )
    
    @task(weight=10)
    def roll_dice(self):
        expressions = ["1d20", "3d6", "2d10+5", "1d12+3"]
        self.ws.send_json({
            "type": "tool",
            "tool": "roll_dice",
            "params": {
                "expression": random.choice(expressions)
            }
        })
    
    @task(weight=5)
    def search_rules(self):
        queries = ["advantage", "fireball", "grapple", "stealth"]
        self.client.post("/api/mcp/tool", 
            json={
                "tool": "search_rules",
                "params": {
                    "query": random.choice(queries)
                }
            },
            headers=self.headers
        )
    
    @task(weight=2)
    def update_character(self):
        self.client.patch(f"/api/characters/{self.character_id}",
            json={
                "hp": random.randint(1, 50)
            },
            headers=self.headers
        )
    
    @task(weight=1)
    def get_monster(self):
        monsters = ["Goblin", "Orc", "Dragon", "Skeleton"]
        self.client.get(
            f"/api/monsters/{random.choice(monsters)}",
            headers=self.headers
        )
```

### 5.2 Performance Metrics Collection
```typescript
// performance-monitor.ts
export class PerformanceMonitor {
  private metrics: Map<string, Metric[]> = new Map();
  
  measureOperation<T>(
    name: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const startMark = `${name}-start`;
    const endMark = `${name}-end`;
    
    performance.mark(startMark);
    
    return operation().finally(() => {
      performance.mark(endMark);
      performance.measure(name, startMark, endMark);
      
      const measure = performance.getEntriesByName(name)[0];
      this.recordMetric(name, {
        duration: measure.duration,
        timestamp: Date.now()
      });
      
      performance.clearMarks(startMark);
      performance.clearMarks(endMark);
      performance.clearMeasures(name);
    });
  }
  
  private recordMetric(name: string, metric: Metric) {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    
    const metrics = this.metrics.get(name)!;
    metrics.push(metric);
    
    // Keep only last 100 measurements
    if (metrics.length > 100) {
      metrics.shift();
    }
    
    // Alert if performance degrades
    this.checkPerformance(name, metrics);
  }
  
  private checkPerformance(name: string, metrics: Metric[]) {
    const recentMetrics = metrics.slice(-10);
    const avgDuration = recentMetrics.reduce(
      (sum, m) => sum + m.duration, 0
    ) / recentMetrics.length;
    
    const thresholds = {
      'dice_roll': 50,
      'rule_search': 200,
      'state_sync': 100
    };
    
    if (thresholds[name] && avgDuration > thresholds[name]) {
      console.warn(`Performance degradation in ${name}: ${avgDuration}ms`);
      this.reportToMonitoring(name, avgDuration);
    }
  }
}
```

## 6. Test Data Management

### 6.1 Test Fixtures
```python
# fixtures.py
import pytest
from typing import Dict, Any

@pytest.fixture
def campaign_fixture() -> Dict[str, Any]:
    return {
        "id": "test_campaign_001",
        "name": "Test Campaign",
        "system": "dnd5e",
        "gm_id": "gm_001",
        "players": ["player_001", "player_002"],
        "created_at": "2024-01-01T00:00:00Z"
    }

@pytest.fixture
def character_fixture() -> Dict[str, Any]:
    return {
        "id": "char_001",
        "name": "Test Character",
        "class": "Fighter",
        "level": 5,
        "hp": {"current": 38, "max": 44},
        "ac": 18,
        "stats": {
            "str": 16, "dex": 14, "con": 15,
            "int": 10, "wis": 12, "cha": 8
        }
    }

@pytest.fixture
async def populated_database(db_manager, campaign_fixture, character_fixture):
    """Populate database with test data"""
    await db_manager.campaigns.create(campaign_fixture)
    await db_manager.characters.create(character_fixture)
    
    # Add some dice rolls
    for i in range(10):
        await db_manager.dice_rolls.create({
            "session_id": "session_001",
            "player_id": "player_001",
            "expression": "1d20+5",
            "result": 15 + i,
            "purpose": "Attack roll"
        })
    
    yield db_manager
    
    # Cleanup
    await db_manager.cleanup_test_data()
```

### 6.2 Mock Data Generators
```typescript
// test-data-generator.ts
import { faker } from '@faker-js/faker';

export class TestDataGenerator {
  generateCharacter(overrides?: Partial<Character>): Character {
    return {
      id: faker.string.uuid(),
      name: faker.person.firstName() + ' ' + faker.person.lastName(),
      race: faker.helpers.arrayElement(['Human', 'Elf', 'Dwarf', 'Halfling']),
      class: faker.helpers.arrayElement(['Fighter', 'Wizard', 'Rogue', 'Cleric']),
      level: faker.number.int({ min: 1, max: 20 }),
      hp: {
        current: faker.number.int({ min: 1, max: 100 }),
        max: faker.number.int({ min: 10, max: 100 })
      },
      ...overrides
    };
  }
  
  generateDiceRoll(expression: string): DiceRollResult {
    const parsed = this.parseExpression(expression);
    const rolls = [];
    let total = 0;
    
    for (const die of parsed.dice) {
      for (let i = 0; i < die.count; i++) {
        const roll = faker.number.int({ min: 1, max: die.sides });
        rolls.push(roll);
        total += roll;
      }
    }
    
    total += parsed.modifier || 0;
    
    return {
      expression,
      rolls,
      total,
      breakdown: rolls.join(' + ') + (parsed.modifier ? ` + ${parsed.modifier}` : '')
    };
  }
}
```

## 7. CI/CD Pipeline

### 7.1 GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: cd frontend && npm ci
      
      - name: Run unit tests
        run: cd frontend && npm run test:unit -- --coverage
      
      - name: Run integration tests
        run: cd frontend && npm run test:integration
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./frontend/coverage/lcov.info
          flags: frontend

  backend-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=term
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: backend

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Wait for services
        run: |
          timeout 60s bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'
      
      - name: Run E2E tests
        run: cd frontend && npx playwright test
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: frontend/playwright-report/
```

## 8. Test Coverage Goals

### Coverage Targets
- **Overall**: 80% minimum
- **Critical paths**: 95% (auth, dice rolling, state sync)
- **UI Components**: 70%
- **Utilities**: 90%
- **MCP Tools**: 100%

### Coverage Report Configuration
```javascript
// vite.config.ts
export default defineConfig({
  test: {
    coverage: {
      provider: 'c8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'tests/',
        '*.config.js',
        '*.config.ts',
        'src/lib/components/ui/**'  // External UI library
      ],
      lines: 80,
      functions: 80,
      branches: 70,
      statements: 80
    }
  }
});
```

## Conclusion

This comprehensive testing strategy ensures the TTRPG MCP Server maintains high quality and reliability through multiple layers of testing, from unit tests to performance monitoring.