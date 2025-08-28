#!/usr/bin/env node
/**
 * Load testing for real-time features
 * Tests WebSocket and SSE under various load conditions
 */

import { EnhancedWebSocketClient } from '../../src/lib/realtime/websocket-client';
import { EnhancedSSEClient } from '../../src/lib/realtime/sse-client';

interface TestMetrics {
  connectTime: number[];
  messageLatency: number[];
  reconnectCount: number;
  errorCount: number;
  messagesReceived: number;
  messagesSent: number;
  memoryUsage: number[];
}

class LoadTester {
  private metrics: TestMetrics = {
    connectTime: [],
    messageLatency: [],
    reconnectCount: 0,
    errorCount: 0,
    messagesReceived: 0,
    messagesSent: 0,
    memoryUsage: []
  };

  async testWebSocketLoad(
    url: string,
    numClients: number,
    messagesPerClient: number,
    messageInterval: number
  ) {
    console.log(`\n🚀 WebSocket Load Test`);
    console.log(`   Clients: ${numClients}`);
    console.log(`   Messages per client: ${messagesPerClient}`);
    console.log(`   Message interval: ${messageInterval}ms\n`);

    const clients: EnhancedWebSocketClient[] = [];
    const startTime = Date.now();

    // Create clients
    for (let i = 0; i < numClients; i++) {
      const client = new EnhancedWebSocketClient({ url });
      const connectStart = Date.now();
      
      client.onOpen(() => {
        this.metrics.connectTime.push(Date.now() - connectStart);
      });

      client.onMessage(() => {
        this.metrics.messagesReceived++;
      });

      client.onError(() => {
        this.metrics.errorCount++;
      });

      // Client connects automatically in constructor
      clients.push(client);
      
      // Stagger client creation
      await this.delay(50);
    }

    // Send messages
    for (let i = 0; i < messagesPerClient; i++) {
      const sendPromises = clients.map(async (client, idx) => {
        const messageStart = Date.now();
        
        try {
          await client.request({
            type: 'echo',
            clientId: idx,
            messageNum: i,
            timestamp: messageStart
          }, { timeout: 5000 });
          
          this.metrics.messageLatency.push(Date.now() - messageStart);
          this.metrics.messagesSent++;
        } catch (error) {
          this.metrics.errorCount++;
        }
      });

      await Promise.all(sendPromises);
      await this.delay(messageInterval);
      
      // Track memory
      if (typeof process !== 'undefined') {
        this.metrics.memoryUsage.push(process.memoryUsage().heapUsed / 1024 / 1024);
      }
    }

    // Cleanup
    await Promise.all(clients.map(c => c.close()));

    const totalTime = Date.now() - startTime;
    this.printMetrics('WebSocket', totalTime);
  }

  async testSSELoad(
    url: string,
    numClients: number,
    duration: number
  ) {
    console.log(`\n📡 SSE Load Test`);
    console.log(`   Clients: ${numClients}`);
    console.log(`   Duration: ${duration}ms\n`);

    const clients: EnhancedSSEClient[] = [];
    const startTime = Date.now();

    // Create clients
    for (let i = 0; i < numClients; i++) {
      const client = new EnhancedSSEClient({ url });
      const connectStart = Date.now();

      client.onOpen(() => {
        this.metrics.connectTime.push(Date.now() - connectStart);
      });

      client.onMessage('message', () => {
        this.metrics.messagesReceived++;
      });

      client.onError(() => {
        this.metrics.errorCount++;
      });

      // Client connects automatically in constructor
      clients.push(client);
      
      // Stagger client creation
      await this.delay(50);
    }

    // Wait for duration
    await this.delay(duration);

    // Cleanup
    clients.forEach(c => c.close());

    const totalTime = Date.now() - startTime;
    this.printMetrics('SSE', totalTime);
  }

  async testConcurrentDrawing(
    url: string,
    numUsers: number,
    drawOperations: number
  ) {
    console.log(`\n🎨 Collaborative Canvas Load Test`);
    console.log(`   Users: ${numUsers}`);
    console.log(`   Operations per user: ${drawOperations}\n`);

    const clients: EnhancedWebSocketClient[] = [];

    // Create users
    for (let i = 0; i < numUsers; i++) {
      const client = new EnhancedWebSocketClient({ url });
      // Client connects automatically in constructor
      clients.push(client);
    }

    // Simulate drawing
    const drawPromises = clients.map(async (client, userIdx) => {
      for (let i = 0; i < drawOperations; i++) {
        await client.send({
          type: 'draw',
          data: {
            tool: 'pen',
            points: Array.from({ length: 10 }, () => ({
              x: Math.random() * 800,
              y: Math.random() * 600
            })),
            color: '#' + Math.floor(Math.random()*16777215).toString(16),
            size: Math.random() * 10 + 1,
            userId: `user_${userIdx}`
          }
        });
        
        // Simulate natural drawing speed
        await this.delay(Math.random() * 100 + 50);
      }
    });

    await Promise.all(drawPromises);
    
    // Cleanup
    await Promise.all(clients.map(c => c.close()));
  }

  private printMetrics(testName: string, totalTime: number) {
    console.log(`\n📊 ${testName} Test Results:`);
    console.log('═'.repeat(40));
    
    if (this.metrics.connectTime.length > 0) {
      const avgConnect = this.average(this.metrics.connectTime);
      const p95Connect = this.percentile(this.metrics.connectTime, 95);
      console.log(`Connection Time: avg=${avgConnect.toFixed(2)}ms, p95=${p95Connect.toFixed(2)}ms`);
    }

    if (this.metrics.messageLatency.length > 0) {
      const avgLatency = this.average(this.metrics.messageLatency);
      const p95Latency = this.percentile(this.metrics.messageLatency, 95);
      console.log(`Message Latency: avg=${avgLatency.toFixed(2)}ms, p95=${p95Latency.toFixed(2)}ms`);
    }

    console.log(`Messages Sent: ${this.metrics.messagesSent}`);
    console.log(`Messages Received: ${this.metrics.messagesReceived}`);
    console.log(`Errors: ${this.metrics.errorCount}`);
    console.log(`Reconnects: ${this.metrics.reconnectCount}`);
    
    if (this.metrics.memoryUsage.length > 0) {
      const maxMemory = Math.max(...this.metrics.memoryUsage);
      console.log(`Max Memory: ${maxMemory.toFixed(2)}MB`);
    }

    const throughput = this.metrics.messagesSent / (totalTime / 1000);
    console.log(`Throughput: ${throughput.toFixed(2)} msg/s`);
    console.log(`Total Time: ${(totalTime / 1000).toFixed(2)}s`);
    
    // Reset metrics
    this.metrics = {
      connectTime: [],
      messageLatency: [],
      reconnectCount: 0,
      errorCount: 0,
      messagesReceived: 0,
      messagesSent: 0,
      memoryUsage: []
    };
  }

  private average(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private percentile(arr: number[], p: number): number {
    const sorted = arr.sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[index];
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Run tests
async function runTests() {
  const tester = new LoadTester();
  const wsUrl = process.env.WS_URL || 'ws://localhost:8080';
  const sseUrl = process.env.SSE_URL || 'http://localhost:8080/sse';

  try {
    // Small load test
    await tester.testWebSocketLoad(wsUrl, 10, 100, 100);
    
    // Medium load test
    await tester.testWebSocketLoad(wsUrl, 50, 50, 200);
    
    // High load test
    await tester.testWebSocketLoad(wsUrl, 100, 20, 500);
    
    // SSE test
    await tester.testSSELoad(sseUrl, 50, 10000);
    
    // Canvas stress test
    await tester.testConcurrentDrawing(wsUrl, 20, 50);
    
    console.log('\n✅ All tests completed successfully!');
  } catch (error) {
    console.error('\n❌ Test failed:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  runTests();
}