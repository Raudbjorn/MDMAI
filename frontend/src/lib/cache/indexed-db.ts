/**
 * IndexedDB wrapper for persistent caching
 */

import type { CacheEntry } from './types';

const DB_NAME = 'TTRPGAssistantCache';
const DB_VERSION = 1;
const STORE_NAME = 'cache';

export class IndexedDBCache {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.db) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('Failed to open IndexedDB:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object store if it doesn't exist
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
          
          // Create indexes for efficient querying
          store.createIndex('expiresAt', 'expiresAt', { unique: false });
          store.createIndex('namespace', 'metadata.namespace', { unique: false });
          store.createIndex('lastAccessed', 'lastAccessed', { unique: false });
          store.createIndex('size', 'size', { unique: false });
        }
      };
    });

    return this.initPromise;
  }

  async get<T>(key: string): Promise<CacheEntry<T> | null> {
    await this.init();
    if (!this.db) return null;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(key);

      request.onsuccess = () => {
        const entry = request.result as CacheEntry<T> | undefined;
        
        // Check if entry exists and hasn't expired
        if (entry && entry.expiresAt > Date.now()) {
          // Update last accessed time
          this.updateLastAccessed(key).catch(console.error);
          resolve(entry);
        } else {
          // Entry expired or doesn't exist
          if (entry) {
            this.delete(key).catch(console.error);
          }
          resolve(null);
        }
      };

      request.onerror = () => {
        reject(request.error);
      };
    });
  }

  async set<T>(entry: CacheEntry<T>): Promise<void> {
    await this.init();
    if (!this.db) throw new Error('Database not initialized');

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.put(entry);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async delete(key: string): Promise<void> {
    await this.init();
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async clear(): Promise<void> {
    await this.init();
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getExpiredKeys(): Promise<string[]> {
    await this.init();
    if (!this.db) return [];

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const index = store.index('expiresAt');
      const now = Date.now();
      const range = IDBKeyRange.upperBound(now);
      const request = index.getAllKeys(range);

      request.onsuccess = () => resolve(request.result as string[]);
      request.onerror = () => reject(request.error);
    });
  }

  async deleteExpired(): Promise<number> {
    const expiredKeys = await this.getExpiredKeys();
    
    for (const key of expiredKeys) {
      await this.delete(key);
    }
    
    return expiredKeys.length;
  }

  async getByNamespace(namespace: string): Promise<CacheEntry[]> {
    await this.init();
    if (!this.db) return [];

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const index = store.index('namespace');
      const request = index.getAll(namespace);

      request.onsuccess = () => {
        const entries = request.result.filter(
          entry => entry.expiresAt > Date.now()
        );
        resolve(entries);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async getTotalSize(): Promise<number> {
    await this.init();
    if (!this.db) return 0;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const total = request.result.reduce(
          (sum, entry) => sum + (entry.size || 0),
          0
        );
        resolve(total);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async evictLRU(maxItems: number): Promise<number> {
    await this.init();
    if (!this.db) return 0;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const index = store.index('lastAccessed');
      const request = index.getAllKeys();

      request.onsuccess = async () => {
        const keys = request.result;
        const currentCount = keys.length;
        
        if (currentCount <= maxItems) {
          resolve(0);
          return;
        }

        const toDelete = currentCount - maxItems;
        const keysToDelete = keys.slice(0, toDelete);
        
        for (const key of keysToDelete) {
          await this.delete(key as string);
        }
        
        resolve(toDelete);
      };

      request.onerror = () => reject(request.error);
    });
  }

  private async updateLastAccessed(key: string): Promise<void> {
    await this.init();
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const getRequest = store.get(key);

      getRequest.onsuccess = () => {
        const entry = getRequest.result;
        if (entry) {
          entry.lastAccessed = Date.now();
          entry.hits = (entry.hits || 0) + 1;
          
          const putRequest = store.put(entry);
          putRequest.onsuccess = () => resolve();
          putRequest.onerror = () => reject(putRequest.error);
        } else {
          resolve();
        }
      };

      getRequest.onerror = () => reject(getRequest.error);
    });
  }

  async getStats(): Promise<{
    totalItems: number;
    totalSize: number;
    oldestEntry: number | null;
    newestEntry: number | null;
  }> {
    await this.init();
    if (!this.db) {
      return { totalItems: 0, totalSize: 0, oldestEntry: null, newestEntry: null };
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => {
        const entries = request.result;
        
        if (entries.length === 0) {
          resolve({
            totalItems: 0,
            totalSize: 0,
            oldestEntry: null,
            newestEntry: null
          });
          return;
        }

        const stats = {
          totalItems: entries.length,
          totalSize: entries.reduce((sum, e) => sum + (e.size || 0), 0),
          oldestEntry: Math.min(...entries.map(e => e.timestamp)),
          newestEntry: Math.max(...entries.map(e => e.timestamp))
        };

        resolve(stats);
      };

      request.onerror = () => reject(request.error);
    });
  }
}