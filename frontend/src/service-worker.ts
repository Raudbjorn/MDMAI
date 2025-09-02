/// <reference types="@sveltejs/kit" />
/// <reference no-default-lib="true"/>
/// <reference lib="esnext" />
/// <reference lib="webworker" />

import { build, files, version } from '$service-worker';

const sw = self as unknown as ServiceWorkerGlobalScope;
const CACHE_NAME = `ttrpg-assistant-v${version}`;

// Resources to cache immediately
const STATIC_RESOURCES = [
  '/',
  '/manifest.json',
  ...build,
  ...files
];

// API endpoints to cache with network-first strategy
const API_CACHE_PATTERNS = [
  /\/api\/campaigns/,
  /\/api\/characters/,
  /\/api\/rules/,
  /\/api\/search/
];

// Resources that should always be fetched from network
const NETWORK_ONLY_PATTERNS = [
  /\/api\/auth/,
  /\/api\/collaboration/,
  /\/ws\//
];

// Install event - cache static resources
sw.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_RESOURCES))
      .then(() => sw.skipWaiting())
  );
});

// Activate event - clean up old caches
sw.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(name => name !== CACHE_NAME)
            .map(name => caches.delete(name))
        );
      })
      .then(() => sw.clients.claim())
  );
});

// Fetch event - implement caching strategies
sw.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // Network-only for specific patterns
  if (NETWORK_ONLY_PATTERNS.some(pattern => pattern.test(url.pathname))) {
    event.respondWith(networkOnly(request));
    return;
  }

  // API requests - network first, fallback to cache
  if (API_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname))) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Static resources - cache first, fallback to network
  if (request.method === 'GET') {
    event.respondWith(cacheFirst(request));
  }
});

// Cache-first strategy
async function cacheFirst(request: Request): Promise<Response> {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  if (cached) {
    // Update cache in background
    fetch(request)
      .then(response => {
        if (response.ok) {
          cache.put(request, response.clone());
        }
      })
      .catch(() => {/* Ignore errors */});
    
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    // Return offline page if available
    const offlinePage = await cache.match('/offline');
    if (offlinePage) {
      return offlinePage;
    }
    throw error;
  }
}

// Network-first strategy
async function networkFirst(request: Request): Promise<Response> {
  const cache = await caches.open(CACHE_NAME);

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    throw error;
  }
}

// Network-only strategy
async function networkOnly(request: Request): Promise<Response> {
  return fetch(request);
}

// Background sync for offline actions
sw.addEventListener('sync', (event: any) => {
  if (event.tag === 'sync-offline-actions') {
    event.waitUntil(syncOfflineActions());
  }
});

async function syncOfflineActions(): Promise<void> {
  // Get offline actions from IndexedDB
  const db = await openDB();
  const tx = db.transaction('offline_actions', 'readonly');
  const store = tx.objectStore('offline_actions');
  const actions = await store.getAll();

  // Process each action
  for (const action of actions) {
    try {
      const response = await fetch(action.url, {
        method: action.method,
        headers: action.headers,
        body: action.body
      });

      if (response.ok) {
        // Remove successful action
        const deleteTx = db.transaction('offline_actions', 'readwrite');
        await deleteTx.objectStore('offline_actions').delete(action.id);
      }
    } catch (error) {
      console.error('Failed to sync action:', error);
    }
  }
}

async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('TTRPGAssistantOffline', 1);
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('offline_actions')) {
        db.createObjectStore('offline_actions', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

// Message handling
sw.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') {
    sw.skipWaiting();
  }

  if (event.data?.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(CACHE_NAME)
        .then(cache => cache.addAll(event.data.urls))
    );
  }

  if (event.data?.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.keys()
        .then(names => Promise.all(names.map(name => caches.delete(name))))
    );
  }
});

// Push notifications
sw.addEventListener('push', (event) => {
  const options = {
    body: event.data?.text() || 'New update from TTRPG Assistant',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/icon-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'Open',
        icon: '/icons/icon-checkmark-32x32.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icons/icon-close-32x32.png'
      }
    ]
  };

  event.waitUntil(
    sw.registration.showNotification('TTRPG Assistant', options)
  );
});

sw.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'explore') {
    event.waitUntil(
      sw.clients.openWindow('/')
    );
  }
});