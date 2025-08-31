# MDMAI Desktop Backend

High-performance Rust backend for the MDMAI desktop application, built with Tauri. This backend provides comprehensive data management capabilities with a focus on memory efficiency, thread safety, and data integrity.

## 🚀 Features

### ✅ Fixed Critical Issues

1. **Thread-Safe Interior Mutability**: Fixed compilation errors with `EncryptionManager` by implementing `ThreadSafeEncryptionManager` wrapper using `Arc<RwLock<T>>`

2. **Memory Leak Prevention**: Eliminated `.leak()` usage in `native_features.rs` with proper string handling and efficient resource management

3. **O(1) Cache Performance**: Replaced O(N) BTreeMap linear scans with O(1) HashMap lookups in `cache.rs` for optimal performance

4. **Streaming File Processing**: Implemented memory-efficient streaming hash calculation for large files in `integrity.rs` and `file_manager.rs`

5. **Type-Safe SQL Operations**: Added proper SQL column type handling in `backup.rs` to prevent data corruption during database export/import

6. **Zero Memory Leaks**: All components use proper RAII patterns and efficient memory management

### 🔐 Security & Encryption

- **AES-256-GCM encryption** with AEAD authentication
- **Argon2 key derivation** for password-based encryption
- **Blake3 + SHA-256 hashing** for data integrity verification
- **Constant-time comparisons** to prevent timing attacks
- **Secure memory cleanup** on Drop

### 📊 Performance Optimizations

- **Concurrent operations** with configurable limits
- **LRU cache** with O(1) access and intelligent eviction
- **Streaming I/O** for memory-efficient large file processing
- **Parallel processing** using Rayon for CPU-intensive tasks
- **Zero-copy operations** where possible

### 💾 Data Management

- **Duplicate detection** with streaming hash calculation
- **File integrity verification** using cryptographic checksums
- **Atomic database operations** with proper transaction handling
- **Type-safe SQL operations** preserving data types
- **Intelligent backup rotation** with configurable retention

### 🖥️ System Integration

- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Native notifications** without memory leaks
- **System resource monitoring** with efficient data collection
- **File system operations** with proper error handling

## 📁 Project Structure

```
src-tauri/
├── src/
│   ├── data_manager/           # Core data management
│   │   ├── encryption.rs       # AES-256-GCM encryption
│   │   ├── cache.rs            # High-performance LRU cache
│   │   ├── integrity.rs        # Streaming integrity verification
│   │   ├── file_manager.rs     # File operations & deduplication
│   │   ├── backup.rs           # Type-safe database backup
│   │   └── mod.rs              # Module exports
│   ├── data_manager_commands.rs # Tauri command handlers
│   ├── native_features.rs      # System integration
│   ├── tests.rs               # Integration tests
│   ├── lib.rs                 # Library root
│   └── main.rs                # Application entry point
├── Cargo.toml                 # Dependencies & configuration
├── tauri.conf.json           # Tauri configuration
└── build.rs                  # Build script
```

## 🔧 Building

### Prerequisites

- Rust 1.75+ 
- Node.js 18+
- Platform-specific dependencies:
  - **Linux**: `build-essential`, `curl`, `wget`, `file`, `libssl-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`
  - **macOS**: Xcode command line tools
  - **Windows**: Microsoft C++ Build Tools

### Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run tauri dev

# Build for production
npm run tauri build

# Run tests
cd src-tauri && cargo test
```

## 🧪 Testing

The backend includes comprehensive tests covering:

- **Unit tests** for individual components
- **Integration tests** for complete data pipelines  
- **Performance benchmarks** for cache and file operations
- **Thread safety tests** for concurrent operations
- **Memory usage verification** to prevent leaks

```bash
cd src-tauri

# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test module
cargo test data_manager

# Run with release optimizations
cargo test --release
```

## 📈 Performance Characteristics

### Cache Performance
- **Insertion**: O(1) amortized
- **Lookup**: O(1) 
- **Eviction**: O(1) amortized
- **Memory usage**: Configurable with intelligent eviction

### File Operations  
- **Hash calculation**: Streaming with configurable chunk size (64KB default)
- **Duplicate detection**: O(N log N) with parallel processing
- **Integrity verification**: Constant memory usage regardless of file size

### Database Operations
- **Backup creation**: Streaming export with type preservation
- **Restoration**: Atomic transactions with rollback support
- **Memory usage**: Constant, independent of database size

## 🔒 Security Considerations

- **No unsafe code** - entirely safe Rust
- **Secure key derivation** using Argon2 with configurable iterations
- **Authentication encryption** prevents tampering
- **Secure memory cleanup** on sensitive data structures
- **Constant-time operations** for cryptographic comparisons

## 🐛 Troubleshooting

### Common Issues

1. **Compilation errors**: Ensure Rust 1.75+ and all system dependencies installed
2. **Permission errors**: Check file system permissions for storage directories
3. **Performance issues**: Adjust cache size and concurrent operation limits in config

### Debugging

Enable detailed logging:
```bash
RUST_LOG=debug npm run tauri dev
```

## 📜 License

This project is part of the MDMAI desktop application. See the root project for license information.

## 🤝 Contributing

1. Follow Rust best practices and clippy suggestions
2. Add tests for new functionality
3. Ensure all tests pass before submitting
4. Document public APIs with rustdoc comments
5. Profile performance-critical code paths

## 🔍 Code Quality

- **Zero warnings** with `#![warn(missing_docs)]`
- **No unsafe code** with `#![deny(unsafe_code)]`
- **Comprehensive error handling** using `anyhow` and `thiserror`
- **Memory leak detection** in tests
- **Performance benchmarking** for critical paths