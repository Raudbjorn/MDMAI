#[cfg(test)]
mod integration_tests {
    use crate::data_manager::*;
    use tempfile::tempdir;
    use tokio;

    #[tokio::test]
    async fn test_complete_data_pipeline() {
        // Test the complete data pipeline: encryption -> caching -> integrity -> file management
        
        // Setup
        let temp_dir = tempdir().unwrap();
        let config = DataManagerConfig {
            cache_size_mb: 10,
            encryption_key_iterations: 10_000, // Lower for testing
            backup_retention_days: 7,
            storage_base_path: Some(temp_dir.path().to_string_lossy().to_string()),
            backup_base_path: Some(temp_dir.path().join("backups").to_string_lossy().to_string()),
            enable_integrity_checking: true,
            max_concurrent_operations: 4,
        };

        // Initialize components
        let cache = CacheManager::new(config.cache_size_mb * 1024 * 1024).unwrap();
        let file_manager = FileManager::with_base_path(temp_dir.path().join("files")).unwrap();
        let integrity_checker = IntegrityChecker::with_base_path(temp_dir.path().join("integrity"));
        let backup_manager = BackupManager::with_backup_path(temp_dir.path().join("backups"), config.backup_retention_days).unwrap();

        // Test data
        let test_data = b"This is comprehensive test data for the complete pipeline.";
        let file_id = "test_pipeline_file";

        // 1. Store file
        let hash = file_manager.store_file(file_id, test_data).await.unwrap();
        assert!(!hash.is_empty());

        // 2. Add to cache
        cache.put(file_id.to_string(), test_data.to_vec()).await;

        // 3. Add integrity record
        integrity_checker.add_file_record(file_id, test_data).await.unwrap();

        // 4. Verify from cache
        let cached_data = cache.get(file_id).await.unwrap();
        assert_eq!(cached_data, test_data);

        // 5. Verify file integrity
        let is_valid = integrity_checker.verify_file(file_id, test_data).await.unwrap();
        assert!(is_valid);

        // 6. Load from file manager
        let loaded_data = file_manager.load_file(file_id).await.unwrap();
        assert_eq!(loaded_data, test_data);

        // 7. Get statistics
        let cache_stats = cache.get_stats().await;
        assert_eq!(cache_stats.hit_count, 1);
        
        let storage_stats = file_manager.get_storage_stats().await;
        assert_eq!(storage_stats["total_files"].as_u64().unwrap(), 1);

        let integrity_stats = integrity_checker.get_integrity_stats().await;
        assert_eq!(integrity_stats["total_files"].as_u64().unwrap(), 1);
        assert_eq!(integrity_stats["healthy_files"].as_u64().unwrap(), 1);

        println!("✓ Complete data pipeline test passed");
    }

    #[tokio::test]
    async fn test_encryption_manager_thread_safety() {
        use crate::data_manager_commands::ThreadSafeEncryptionManager;
        use std::sync::Arc;
        use tokio::task;

        let manager = Arc::new(ThreadSafeEncryptionManager::new());
        let password = "test_password_for_thread_safety";
        let salt = vec![1u8; 32];

        // Initialize encryption
        manager.initialize_with_password(password, &salt).unwrap();
        assert!(manager.is_initialized());

        // Test concurrent operations
        let mut handles = Vec::new();
        
        for i in 0..10 {
            let manager_clone = Arc::clone(&manager);
            let test_data = format!("Test data for thread {}", i);
            
            let handle = task::spawn(async move {
                let data_bytes = test_data.as_bytes();
                let encrypted = manager_clone.encrypt_data(data_bytes).unwrap();
                let decrypted = manager_clone.decrypt_data(&encrypted).unwrap();
                assert_eq!(data_bytes, decrypted.as_slice());
                i
            });
            
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let thread_id = handle.await.unwrap();
            println!("✓ Thread {} encryption test completed", thread_id);
        }

        println!("✓ Encryption manager thread safety test passed");
    }

    #[tokio::test]
    async fn test_cache_performance_under_load() {
        let cache = CacheManager::new(1024 * 1024).unwrap(); // 1MB cache
        
        let start_time = std::time::Instant::now();
        
        // Add many entries to test performance
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let data = format!("data_for_key_{}", i).into_bytes();
            cache.put(key, data).await;
        }
        
        let insert_time = start_time.elapsed();
        
        // Test retrieval performance
        let retrieval_start = std::time::Instant::now();
        
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let _data = cache.get(&key).await;
        }
        
        let retrieval_time = retrieval_start.elapsed();
        
        let stats = cache.get_stats().await;
        
        println!("Cache performance test:");
        println!("  Insert time for 1000 entries: {:?}", insert_time);
        println!("  Retrieval time for 1000 entries: {:?}", retrieval_time);
        println!("  Final cache size: {} entries", stats.total_entries);
        println!("  Hit ratio: {:.2}%", stats.hit_ratio * 100.0);
        println!("  Evicted entries: {}", stats.eviction_count);
        
        // Performance assertions
        assert!(insert_time.as_millis() < 1000, "Insert performance too slow");
        assert!(retrieval_time.as_millis() < 100, "Retrieval performance too slow");
        
        println!("✓ Cache performance test passed");
    }

    #[tokio::test]
    async fn test_file_manager_duplicate_detection() {
        let temp_dir = tempdir().unwrap();
        let file_manager = FileManager::with_base_path(temp_dir.path()).unwrap();

        let test_data = b"Data for duplicate detection test";
        
        // Store multiple files with same content
        file_manager.store_file("file1", test_data).await.unwrap();
        file_manager.store_file("file2", test_data).await.unwrap();
        file_manager.store_file("file3", test_data).await.unwrap();

        // Add one unique file
        let unique_data = b"Unique data that should not be detected as duplicate";
        file_manager.store_file("unique", unique_data).await.unwrap();

        let duplicate_groups = file_manager.get_duplicate_groups().await;
        
        // Should have one duplicate group with 3 files
        assert_eq!(duplicate_groups.len(), 1);
        assert_eq!(duplicate_groups[0].file_ids.len(), 3);
        assert!(duplicate_groups[0].total_wasted_space > 0);

        let storage_stats = file_manager.get_storage_stats().await;
        let wasted_space = storage_stats["wasted_space_bytes"].as_u64().unwrap();
        assert!(wasted_space > 0);

        println!("✓ File manager duplicate detection test passed");
    }

    #[test]
    fn test_configuration_validation() {
        // Test valid configuration
        let valid_config = DataManagerConfig::default();
        assert!(validate_config(&valid_config).is_ok());

        // Test invalid configurations
        let mut invalid_config = DataManagerConfig::default();
        invalid_config.cache_size_mb = 0;
        assert!(validate_config(&invalid_config).is_err());

        invalid_config.cache_size_mb = 256;
        invalid_config.encryption_key_iterations = 1000;
        assert!(validate_config(&invalid_config).is_err());

        invalid_config.encryption_key_iterations = 100_000;
        invalid_config.backup_retention_days = 0;
        assert!(validate_config(&invalid_config).is_err());

        println!("✓ Configuration validation test passed");
    }
}