// Standalone syntax check for our modules
// This compiles only our data management modules without Tauri dependencies

#[allow(unused_imports)]
mod data_manager {
    pub mod encryption {
        include!("data_manager/encryption.rs");
    }
    
    pub mod cache {
        include!("data_manager/cache.rs");
    }
    
    pub mod integrity {
        include!("data_manager/integrity.rs");
    }
    
    pub mod file_manager {
        include!("data_manager/file_manager.rs");
    }
    
    pub mod backup {
        include!("data_manager/backup.rs");
    }
}

// Test compilation
#[cfg(test)]
mod tests {
    use super::data_manager::*;

    #[test]
    fn test_compilation() {
        // Basic compilation test for each module
        let _encryption_manager = encryption::EncryptionManager::new();
        println!("✓ EncryptionManager compiles");

        // Cache manager
        if let Ok(_cache) = cache::CacheManager::new(1024) {
            println!("✓ CacheManager compiles");
        }

        // IntegrityChecker
        let _checker = integrity::IntegrityChecker::new();
        println!("✓ IntegrityChecker compiles");

        // FileManager  
        if let Ok(_fm) = file_manager::FileManager::new() {
            println!("✓ FileManager compiles");
        }

        // BackupManager
        if let Ok(_bm) = backup::BackupManager::new(30) {
            println!("✓ BackupManager compiles");
        }

        println!("All modules compile successfully!");
    }
}