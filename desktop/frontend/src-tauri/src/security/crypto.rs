//! Additional Cryptographic Utilities
//! 
//! This module provides additional cryptographic functions beyond the main encryption manager:
//! - Digital signatures and verification
//! - Cryptographic hash functions
//! - Secure random number generation
//! - Key derivation functions
//! - Certificate management

use super::*;
use rand::RngCore;
use sha2::{Sha256, Sha512, Digest};
use hmac::{Hmac, Mac};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Additional cryptographic manager
pub struct CryptoManager {
    config: SecurityConfig,
    signing_keys: Arc<RwLock<HashMap<String, SigningKey>>>,
    verification_keys: Arc<RwLock<HashMap<String, VerificationKey>>>,
}

#[derive(Debug, Clone)]
struct SigningKey {
    key_id: String,
    private_key: Vec<u8>,
    algorithm: SignatureAlgorithm,
    created_at: SystemTime,
}

#[derive(Debug, Clone)]
struct VerificationKey {
    key_id: String,
    public_key: Vec<u8>,
    algorithm: SignatureAlgorithm,
    created_at: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    HMACSHA256,
    Ed25519,
    ECDSAP256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignature {
    pub signature: Vec<u8>,
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashResult {
    pub hash: String,
    pub algorithm: String,
    pub input_size: usize,
    pub timestamp: SystemTime,
}

impl CryptoManager {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        Ok(Self {
            config: config.clone(),
            signing_keys: Arc::new(RwLock::new(HashMap::new())),
            verification_keys: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate a cryptographically secure random byte array
    pub fn generate_random_bytes(&self, length: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; length];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        bytes
    }

    /// Generate a secure random string (base64 encoded)
    pub fn generate_random_string(&self, byte_length: usize) -> String {
        let bytes = self.generate_random_bytes(byte_length);
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// Generate a UUID v4
    pub fn generate_uuid(&self) -> String {
        Uuid::new_v4().to_string()
    }

    /// Compute SHA-256 hash
    pub fn hash_sha256(&self, data: &[u8]) -> HashResult {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();

        HashResult {
            hash: hex::encode(hash),
            algorithm: "SHA-256".to_string(),
            input_size: data.len(),
            timestamp: SystemTime::now(),
        }
    }

    /// Compute SHA-512 hash
    pub fn hash_sha512(&self, data: &[u8]) -> HashResult {
        let mut hasher = Sha512::new();
        hasher.update(data);
        let hash = hasher.finalize();

        HashResult {
            hash: hex::encode(hash),
            algorithm: "SHA-512".to_string(),
            input_size: data.len(),
            timestamp: SystemTime::now(),
        }
    }

    /// Compute BLAKE3 hash (faster alternative)
    pub fn hash_blake3(&self, data: &[u8]) -> HashResult {
        let hash = blake3::hash(data);

        HashResult {
            hash: hex::encode(hash.as_bytes()),
            algorithm: "BLAKE3".to_string(),
            input_size: data.len(),
            timestamp: SystemTime::now(),
        }
    }

    /// Key derivation using PBKDF2
    pub fn derive_key_pbkdf2(&self, password: &[u8], salt: &[u8], iterations: u32, key_length: usize) -> SecurityResult<Vec<u8>> {
        use pbkdf2::pbkdf2_hmac;
        
        let mut key = vec![0u8; key_length];
        pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut key);
        Ok(key)
    }

    /// Key derivation using Argon2
    pub fn derive_key_argon2(&self, password: &[u8], salt: &[u8], key_length: usize) -> SecurityResult<Vec<u8>> {
        use argon2::{Argon2, Config, Variant, Version};
        
        let config = Config {
            variant: Variant::Argon2id,
            version: Version::Version13,
            mem_cost: 4096,      // 4 MB memory
            time_cost: 3,        // 3 iterations
            lanes: 1,            // Single thread
            secret: &[],
            ad: &[],
            hash_length: key_length as u32,
        };

        argon2::hash_raw(password, salt, &config)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Argon2 key derivation failed: {}", e),
            })
    }

    /// Generate HMAC-SHA256 signature
    pub fn sign_hmac_sha256(&self, data: &[u8], key: &[u8]) -> SecurityResult<Vec<u8>> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("HMAC key setup failed: {}", e),
            })?;
        
        mac.update(data);
        let signature = mac.finalize().into_bytes();
        Ok(signature.to_vec())
    }

    /// Verify HMAC-SHA256 signature
    pub fn verify_hmac_sha256(&self, data: &[u8], signature: &[u8], key: &[u8]) -> SecurityResult<bool> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("HMAC key setup failed: {}", e),
            })?;
        
        mac.update(data);
        
        match mac.verify_slice(signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Create a signing key pair
    pub async fn create_signing_key(&self, key_id: &str, algorithm: SignatureAlgorithm) -> SecurityResult<String> {
        let (private_key, public_key) = match algorithm {
            SignatureAlgorithm::HMACSHA256 => {
                let key = self.generate_random_bytes(32);
                (key.clone(), key) // HMAC uses the same key for signing and verification
            }
            SignatureAlgorithm::Ed25519 => {
                return Err(SecurityError::CryptographicError {
                    message: "Ed25519 algorithm not yet implemented. Please use HMACSHA256 instead.".to_string(),
                });
            }
            SignatureAlgorithm::ECDSAP256 => {
                return Err(SecurityError::CryptographicError {
                    message: "ECDSA-P256 algorithm not yet implemented. Please use HMACSHA256 instead.".to_string(),
                });
            }
        };

        let signing_key = SigningKey {
            key_id: key_id.to_string(),
            private_key,
            algorithm: algorithm.clone(),
            created_at: SystemTime::now(),
        };

        let verification_key = VerificationKey {
            key_id: key_id.to_string(),
            public_key,
            algorithm,
            created_at: SystemTime::now(),
        };

        // Store keys
        self.signing_keys.write().await.insert(key_id.to_string(), signing_key);
        self.verification_keys.write().await.insert(key_id.to_string(), verification_key);

        Ok(key_id.to_string())
    }

    /// Sign data with a stored key
    pub async fn sign_data(&self, data: &[u8], key_id: &str) -> SecurityResult<DigitalSignature> {
        let signing_keys = self.signing_keys.read().await;
        let signing_key = signing_keys.get(key_id)
            .ok_or_else(|| SecurityError::CryptographicError {
                message: format!("Signing key not found: {}", key_id),
            })?;

        let signature_bytes = match signing_key.algorithm {
            SignatureAlgorithm::HMACSHA256 => {
                self.sign_hmac_sha256(data, &signing_key.private_key)?
            }
            SignatureAlgorithm::Ed25519 => {
                return Err(SecurityError::CryptographicError {
                    message: "Ed25519 algorithm not implemented".to_string(),
                });
            }
            SignatureAlgorithm::ECDSAP256 => {
                return Err(SecurityError::CryptographicError {
                    message: "ECDSA-P256 algorithm not implemented".to_string(),
                });
            }
        };

        Ok(DigitalSignature {
            signature: signature_bytes,
            algorithm: signing_key.algorithm.clone(),
            key_id: key_id.to_string(),
            timestamp: SystemTime::now(),
        })
    }

    /// Verify a digital signature
    pub async fn verify_signature(&self, data: &[u8], signature: &DigitalSignature) -> SecurityResult<bool> {
        let verification_keys = self.verification_keys.read().await;
        let verification_key = verification_keys.get(&signature.key_id)
            .ok_or_else(|| SecurityError::CryptographicError {
                message: format!("Verification key not found: {}", signature.key_id),
            })?;

        match signature.algorithm {
            SignatureAlgorithm::HMACSHA256 => {
                self.verify_hmac_sha256(data, &signature.signature, &verification_key.public_key)
            }
            SignatureAlgorithm::Ed25519 => {
                Err(SecurityError::CryptographicError {
                    message: "Ed25519 algorithm not implemented".to_string(),
                })
            }
            SignatureAlgorithm::ECDSAP256 => {
                Err(SecurityError::CryptographicError {
                    message: "ECDSA-P256 algorithm not implemented".to_string(),
                })
            }
        }
    }

    /// Constant-time comparison to prevent timing attacks
    pub fn constant_time_compare(&self, a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (byte_a, byte_b) in a.iter().zip(b.iter()) {
            result |= byte_a ^ byte_b;
        }

        result == 0
    }

    /// Generate a secure session token
    pub fn generate_session_token(&self) -> SecurityResult<String> {
        let token_data = SessionTokenData {
            random_bytes: self.generate_random_bytes(32),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)
                .map_err(|e| SecurityError::CryptographicError {
                    message: format!("Time calculation failed: {}", e),
                })?
                .as_secs(),
            version: 1,
        };

        let serialized = serde_json::to_vec(&token_data)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Token serialization failed: {}", e),
            })?;

        Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serialized))
    }

    /// Verify a session token
    pub fn verify_session_token(&self, token: &str) -> SecurityResult<bool> {
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(token)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Token decoding failed: {}", e),
            })?;

        let token_data: SessionTokenData = serde_json::from_slice(&decoded)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Token deserialization failed: {}", e),
            })?;

        // Check token age (24 hours)
        let current_timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Time calculation failed: {}", e),
            })?
            .as_secs();

        if current_timestamp - token_data.timestamp > 24 * 3600 {
            return Ok(false); // Token expired
        }

        Ok(true)
    }

    /// Generate a cryptographic nonce
    pub fn generate_nonce(&self, length: usize) -> String {
        let nonce = self.generate_random_bytes(length);
        hex::encode(nonce)
    }

    /// Time-based one-time password (TOTP) generation
    pub fn generate_totp(&self, secret: &[u8], time_step: u64) -> SecurityResult<u32> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Time calculation failed: {}", e),
            })?
            .as_secs();

        let time_counter = current_time / time_step;
        let time_bytes = time_counter.to_be_bytes();

        let signature = self.sign_hmac_sha256(&time_bytes, secret)?;
        
        // Extract 4 bytes from the HMAC result
        let offset = (signature[signature.len() - 1] & 0x0f) as usize;
        let code = u32::from_be_bytes([
            signature[offset] & 0x7f,
            signature[offset + 1],
            signature[offset + 2],
            signature[offset + 3],
        ]) % 1_000_000;

        Ok(code)
    }

    /// Verify TOTP code
    pub fn verify_totp(&self, secret: &[u8], code: u32, time_step: u64, window: u32) -> SecurityResult<bool> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|e| SecurityError::CryptographicError {
                message: format!("Time calculation failed: {}", e),
            })?
            .as_secs();

        // Check current time and nearby time windows
        for i in 0..=window {
            for direction in [-1i64, 0, 1] {
                let test_time = current_time as i64 + (direction * i as i64 * time_step as i64);
                if test_time < 0 {
                    continue;
                }

                let time_counter = (test_time as u64) / time_step;
                let time_bytes = time_counter.to_be_bytes();

                let signature = self.sign_hmac_sha256(&time_bytes, secret)?;
                let offset = (signature[signature.len() - 1] & 0x0f) as usize;
                let generated_code = u32::from_be_bytes([
                    signature[offset] & 0x7f,
                    signature[offset + 1],
                    signature[offset + 2],
                    signature[offset + 3],
                ]) % 1_000_000;

                if generated_code == code {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// List available signing keys
    pub async fn list_signing_keys(&self) -> Vec<String> {
        self.signing_keys.read().await.keys().cloned().collect()
    }

    /// Delete a key pair
    pub async fn delete_key_pair(&self, key_id: &str) -> SecurityResult<()> {
        self.signing_keys.write().await.remove(key_id);
        self.verification_keys.write().await.remove(key_id);
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionTokenData {
    random_bytes: Vec<u8>,
    timestamp: u64,
    version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hash_functions() {
        let config = SecurityConfig::default();
        let crypto = CryptoManager::new(&config).unwrap();

        let data = b"Hello, World!";
        
        let sha256_result = crypto.hash_sha256(data);
        assert_eq!(sha256_result.algorithm, "SHA-256");
        assert!(!sha256_result.hash.is_empty());

        let blake3_result = crypto.hash_blake3(data);
        assert_eq!(blake3_result.algorithm, "BLAKE3");
        assert!(!blake3_result.hash.is_empty());
    }

    #[tokio::test]
    async fn test_hmac_signing() {
        let config = SecurityConfig::default();
        let crypto = CryptoManager::new(&config).unwrap();

        let data = b"test message";
        let key = b"secret key";

        let signature = crypto.sign_hmac_sha256(data, key).unwrap();
        let is_valid = crypto.verify_hmac_sha256(data, &signature, key).unwrap();
        
        assert!(is_valid);

        // Test with wrong key
        let wrong_key = b"wrong key";
        let is_valid_wrong = crypto.verify_hmac_sha256(data, &signature, wrong_key).unwrap();
        assert!(!is_valid_wrong);
    }

    #[tokio::test]
    async fn test_random_generation() {
        let config = SecurityConfig::default();
        let crypto = CryptoManager::new(&config).unwrap();

        let bytes1 = crypto.generate_random_bytes(32);
        let bytes2 = crypto.generate_random_bytes(32);
        
        assert_eq!(bytes1.len(), 32);
        assert_eq!(bytes2.len(), 32);
        assert_ne!(bytes1, bytes2); // Should be different

        let string1 = crypto.generate_random_string(16);
        let string2 = crypto.generate_random_string(16);
        
        assert_ne!(string1, string2);
    }

    #[tokio::test]
    async fn test_session_token() {
        let config = SecurityConfig::default();
        let crypto = CryptoManager::new(&config).unwrap();

        let token = crypto.generate_session_token().unwrap();
        let is_valid = crypto.verify_session_token(&token).unwrap();
        
        assert!(is_valid);
        
        // Test invalid token
        let is_invalid = crypto.verify_session_token("invalid_token").unwrap_or(false);
        assert!(!is_invalid);
    }

    #[tokio::test]
    async fn test_totp() {
        let config = SecurityConfig::default();
        let crypto = CryptoManager::new(&config).unwrap();

        let secret = b"test_secret_key_for_totp";
        let time_step = 30;
        
        let code = crypto.generate_totp(secret, time_step).unwrap();
        let is_valid = crypto.verify_totp(secret, code, time_step, 1).unwrap();
        
        assert!(is_valid);
        
        // Test wrong code
        let is_invalid = crypto.verify_totp(secret, code + 1, time_step, 1).unwrap();
        assert!(!is_invalid);
    }
}