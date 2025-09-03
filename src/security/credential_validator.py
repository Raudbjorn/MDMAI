"""
API key validation service for different AI providers.

This module validates API keys before storage to ensure they are:
- Properly formatted for each provider
- Active and functional
- Have appropriate permissions
- Not rate-limited or blocked

Supports validation for:
- Anthropic (Claude)
- OpenAI (GPT models)
- Google (Gemini)
- Custom validation rules
"""

import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import httpx
from structlog import get_logger
from returns.result import Result, Success, Failure

from ..ai_providers.models import ProviderType

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of API key validation."""
    
    is_valid: bool
    provider_type: ProviderType
    key_format_valid: bool
    key_active: bool
    has_required_permissions: bool
    rate_limit_status: Optional[Dict[str, Any]] = None
    account_info: Optional[Dict[str, Any]] = None
    issues: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    validated_at: datetime = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
        if self.validated_at is None:
            self.validated_at = datetime.utcnow()
    
    def add_issue(self, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add a validation issue."""
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.issues.append(f"[{severity.value.upper()}] {message}")
            self.is_valid = False
        else:
            self.warnings.append(f"[{severity.value.upper()}] {message}")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        if self.is_valid:
            summary = f"✓ Valid {self.provider_type.value} API key"
            if self.warnings:
                summary += f" (with {len(self.warnings)} warnings)"
        else:
            summary = f"✗ Invalid {self.provider_type.value} API key"
            if self.issues:
                summary += f" ({len(self.issues)} issues)"
        
        return summary


class APIKeyValidator(ABC):
    """Abstract base class for API key validators."""
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Get the provider type this validator handles."""
        pass
    
    @abstractmethod
    def validate_format(self, api_key: str) -> Tuple[bool, List[str]]:
        """Validate the format of the API key."""
        pass
    
    @abstractmethod
    async def validate_functionality(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate that the API key is functional."""
        pass
    
    @abstractmethod
    async def get_account_info(self, api_key: str) -> Dict[str, Any]:
        """Get account information for the API key."""
        pass
    
    async def validate(self, api_key: str) -> ValidationResult:
        """Perform complete validation of API key."""
        result = ValidationResult(
            is_valid=True,
            provider_type=self.provider_type,
            key_format_valid=False,
            key_active=False,
            has_required_permissions=False
        )
        
        try:
            # Step 1: Format validation
            format_valid, format_issues = self.validate_format(api_key)
            result.key_format_valid = format_valid
            
            for issue in format_issues:
                result.add_issue(issue, ValidationSeverity.ERROR)
            
            if not format_valid:
                return result
            
            # Step 2: Functionality validation
            func_valid, func_info = await self.validate_functionality(api_key)
            result.key_active = func_valid
            result.metadata.update(func_info)
            
            if not func_valid:
                result.add_issue("API key is not functional", ValidationSeverity.ERROR)
                return result
            
            # Step 3: Get account info
            try:
                account_info = await self.get_account_info(api_key)
                result.account_info = account_info
                result.has_required_permissions = True  # If we got this far, permissions are likely OK
                
                # Check for account-specific warnings
                if account_info.get('usage_exceeded'):
                    result.add_issue("Account usage limits may be exceeded", ValidationSeverity.WARNING)
                
                if account_info.get('rate_limited'):
                    result.add_issue("Account is currently rate limited", ValidationSeverity.WARNING)
                
            except Exception as e:
                result.add_issue(f"Could not retrieve account info: {str(e)}", ValidationSeverity.WARNING)
            
            logger.info(
                "API key validation completed",
                provider=self.provider_type.value,
                valid=result.is_valid,
                issues=len(result.issues),
                warnings=len(result.warnings)
            )
            
        except Exception as e:
            result.add_issue(f"Validation failed with exception: {str(e)}", ValidationSeverity.CRITICAL)
            logger.error("API key validation failed", provider=self.provider_type.value, error=str(e))
        
        return result


class AnthropicValidator(APIKeyValidator):
    """Validator for Anthropic API keys."""
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC
    
    def validate_format(self, api_key: str) -> Tuple[bool, List[str]]:
        """Validate Anthropic API key format."""
        issues = []
        
        # Anthropic keys start with 'sk-ant-'
        if not api_key.startswith('sk-ant-'):
            issues.append("Anthropic API keys must start with 'sk-ant-'")
            return False, issues
        
        # Check length (approximately 108 characters total)
        if len(api_key) < 100 or len(api_key) > 120:
            issues.append(f"Anthropic API key length ({len(api_key)}) appears invalid (expected ~108 chars)")
        
        # Check for valid base64-like characters after prefix
        key_body = api_key[7:]  # Remove 'sk-ant-' prefix
        if not re.match(r'^[A-Za-z0-9+/=_-]+$', key_body):
            issues.append("Anthropic API key contains invalid characters")
            return False, issues
        
        return len(issues) == 0, issues
    
    async def validate_functionality(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Test Anthropic API key functionality."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                }
                
                # Test with a minimal completion request
                test_data = {
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 1,
                    'messages': [{'role': 'user', 'content': 'Hi'}]
                }
                
                response = await client.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=test_data
                )
                
                if response.status_code == 200:
                    return True, {'test_response_status': 200}
                elif response.status_code == 401:
                    return False, {'error': 'Invalid API key', 'status_code': 401}
                elif response.status_code == 403:
                    return False, {'error': 'Access forbidden', 'status_code': 403}
                elif response.status_code == 429:
                    # Rate limited but key might be valid
                    return True, {'warning': 'Rate limited', 'status_code': 429}
                else:
                    return False, {'error': f'Unexpected status: {response.status_code}'}
                
        except httpx.TimeoutException:
            return False, {'error': 'Request timed out'}
        except Exception as e:
            return False, {'error': f'Request failed: {str(e)}'}
    
    async def get_account_info(self, api_key: str) -> Dict[str, Any]:
        """Get Anthropic account information."""
        # Anthropic doesn't have a dedicated account info endpoint
        # So we'll use the successful test call info
        return {
            'provider': 'anthropic',
            'verified': True,
            'note': 'Anthropic does not provide account info endpoint'
        }


class OpenAIValidator(APIKeyValidator):
    """Validator for OpenAI API keys."""
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI
    
    def validate_format(self, api_key: str) -> Tuple[bool, List[str]]:
        """Validate OpenAI API key format."""
        issues = []
        
        # OpenAI keys start with 'sk-'
        if not api_key.startswith('sk-'):
            issues.append("OpenAI API keys must start with 'sk-'")
            return False, issues
        
        # Check length (typically 51 characters for current format)
        if len(api_key) < 40 or len(api_key) > 60:
            issues.append(f"OpenAI API key length ({len(api_key)}) appears invalid (expected ~51 chars)")
        
        # Check for valid characters after prefix
        key_body = api_key[3:]  # Remove 'sk-' prefix
        if not re.match(r'^[A-Za-z0-9]+$', key_body):
            issues.append("OpenAI API key contains invalid characters")
            return False, issues
        
        return len(issues) == 0, issues
    
    async def validate_functionality(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Test OpenAI API key functionality."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Test with a minimal completion request
                test_data = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                    'max_tokens': 1
                }
                
                response = await client.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=test_data
                )
                
                if response.status_code == 200:
                    return True, {'test_response_status': 200}
                elif response.status_code == 401:
                    return False, {'error': 'Invalid API key', 'status_code': 401}
                elif response.status_code == 403:
                    return False, {'error': 'Access forbidden', 'status_code': 403}
                elif response.status_code == 429:
                    return True, {'warning': 'Rate limited', 'status_code': 429}
                else:
                    return False, {'error': f'Unexpected status: {response.status_code}'}
                
        except httpx.TimeoutException:
            return False, {'error': 'Request timed out'}
        except Exception as e:
            return False, {'error': f'Request failed: {str(e)}'}
    
    async def get_account_info(self, api_key: str) -> Dict[str, Any]:
        """Get OpenAI account information."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Get account usage (if available)
                response = await client.get(
                    'https://api.openai.com/v1/usage',
                    headers=headers,
                    params={'date': datetime.utcnow().strftime('%Y-%m-%d')}
                )
                
                account_info = {'provider': 'openai'}
                
                if response.status_code == 200:
                    usage_data = response.json()
                    account_info.update({
                        'usage_data': usage_data,
                        'verified': True
                    })
                else:
                    account_info['note'] = 'Usage data not available'
                
                return account_info
                
        except Exception as e:
            return {
                'provider': 'openai',
                'error': f'Could not retrieve account info: {str(e)}'
            }


class GoogleValidator(APIKeyValidator):
    """Validator for Google AI API keys."""
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE
    
    def validate_format(self, api_key: str) -> Tuple[bool, List[str]]:
        """Validate Google AI API key format."""
        issues = []
        
        # Google AI keys are typically 39 characters long
        if len(api_key) < 35 or len(api_key) > 45:
            issues.append(f"Google API key length ({len(api_key)}) appears invalid (expected ~39 chars)")
        
        # Check for valid characters (alphanumeric and some special chars)
        if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
            issues.append("Google API key contains invalid characters")
            return False, issues
        
        return len(issues) == 0, issues
    
    async def validate_functionality(self, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Test Google AI API key functionality."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test with Gemini API
                url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}'
                
                test_data = {
                    'contents': [{
                        'parts': [{'text': 'Hi'}]
                    }],
                    'generationConfig': {
                        'maxOutputTokens': 1
                    }
                }
                
                response = await client.post(url, json=test_data)
                
                if response.status_code == 200:
                    return True, {'test_response_status': 200}
                elif response.status_code == 400:
                    # Check if it's an API key issue
                    error_data = response.json()
                    if 'API_KEY_INVALID' in str(error_data):
                        return False, {'error': 'Invalid API key', 'status_code': 400}
                    return True, {'warning': 'Request format issue', 'status_code': 400}
                elif response.status_code == 403:
                    return False, {'error': 'Access forbidden', 'status_code': 403}
                elif response.status_code == 429:
                    return True, {'warning': 'Rate limited', 'status_code': 429}
                else:
                    return False, {'error': f'Unexpected status: {response.status_code}'}
                
        except httpx.TimeoutException:
            return False, {'error': 'Request timed out'}
        except Exception as e:
            return False, {'error': f'Request failed: {str(e)}'}
    
    async def get_account_info(self, api_key: str) -> Dict[str, Any]:
        """Get Google AI account information."""
        # Google AI doesn't provide account info endpoint for API keys
        return {
            'provider': 'google',
            'verified': True,
            'note': 'Google AI does not provide account info endpoint'
        }


class CredentialValidationService:
    """Service for validating API credentials across providers."""
    
    def __init__(self):
        """Initialize the validation service."""
        self.validators: Dict[ProviderType, APIKeyValidator] = {
            ProviderType.ANTHROPIC: AnthropicValidator(),
            ProviderType.OPENAI: OpenAIValidator(),
            ProviderType.GOOGLE: GoogleValidator()
        }
        
        # Validation cache to avoid repeated validations
        self._validation_cache: Dict[str, Tuple[ValidationResult, datetime]] = {}
        self.cache_ttl_minutes = 30
        
        logger.info(
            "Initialized credential validation service",
            supported_providers=[p.value for p in self.validators.keys()]
        )
    
    async def validate_credential(
        self,
        api_key: str,
        provider_type: ProviderType,
        skip_cache: bool = False
    ) -> Result[ValidationResult, str]:
        """
        Validate an API credential.
        
        Args:
            api_key: The API key to validate
            provider_type: The provider type
            skip_cache: Whether to skip validation cache
            
        Returns:
            Result containing ValidationResult or error message
        """
        try:
            # Check cache first
            cache_key = f"{provider_type.value}:{hash(api_key)}"
            if not skip_cache and cache_key in self._validation_cache:
                cached_result, cached_at = self._validation_cache[cache_key]
                if datetime.utcnow() - cached_at < timedelta(minutes=self.cache_ttl_minutes):
                    logger.debug("Returning cached validation result", provider=provider_type.value)
                    return Success(cached_result)
            
            # Get validator
            validator = self.validators.get(provider_type)
            if not validator:
                return Failure(f"No validator available for provider: {provider_type.value}")
            
            # Perform validation
            logger.info("Validating API key", provider=provider_type.value)
            validation_result = await validator.validate(api_key)
            
            # Cache result
            self._validation_cache[cache_key] = (validation_result, datetime.utcnow())
            
            # Cleanup old cache entries
            self._cleanup_cache()
            
            return Success(validation_result)
            
        except Exception as e:
            logger.error("Credential validation failed", error=str(e))
            return Failure(f"Validation failed: {str(e)}")
    
    async def validate_multiple_credentials(
        self,
        credentials: List[Tuple[str, ProviderType]]
    ) -> Dict[ProviderType, ValidationResult]:
        """
        Validate multiple credentials in parallel.
        
        Args:
            credentials: List of (api_key, provider_type) tuples
            
        Returns:
            Dictionary mapping provider types to validation results
        """
        results = {}
        
        # Create validation tasks
        tasks = []
        provider_types = []
        for api_key, provider_type in credentials:
            task = self.validate_credential(api_key, provider_type)
            tasks.append(task)
            provider_types.append(provider_type)
        
        # Execute in parallel using asyncio.gather
        try:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result, provider_type in zip(task_results, provider_types):
                if isinstance(result, Exception):
                    # Handle exceptions
                    failed_result = ValidationResult(
                        is_valid=False,
                        provider_type=provider_type,
                        key_format_valid=False,
                        key_active=False,
                        has_required_permissions=False
                    )
                    failed_result.add_issue(str(result), ValidationSeverity.CRITICAL)
                    results[provider_type] = failed_result
                elif result.is_success():
                    results[provider_type] = result.unwrap()
                else:
                    # Create failed validation result  
                    failed_result = ValidationResult(
                        is_valid=False,
                        provider_type=provider_type,
                        key_format_valid=False,
                        key_active=False,
                        has_required_permissions=False
                    )
                    failed_result.add_issue(result.failure(), ValidationSeverity.CRITICAL)
                    results[provider_type] = failed_result
        except Exception as e:
            logger.error("Failed to validate credentials in parallel", error=str(e))
        
        return results
    
    def get_supported_providers(self) -> List[ProviderType]:
        """Get list of supported provider types."""
        return list(self.validators.keys())
    
    def add_validator(self, validator: APIKeyValidator) -> None:
        """Add a custom validator."""
        self.validators[validator.provider_type] = validator
        logger.info("Added custom validator", provider=validator.provider_type.value)
    
    def _cleanup_cache(self) -> None:
        """Remove expired entries from validation cache."""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for cache_key, (_, cached_at) in self._validation_cache.items():
            if current_time - cached_at >= timedelta(minutes=self.cache_ttl_minutes):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._validation_cache[key]
        
        if expired_keys:
            logger.debug("Cleaned up validation cache", expired_entries=len(expired_keys))
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation service statistics."""
        return {
            'supported_providers': [p.value for p in self.validators.keys()],
            'cache_entries': len(self._validation_cache),
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'validators': {
                provider.value: type(validator).__name__ 
                for provider, validator in self.validators.items()
            }
        }