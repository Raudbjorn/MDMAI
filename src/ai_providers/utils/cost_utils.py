"""
Shared Cost Utilities Module
Addresses PR #59 review issue: deduplicate cost estimation code
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from ..models import ProviderType, ModelSpec, AIRequest


class ErrorClassification(Enum):
    """
    Standardized error classifications.
    Addresses review issue: improve error classification using specific types.
    """
    
    # Rate limiting errors
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    TOO_MANY_REQUESTS = "too_many_requests"
    
    # Connection errors
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    DNS_RESOLUTION_FAILED = "dns_resolution"
    
    # Service errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    BAD_GATEWAY = "bad_gateway"
    GATEWAY_TIMEOUT = "gateway_timeout"
    
    # Authentication/Authorization
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_DENIED = "permission_denied"
    INVALID_API_KEY = "invalid_api_key"
    
    # Request errors
    INVALID_REQUEST = "invalid_request"
    VALIDATION_ERROR = "validation_error"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    
    # Model errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_OVERLOADED = "model_overloaded"
    
    # Other
    UNKNOWN_ERROR = "unknown_error"
    TRANSIENT_ERROR = "transient_error"


def classify_error(error: Exception) -> ErrorClassification:
    """
    Classify an error into a standardized category.
    Uses exception type and attributes instead of string matching.
    """
    # Check exception type first
    error_type = type(error).__name__
    error_message = str(error).lower()
    
    # Check for specific exception types (these should be defined in error_handler.py)
    if hasattr(error, '__class__'):
        # Rate limiting
        if error_type in ['RateLimitError', 'RateLimitException']:
            return ErrorClassification.RATE_LIMIT
        if error_type in ['QuotaExceededError', 'QuotaException']:
            return ErrorClassification.QUOTA_EXCEEDED
            
        # Timeout
        if error_type in ['TimeoutError', 'AsyncTimeoutError', 'ReadTimeoutError']:
            return ErrorClassification.TIMEOUT
            
        # Connection
        if error_type in ['ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError']:
            return ErrorClassification.CONNECTION_ERROR
            
        # Authentication
        if error_type in ['AuthenticationError', 'AuthorizationError']:
            return ErrorClassification.AUTHENTICATION_ERROR
        if error_type in ['PermissionError', 'ForbiddenError']:
            return ErrorClassification.PERMISSION_DENIED
    
    # Check HTTP status codes if available
    if hasattr(error, 'status_code'):
        status_code = error.status_code
        if status_code == 429:
            return ErrorClassification.TOO_MANY_REQUESTS
        elif status_code == 401:
            return ErrorClassification.AUTHENTICATION_ERROR
        elif status_code == 403:
            return ErrorClassification.PERMISSION_DENIED
        elif status_code == 500:
            return ErrorClassification.INTERNAL_SERVER_ERROR
        elif status_code == 502:
            return ErrorClassification.BAD_GATEWAY
        elif status_code == 503:
            return ErrorClassification.SERVICE_UNAVAILABLE
        elif status_code == 504:
            return ErrorClassification.GATEWAY_TIMEOUT
        elif 400 <= status_code < 500:
            return ErrorClassification.INVALID_REQUEST
    
    # Check error attributes for more specific classification
    if hasattr(error, 'error_type'):
        error_type_attr = str(error.error_type).lower()
        if 'rate' in error_type_attr or 'limit' in error_type_attr:
            return ErrorClassification.RATE_LIMIT
        if 'quota' in error_type_attr:
            return ErrorClassification.QUOTA_EXCEEDED
        if 'timeout' in error_type_attr:
            return ErrorClassification.TIMEOUT
        if 'auth' in error_type_attr:
            return ErrorClassification.AUTHENTICATION_ERROR
        if 'model' in error_type_attr:
            return ErrorClassification.MODEL_NOT_FOUND
    
    # Fallback to message inspection as last resort
    if 'rate limit' in error_message or 'too many requests' in error_message:
        return ErrorClassification.RATE_LIMIT
    if 'quota' in error_message or 'usage limit' in error_message:
        return ErrorClassification.QUOTA_EXCEEDED
    if 'timeout' in error_message or 'timed out' in error_message:
        return ErrorClassification.TIMEOUT
    if 'connection' in error_message or 'network' in error_message:
        return ErrorClassification.CONNECTION_ERROR
    if 'authentication' in error_message or 'unauthorized' in error_message:
        return ErrorClassification.AUTHENTICATION_ERROR
    if 'permission' in error_message or 'forbidden' in error_message:
        return ErrorClassification.PERMISSION_DENIED
    if 'not found' in error_message and 'model' in error_message:
        return ErrorClassification.MODEL_NOT_FOUND
    if 'context' in error_message and ('length' in error_message or 'too long' in error_message):
        return ErrorClassification.CONTEXT_LENGTH_EXCEEDED
    if 'service' in error_message and 'unavailable' in error_message:
        return ErrorClassification.SERVICE_UNAVAILABLE
    if 'internal' in error_message and 'error' in error_message:
        return ErrorClassification.INTERNAL_SERVER_ERROR
    
    # Check if it's likely a transient error
    transient_keywords = ['temporary', 'transient', 'retry', 'try again']
    if any(keyword in error_message for keyword in transient_keywords):
        return ErrorClassification.TRANSIENT_ERROR
    
    return ErrorClassification.UNKNOWN_ERROR


def estimate_input_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate input token count from messages.
    Shared utility to avoid duplication.
    """
    total_chars = 0
    
    for message in messages:
        # Handle content field
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Handle multi-modal content
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
                    elif item.get("type") == "image":
                        # Rough estimate for image tokens
                        total_chars += 1000  # Approximate token count for images
        
        # Add role overhead
        role = message.get("role", "")
        total_chars += len(role) + 10  # Role + formatting overhead
        
        # Handle function/tool calls if present
        if "function_call" in message:
            func_call = message["function_call"]
            total_chars += len(str(func_call.get("name", ""))) + len(str(func_call.get("arguments", "")))
        
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                total_chars += len(str(tool_call))
    
    # Rough estimation: 4 characters per token (GPT-3/4 average)
    # This can be refined based on specific tokenizer if needed
    return max(1, total_chars // 4)


def estimate_output_tokens(request: AIRequest, model_spec: Optional[ModelSpec] = None) -> int:
    """
    Estimate output token count for a request.
    """
    # Use request's max_tokens if specified
    if request.max_tokens:
        output_tokens = request.max_tokens
    elif model_spec:
        # Use model's default or max output tokens
        output_tokens = min(model_spec.max_output_tokens, 1000)  # Default to 1000
    else:
        output_tokens = 1000  # Fallback default
    
    # Apply model-specific limits if available
    if model_spec:
        output_tokens = min(output_tokens, model_spec.max_output_tokens)
    
    return output_tokens


def estimate_request_cost(
    provider_type: ProviderType,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    config_manager: Optional[Any] = None
) -> float:
    """
    Centralized cost estimation function.
    Uses configuration manager if available, otherwise uses defaults.
    """
    # Try to use config manager if provided
    if config_manager:
        try:
            from ..config.model_config import get_model_config_manager
            config_mgr = config_manager or get_model_config_manager()
            return config_mgr.get_model_cost(model_id, input_tokens, output_tokens)
        except ImportError:
            pass
    
    # Fallback to hardcoded rates (these should match the config defaults)
    default_rates = {
        ProviderType.ANTHROPIC: {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        },
        ProviderType.OPENAI: {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        },
        ProviderType.GOOGLE: {
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-ultra": {"input": 0.001, "output": 0.003},
        },
    }
    
    # Get rates for provider and model
    provider_rates = default_rates.get(provider_type, {})
    model_rates = provider_rates.get(model_id, {"input": 0.001, "output": 0.002})
    
    # Calculate cost
    input_cost = (input_tokens / 1000.0) * model_rates["input"]
    output_cost = (output_tokens / 1000.0) * model_rates["output"]
    
    return input_cost + output_cost


def calculate_token_efficiency(
    input_tokens: int,
    output_tokens: int,
    processing_time_ms: float,
    cost: float
) -> Dict[str, float]:
    """
    Calculate various efficiency metrics for a request.
    """
    total_tokens = input_tokens + output_tokens
    processing_time_seconds = processing_time_ms / 1000.0
    
    return {
        "tokens_per_second": total_tokens / processing_time_seconds if processing_time_seconds > 0 else 0,
        "cost_per_1k_tokens": (cost / total_tokens) * 1000 if total_tokens > 0 else 0,
        "input_output_ratio": input_tokens / output_tokens if output_tokens > 0 else float('inf'),
        "processing_efficiency": total_tokens / cost if cost > 0 else 0,  # Tokens per dollar
    }


def assess_request_complexity(messages: List[Dict[str, Any]]) -> float:
    """
    Assess the complexity of a request based on message content.
    Returns a score between 0.0 (simple) and 1.0 (complex).
    """
    complexity_score = 0.0
    
    # Analyze message content
    total_content = ""
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_content += content.lower()
    
    # Complexity indicators with weights
    complexity_indicators = [
        # Technical/specialized terms
        (["algorithm", "optimize", "implement", "architecture", "framework"], 0.15),
        (["analyze", "evaluate", "compare", "assess", "investigate"], 0.12),
        (["code", "programming", "function", "class", "method"], 0.15),
        (["complex", "complicated", "sophisticated", "advanced"], 0.10),
        (["detailed", "comprehensive", "thorough", "extensive"], 0.08),
        (["explain", "describe", "elaborate", "clarify"], 0.05),
        
        # Multi-step tasks
        (["first", "then", "next", "finally", "step"], 0.10),
        (["multiple", "several", "various", "different"], 0.08),
        
        # Data processing
        (["data", "dataset", "database", "table", "query"], 0.10),
        (["calculate", "compute", "formula", "equation"], 0.12),
    ]
    
    for keywords, weight in complexity_indicators:
        if any(keyword in total_content for keyword in keywords):
            complexity_score += weight
    
    # Length-based complexity
    content_length = len(total_content)
    if content_length > 5000:
        complexity_score += 0.25
    elif content_length > 2000:
        complexity_score += 0.15
    elif content_length > 1000:
        complexity_score += 0.10
    elif content_length > 500:
        complexity_score += 0.05
    
    # Number of messages (conversation depth)
    if len(messages) > 10:
        complexity_score += 0.15
    elif len(messages) > 5:
        complexity_score += 0.10
    elif len(messages) > 3:
        complexity_score += 0.05
    
    # Check for code blocks
    if "```" in total_content:
        complexity_score += 0.20
    
    # Check for structured data (JSON, tables, etc.)
    if any(char in total_content for char in ["{", "[", "|"]):
        complexity_score += 0.10
    
    # Normalize to 0-1 range
    return min(1.0, complexity_score)


def get_cost_tier_for_budget(
    remaining_budget: float,
    total_budget: float,
    aggressive_savings: bool = False
) -> str:
    """
    Determine appropriate cost tier based on remaining budget.
    """
    budget_percentage = (remaining_budget / total_budget) * 100 if total_budget > 0 else 0
    
    if aggressive_savings:
        # More conservative thresholds
        if budget_percentage < 10:
            return "LOW"  # Emergency mode
        elif budget_percentage < 25:
            return "LOW"
        elif budget_percentage < 50:
            return "MEDIUM"
        else:
            return "HIGH"
    else:
        # Normal thresholds
        if budget_percentage < 5:
            return "LOW"  # Emergency mode
        elif budget_percentage < 20:
            return "MEDIUM"
        elif budget_percentage < 50:
            return "HIGH"
        else:
            return "PREMIUM"


def calculate_provider_adjustment_factor(
    provider_type: ProviderType,
    historical_performance: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate provider-specific cost adjustment factor based on performance.
    """
    # Base adjustments
    base_adjustments = {
        ProviderType.ANTHROPIC: 1.0,   # Baseline
        ProviderType.OPENAI: 1.05,     # Slight premium for reliability
        ProviderType.GOOGLE: 0.95,     # Slight discount
    }
    
    adjustment = base_adjustments.get(provider_type, 1.0)
    
    # Apply historical performance adjustments if available
    if historical_performance:
        # Adjust based on success rate
        success_rate = historical_performance.get("success_rate", 0.95)
        if success_rate < 0.9:
            adjustment *= 1.1  # Penalty for low reliability
        elif success_rate > 0.98:
            adjustment *= 0.95  # Reward for high reliability
        
        # Adjust based on average latency
        avg_latency = historical_performance.get("avg_latency_ms", 3000)
        if avg_latency > 10000:
            adjustment *= 1.05  # Penalty for slow response
        elif avg_latency < 2000:
            adjustment *= 0.98  # Reward for fast response
    
    return adjustment