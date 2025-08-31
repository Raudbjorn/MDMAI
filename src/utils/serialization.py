"""Centralized JSON serialization utilities for consistent data handling."""

import json
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import UUID

from pydantic import BaseModel


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex types used throughout the application."""
    
    def default(self, obj: Any) -> Any:
        """Convert non-JSON serializable objects to serializable format.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
        """
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle timedelta
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        
        # Handle UUID
        if isinstance(obj, UUID):
            return str(obj)
        
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle Enum
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle Path
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            return obj.dict()
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        
        # Fallback to default encoder
        return super().default(obj)


def serialize_to_json(
    data: Any,
    pretty: bool = False,
    sort_keys: bool = False,
) -> str:
    """Serialize data to JSON string with custom encoding support.
    
    Args:
        data: Data to serialize
        pretty: Whether to format with indentation
        sort_keys: Whether to sort dictionary keys
        
    Returns:
        JSON string representation
    """
    return json.dumps(
        data,
        cls=JSONEncoder,
        indent=2 if pretty else None,
        sort_keys=sort_keys,
        ensure_ascii=False,
    )


def deserialize_from_json(
    json_str: str,
    target_type: Optional[type] = None,
) -> Any:
    """Deserialize JSON string to Python object.
    
    Args:
        json_str: JSON string to deserialize
        target_type: Optional target type for deserialization
        
    Returns:
        Deserialized Python object
        
    Raises:
        json.JSONDecodeError: If JSON is invalid
        ValueError: If type conversion fails
    """
    data = json.loads(json_str)
    
    if target_type and isinstance(data, dict):
        # Attempt to construct target type if it's a Pydantic model
        if issubclass(target_type, BaseModel):
            return target_type(**data)
    
    return data


def safe_json_response(
    data: Any,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a safe JSON response structure.
    
    Args:
        data: Response data
        status_code: HTTP status code
        headers: Optional response headers
        
    Returns:
        Standardized response dictionary
    """
    return {
        "status_code": status_code,
        "data": json.loads(serialize_to_json(data)),
        "headers": headers or {},
        "timestamp": datetime.now().isoformat(),
    }


def merge_json_configs(*configs: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple JSON configurations with deep merging.
    
    Args:
        *configs: Variable number of JSON strings or dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        if isinstance(config, str):
            config = json.loads(config)
        
        if isinstance(config, dict):
            _deep_merge(result, config)
    
    return result


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Deep merge update dictionary into base dictionary.
    
    Args:
        base: Base dictionary to merge into
        update: Dictionary with updates
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def json_diff(
    old_data: Union[str, Dict[str, Any]],
    new_data: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate difference between two JSON structures.
    
    Args:
        old_data: Original JSON data
        new_data: New JSON data
        
    Returns:
        Dictionary showing additions, deletions, and modifications
    """
    if isinstance(old_data, str):
        old_data = json.loads(old_data)
    if isinstance(new_data, str):
        new_data = json.loads(new_data)
    
    diff = {
        "added": {},
        "deleted": {},
        "modified": {},
    }
    
    # Find added and modified keys
    for key, value in new_data.items():
        if key not in old_data:
            diff["added"][key] = value
        elif old_data[key] != value:
            diff["modified"][key] = {
                "old": old_data[key],
                "new": value,
            }
    
    # Find deleted keys
    for key, value in old_data.items():
        if key not in new_data:
            diff["deleted"][key] = value
    
    return diff


def validate_json_schema(
    data: Union[str, Dict[str, Any]],
    schema: Dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """Validate JSON data against a schema.
    
    Args:
        data: JSON data to validate
        schema: JSON schema for validation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import jsonschema
        
        if isinstance(data, str):
            data = json.loads(data)
        
        jsonschema.validate(instance=data, schema=schema)
        return True, None
        
    except jsonschema.ValidationError as e:
        return False, str(e)
    except ImportError:
        # jsonschema not available, do basic validation
        try:
            if isinstance(data, str):
                json.loads(data)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)


# Export commonly used functions
__all__ = [
    "JSONEncoder",
    "serialize_to_json",
    "deserialize_from_json",
    "safe_json_response",
    "merge_json_configs",
    "json_diff",
    "validate_json_schema",
]