"""MCP Protocol translator for converting between client formats and MCP."""

import json
from typing import Any, Dict, List, Optional, Union

from structlog import get_logger

from .models import (
    MCPError,
    MCPErrorCode,
    MCPNotification,
    MCPRequest,
    MCPResponse,
)

logger = get_logger(__name__)


class MCPProtocolTranslator:
    """Translates between different protocol formats and MCP JSON-RPC 2.0."""
    
    def __init__(self):
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        self.resource_registry: Dict[str, Dict[str, Any]] = {}
        self.prompt_registry: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a tool schema for translation."""
        self.tool_registry[name] = schema
        logger.debug("Registered tool", name=name)
    
    def register_resource(self, uri: str, schema: Dict[str, Any]) -> None:
        """Register a resource schema for translation."""
        self.resource_registry[uri] = schema
        logger.debug("Registered resource", uri=uri)
    
    def register_prompt(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a prompt template for translation."""
        self.prompt_registry[name] = schema
        logger.debug("Registered prompt", name=name)
    
    def parse_client_message(
        self,
        message: Union[str, Dict[str, Any]],
    ) -> Union[MCPRequest, MCPNotification, List[Union[MCPRequest, MCPNotification]]]:
        """Parse a message from a client into MCP format."""
        # Handle string messages
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON message: {e}")
        
        # Check for batch request
        if isinstance(message, list):
            return [self._parse_single_message(m) for m in message]
        
        return self._parse_single_message(message)
    
    def _parse_single_message(
        self,
        message: Dict[str, Any],
    ) -> Union[MCPRequest, MCPNotification]:
        """Parse a single message into MCP format."""
        # Check if it's already in JSON-RPC format
        if "jsonrpc" in message and message["jsonrpc"] == "2.0":
            if "id" in message:
                return MCPRequest(**message)
            else:
                return MCPNotification(**message)
        
        # Try to detect the format and translate
        if "tool" in message or "function" in message:
            # Tool/function call format
            return self._translate_tool_call(message)
        elif "resource" in message:
            # Resource request format
            return self._translate_resource_request(message)
        elif "prompt" in message:
            # Prompt request format
            return self._translate_prompt_request(message)
        else:
            # Default to generic request
            method = message.get("method", message.get("action", "unknown"))
            params = message.get("params", message.get("arguments", {}))
            
            if message.get("id"):
                return MCPRequest(id=message["id"], method=method, params=params)
            else:
                return MCPNotification(method=method, params=params)
    
    def _translate_tool_call(self, message: Dict[str, Any]) -> MCPRequest:
        """Translate a tool/function call to MCP format."""
        tool_name = message.get("tool") or message.get("function")
        params = message.get("params") or message.get("arguments") or {}
        
        # Validate against registered schema if available
        if tool_name in self.tool_registry:
            params = self._validate_params(params, self.tool_registry[tool_name])
        
        return MCPRequest(
            id=message.get("id"),
            method=f"tools/{tool_name}",
            params=params,
        )
    
    def _translate_resource_request(self, message: Dict[str, Any]) -> MCPRequest:
        """Translate a resource request to MCP format."""
        resource_uri = message.get("resource")
        
        return MCPRequest(
            id=message.get("id"),
            method="resources/read",
            params={"uri": resource_uri},
        )
    
    def _translate_prompt_request(self, message: Dict[str, Any]) -> MCPRequest:
        """Translate a prompt request to MCP format."""
        prompt_name = message.get("prompt")
        params = message.get("params") or message.get("arguments") or {}
        
        return MCPRequest(
            id=message.get("id"),
            method="prompts/get",
            params={
                "name": prompt_name,
                "arguments": params,
            },
        )
    
    def _validate_params(
        self,
        params: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and coerce parameters against a schema."""
        # Simple validation - can be enhanced with jsonschema
        validated = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in params:
                    validated[prop_name] = params[prop_name]
                elif prop_schema.get("default") is not None:
                    validated[prop_name] = prop_schema["default"]
                elif prop_name in schema.get("required", []):
                    raise ValueError(f"Missing required parameter: {prop_name}")
        else:
            validated = params
        
        return validated
    
    def format_response(
        self,
        response: Union[MCPResponse, MCPError],
        format_type: str = "json-rpc",
    ) -> Dict[str, Any]:
        """Format an MCP response for a client."""
        if format_type == "json-rpc":
            return response.dict() if hasattr(response, "dict") else response
        
        elif format_type == "openai":
            # Format for OpenAI-style function calling
            if isinstance(response, MCPResponse):
                return {
                    "role": "function",
                    "name": response.id,
                    "content": json.dumps(response.result),
                }
            else:
                return {
                    "error": {
                        "message": response.message if hasattr(response, "message") else str(response),
                        "type": "function_error",
                    }
                }
        
        elif format_type == "anthropic":
            # Format for Anthropic-style tool use
            if isinstance(response, MCPResponse):
                return {
                    "type": "tool_result",
                    "tool_use_id": response.id,
                    "content": response.result,
                }
            else:
                return {
                    "type": "error",
                    "error": {
                        "type": "tool_error",
                        "message": response.message if hasattr(response, "message") else str(response),
                    }
                }
        
        else:
            # Default format
            if isinstance(response, MCPResponse):
                return {
                    "success": True,
                    "id": response.id,
                    "result": response.result,
                }
            else:
                return {
                    "success": False,
                    "error": str(response),
                }
    
    def translate_tool_discovery(
        self,
        tools: List[Dict[str, Any]],
        format_type: str = "mcp",
    ) -> List[Dict[str, Any]]:
        """Translate MCP tool discoveries to different formats."""
        if format_type == "mcp":
            return tools
        
        translated = []
        
        for tool in tools:
            if format_type == "openai":
                # OpenAI function format
                translated.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {}),
                    },
                })
            
            elif format_type == "anthropic":
                # Anthropic tool format
                translated.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {}),
                })
            
            else:
                # Generic format
                translated.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                })
        
        return translated
    
    def create_error_response(
        self,
        request_id: Optional[Union[str, int]],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> MCPResponse:
        """Create an MCP error response."""
        return MCPResponse(
            id=request_id,
            error={
                "code": code,
                "message": message,
                "data": data,
            },
        )
    
    def validate_mcp_message(
        self,
        message: Dict[str, Any],
    ) -> bool:
        """Validate if a message conforms to MCP JSON-RPC 2.0 spec."""
        # Check JSON-RPC version
        if message.get("jsonrpc") != "2.0":
            return False
        
        # Check if it's a request, response, or notification
        has_method = "method" in message
        has_result = "result" in message
        has_error = "error" in message
        has_id = "id" in message
        
        # Notification: has method, no id
        if has_method and not has_id and not has_result and not has_error:
            return True
        
        # Request: has method and id
        if has_method and has_id and not has_result and not has_error:
            return True
        
        # Response: has id and (result XOR error)
        if has_id and not has_method and (has_result ^ has_error):
            return True
        
        return False