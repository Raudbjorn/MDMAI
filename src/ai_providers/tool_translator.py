"""Tool format translator for converting between MCP and provider-specific formats."""

from typing import Any, Dict, List

from structlog import get_logger

from .models import MCPTool, ProviderType

logger = get_logger(__name__)


class ToolTranslator:
    """Translates tool definitions between MCP format and provider-specific formats.
    
    Handles the conversion of tool schemas, parameter types, and descriptions
    to ensure compatibility across different AI providers.
    """
    
    @staticmethod
    def mcp_to_provider(tools: List[MCPTool], provider_type: ProviderType) -> List[Dict[str, Any]]:
        """Convert MCP tools to provider-specific format.
        
        Args:
            tools: List of MCP tool definitions
            provider_type: Target provider type
            
        Returns:
            List of tools in provider-specific format
        """
        if provider_type == ProviderType.ANTHROPIC:
            return ToolTranslator._mcp_to_anthropic(tools)
        elif provider_type == ProviderType.OPENAI:
            return ToolTranslator._mcp_to_openai(tools)
        elif provider_type == ProviderType.GOOGLE:
            return ToolTranslator._mcp_to_google(tools)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def provider_to_mcp(tools: List[Dict[str, Any]], provider_type: ProviderType) -> List[MCPTool]:
        """Convert provider-specific tools to MCP format.
        
        Args:
            tools: List of provider-specific tool definitions
            provider_type: Source provider type
            
        Returns:
            List of MCP tool definitions
        """
        if provider_type == ProviderType.ANTHROPIC:
            return ToolTranslator._anthropic_to_mcp(tools)
        elif provider_type == ProviderType.OPENAI:
            return ToolTranslator._openai_to_mcp(tools)
        elif provider_type == ProviderType.GOOGLE:
            return ToolTranslator._google_to_mcp(tools)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def _mcp_to_anthropic(tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic format."""
        anthropic_tools = []
        
        for tool in tools:
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": ToolTranslator._normalize_schema(tool.inputSchema),
            }
            anthropic_tools.append(anthropic_tool)
        
        logger.debug("Converted MCP tools to Anthropic format", count=len(anthropic_tools))
        return anthropic_tools
    
    @staticmethod
    def _mcp_to_openai(tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI format."""
        openai_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": ToolTranslator._normalize_schema(tool.inputSchema),
                },
            }
            openai_tools.append(openai_tool)
        
        logger.debug("Converted MCP tools to OpenAI format", count=len(openai_tools))
        return openai_tools
    
    @staticmethod
    def _mcp_to_google(tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Google format."""
        function_declarations = []
        
        for tool in tools:
            function_declaration = {
                "name": tool.name,
                "description": tool.description,
                "parameters": ToolTranslator._normalize_schema(tool.inputSchema),
            }
            function_declarations.append(function_declaration)
        
        # Google expects tools wrapped in a tools array with functionDeclarations
        google_tools = [{
            "functionDeclarations": function_declarations
        }]
        
        logger.debug("Converted MCP tools to Google format", count=len(function_declarations))
        return google_tools
    
    @staticmethod
    def _anthropic_to_mcp(tools: List[Dict[str, Any]]) -> List[MCPTool]:
        """Convert Anthropic tools to MCP format."""
        mcp_tools = []
        
        for tool in tools:
            mcp_tool = MCPTool(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                inputSchema=tool.get("input_schema", {}),
            )
            mcp_tools.append(mcp_tool)
        
        logger.debug("Converted Anthropic tools to MCP format", count=len(mcp_tools))
        return mcp_tools
    
    @staticmethod
    def _openai_to_mcp(tools: List[Dict[str, Any]]) -> List[MCPTool]:
        """Convert OpenAI tools to MCP format."""
        mcp_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                mcp_tool = MCPTool(
                    name=function.get("name", ""),
                    description=function.get("description", ""),
                    inputSchema=function.get("parameters", {}),
                )
                mcp_tools.append(mcp_tool)
        
        logger.debug("Converted OpenAI tools to MCP format", count=len(mcp_tools))
        return mcp_tools
    
    @staticmethod
    def _google_to_mcp(tools: List[Dict[str, Any]]) -> List[MCPTool]:
        """Convert Google tools to MCP format."""
        mcp_tools = []
        
        for tool_group in tools:
            function_declarations = tool_group.get("functionDeclarations", [])
            for function in function_declarations:
                mcp_tool = MCPTool(
                    name=function.get("name", ""),
                    description=function.get("description", ""),
                    inputSchema=function.get("parameters", {}),
                )
                mcp_tools.append(mcp_tool)
        
        logger.debug("Converted Google tools to MCP format", count=len(mcp_tools))
        return mcp_tools
    
    @staticmethod
    def _normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize JSON schema to ensure compatibility across providers.
        
        Args:
            schema: Original JSON schema
            
        Returns:
            Normalized schema compatible with all providers
        """
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        
        # Ensure we have basic schema structure
        normalized = {
            "type": schema.get("type", "object"),
        }
        
        # Handle properties
        if "properties" in schema:
            normalized["properties"] = {}
            for prop_name, prop_def in schema["properties"].items():
                normalized["properties"][prop_name] = ToolTranslator._normalize_property(prop_def)
        
        # Handle required fields
        if "required" in schema and isinstance(schema["required"], list):
            normalized["required"] = schema["required"]
        
        # Handle description
        if "description" in schema:
            normalized["description"] = schema["description"]
        
        # Handle additional properties
        if "additionalProperties" in schema:
            normalized["additionalProperties"] = schema["additionalProperties"]
        
        return normalized
    
    @staticmethod
    def _normalize_property(prop_def: Any) -> Dict[str, Any]:
        """Normalize a single property definition.
        
        Args:
            prop_def: Property definition
            
        Returns:
            Normalized property definition
        """
        if not isinstance(prop_def, dict):
            return {"type": "string", "description": str(prop_def)}
        
        normalized = {}
        
        # Handle type
        prop_type = prop_def.get("type", "string")
        if prop_type in ["string", "number", "integer", "boolean", "array", "object"]:
            normalized["type"] = prop_type
        else:
            # Map custom types to standard types
            type_mapping = {
                "str": "string",
                "int": "integer", 
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            normalized["type"] = type_mapping.get(prop_type, "string")
        
        # Handle description
        if "description" in prop_def:
            normalized["description"] = prop_def["description"]
        
        # Handle enum values
        if "enum" in prop_def:
            normalized["enum"] = prop_def["enum"]
        
        # Handle array items
        if prop_type == "array" and "items" in prop_def:
            normalized["items"] = ToolTranslator._normalize_property(prop_def["items"])
        
        # Handle object properties
        if prop_type == "object":
            if "properties" in prop_def:
                normalized["properties"] = {}
                for sub_prop_name, sub_prop_def in prop_def["properties"].items():
                    normalized["properties"][sub_prop_name] = ToolTranslator._normalize_property(
                        sub_prop_def
                    )
            
            if "required" in prop_def:
                normalized["required"] = prop_def["required"]
        
        # Handle format constraints
        if "format" in prop_def:
            normalized["format"] = prop_def["format"]
        
        # Handle numeric constraints
        for constraint in ["minimum", "maximum", "minLength", "maxLength", "pattern"]:
            if constraint in prop_def:
                normalized[constraint] = prop_def[constraint]
        
        return normalized
    
    @staticmethod
    def validate_tool_compatibility(
        tool: MCPTool, provider_type: ProviderType
    ) -> List[str]:
        """Validate if a tool is compatible with a specific provider.
        
        Args:
            tool: MCP tool to validate
            provider_type: Target provider type
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check tool name
        if not tool.name or not isinstance(tool.name, str):
            errors.append("Tool name is required and must be a string")
        elif not tool.name.isidentifier():
            errors.append("Tool name must be a valid identifier")
        
        # Check description
        if not tool.description or not isinstance(tool.description, str):
            errors.append("Tool description is required and must be a string")
        
        # Check schema
        if not isinstance(tool.inputSchema, dict):
            errors.append("Input schema must be a dictionary")
        else:
            schema_errors = ToolTranslator._validate_schema(tool.inputSchema, provider_type)
            errors.extend(schema_errors)
        
        # Provider-specific validations
        if provider_type == ProviderType.ANTHROPIC:
            errors.extend(ToolTranslator._validate_anthropic_tool(tool))
        elif provider_type == ProviderType.OPENAI:
            errors.extend(ToolTranslator._validate_openai_tool(tool))
        elif provider_type == ProviderType.GOOGLE:
            errors.extend(ToolTranslator._validate_google_tool(tool))
        
        return errors
    
    @staticmethod
    def _validate_schema(schema: Dict[str, Any], provider_type: ProviderType) -> List[str]:
        """Validate JSON schema for provider compatibility."""
        errors = []
        
        # Check basic structure
        if "type" not in schema:
            errors.append("Schema must have a 'type' field")
        
        schema_type = schema.get("type")
        if schema_type not in ["object", "array", "string", "number", "integer", "boolean"]:
            errors.append(f"Unsupported schema type: {schema_type}")
        
        # Validate properties if object type
        if schema_type == "object" and "properties" in schema:
            properties = schema["properties"]
            if not isinstance(properties, dict):
                errors.append("Properties must be a dictionary")
            else:
                for prop_name, prop_def in properties.items():
                    if not isinstance(prop_def, dict):
                        errors.append(f"Property '{prop_name}' must be a dictionary")
        
        return errors
    
    @staticmethod
    def _validate_anthropic_tool(tool: MCPTool) -> List[str]:
        """Validate tool for Anthropic compatibility."""
        errors = []
        
        # Anthropic-specific constraints
        if len(tool.name) > 64:
            errors.append("Tool name must be 64 characters or less for Anthropic")
        
        if len(tool.description) > 1024:
            errors.append("Tool description must be 1024 characters or less for Anthropic")
        
        return errors
    
    @staticmethod
    def _validate_openai_tool(tool: MCPTool) -> List[str]:
        """Validate tool for OpenAI compatibility."""
        errors = []
        
        # OpenAI-specific constraints
        if len(tool.name) > 64:
            errors.append("Tool name must be 64 characters or less for OpenAI")
        
        if len(tool.description) > 1024:
            errors.append("Tool description must be 1024 characters or less for OpenAI")
        
        return errors
    
    @staticmethod
    def _validate_google_tool(tool: MCPTool) -> List[str]:
        """Validate tool for Google compatibility."""
        errors = []
        
        # Google-specific constraints
        if len(tool.name) > 64:
            errors.append("Tool name must be 64 characters or less for Google")
        
        if len(tool.description) > 1024:
            errors.append("Tool description must be 1024 characters or less for Google")
        
        return errors
    
    @staticmethod
    def get_tool_call_format(provider_type: ProviderType) -> str:
        """Get the expected tool call format for a provider.
        
        Args:
            provider_type: Provider type
            
        Returns:
            Format description string
        """
        formats = {
            ProviderType.ANTHROPIC: "Claude tool_use content blocks",
            ProviderType.OPENAI: "OpenAI function calling format",
            ProviderType.GOOGLE: "Google function calling format",
        }
        return formats.get(provider_type, "Unknown format")
    
    @staticmethod
    def convert_tool_call_result(
        result: Any, provider_type: ProviderType, target_format: str = "mcp"
    ) -> Dict[str, Any]:
        """Convert tool call result between provider formats.
        
        Args:
            result: Tool call result from provider
            provider_type: Source provider type
            target_format: Target format ("mcp", "anthropic", "openai", "google")
            
        Returns:
            Converted result
        """
        # Normalize result to standard format first
        if isinstance(result, dict):
            normalized_result = result
        else:
            normalized_result = {"result": result}
        
        # Convert to target format
        if target_format == "mcp":
            return normalized_result
        elif target_format == "anthropic":
            return {
                "type": "tool_result",
                "content": [{"type": "text", "text": str(normalized_result)}],
            }
        elif target_format == "openai":
            return normalized_result
        elif target_format == "google":
            return {
                "functionResponse": {
                    "name": "tool_result",
                    "response": normalized_result,
                }
            }
        else:
            return normalized_result