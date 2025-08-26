# TTRPG Assistant MCP Server - API Documentation

## Overview

The TTRPG Assistant MCP Server provides a comprehensive set of tools for managing tabletop role-playing games through the Model Context Protocol (MCP). This API documentation covers all available tools, their parameters, and expected responses.

## Table of Contents

1. [Search Tools](./search_tools.md)
2. [Source Management](./source_management.md)
3. [Campaign Management](./campaign_management.md)
4. [Session Management](./session_management.md)
5. [Character Generation](./character_generation.md)
6. [Personality Management](./personality_management.md)

## Architecture

The MCP server operates via stdin/stdout communication and is built using FastMCP for Python. All tools are exposed as async functions and follow a consistent pattern for request/response handling.

### Communication Protocol

- **Transport**: stdin/stdout (local operations)
- **Protocol**: Model Context Protocol (MCP) standard
- **Format**: JSON-RPC 2.0
- **Encoding**: UTF-8

### Response Format

All tools follow a standardized response pattern:

#### Success Response
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {...},
  "id": "entity-uuid"
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Descriptive error message",
  "details": {...}
}
```

## Authentication

Currently, the MCP server operates locally without authentication. Future versions may include:
- API key authentication
- OAuth2 support
- JWT tokens for session management

## Rate Limiting

No rate limiting is currently implemented for local operations. When the web UI integration is added, the following limits will apply:
- 60 requests per minute
- 1000 requests per hour
- Configurable per-tool limits

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Invalid request parameters |
| 404 | Resource not found |
| 409 | Conflict (e.g., duplicate resource) |
| 500 | Internal server error |
| 503 | Service unavailable |

## Quick Start

### Starting the Server

```bash
python src/main.py
```

### Example Request

```json
{
  "jsonrpc": "2.0",
  "method": "search",
  "params": {
    "query": "fireball spell",
    "rulebook": "D&D 5e",
    "max_results": 5
  },
  "id": 1
}
```

### Example Response

```json
{
  "jsonrpc": "2.0",
  "result": {
    "success": true,
    "query": "fireball spell",
    "results": [
      {
        "content": "Fireball: 3rd-level evocation spell...",
        "source": "Player's Handbook",
        "page": 241,
        "relevance_score": 0.95
      }
    ],
    "total_results": 1
  },
  "id": 1
}
```

## Database Schema

The system uses ChromaDB for vector storage with the following collections:

- **rulebooks**: Game system rules and mechanics
- **flavor_sources**: Novels and narrative content
- **campaigns**: Campaign data and metadata
- **sessions**: Game session tracking
- **personalities**: System-specific personality profiles

## Performance Considerations

- Vector search operations typically complete in < 100ms
- PDF processing is performed asynchronously
- Results are cached with LRU eviction
- Batch operations are supported for efficiency

## Versioning

API Version: 1.0.0

The API follows semantic versioning:
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes

## Support

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/Raudbjorn/MDMAI/issues
- Documentation: This directory
- Troubleshooting: See [troubleshooting guide](../troubleshooting/README.md)