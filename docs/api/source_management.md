# Source Management API Documentation

## Overview

Source management tools handle adding, organizing, and managing PDF sources for the TTRPG Assistant.

## Tools

### `add_source`

Adds a new PDF source to the knowledge base with automatic content extraction and indexing.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pdf_path` | string | Yes | - | Absolute path to the PDF file |
| `rulebook_name` | string | Yes | - | Name of the rulebook/source |
| `system` | string | Yes | - | Game system (e.g., "D&D 5e", "Pathfinder") |
| `source_type` | string | No | "rulebook" | Type: "rulebook", "flavor", "supplement", "adventure" |

#### Response

```json
{
  "success": true,
  "message": "Source added successfully",
  "source_id": "src_abc123",
  "metadata": {
    "pages": 320,
    "chunks_created": 1543,
    "processing_time_seconds": 45.3,
    "content_types_detected": ["rules", "spells", "monsters", "tables"]
  }
}
```

#### Processing Pipeline

1. **Validation**: File existence and format checking
2. **Extraction**: PDF text and structure extraction
3. **Chunking**: Semantic chunking with overlap
4. **Classification**: Content type detection
5. **Embedding**: Vector embedding generation
6. **Storage**: ChromaDB storage with metadata
7. **Indexing**: Search index updates

### `list_sources`

Lists all available sources in the system with optional filtering.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system` | string | No | null | Filter by game system |
| `source_type` | string | No | null | Filter by source type |

#### Response

```json
{
  "success": true,
  "sources": [
    {
      "source_id": "src_phb5e",
      "name": "Player's Handbook",
      "system": "D&D 5e",
      "source_type": "rulebook",
      "metadata": {
        "pages": 320,
        "chunks": 1543,
        "added_date": "2024-01-15T10:30:00Z",
        "file_size_mb": 45.2,
        "content_categories": ["rules", "spells", "classes", "equipment"]
      }
    },
    {
      "source_id": "src_mm5e",
      "name": "Monster Manual",
      "system": "D&D 5e",
      "source_type": "rulebook",
      "metadata": {
        "pages": 352,
        "chunks": 2104,
        "added_date": "2024-01-15T11:00:00Z",
        "file_size_mb": 52.8,
        "content_categories": ["monsters", "lore"]
      }
    }
  ],
  "total_sources": 2
}
```

### `remove_source`

Removes a source from the system (soft delete with option for hard delete).

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Source identifier |
| `hard_delete` | boolean | No | false | Permanently delete (true) or soft delete (false) |

#### Response

```json
{
  "success": true,
  "message": "Source removed successfully",
  "source_id": "src_abc123",
  "action": "soft_delete"
}
```

### `update_source`

Updates source metadata or reprocesses content.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | Yes | - | Source identifier |
| `metadata` | object | No | {} | Metadata to update |
| `reprocess` | boolean | No | false | Reprocess the PDF content |

#### Response

```json
{
  "success": true,
  "message": "Source updated successfully",
  "source_id": "src_abc123",
  "changes": {
    "metadata": ["system", "tags"],
    "reprocessed": false
  }
}
```

## Source Types

### Rulebook
Core game rules and mechanics. Highest priority for rule searches.

### Flavor
Novels, stories, and narrative content. Used for atmosphere and backstory generation.

### Supplement
Additional rules and options that extend the base game.

### Adventure
Pre-written adventures and campaigns with maps, encounters, and storylines.

## Content Categories

Sources are automatically categorized into:

- **Rules**: Core game mechanics
- **Spells**: Spell descriptions and mechanics
- **Monsters**: Creature statistics and lore
- **Items**: Equipment, magic items, artifacts
- **Classes**: Character classes and features
- **Races**: Playable races and ancestries
- **Lore**: World building and history
- **Tables**: Random tables and generators
- **Maps**: Geographic and dungeon maps
- **NPCs**: Non-player character templates

## Duplicate Detection

The system detects duplicates using:
1. File hash comparison
2. Title similarity matching
3. Content fingerprinting

## Processing Statistics

Typical processing times:
- Small PDF (< 100 pages): 10-20 seconds
- Medium PDF (100-300 pages): 30-60 seconds
- Large PDF (> 300 pages): 60-120 seconds

## Best Practices

1. **File Naming**: Use descriptive filenames for easier management
2. **System Consistency**: Use consistent system names (e.g., always "D&D 5e")
3. **Source Types**: Correctly classify sources for optimal search results
4. **Regular Updates**: Reprocess sources after major system updates
5. **Backup**: Keep original PDFs as the system stores only processed data

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | PDF path doesn't exist | Verify file path |
| `InvalidPDFError` | Corrupted or encrypted PDF | Use a valid, unencrypted PDF |
| `DuplicateSourceError` | Source already exists | Use update_source or remove first |
| `ProcessingError` | PDF parsing failed | Check PDF format and try again |

## Related Tools

- [`search`](./search_tools.md#search) - Search across sources
- [`get_system_personality`](./personality_management.md#get_system_personality) - Get personality from sources
