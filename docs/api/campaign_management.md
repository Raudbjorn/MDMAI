# Campaign Management API Documentation

## Overview

Campaign management tools provide comprehensive functionality for creating, managing, and organizing TTRPG campaigns.

## Tools

### `create_campaign`

Creates a new campaign with associated metadata.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Campaign name |
| `system` | string | Yes | - | Game system (e.g., "D&D 5e") |
| `description` | string | No | null | Campaign description |

#### Response

```json
{
  "success": true,
  "message": "Campaign created successfully",
  "campaign_id": "camp_xyz789",
  "data": {
    "name": "Curse of Strahd",
    "system": "D&D 5e",
    "description": "A gothic horror campaign in Barovia",
    "created_at": "2024-01-20T14:30:00Z",
    "updated_at": "2024-01-20T14:30:00Z"
  }
}
```

### `get_campaign_data`

Retrieves campaign-specific data with optional filtering.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `campaign_id` | string | Yes | - | Campaign identifier |
| `data_type` | string | No | null | Filter: "characters", "npcs", "locations", "plot_points" |

#### Response

```json
{
  "success": true,
  "campaign_id": "camp_xyz789",
  "data": {
    "characters": [
      {
        "id": "char_001",
        "name": "Elara Moonwhisper",
        "class": "Wizard",
        "level": 5,
        "player": "Alice",
        "status": "active"
      }
    ],
    "npcs": [
      {
        "id": "npc_001",
        "name": "Strahd von Zarovich",
        "role": "Main Antagonist",
        "location": "Castle Ravenloft",
        "description": "The vampire lord of Barovia"
      }
    ],
    "locations": [
      {
        "id": "loc_001",
        "name": "Village of Barovia",
        "type": "settlement",
        "description": "A gloomy village under Strahd's shadow",
        "notable_features": ["Blood of the Vine Tavern", "Church", "Burgomaster's Mansion"]
      }
    ],
    "plot_points": [
      {
        "id": "plot_001",
        "title": "The Tarokka Reading",
        "status": "completed",
        "description": "Madam Eva's fortune telling reveals the location of key items",
        "session_id": "sess_003"
      }
    ]
  }
}
```

### `update_campaign_data`

Updates campaign information including characters, NPCs, locations, and plot points.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `campaign_id` | string | Yes | - | Campaign identifier |
| `data_type` | string | Yes | - | Type: "character", "npc", "location", "plot_point" |
| `data` | object | Yes | - | Data to add or update |

#### Response

```json
{
  "success": true,
  "message": "Campaign data updated successfully",
  "campaign_id": "camp_xyz789",
  "data_type": "character",
  "entity_id": "char_002",
  "version": 3
}
```

### `delete_campaign`

Archives or permanently deletes a campaign.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `campaign_id` | string | Yes | - | Campaign identifier |
| `hard_delete` | boolean | No | false | Permanent deletion if true |

#### Response

```json
{
  "success": true,
  "message": "Campaign archived successfully",
  "campaign_id": "camp_xyz789",
  "action": "archived",
  "archived_at": "2024-01-25T10:00:00Z"
}
```

### `link_to_rulebook`

Creates links between campaign elements and rulebook content.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `campaign_id` | string | Yes | - | Campaign identifier |
| `entity_type` | string | Yes | - | Type: "character", "npc", "item", "spell" |
| `entity_id` | string | Yes | - | Entity identifier |
| `rulebook_reference` | string | Yes | - | Reference to rulebook content |

#### Response

```json
{
  "success": true,
  "message": "Link created successfully",
  "link_id": "link_001",
  "entity": "char_001",
  "reference": "PHB p.112 (Wizard Class)"
}
```

## Data Models

### Campaign
```python
{
  "id": "camp_xyz789",
  "name": "Campaign Name",
  "system": "D&D 5e",
  "description": "Campaign description",
  "created_at": "2024-01-20T14:30:00Z",
  "updated_at": "2024-01-25T16:45:00Z",
  "status": "active",
  "metadata": {
    "sessions_count": 12,
    "players_count": 4,
    "current_level": 5
  }
}
```

### Character
```python
{
  "id": "char_001",
  "campaign_id": "camp_xyz789",
  "name": "Character Name",
  "player": "Player Name",
  "class": "Fighter",
  "level": 5,
  "race": "Human",
  "alignment": "Lawful Good",
  "stats": {
    "str": 16, "dex": 14, "con": 15,
    "int": 10, "wis": 12, "cha": 13
  },
  "hp": {"current": 42, "max": 44},
  "equipment": [],
  "notes": []
}
```

### NPC
```python
{
  "id": "npc_001",
  "campaign_id": "camp_xyz789",
  "name": "NPC Name",
  "role": "merchant",
  "location": "Town Square",
  "description": "A friendly merchant",
  "personality_traits": ["helpful", "gossip"],
  "relationships": {},
  "quest_giver": false
}
```

### Location
```python
{
  "id": "loc_001",
  "campaign_id": "camp_xyz789",
  "name": "Location Name",
  "type": "dungeon",
  "description": "A dangerous dungeon",
  "parent_location": null,
  "notable_features": [],
  "npcs": [],
  "encounters": []
}
```

### Plot Point
```python
{
  "id": "plot_001",
  "campaign_id": "camp_xyz789",
  "title": "The Mystery Deepens",
  "description": "Investigation reveals...",
  "status": "active",
  "session_id": "sess_005",
  "related_npcs": ["npc_002", "npc_003"],
  "related_locations": ["loc_002"],
  "notes": []
}
```

## Versioning

All campaign data supports versioning:
- Automatic version creation on updates
- Rollback to any previous version
- Version comparison and diff viewing
- Maximum 50 versions retained by default

## Best Practices

1. **Regular Updates**: Update campaign data after each session
2. **Descriptive Names**: Use clear, memorable names for entities
3. **Link References**: Connect campaign elements to rules for quick lookup
4. **Status Tracking**: Keep character and plot status current
5. **Notes**: Add session notes for continuity

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `CampaignNotFoundError` | Invalid campaign_id | Verify campaign exists |
| `InvalidDataTypeError` | Unknown data_type | Use valid type |
| `VersionConflictError` | Concurrent updates | Retry with latest version |
| `ValidationError` | Invalid data format | Check data structure |

## Related Tools

- [`start_session`](./session_management.md#start_session) - Start a game session
- [`generate_npc`](./character_generation.md#generate_npc) - Generate NPCs