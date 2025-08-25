# Session Management API Documentation

## Overview

Session management tools help Game Masters track and manage active game sessions including initiative, combat, and notes.

## Tools

### `start_session`

Starts a new game session for a campaign.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `campaign_id` | string | Yes | - | Campaign identifier |
| `session_name` | string | No | "Session [date]" | Custom session name |

#### Response

```json
{
  "success": true,
  "message": "Session started successfully",
  "session_id": "sess_abc123",
  "data": {
    "campaign_id": "camp_xyz789",
    "name": "Session 12: Into the Underdark",
    "status": "active",
    "started_at": "2024-01-25T18:00:00Z",
    "initiative_order": [],
    "monsters": [],
    "notes": []
  }
}
```

### `add_session_note`

Adds a note to the current session for tracking important events.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |
| `note` | string | Yes | - | Note content |
| `category` | string | No | "general" | Note category: "general", "combat", "roleplay", "loot" |

#### Response

```json
{
  "success": true,
  "message": "Note added successfully",
  "note_id": "note_001",
  "timestamp": "2024-01-25T18:15:00Z"
}
```

### `set_initiative`

Sets the initiative order for combat encounters.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |
| `initiative_order` | array | Yes | - | Array of initiative entries |

#### Initiative Entry Structure

```json
{
  "name": "Elara",
  "initiative": 18,
  "type": "player",
  "hp": {"current": 35, "max": 35},
  "conditions": [],
  "notes": ""
}
```

#### Response

```json
{
  "success": true,
  "message": "Initiative order set",
  "current_turn": "Elara",
  "round": 1,
  "order": [
    {"name": "Goblin 1", "initiative": 20, "type": "monster"},
    {"name": "Elara", "initiative": 18, "type": "player"},
    {"name": "Thorin", "initiative": 15, "type": "player"},
    {"name": "Goblin 2", "initiative": 12, "type": "monster"}
  ]
}
```

### `next_turn`

Advances to the next turn in initiative order.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |

#### Response

```json
{
  "success": true,
  "current_turn": "Thorin",
  "previous_turn": "Elara",
  "round": 1,
  "turn_number": 3
}
```

### `add_monster`

Adds a monster to the current session.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |
| `name` | string | Yes | - | Monster name |
| `hp` | integer | Yes | - | Hit points |
| `ac` | integer | No | 10 | Armor class |
| `initiative` | integer | No | null | Initiative roll |

#### Response

```json
{
  "success": true,
  "message": "Monster added successfully",
  "monster_id": "mon_001",
  "data": {
    "name": "Ancient Red Dragon",
    "hp": {"current": 546, "max": 546},
    "ac": 22,
    "status": "healthy",
    "conditions": []
  }
}
```

### `update_monster_hp`

Updates a monster's hit points during combat.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |
| `monster_id` | string | Yes | - | Monster identifier |
| `new_hp` | integer | Yes | - | New HP value |

#### Response

```json
{
  "success": true,
  "message": "Monster HP updated",
  "monster_id": "mon_001",
  "hp": {"current": 423, "max": 546},
  "status": "injured",
  "percentage": 77.5
}
```

### `end_session`

Ends the current session and generates a summary.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |
| `summary` | string | No | null | Optional session summary |

#### Response

```json
{
  "success": true,
  "message": "Session ended successfully",
  "session_id": "sess_abc123",
  "summary": {
    "duration_hours": 3.5,
    "combats": 2,
    "monsters_defeated": 8,
    "notes_count": 15,
    "xp_awarded": 2400,
    "highlights": [
      "Defeated the goblin ambush",
      "Discovered the hidden entrance",
      "Rescued the merchant"
    ]
  }
}
```

### `get_session_data`

Retrieves complete session information.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | Yes | - | Session identifier |

#### Response

```json
{
  "success": true,
  "session": {
    "id": "sess_abc123",
    "campaign_id": "camp_xyz789",
    "name": "Session 12",
    "status": "active",
    "started_at": "2024-01-25T18:00:00Z",
    "round": 3,
    "turn": 2,
    "initiative_order": [...],
    "monsters": [...],
    "notes": [...],
    "combat_log": [...]
  }
}
```

## Session Status

Sessions can have the following statuses:

- **planned**: Scheduled but not started
- **active**: Currently in progress
- **completed**: Finished normally
- **archived**: Older completed session

## Monster Status

Monsters are automatically assigned status based on HP:

- **healthy**: 100% HP
- **injured**: 75-99% HP
- **bloodied**: 25-74% HP
- **critical**: 1-24% HP
- **unconscious**: 0 HP
- **dead**: Below 0 HP

## Combat Tracking

The system tracks:
- Initiative order with automatic sorting
- Round and turn counters
- HP changes with history
- Status conditions
- Combat duration
- Action economy

## Best Practices

1. **Session Names**: Use descriptive names for easy reference
2. **Regular Notes**: Add notes during play for better summaries
3. **Initiative Setup**: Set initiative before combat starts
4. **HP Tracking**: Update HP immediately after damage/healing
5. **Session Summaries**: End sessions properly for complete records

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `SessionNotFoundError` | Invalid session_id | Verify session exists |
| `SessionNotActiveError` | Session already ended | Start new session |
| `MonsterNotFoundError` | Invalid monster_id | Check monster list |
| `InvalidInitiativeError` | Malformed initiative data | Check data format |

## Related Tools

- [`create_campaign`](./campaign_management.md#create_campaign) - Create a campaign
- [`generate_npc`](./character_generation.md#generate_npc) - Generate NPCs for encounters