# Character Generation API Documentation

## Overview

Character generation tools create player characters and NPCs with appropriate statistics, backstories, and personalities.

## Tools

### `generate_character`

Generates a complete player character with stats, equipment, and backstory.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system` | string | Yes | - | Game system (e.g., "D&D 5e") |
| `level` | integer | No | 1 | Character level |
| `class_type` | string | No | random | Character class |
| `race` | string | No | random | Character race |
| `backstory_hints` | string | No | null | Guidance for backstory generation |
| `stat_method` | string | No | "standard" | Method: "standard", "random", "point_buy" |

#### Response

```json
{
  "success": true,
  "character": {
    "id": "char_gen_001",
    "name": "Lyra Silverwind",
    "race": "Elf",
    "class": "Ranger",
    "level": 3,
    "alignment": "Chaotic Good",
    "background": "Outlander",
    "stats": {
      "strength": 13,
      "dexterity": 17,
      "constitution": 14,
      "intelligence": 12,
      "wisdom": 16,
      "charisma": 10
    },
    "hp": {"current": 28, "max": 28},
    "ac": 15,
    "skills": ["Survival", "Nature", "Perception", "Stealth"],
    "equipment": {
      "weapons": ["Longbow", "Two Shortswords"],
      "armor": "Studded Leather",
      "items": ["Explorer's Pack", "Healing Potion x2"]
    },
    "backstory": {
      "summary": "Raised by wolves in the Silverwood Forest after her village was destroyed...",
      "personality_traits": [
        "I'm always picking things up and fidgeting with them",
        "I feel more comfortable around animals than people"
      ],
      "ideals": ["The natural world is more important than civilization"],
      "bonds": ["I will avenge my destroyed village"],
      "flaws": ["I have trouble trusting members of other races"]
    }
  }
}
```

### `generate_npc`

Generates an NPC with role-appropriate stats and personality.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system` | string | Yes | - | Game system |
| `role` | string | Yes | - | NPC role (see roles below) |
| `level` | integer | No | null | Level (scales to party if not specified) |
| `personality_traits` | array | No | [] | Specific personality traits |
| `name` | string | No | generated | NPC name |

#### Available Roles

- **merchant**: Shop owner, trader
- **guard**: City watch, soldier
- **noble**: Aristocrat, courtier
- **commoner**: Farmer, laborer
- **thief**: Pickpocket, burglar
- **priest**: Cleric, acolyte
- **mage**: Wizard, scholar
- **innkeeper**: Tavern owner
- **blacksmith**: Weaponsmith, armorer
- **bard**: Entertainer, storyteller
- **assassin**: Hired killer
- **bandit**: Highway robber
- **cultist**: Dark worshipper
- **scout**: Ranger, tracker

#### Response

```json
{
  "success": true,
  "npc": {
    "id": "npc_gen_001",
    "name": "Marcus Ironforge",
    "role": "blacksmith",
    "race": "Dwarf",
    "stats": {
      "strength": 16,
      "dexterity": 10,
      "constitution": 15,
      "intelligence": 12,
      "wisdom": 11,
      "charisma": 13
    },
    "skills": {
      "smithing": "+8",
      "appraisal": "+5",
      "persuasion": "+4"
    },
    "personality": {
      "traits": ["Perfectionist", "Gruff but fair", "Takes pride in work"],
      "motivation": "Create a legendary weapon",
      "fear": "Losing his forge",
      "quirk": "Hums while working"
    },
    "appearance": {
      "description": "Stocky dwarf with soot-stained beard and powerful arms",
      "distinguishing_features": ["Burn scar on left hand", "Always wears leather apron"]
    },
    "inventory": {
      "weapons": ["Smith's hammer", "Hand axe"],
      "items": ["Various smith tools", "50 gp in materials"],
      "special": ["Masterwork dagger (sample)"]
    },
    "services": [
      {"service": "Weapon repair", "cost": "1-5 gp"},
      {"service": "Armor repair", "cost": "2-10 gp"},
      {"service": "Custom weapon", "cost": "Base price +50%"}
    ]
  }
}
```

### `generate_backstory`

Generates a detailed backstory for an existing character.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `character_data` | object | Yes | - | Character information |
| `depth` | string | No | "medium" | Depth: "brief", "medium", "detailed" |
| `tone` | string | No | system default | Tone: "tragic", "heroic", "mysterious", "comedic" |
| `include_secrets` | boolean | No | false | Include hidden motivations |

#### Response

```json
{
  "success": true,
  "backstory": {
    "origin": "Born in the merchant district of Waterdeep...",
    "childhood": "Showed early aptitude for magic when...",
    "defining_moment": "Everything changed when the dragon attacked...",
    "motivation": "Seeks to prevent others from suffering as they did",
    "relationships": {
      "family": "Parents deceased, sister missing",
      "mentor": "Archmage Elminster (complicated relationship)",
      "rival": "Former classmate turned dark wizard"
    },
    "secrets": [
      "Accidentally caused the fire that killed their parents",
      "Has a forbidden romance with a demon"
    ],
    "future_hooks": [
      "Sister may still be alive",
      "Demon lover will demand a favor",
      "Rival knows the truth about the fire"
    ]
  }
}
```

### `generate_party`

Generates a balanced party of characters.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system` | string | Yes | - | Game system |
| `size` | integer | No | 4 | Party size (3-6) |
| `level` | integer | No | 1 | Starting level |
| `composition` | string | No | "balanced" | Type: "balanced", "combat", "roleplay", "random" |

#### Response

```json
{
  "success": true,
  "party": [
    {
      "name": "Thorin Battlehammer",
      "role": "tank",
      "class": "Fighter",
      "race": "Dwarf"
    },
    {
      "name": "Elara Moonwhisper",
      "role": "damage",
      "class": "Wizard",
      "race": "Elf"
    },
    {
      "name": "Pip Lightfingers",
      "role": "skill",
      "class": "Rogue",
      "race": "Halfling"
    },
    {
      "name": "Brother Marcus",
      "role": "support",
      "class": "Cleric",
      "race": "Human"
    }
  ],
  "party_dynamics": {
    "strengths": ["Good balance of combat and utility", "Strong healing capability"],
    "weaknesses": ["Limited ranged physical damage", "Low charisma overall"],
    "potential_conflicts": ["Rogue and Cleric moral differences"],
    "synergies": ["Fighter can protect squishy Wizard", "Cleric buffs enhance Rogue stealth"]
  }
}
```

## Generation Methods

### Stat Generation

#### Standard Array
Uses the system's default array (e.g., 15, 14, 13, 12, 10, 8 for D&D 5e)

#### Random
Rolls dice according to system rules (e.g., 4d6 drop lowest)

#### Point Buy
Allocates points from a pool with costs per system

### Personality Generation

Personalities are generated using:
1. Role-based templates
2. System personality profiles
3. Random trait combinations
4. Cultural backgrounds

## Integration with Campaigns

Generated characters can be immediately added to campaigns:

```python
# Generate character
character = await generate_character(system="D&D 5e", level=3)

# Add to campaign
await update_campaign_data(
    campaign_id="camp_xyz789",
    data_type="character",
    data=character["character"]
)
```

## Best Practices

1. **Level Appropriateness**: Generate NPCs at appropriate levels for the party
2. **Role Consistency**: Use specific roles for better NPC generation
3. **Backstory Integration**: Provide hints to connect backstories to your campaign
4. **Personality Depth**: Request detailed personalities for important NPCs
5. **Party Balance**: Use balanced composition for new player groups

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `InvalidSystemError` | Unknown game system | Check supported systems |
| `InvalidRoleError` | Unknown NPC role | Use valid role from list |
| `InvalidLevelError` | Level out of range | Use level 1-20 (D&D) |
| `GenerationFailedError` | Random generation failed | Retry or use different method |

## Related Tools

- [`update_campaign_data`](./campaign_management.md#update_campaign_data) - Add to campaign
- [`get_system_personality`](./personality_management.md#get_system_personality) - Get personality profile
