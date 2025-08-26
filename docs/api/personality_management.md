# Personality Management API Documentation

## Overview

Personality management tools control how the system adapts its responses to match different game systems' tones and styles.

## Tools

### `get_system_personality`

Retrieves the personality profile for a specific game system.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system` | string | Yes | - | Game system (e.g., "D&D 5e") |

#### Response

```json
{
  "success": true,
  "personality": {
    "system": "D&D 5e",
    "name": "Wise Sage",
    "tone": "authoritative",
    "perspective": "omniscient",
    "style_descriptors": [
      "academic",
      "mystical",
      "formal",
      "knowledgeable"
    ],
    "common_phrases": [
      "According to ancient lore...",
      "The arcane arts dictate...",
      "In the annals of history...",
      "As written in the sacred texts..."
    ],
    "vocabulary": {
      "arcane": 0.85,
      "mystical": 0.72,
      "ancient": 0.68,
      "legendary": 0.61,
      "prophetic": 0.55
    },
    "response_templates": {
      "greeting": "Greetings, adventurer. How may this humble sage assist you?",
      "confirmation": "Indeed, your understanding is correct.",
      "correction": "Ah, a common misconception. Allow me to clarify...",
      "uncertainty": "The ancient texts are unclear on this matter..."
    }
  }
}
```

### `set_active_personality`

Sets the active personality for system responses.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system` | string | Yes | - | Game system to activate |

#### Response

```json
{
  "success": true,
  "message": "Personality activated",
  "active_personality": "D&D 5e - Wise Sage",
  "previous_personality": "Neutral"
}
```

### `create_custom_personality`

Creates a custom personality profile for a campaign or system.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Personality name |
| `base_system` | string | No | null | Base system to inherit from |
| `tone` | string | Yes | - | Overall tone |
| `style_descriptors` | array | Yes | - | Style descriptors |
| `custom_phrases` | array | No | [] | Custom phrases |

#### Response

```json
{
  "success": true,
  "message": "Custom personality created",
  "personality_id": "pers_custom_001",
  "name": "Sarcastic Narrator"
}
```

### `apply_personality`

Applies a personality to a given text response.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Original text |
| `personality_id` | string | Yes | - | Personality to apply |
| `intensity` | float | No | 1.0 | Intensity (0.0-1.0) |

#### Response

```json
{
  "success": true,
  "original_text": "The spell deals 8d6 fire damage.",
  "styled_text": "As the ancient incantations speak, this formidable evocation shall unleash the fury of eight dice worth of searing flame upon thy foes.",
  "personality_applied": "D&D 5e - Wise Sage"
}
```

## Personality Profiles

### Default System Personalities

#### D&D 5e - "Wise Sage"
- **Tone**: Authoritative, omniscient
- **Style**: Academic with magical terminology
- **Usage**: Default for D&D 5th Edition content

#### Pathfinder - "Chronicler"
- **Tone**: Detailed, encyclopedic
- **Style**: Precise, rule-focused
- **Usage**: Pathfinder rules and lore

#### Call of Cthulhu - "Antiquarian Scholar"
- **Tone**: Ominous, scholarly
- **Style**: Victorian academic with eldritch undertones
- **Usage**: Cosmic horror investigations

#### Blades in the Dark - "Shadowy Informant"
- **Tone**: Mysterious, conspiratorial
- **Style**: Criminal underworld slang
- **Usage**: Heist planning and underworld dealings

#### Delta Green - "Classified Handler"
- **Tone**: Formal, authoritative
- **Style**: Government/military briefing
- **Usage**: Modern conspiracy and horror

#### Cyberpunk - "Net Runner"
- **Tone**: Tech-savvy, cynical
- **Style**: Futuristic slang and technical jargon
- **Usage**: Cyberpunk settings

## Personality Components

### Tone Categories

- **Authoritative**: Command and expertise
- **Mysterious**: Enigmatic and secretive
- **Scholarly**: Academic and analytical
- **Casual**: Relaxed and approachable
- **Dramatic**: Theatrical and intense
- **Cynical**: Skeptical and world-weary

### Style Descriptors

- **Formal**: Proper grammar, complex sentences
- **Casual**: Conversational, contractions
- **Technical**: Jargon-heavy, precise
- **Poetic**: Metaphorical, flowery
- **Terse**: Short, direct statements
- **Verbose**: Elaborate, detailed

### Perspective Types

- **Omniscient**: All-knowing narrator
- **Limited**: Specific viewpoint
- **First-person**: Personal experience
- **Instructional**: Teaching approach
- **Conversational**: Dialogue style

## Personality Extraction

When processing source materials, the system extracts:

1. **Writing Style**: Sentence structure, complexity
2. **Vocabulary**: Common terms and frequencies
3. **Tone Markers**: Emotional indicators
4. **Cultural Elements**: Setting-specific references
5. **Speech Patterns**: Dialogue characteristics

## Custom Personality Creation

### Example: Creating a Pirate Campaign Personality

```python
personality = await create_custom_personality(
    name="Salty Sea Dog",
    base_system="D&D 5e",
    tone="boisterous",
    style_descriptors=["nautical", "crude", "adventurous"],
    custom_phrases=[
        "Arr, me hearty!",
        "By Davy Jones' locker!",
        "Avast ye landlubber!"
    ]
)
```

## Intensity Levels

Personality application can be adjusted:

- **0.0**: No personality (neutral)
- **0.25**: Subtle hints
- **0.5**: Moderate application
- **0.75**: Strong personality
- **1.0**: Full immersion

## Best Practices

1. **Consistency**: Use the same personality throughout a session
2. **Appropriateness**: Match personality to content type
3. **Gradual Introduction**: Start with lower intensity for new players
4. **Custom Personalities**: Create unique personalities for special campaigns
5. **Player Preference**: Adjust based on group preferences

## Integration Examples

### With Search Results

```python
# Search with personality
results = await search(query="fireball spell")
personality = await get_system_personality("D&D 5e")
styled_results = await apply_personality(
    text=results["content"],
    personality_id=personality["id"]
)
```

### With NPC Generation

```python
# Generate NPC with specific personality
npc = await generate_npc(
    system="Blades in the Dark",
    role="informant"
)
# NPC automatically uses "Shadowy Informant" personality
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `PersonalityNotFoundError` | Unknown personality | Check available personalities |
| `InvalidIntensityError` | Intensity out of range | Use 0.0-1.0 |
| `SystemNotSupportedError` | No personality for system | Create custom personality |
| `ExtractionFailedError` | Could not extract from source | Manually define personality |

## Related Tools

- [`add_source`](./source_management.md#add_source) - Adds sources for personality extraction
- [`generate_npc`](./character_generation.md#generate_npc) - Uses personality for NPCs
