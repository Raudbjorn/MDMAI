# Campaign Management Guide

This guide covers everything you need to know about managing campaigns in the TTRPG Assistant.

## Understanding Campaigns

A campaign is a container for all the data related to your ongoing TTRPG story:
- **Characters**: Player characters with full stats
- **NPCs**: Non-player characters the party encounters
- **Locations**: Places in your world
- **Plot Points**: Story developments and quests
- **Sessions**: Individual game meetings

## Creating a Campaign

### Basic Campaign Creation

```python
campaign = await create_campaign(
    name="Waterdeep: Dragon Heist",
    system="D&D 5e",
    description="Urban adventure in the City of Splendors"
)
```

### Campaign with Custom Settings

```python
campaign = await create_campaign(
    name="Homebrew World",
    system="D&D 5e",
    description="A world where magic is dying",
    metadata={
        "setting": "low_magic",
        "starting_level": 3,
        "tone": "dark_fantasy",
        "house_rules": ["gritty_realism", "slow_healing"]
    }
)
```

## Managing Campaign Data

### Adding Player Characters

#### Option 1: Generate New Character

```python
# Generate a character
character = await generate_character(
    system="D&D 5e",
    level=3,
    class_type="Paladin",
    race="Dragonborn",
    backstory_hints="Former city guard seeking redemption"
)

# Add to campaign
await update_campaign_data(
    campaign_id=campaign_id,
    data_type="character",
    data=character["character"]
)
```

#### Option 2: Add Existing Character

```python
character_data = {
    "name": "Aramil Galanodel",
    "player": "John",
    "class": "Wizard",
    "level": 5,
    "race": "High Elf",
    "stats": {
        "strength": 8,
        "dexterity": 14,
        "constitution": 13,
        "intelligence": 18,
        "wisdom": 12,
        "charisma": 10
    },
    "hp": {"current": 32, "max": 32},
    "equipment": ["Spellbook", "Component pouch", "Scholar's pack"]
}

await update_campaign_data(
    campaign_id=campaign_id,
    data_type="character",
    data=character_data
)
```

### Managing NPCs

#### Creating Important NPCs

```python
# Generate a villain
villain = await generate_npc(
    system="D&D 5e",
    role="noble",
    name="Lord Neverember",
    personality_traits=["ambitious", "manipulative", "charming"]
)

# Add with additional details
villain_data = villain["npc"]
villain_data.update({
    "role": "Main Antagonist",
    "secrets": ["Actually a dragon in disguise"],
    "goals": ["Control the city's trade"],
    "relationships": {
        "Lady Silverhand": "rival",
        "Captain Stormwind": "unwitting pawn"
    }
})

await update_campaign_data(
    campaign_id=campaign_id,
    data_type="npc",
    data=villain_data
)
```

#### Batch NPC Creation

```python
# Create a tavern's worth of NPCs
tavern_npcs = []

for role in ["innkeeper", "bard", "merchant", "guard", "commoner"]:
    npc = await generate_npc(system="D&D 5e", role=role)
    tavern_npcs.append(npc["npc"])

# Add all to campaign
for npc in tavern_npcs:
    await update_campaign_data(
        campaign_id=campaign_id,
        data_type="npc",
        data=npc
    )
```

### Managing Locations

#### Adding Detailed Locations

```python
location = {
    "name": "The Yawning Portal",
    "type": "tavern",
    "description": "Famous tavern with entrance to Undermountain",
    "parent_location": "Castle Ward",
    "notable_features": [
        "40-foot wide well to dungeon",
        "Adventurer's board",
        "Private meeting rooms"
    ],
    "npcs": ["Durnan", "Bonnie", "Threestrings"],
    "services": [
        {"service": "Room", "cost": "2 gp/night"},
        {"service": "Meal", "cost": "4 sp"},
        {"service": "Dungeon entry", "cost": "1 gp"}
    ],
    "secrets": ["Hidden smuggling tunnel in cellar"]
}

await update_campaign_data(
    campaign_id=campaign_id,
    data_type="location",
    data=location
)
```

#### Creating Location Hierarchies

```python
# Create nested locations
city = {
    "name": "Waterdeep",
    "type": "city",
    "description": "The City of Splendors"
}

ward = {
    "name": "Castle Ward",
    "type": "district",
    "parent_location": "Waterdeep",
    "description": "The heart of the city"
}

tavern = {
    "name": "The Yawning Portal",
    "type": "tavern",
    "parent_location": "Castle Ward"
}

# Add in order
for location in [city, ward, tavern]:
    await update_campaign_data(
        campaign_id=campaign_id,
        data_type="location",
        data=location
    )
```

### Managing Plot Points

#### Creating Quest Lines

```python
main_quest = {
    "title": "The Stone of Golorr",
    "description": "Find the legendary artifact before the villains",
    "status": "active",
    "priority": "high",
    "chapters": [
        {
            "title": "A Friend in Need",
            "status": "completed",
            "summary": "Rescued Renaer from Xanathar Guild"
        },
        {
            "title": "Trollskull Manor",
            "status": "active",
            "summary": "Investigating the haunted tavern"
        },
        {
            "title": "The Nimblewright",
            "status": "pending",
            "summary": "Track down the mechanical assassin"
        }
    ],
    "rewards": ["500pp", "Stone of Golorr", "Noble title"],
    "npcs_involved": ["Renaer", "Volo", "Laeral Silverhand"]
}

await update_campaign_data(
    campaign_id=campaign_id,
    data_type="plot_point",
    data=main_quest
)
```

#### Tracking Side Quests

```python
side_quest = {
    "title": "The Haunting of Trollskull",
    "description": "Cleanse the manor of its ghostly inhabitant",
    "status": "active",
    "priority": "medium",
    "clues": [
        "Ghost appears at midnight",
        "Died in the explosion",
        "Seeking justice for murder"
    ],
    "resolution": "Help ghost find peace by catching killer",
    "reward": "Ghost becomes friendly, provides information"
}

await update_campaign_data(
    campaign_id=campaign_id,
    data_type="plot_point",
    data=side_quest
)
```

## Retrieving Campaign Data

### Get Everything

```python
all_data = await get_campaign_data(campaign_id=campaign_id)
```

### Get Specific Data Types

```python
# Just characters
characters = await get_campaign_data(
    campaign_id=campaign_id,
    data_type="characters"
)

# Just active plot points
plots = await get_campaign_data(
    campaign_id=campaign_id,
    data_type="plot_points"
)
active_plots = [p for p in plots if p["status"] == "active"]
```

## Campaign Organization

### Using Tags and Categories

```python
# Tag NPCs by faction
npc_data["tags"] = ["Zhentarim", "Merchant", "Information Broker"]

# Categorize locations
location_data["category"] = "Safe Haven"
location_data["danger_level"] = "Low"

# Priority levels for plots
plot_data["urgency"] = "Time-sensitive"
plot_data["deadline"] = "3 days game time"
```

### Creating Relationships

```python
# Link characters to NPCs
character_data["relationships"] = {
    "Durnan": "mentor",
    "Renaer": "friend",
    "Xanathar": "enemy"
}

# Connect locations
location_data["connections"] = {
    "Castle Ward": "10 minute walk",
    "Dock Ward": "20 minute walk",
    "Undermountain": "Direct access"
}
```

## Version Control

### Automatic Versioning

Every update creates a version automatically:

```python
# Make a change
result = await update_campaign_data(
    campaign_id=campaign_id,
    data_type="character",
    data=updated_character
)

print(f"Created version {result['version']}")
```

### Rolling Back Changes

```python
# View version history
history = await get_campaign_versions(campaign_id=campaign_id)

# Rollback to specific version
await rollback_campaign(
    campaign_id=campaign_id,
    version=5
)
```

### Comparing Versions

```python
diff = await compare_campaign_versions(
    campaign_id=campaign_id,
    version1=5,
    version2=8
)

print(f"Changes: {diff['changes']}")
```

## Campaign Templates

### Creating Reusable Templates

```python
# Save campaign as template
template = await export_campaign_template(
    campaign_id=campaign_id,
    name="Urban Mystery Template",
    exclude=["characters"]  # Don't include PCs
)

# Create new campaign from template
new_campaign = await create_campaign_from_template(
    template_id=template["template_id"],
    name="Baldur's Gate Mystery",
    system="D&D 5e"
)
```

## Best Practices

### 1. Regular Updates
- Update after every session
- Track character progression
- Note important NPC interactions

### 2. Detailed NPCs
- Give NPCs motivations and goals
- Track relationships between NPCs
- Note secrets and hidden information

### 3. Location Details
- Include sensory descriptions
- Note available services
- Track which PCs have visited

### 4. Plot Management
- Break large quests into chapters
- Track multiple plot threads
- Note player decisions and consequences

### 5. Use Version Control
- Review changes before sessions
- Rollback if needed
- Keep history for reference

## Advanced Features

### Campaign Analytics

```python
stats = await get_campaign_statistics(campaign_id=campaign_id)
print(f"Sessions run: {stats['session_count']}")
print(f"Total NPCs: {stats['npc_count']}")
print(f"Locations visited: {stats['visited_locations']}")
print(f"Active quests: {stats['active_quests']}")
```

### Cross-Campaign References

```python
# Reference NPCs from other campaigns
await import_npc_from_campaign(
    source_campaign_id=old_campaign_id,
    target_campaign_id=campaign_id,
    npc_id="npc_memorable_villain"
)
```

### Campaign Archival

```python
# Archive completed campaign
await archive_campaign(
    campaign_id=campaign_id,
    summary="Epic 2-year campaign, levels 1-15"
)

# Export for backup
backup = await export_campaign(
    campaign_id=campaign_id,
    format="json",
    include_versions=True
)
```

## Troubleshooting

### Common Issues

**Issue**: Can't find campaign data
**Solution**: Check campaign_id is correct, ensure data was saved

**Issue**: Updates not saving
**Solution**: Check for validation errors, ensure proper data format

**Issue**: Version conflicts
**Solution**: Fetch latest data before updating, use version numbers

## Next Steps

<!-- Note: The following guides are coming soon:
- [Running Sessions](./sessions.md) - Learn session management
- [Character Creation](./characters.md) - Master character generation
- [Search Guide](./search.md) - Find information quickly
-->
