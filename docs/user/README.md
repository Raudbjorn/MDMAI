# TTRPG Assistant User Guide

Welcome to the TTRPG Assistant MCP Server! This guide will help you get started and make the most of the system's features.

## Table of Contents

1. [Getting Started](./getting_started.md)
2. [Managing Campaigns](./campaigns.md)
3. [Running Sessions](./sessions.md)
4. [Character Creation](./characters.md)
5. [Using Search](./search.md)
6. [Managing Sources](./sources.md)
7. [Tips and Tricks](./tips.md)

## What is TTRPG Assistant?

The TTRPG Assistant is a comprehensive tool designed to help Dungeon Masters and Game Runners manage their tabletop role-playing games more efficiently. It provides:

- **Quick Rule Lookups**: Search across all your rulebooks instantly
- **Campaign Management**: Track characters, NPCs, locations, and plot points
- **Session Tracking**: Manage initiative, combat, and session notes
- **Character Generation**: Create PCs and NPCs with full stats and backstories
- **Smart Search**: Find information using natural language queries
- **Personality System**: Responses tailored to your game system's style

## Quick Start

### 1. Start the Server

```bash
python src/main.py
```

### 2. Add Your First Rulebook

Use the Model Context Protocol client to add a PDF:

```python
await add_source(
    pdf_path="/path/to/rulebook.pdf",
    rulebook_name="Player's Handbook",
    system="D&D 5e",
    source_type="rulebook"
)
```

### 3. Create a Campaign

```python
await create_campaign(
    name="My First Campaign",
    system="D&D 5e",
    description="An epic adventure begins..."
)
```

### 4. Start Searching

```python
results = await search(
    query="How does advantage work?",
    max_results=3
)
```

## Core Concepts

### Sources

Sources are PDF documents containing game rules, lore, or narrative content. They're categorized as:
- **Rulebooks**: Core game mechanics and rules
- **Flavor**: Novels and narrative materials
- **Supplements**: Additional rules and options
- **Adventures**: Pre-written campaigns

### Campaigns

Campaigns are containers for all your game data:
- Player characters
- NPCs
- Locations
- Plot points
- Session history

### Sessions

Sessions track individual game meetings:
- Initiative order
- Monster HP
- Combat rounds
- Session notes
- Important events

### Personalities

Each game system has a unique personality that affects how the assistant responds:
- D&D 5e speaks like a wise sage
- Call of Cthulhu uses ominous scholarly tones
- Blades in the Dark employs criminal slang

## Common Workflows

### Preparing for a Session

1. Review campaign data
2. Generate any needed NPCs
3. Look up relevant rules
4. Set up encounter monsters
5. Start the session tracker

### During Play

1. Quick rule lookups
2. Track initiative
3. Update monster HP
4. Add session notes
5. Generate NPCs on the fly

### After the Session

1. End session with summary
2. Update campaign data
3. Add plot developments
4. Plan next session

## Feature Highlights

### Intelligent Search

The search system understands context and intent:
- "fireball damage" finds spell details
- "AC calculation" explains armor class
- "dragon stats" returns monster entries

### Automatic Personality

Responses automatically match your game's tone:
- Formal and mystical for D&D
- Technical and military for Delta Green
- Streetwise and gritty for Cyberpunk

### Version Control

All campaign data is versioned:
- Automatic backups on changes
- Rollback to any previous state
- Track change history

### Performance Optimization

- Search results in < 100ms
- Cached frequent queries
- Parallel processing for PDFs
- Efficient database indexing

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space per 10 PDFs
- Local storage for ChromaDB

## Getting Help

- Check the [Troubleshooting Guide](../troubleshooting/README.md)
- Review the [API Documentation](../api/README.md)
- Submit issues on [GitHub](https://github.com/Raudbjorn/MDMAI/issues)

## Privacy and Security

- All data stored locally
- No internet connection required
- No telemetry or tracking
- Your campaigns remain private

## Next Steps

Ready to dive deeper? Check out:
1. [Getting Started Guide](./getting_started.md) for detailed setup
2. [Campaign Management](./campaigns.md) for organizing your game
3. [Search Guide](./search.md) for mastering queries

Happy gaming!