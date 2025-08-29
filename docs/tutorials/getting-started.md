# TTRPG Assistant - Complete Getting Started Guide

## Table of Contents

1. [Welcome & Overview](#welcome--overview)
2. [Installation Methods](#installation-methods)
3. [Quick Start Guide](#quick-start-guide)
4. [First-Time Setup](#first-time-setup)
5. [Adding Your First Sources](#adding-your-first-sources)
6. [Setting Up a Campaign](#setting-up-a-campaign)
7. [Running Your First Session](#running-your-first-session)
8. [Using the Web Interface](#using-the-web-interface)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Next Steps](#next-steps)

## Welcome & Overview

Welcome to the TTRPG Assistant! This comprehensive guide will help you get started with a powerful AI-powered assistant designed specifically for tabletop role-playing games. Whether you're a Dungeon Master running D&D 5e, a Keeper managing Call of Cthulhu, or a GM for any other system, this tool will help you quickly find rules, generate content, and manage your campaigns.

### What You'll Learn

By the end of this guide, you'll be able to:
- Install and configure the TTRPG Assistant
- Import your PDF rulebooks and sources
- Create and manage campaigns
- Search for rules and content instantly
- Generate NPCs and characters
- Run game sessions with real-time tracking
- Use advanced features like collaborative tools

### What You'll Need

**System Requirements:**
- Computer with at least 4GB RAM (8GB recommended)
- 10GB free disk space (more for large PDF collections)
- Internet connection for initial setup
- PDF files of your TTRPG rulebooks

**Optional but Recommended:**
- GPU for faster processing (NVIDIA preferred)
- SSD storage for better performance
- Multiple monitors for DM screens

## Installation Methods

Choose the installation method that best suits your needs:

### Method 1: Docker (Recommended for Most Users)

Docker provides the easiest setup with all dependencies included.

**Prerequisites:**
- Docker and Docker Compose installed

**Installation:**

```bash
# 1. Clone the repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# 2. Start the application
docker-compose up -d

# 3. Check if it's running
docker-compose ps
```

**Verification:**
Open your browser to `http://localhost:8000/health` - you should see a health status page.

### Method 2: Manual Installation (For Developers)

For those who want full control or plan to contribute to development.

**Prerequisites:**
- Python 3.11 or higher
- Git
- pip package manager

**Installation:**

```bash
# 1. Clone the repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run initial setup
python deploy/scripts/setup_environment.py

# 5. Start the application
python src/main.py
```

### Method 3: Quick Setup Script (Linux/macOS)

For a completely automated installation:

```bash
# Download and run the quick setup script
curl -sSL https://raw.githubusercontent.com/Raudbjorn/MDMAI/main/quick_setup.sh | bash

# Or if you prefer to inspect first:
wget https://raw.githubusercontent.com/Raudbjorn/MDMAI/main/quick_setup.sh
chmod +x quick_setup.sh
./quick_setup.sh
```

### Method 4: Cloud Deployment

Deploy to cloud platforms for remote access:

**Heroku:**
```bash
# Install Heroku CLI, then:
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI
heroku create your-ttrpg-assistant
git push heroku main
```

**DigitalOcean:**
Use the one-click app from the DigitalOcean marketplace or deploy via Docker.

**AWS/Google Cloud:**
See the [deployment guide](../deployment/README.md) for detailed instructions.

## Quick Start Guide

### Step 1: Verify Installation

After installation, verify everything is working:

```bash
# Check service status
curl http://localhost:8000/health

# You should see:
{
  "status": "healthy",
  "uptime": 45.2,
  "version": "1.0.0"
}
```

### Step 2: Access the Web Interface

1. Open your browser to `http://localhost:8000`
2. You'll see the TTRPG Assistant welcome page
3. Click "Get Started" to begin setup

### Step 3: Complete Initial Configuration

The setup wizard will guide you through:
1. **Basic Settings**: Choose your timezone, preferred game systems
2. **Data Directory**: Confirm where to store your data
3. **Performance Settings**: Optimize for your system
4. **Optional Features**: Enable AI providers, collaboration tools

### Step 4: First Search Test

Try a basic search to confirm everything works:

```bash
# Using curl (if comfortable with command line)
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search",
    "params": {
      "query": "what is a saving throw",
      "max_results": 3
    }
  }'
```

Or use the web interface search box.

## First-Time Setup

### Understanding the Interface

The TTRPG Assistant has several main components:

1. **Search Interface**: Find rules, spells, monsters quickly
2. **Source Manager**: Add and organize your PDF rulebooks  
3. **Campaign Manager**: Create and track campaigns
4. **Session Tools**: Initiative tracking, notes, dice rolling
5. **Content Generators**: Create NPCs, characters, encounters

### Initial Configuration

#### 1. System Preferences

Navigate to Settings → General:

```yaml
# Basic preferences
preferred_systems: ["D&D 5e", "Pathfinder 2e"]  # Your main game systems
timezone: "America/New_York"
date_format: "MM/DD/YYYY"
time_format: "12-hour"

# Search preferences  
default_search_results: 5
enable_hybrid_search: true
search_timeout_seconds: 30

# Performance settings
max_workers: 4  # Adjust based on your CPU
cache_size_mb: 500
enable_gpu: true  # If you have a GPU
```

#### 2. Directory Structure

The system creates these directories:
```
/var/lib/ttrpg-assistant/          # Main data directory
├── chromadb/                      # Vector database
├── cache/                         # Cached results
├── uploads/                       # Uploaded PDFs
├── campaigns/                     # Campaign data
└── sessions/                      # Session logs
```

#### 3. Security Settings

If you plan to access remotely or have multiple users:

```bash
# Enable authentication
export ENABLE_AUTHENTICATION=true
export AUTH_SECRET_KEY="your-secure-random-key"

# Create first admin user
python src/utils/create_user.py --username admin --email admin@example.com --role admin
```

### Performance Optimization

#### For Systems with Limited RAM (4GB or less):

```bash
# Reduce memory usage
export CACHE_MAX_MEMORY_MB=200
export MAX_WORKERS=2
export EMBEDDING_BATCH_SIZE=16
```

#### For High-Performance Systems (16GB+ RAM):

```bash
# Maximize performance
export CACHE_MAX_MEMORY_MB=2000
export MAX_WORKERS=8
export EMBEDDING_BATCH_SIZE=64
export ENABLE_PARALLEL_PROCESSING=true
```

#### For GPU Systems:

```bash
# Enable GPU acceleration
export ENABLE_GPU=true
export GPU_TYPE=cuda  # or 'mps' for Apple Silicon
export CUDA_VISIBLE_DEVICES=0
```

## Adding Your First Sources

### Understanding Source Types

The TTRPG Assistant supports several source types:

- **Rulebooks**: Core rules, player handbooks, monster manuals
- **Adventures**: Published adventures and modules
- **Supplements**: Additional rules, spells, items
- **Homebrew**: Custom content and house rules
- **Reference**: Quick reference guides, spell cards

### Method 1: Using the Web Interface

1. **Navigate to Sources**: Click "Manage Sources" in the sidebar
2. **Click "Add New Source"**: Opens the source upload dialog
3. **Upload PDF**: Drag and drop or browse for your PDF file
4. **Fill Details**:
   - **Name**: "Player's Handbook" 
   - **System**: "D&D 5e"
   - **Type**: "Rulebook"
   - **Description**: Brief description of the content
5. **Start Processing**: Click "Upload and Process"

The system will:
- Extract text from the PDF (may take several minutes)
- Create searchable chunks of content
- Generate embeddings for semantic search
- Build search indexes

### Method 2: Using the API

For batch processing multiple files:

```python
import asyncio
import requests

async def add_multiple_sources():
    sources = [
        {
            "pdf_path": "/path/to/players_handbook.pdf",
            "rulebook_name": "Player's Handbook",
            "system": "D&D 5e",
            "source_type": "rulebook"
        },
        {
            "pdf_path": "/path/to/monster_manual.pdf", 
            "rulebook_name": "Monster Manual",
            "system": "D&D 5e",
            "source_type": "rulebook"
        }
    ]
    
    for source in sources:
        response = requests.post(
            "http://localhost:8000/tools/call",
            json={
                "tool": "add_source",
                "params": source
            }
        )
        print(f"Added {source['rulebook_name']}: {response.json()}")

# Run the async function
asyncio.run(add_multiple_sources())
```

### Method 3: Batch Import Script

For large collections:

```bash
# Create a directory structure
mkdir -p ~/ttrpg-pdfs/{dnd5e,pathfinder,call-of-cthulhu}

# Place your PDFs in appropriate directories
# Then run the batch import
python tools/batch_import.py \
  --source-dir ~/ttrpg-pdfs/dnd5e \
  --system "D&D 5e" \
  --type rulebook
```

### Monitoring Import Progress

Watch the processing progress:

```bash
# Check processing status
curl http://localhost:8000/sources/status

# View logs
tail -f /var/log/ttrpg-assistant/processing.log

# Check database size
du -sh /var/lib/ttrpg-assistant/chromadb/
```

### What to Expect During Processing

**Small PDFs (< 50 pages)**: 2-5 minutes
**Medium PDFs (50-200 pages)**: 5-15 minutes  
**Large PDFs (200+ pages)**: 15-45 minutes

The process includes:
1. **PDF Text Extraction**: 40% of time
2. **Content Chunking**: 20% of time
3. **Embedding Generation**: 35% of time
4. **Index Creation**: 5% of time

### Verifying Your Sources

After processing, verify your sources are searchable:

```bash
# List all sources
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_sources"}'

# Test search
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search", 
    "params": {
      "query": "fireball",
      "system": "D&D 5e",
      "max_results": 3
    }
  }'
```

## Setting Up a Campaign

### Creating Your First Campaign

#### Using the Web Interface

1. **Navigate to Campaigns**: Click "Campaigns" in the sidebar
2. **Create New Campaign**: Click the "+" button
3. **Fill Campaign Details**:
   - **Name**: "Lost Mine of Phandelver"
   - **System**: "D&D 5e" 
   - **Description**: "A starter adventure for new players"
   - **Starting Level**: 1
   - **Max Players**: 5
4. **Configure Settings**:
   - **House Rules**: Any special rules you use
   - **Allowed Sources**: Which books/supplements to include
   - **Campaign Themes**: Horror, heroic, political, etc.

#### Using the API

```python
import requests

# Create campaign
response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "create_campaign",
        "params": {
            "name": "Lost Mine of Phandelver",
            "system": "D&D 5e",
            "description": "Starter set adventure",
            "settings": {
                "starting_level": 1,
                "max_players": 5,
                "house_rules": ["Critical hit damage maximizes base dice"],
                "themes": ["heroic", "exploration"]
            }
        }
    }
)

campaign = response.json()
campaign_id = campaign["result"]["campaign_id"]
print(f"Created campaign: {campaign_id}")
```

### Adding Player Characters

#### Method 1: Generate Characters

```python
# Generate a balanced party
characters = []
classes = ["Fighter", "Wizard", "Cleric", "Rogue"]

for char_class in classes:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "generate_character",
            "params": {
                "system": "D&D 5e",
                "level": 1,
                "class_type": char_class,
                "backstory_hints": f"A {char_class.lower()} seeking adventure"
            }
        }
    )
    character = response.json()["result"]["character"]
    characters.append(character)

print(f"Generated {len(characters)} characters")
```

#### Method 2: Import Existing Characters

```python
# Import character from D&D Beyond or other sources
character_data = {
    "name": "Thorin Ironbeard",
    "race": "Mountain Dwarf",
    "class": "Fighter",
    "level": 3,
    "stats": {
        "strength": 16,
        "dexterity": 12,
        "constitution": 15,
        "intelligence": 10,
        "wisdom": 13,
        "charisma": 8
    },
    "backstory": "A gruff but loyal warrior...",
    "equipment": ["Chain mail", "Warhammer", "Shield"]
}

# Add to campaign
response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "add_campaign_character",
        "params": {
            "campaign_id": campaign_id,
            "character": character_data
        }
    }
)
```

### Creating NPCs and Locations

#### Generate Key NPCs

```python
# Generate important NPCs for your campaign
npcs = [
    {"role": "innkeeper", "name": "Toblen Stonehill"},
    {"role": "blacksmith", "name": "Linene Graywind"},
    {"role": "villain", "importance": "major"},
    {"role": "merchant", "personality": "greedy but cowardly"}
]

for npc_info in npcs:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "generate_npc",
            "params": {
                "system": "D&D 5e",
                **npc_info
            }
        }
    )
    npc = response.json()["result"]["npc"]
    print(f"Generated NPC: {npc['name']} ({npc['role']})")
```

#### Add Locations and Map Data

```python
# Add key locations
locations = [
    {
        "name": "Cragmaw Hideout",
        "type": "dungeon",
        "description": "A cave complex inhabited by goblins",
        "map_url": "https://example.com/cragmaw-map.jpg"
    },
    {
        "name": "Phandalin",
        "type": "town", 
        "description": "A small frontier town",
        "population": 500,
        "notable_npcs": ["Toblen Stonehill", "Linene Graywind"]
    }
]

for location in locations:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "add_campaign_location",
            "params": {
                "campaign_id": campaign_id,
                "location": location
            }
        }
    )
```

### Campaign Notes and Planning

#### Organize Your Campaign Notes

```python
# Add campaign notes with categories
notes = [
    {
        "title": "Session 1 Plan",
        "content": "Party meets at inn, hears about missing supplies",
        "category": "session_planning",
        "tags": ["session1", "introduction"]
    },
    {
        "title": "House Rules",
        "content": "Flanking gives advantage, critical fumbles on nat 1",
        "category": "rules",
        "tags": ["houserules"]
    },
    {
        "title": "Player Backgrounds",
        "content": "Thorin is seeking his missing brother",
        "category": "character_notes",
        "tags": ["thorin", "background"]
    }
]

for note in notes:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "add_campaign_note",
            "params": {
                "campaign_id": campaign_id,
                **note
            }
        }
    )
```

#### Set Up Encounter Tables

```python
# Create random encounter tables
encounter_tables = {
    "road_encounters": [
        {"description": "1d4 goblins ambush", "cr": 1},
        {"description": "Traveling merchant", "cr": 0},
        {"description": "Pack of wolves", "cr": 2}
    ],
    "town_encounters": [
        {"description": "Pickpocket attempt", "cr": 0},
        {"description": "Drunk starts fight", "cr": 0},
        {"description": "Mysterious hooded figure", "cr": 0}
    ]
}

response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "add_campaign_data",
        "params": {
            "campaign_id": campaign_id,
            "data_type": "encounter_tables",
            "data": encounter_tables
        }
    }
)
```

## Running Your First Session

### Session Setup

#### 1. Start a New Session

```python
# Create a new game session
response = requests.post(
    "http://localhost:8000/tools/call", 
    json={
        "tool": "start_session",
        "params": {
            "campaign_id": campaign_id,
            "session_name": "Session 1: Goblin Ambush",
            "planned_duration": 240,  # 4 hours in minutes
            "players_expected": ["Alice", "Bob", "Carol", "Dave"]
        }
    }
)

session = response.json()["result"]
session_id = session["session_id"]
print(f"Started session: {session_id}")
```

#### 2. Prepare Session Materials

```python
# Pre-load relevant content for quick access
session_prep = {
    "monsters": ["Goblin", "Hobgoblin", "Wolf"],
    "spells": ["Cure Wounds", "Magic Missile", "Shield"],
    "conditions": ["Prone", "Unconscious", "Poisoned"],
    "locations": ["Cragmaw Hideout - Entrance"]
}

for content_type, items in session_prep.items():
    for item in items:
        # Pre-cache search results
        requests.post(
            "http://localhost:8000/tools/call",
            json={
                "tool": "search",
                "params": {
                    "query": item,
                    "content_type": content_type[:-1],  # Remove 's'
                    "max_results": 1
                }
            }
        )
```

### Combat Management

#### 1. Initiative Tracking

```python
# Set initiative order
initiative_order = [
    {"name": "Goblin 1", "initiative": 15, "type": "monster", "hp": 7, "ac": 15},
    {"name": "Thorin", "initiative": 12, "type": "player"},
    {"name": "Elara", "initiative": 10, "type": "player"}, 
    {"name": "Goblin 2", "initiative": 8, "type": "monster", "hp": 7, "ac": 15}
]

response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "set_initiative",
        "params": {
            "session_id": session_id,
            "initiative_order": initiative_order
        }
    }
)

print("Initiative set! Current turn:", response.json()["result"]["current_turn"])
```

#### 2. Managing Monsters

```python
# Add monsters to the encounter
monsters = [
    {"name": "Goblin Boss", "hp": 21, "ac": 17, "initiative": 14},
    {"name": "Goblin 1", "hp": 7, "ac": 15, "initiative": 15},
    {"name": "Goblin 2", "hp": 7, "ac": 15, "initiative": 8}
]

for monster in monsters:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "add_monster",
            "params": {
                "session_id": session_id,
                **monster
            }
        }
    )
    print(f"Added {monster['name']}")

# Update HP during combat
response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "update_monster_hp",
        "params": {
            "session_id": session_id,
            "monster_name": "Goblin 1",
            "damage": 5  # Or use "new_hp": 2
        }
    }
)
```

#### 3. Quick Rule Lookups

```python
# Common searches during combat
common_searches = [
    "grapple rules",
    "opportunity attack",
    "help action",
    "dodge action", 
    "ready action",
    "two weapon fighting"
]

# Pre-load these for instant access
for query in common_searches:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "search",
            "params": {
                "query": query,
                "max_results": 1
            }
        }
    )
    # Results are cached for instant retrieval
```

### Session Notes and Tracking

#### 1. Live Note Taking

```python
# Add notes throughout the session
session_notes = [
    {
        "timestamp": "2024-01-15T19:15:00",
        "note": "Party successfully ambushed goblins on road",
        "category": "combat",
        "tags": ["ambush", "goblins"]
    },
    {
        "timestamp": "2024-01-15T19:45:00", 
        "note": "Thorin found mysterious map in goblin pouch",
        "category": "discovery",
        "tags": ["map", "thorin", "treasure"]
    },
    {
        "timestamp": "2024-01-15T20:10:00",
        "note": "Elara convinced goblin to surrender with Persuasion (18)",
        "category": "roleplay", 
        "tags": ["elara", "persuasion", "prisoner"]
    }
]

for note in session_notes:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "add_session_note",
            "params": {
                "session_id": session_id,
                **note
            }
        }
    )
```

#### 2. Experience and Rewards

```python
# Track XP and treasure
session_rewards = {
    "xp_total": 300,
    "xp_per_player": 75,
    "treasure": {
        "gold": 15,
        "items": ["Potion of Healing", "Silver Dagger"]
    },
    "achievements": ["First Blood", "Spared an Enemy"]
}

response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "add_session_rewards",
        "params": {
            "session_id": session_id,
            **session_rewards
        }
    }
)
```

### End Session

```python
# Wrap up the session
response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "end_session",
        "params": {
            "session_id": session_id,
            "summary": "Party defeated goblin ambush, found map leading to Cragmaw Hideout",
            "next_session_notes": "Party plans to investigate the hideout",
            "player_feedback": {
                "enjoyed": ["Combat", "Roleplay opportunities"],
                "improve": ["Need more description of environments"]
            }
        }
    }
)

print("Session ended successfully!")
```

## Using the Web Interface

### Dashboard Overview

The web interface provides an intuitive way to manage all aspects of your TTRPG assistant:

#### Main Dashboard Features

1. **Quick Search Bar**: Global search across all your sources
2. **Recent Activity**: Latest searches, campaigns, sessions
3. **Campaign Cards**: Visual overview of active campaigns
4. **System Status**: Health, performance metrics
5. **Quick Actions**: Common tasks like adding sources, starting sessions

### Search Interface

#### Basic Search

1. **Enter Query**: Type your search in natural language
   - "How does stealth work?"
   - "Fireball spell details"
   - "CR 5 monsters for forest encounter"

2. **Filter Options**:
   - **System**: D&D 5e, Pathfinder, etc.
   - **Source**: Specific books
   - **Content Type**: Spells, monsters, rules, etc.
   - **Page Range**: Search specific sections

3. **Search Results**:
   - **Relevance Score**: How well it matches your query
   - **Source Info**: Book name and page number
   - **Preview**: First few lines of content
   - **Quick Actions**: Copy, save to campaign, share

#### Advanced Search Features

**Boolean Operators**:
```
"saving throw" AND constitution
fireball OR lightning bolt
magic -ritual (exclude ritual spells)
```

**Field-Specific Search**:
```
title:fireball (search only titles)
author:crawford (search by author)  
level:3 (spells of specific level)
```

**Fuzzy Search**:
```
~fireball (finds similar terms like "fire ball")
```

### Campaign Management Interface

#### Campaign Dashboard

Each campaign has its own dashboard showing:

1. **Overview Panel**:
   - Campaign name and system
   - Session count and next session date
   - Player character portraits
   - Recent activity

2. **Quick Reference**:
   - House rules
   - Important NPCs
   - Key locations
   - Current plot threads

3. **Session Prep**:
   - Encounter builder
   - Random generators
   - Music and ambiance controls
   - Handout manager

#### Session Runner Interface

During active sessions:

1. **Initiative Tracker**:
   - Drag-and-drop reordering
   - HP/status tracking
   - Turn timers
   - Action reminders

2. **Search Panel**:
   - Always available quick search
   - Recent searches
   - Bookmarked rules

3. **Notes Panel**:
   - Live note-taking
   - Player quotes
   - Plot development tracking
   - Action item lists

4. **Dice Roller**:
   - Various dice configurations
   - Advantage/disadvantage
   - Modifier shortcuts
   - Roll history

### Collaboration Features

#### Multi-User Sessions

**Setting Up Shared Sessions**:

1. **Enable Collaboration**: In campaign settings
2. **Invite Players**: Send join links or invite codes
3. **Set Permissions**: View-only, contribute, co-DM
4. **Start Collaborative Session**: All participants see real-time updates

**Real-time Features**:
- Shared initiative tracker
- Collaborative note-taking  
- Group dice rolling
- Character sheet sharing
- Map annotations

#### Player Interface

Players get a simplified interface with:
- Character sheet viewer
- Spell/ability lookup
- Dice rolling
- Note sharing with DM
- Initiative status

### Mobile Interface

The web interface is fully responsive and works great on tablets and phones:

**Tablet Mode**:
- Split-screen search and session tools
- Gesture-based navigation
- Optimized touch targets

**Phone Mode**:
- Collapsible panels
- Swipe navigation
- Voice search support
- Offline reading mode

## Advanced Features

### AI Provider Integration

#### Setting Up AI Providers

```python
# Configure multiple AI providers for different tasks
ai_config = {
    "providers": {
        "anthropic": {
            "api_key": "your-anthropic-key",
            "model": "claude-3-sonnet-20240229",
            "use_for": ["content_generation", "complex_queries"]
        },
        "openai": {
            "api_key": "your-openai-key", 
            "model": "gpt-4-turbo-preview",
            "use_for": ["quick_responses", "image_analysis"]
        },
        "google": {
            "api_key": "your-google-key",
            "model": "gemini-pro",
            "use_for": ["fact_checking", "multilingual"]
        }
    },
    "fallback_order": ["anthropic", "openai", "google"]
}
```

#### AI-Enhanced Content Generation

```python
# Generate enhanced NPCs with AI
response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "ai_generate_npc",
        "params": {
            "system": "D&D 5e",
            "role": "tavern keeper",
            "personality_traits": ["secretive", "knows local gossip"],
            "plot_relevance": "has information about missing caravan",
            "ai_provider": "anthropic",
            "detail_level": "high"
        }
    }
)
```

#### Context-Aware Responses

```python
# AI can reference your campaign context
response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "ai_campaign_advice",
        "params": {
            "campaign_id": campaign_id,
            "question": "How should I handle a player who wants to multiclass?",
            "context": {
                "current_level": 3,
                "player_experience": "new",
                "campaign_complexity": "beginner"
            }
        }
    }
)
```

### Custom Content Creation

#### Homebrew Rules

```python
# Add custom rules to your system
homebrew_rules = [
    {
        "name": "Exhaustion Recovery",
        "description": "Long rest removes 2 levels of exhaustion instead of 1",
        "system": "D&D 5e",
        "category": "house_rule",
        "tags": ["exhaustion", "rest", "houserule"]
    },
    {
        "name": "Critical Hit Damage",
        "description": "Critical hits maximize base damage dice then roll additional dice",
        "system": "D&D 5e", 
        "category": "house_rule",
        "tags": ["critical", "damage", "houserule"]
    }
]

for rule in homebrew_rules:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "add_custom_rule",
            "params": rule
        }
    )
```

#### Custom Monsters and NPCs

```python
# Create custom monsters
custom_monster = {
    "name": "Shadow Wolf",
    "type": "monstrosity",
    "size": "Medium",
    "alignment": "chaotic evil",
    "ac": 14,
    "hp": 37,
    "speed": "50 ft.",
    "stats": {
        "str": 12,
        "dex": 18,
        "con": 13,
        "int": 7,
        "wis": 12,
        "cha": 6
    },
    "skills": "Perception +3, Stealth +8",
    "damage_resistances": "necrotic",
    "senses": "darkvision 120 ft., passive Perception 13",
    "languages": "-",
    "cr": 2,
    "abilities": [
        {
            "name": "Shadow Blend",
            "description": "In dim light or darkness, has advantage on Stealth checks"
        }
    ],
    "actions": [
        {
            "name": "Bite",
            "description": "+6 to hit, reach 5 ft., 2d6 + 4 piercing + 1d4 necrotic"
        }
    ]
}

response = requests.post(
    "http://localhost:8000/tools/call",
    json={
        "tool": "add_custom_monster",
        "params": custom_monster
    }
)
```

### Automation and Scripting

#### Custom Macros

```python
# Create macros for common actions
macros = {
    "initiative_setup": {
        "description": "Set up combat with standard monsters",
        "steps": [
            {"action": "clear_initiative"},
            {"action": "add_players", "source": "campaign"},
            {"action": "roll_monster_initiative", "monsters": ["Goblin", "Hobgoblin"]}
        ]
    },
    "end_combat": {
        "description": "Clean up after combat",
        "steps": [
            {"action": "award_xp", "method": "encounter_budget"},
            {"action": "reset_hp", "target": "all_players"},
            {"action": "clear_conditions"}
        ]
    }
}
```

#### Automated Content Import

```python
# Set up automated processing of new PDFs
import watchdog.events
import watchdog.observers

class PDFHandler(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.pdf'):
            # Automatically process new PDFs
            requests.post(
                "http://localhost:8000/tools/call",
                json={
                    "tool": "add_source",
                    "params": {
                        "pdf_path": event.src_path,
                        "auto_detect_metadata": True
                    }
                }
            )

# Monitor a directory for new PDFs
observer = watchdog.observers.Observer()
observer.schedule(PDFHandler(), "/path/to/pdf/dropbox", recursive=True)
observer.start()
```

### Integration with External Tools

#### Discord Bot Integration

```python
# Example Discord bot command
import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='!')

@bot.command(name='search')
async def search_rules(ctx, *, query):
    """Search TTRPG rules from Discord"""
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "search",
            "params": {
                "query": query,
                "max_results": 1
            }
        }
    )
    
    result = response.json()["result"]
    if result["results"]:
        item = result["results"][0]
        embed = discord.Embed(
            title=f"Found: {query}",
            description=item["content"][:500] + "...",
            color=0x00ff00
        )
        embed.add_field(name="Source", value=item["source"], inline=True)
        embed.add_field(name="Page", value=item["page"], inline=True)
        await ctx.send(embed=embed)
    else:
        await ctx.send(f"No results found for: {query}")
```

#### Virtual Tabletop Integration

```python
# Export encounter data to Roll20/Foundry format
def export_encounter_to_vtt(session_id, format="roll20"):
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "export_encounter",
            "params": {
                "session_id": session_id,
                "format": format,
                "include_tokens": True,
                "include_maps": True
            }
        }
    )
    
    return response.json()["result"]["export_data"]
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Application Won't Start

**Symptoms**: Error messages on startup, service fails to start

**Common Causes**:
- Port already in use
- Missing dependencies
- Permission issues
- Corrupted configuration

**Solutions**:

```bash
# Check if port is in use
sudo lsof -i :8000
# Kill process if needed
sudo kill -9 <PID>

# Check dependencies
pip check
pip install -r requirements.txt --upgrade

# Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER /var/lib/ttrpg-assistant
chmod -R 755 /var/lib/ttrpg-assistant

# Reset configuration
mv /etc/ttrpg-assistant/config.yml /etc/ttrpg-assistant/config.yml.backup
cp config/config.yml.template /etc/ttrpg-assistant/config.yml
```

#### 2. Search Returns No Results

**Symptoms**: Searches return empty results even for known content

**Common Causes**:
- No sources have been added
- PDF processing failed
- Database corruption
- Search index issues

**Solutions**:

```bash
# Check if sources exist
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "list_sources"}'

# Check database status
ls -la /var/lib/ttrpg-assistant/chromadb/

# Rebuild search index
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "rebuild_index"}'

# Re-process a source if needed
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "reprocess_source", 
    "params": {"source_id": "your-source-id"}
  }'
```

#### 3. High Memory Usage

**Symptoms**: System becomes slow, out of memory errors

**Common Causes**:
- Large PDF collections
- High cache settings  
- Memory leaks
- Too many concurrent processes

**Solutions**:

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Reduce cache size
export CACHE_MAX_MEMORY_MB=200
export MAX_WORKERS=2

# Restart service to clear memory
sudo systemctl restart ttrpg-assistant

# Monitor memory usage
watch -n 1 'ps aux --sort=-%mem | head -10'
```

#### 4. PDF Processing Fails

**Symptoms**: PDFs stuck in "processing" state, extraction errors

**Common Causes**:
- Corrupted PDF files
- Password-protected PDFs
- Scanned images without OCR
- Insufficient disk space

**Solutions**:

```bash
# Check PDF validity
python -c "
import pypdf
with open('your-file.pdf', 'rb') as f:
    reader = pypdf.PdfReader(f)
    print(f'Pages: {len(reader.pages)}')
    print(f'Encrypted: {reader.is_encrypted}')
"

# Check disk space
df -h

# Enable OCR for scanned PDFs
export ENABLE_OCR=true
export OCR_LANGUAGE=eng

# Process with debug logging
export LOG_LEVEL=DEBUG
python src/pdf_processing/pdf_parser.py --file your-file.pdf
```

#### 5. Web Interface Not Loading

**Symptoms**: Blank page, JavaScript errors, style issues

**Common Causes**:
- Browser cache issues
- JavaScript disabled
- CORS problems
- Reverse proxy misconfiguration

**Solutions**:

```bash
# Clear browser cache and cookies
# Or try incognito/private mode

# Check browser console for errors (F12)

# Test API directly
curl http://localhost:8000/health

# Check CORS settings
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: X-Requested-With" \
     -X OPTIONS http://localhost:8000/tools/call
```

### Getting Help

#### Log Files

Important log locations:
```bash
# Application logs
tail -f /var/log/ttrpg-assistant/app.log

# Error logs  
tail -f /var/log/ttrpg-assistant/error.log

# Processing logs
tail -f /var/log/ttrpg-assistant/processing.log

# Web server logs (if using nginx)
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

#### Debug Mode

Enable detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export SQL_ECHO=true  # Database queries
export TRACE_REQUESTS=true  # HTTP requests
```

#### Health Check Endpoint

Monitor system health:
```bash
curl http://localhost:8000/health | jq
```

Expected response:
```json
{
  "status": "healthy",
  "uptime": 3600.5,
  "version": "1.0.0",
  "database": "connected",
  "memory_usage": "45%",
  "disk_usage": "12%"
}
```

#### Community Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the `/docs` directory
- **Discord Community**: Join for real-time help
- **Wiki**: Community-contributed guides and tips

## Next Steps

### Expanding Your Setup

#### 1. Add More Content

**Official Sources**:
- Core rulebooks for your system
- Monster manuals and bestiaries
- Adventure modules
- Supplemental rules

**Third-Party Content**:
- Kobold Press supplements
- Paizo Adventure Paths  
- Independent publishers
- Community homebrew

**Homebrew Content**:
- Your own house rules
- Custom monsters and NPCs
- Original adventures
- Player-created content

#### 2. Advanced Campaign Management

**Multi-Campaign Setup**:
```python
# Create campaigns for different systems
campaigns = [
    {"name": "Curse of Strahd", "system": "D&D 5e", "theme": "horror"},
    {"name": "Extinction Curse", "system": "Pathfinder 2e", "theme": "circus"},
    {"name": "Call of Cthulhu Investigation", "system": "Call of Cthulhu", "theme": "mystery"}
]

for campaign_data in campaigns:
    response = requests.post(
        "http://localhost:8000/tools/call",
        json={"tool": "create_campaign", "params": campaign_data}
    )
```

**Cross-Campaign Sharing**:
- Reuse NPCs and locations
- Import characters between campaigns
- Share house rules and content
- Build a persistent world

#### 3. Collaboration Features

**Setting Up for Multiple DMs**:
```python
# Create DM accounts with different permissions
dm_accounts = [
    {"username": "alice_dm", "role": "co_dm", "campaigns": ["campaign_1"]},
    {"username": "bob_dm", "role": "assistant_dm", "campaigns": ["campaign_1", "campaign_2"]}
]
```

**Player Access**:
- Give players read-only access to campaign info
- Allow character sheet updates
- Enable collaborative note-taking
- Set up player-only communications

### Integration Projects

#### 1. Discord Bot

Create a Discord bot for your group:

```python
# Full-featured Discord integration
@bot.command(name='roll')
async def roll_dice(ctx, dice_expression):
    """Roll dice with full D&D support"""
    result = requests.post(
        "http://localhost:8000/tools/call",
        json={
            "tool": "roll_dice",
            "params": {"expression": dice_expression}
        }
    )
    await ctx.send(f"🎲 {dice_expression}: {result['total']}")

@bot.command(name='monster')  
async def monster_lookup(ctx, *, monster_name):
    """Look up monster stats"""
    # Implementation here

@bot.command(name='spell')
async def spell_lookup(ctx, *, spell_name):
    """Look up spell details"""  
    # Implementation here
```

#### 2. Streaming Integration

**OBS Studio Integration**:
- Display search results on stream
- Show initiative tracker
- Overlay campaign information
- Real-time dice rolls

**Twitch Bot Integration**:
- Let viewers search rules
- Interactive polls for decisions
- Chat command integration

#### 3. Mobile Apps

**Progressive Web App**:
The web interface works great as a mobile app:

```javascript
// Install as PWA on mobile devices
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}

// Add to home screen prompt
window.addEventListener('beforeinstallprompt', (e) => {
  // Show install button
});
```

### Performance Optimization

#### 1. Hardware Upgrades

**Recommended Upgrades by Usage**:

*Light Usage (1-2 campaigns, basic search)*:
- 8GB RAM minimum
- SSD storage recommended
- Dual-core CPU sufficient

*Medium Usage (3-5 campaigns, regular sessions)*:
- 16GB RAM recommended  
- NVMe SSD for database
- Quad-core CPU
- Dedicated GPU helpful

*Heavy Usage (10+ campaigns, multiple concurrent users)*:
- 32GB+ RAM
- Multiple SSDs (separate OS, database, cache)
- 8+ core CPU
- High-end GPU for AI processing

#### 2. Advanced Caching

```python
# Multi-tier caching strategy
cache_config = {
    "tiers": {
        "l1_memory": {
            "type": "lru",
            "max_entries": 10000,
            "max_memory_mb": 500,
            "ttl_seconds": 300
        },
        "l2_redis": {
            "host": "localhost", 
            "port": 6379,
            "max_memory_mb": 1000,
            "ttl_seconds": 3600
        },
        "l3_disk": {
            "path": "/var/cache/ttrpg-assistant",
            "max_size_gb": 10,
            "ttl_seconds": 86400
        }
    }
}
```

#### 3. Database Optimization

**ChromaDB Tuning**:
```python
# Optimize for your usage pattern
chroma_config = {
    "implementation": "clickhouse",  # For high-performance
    "embedding_batch_size": 100,
    "query_batch_size": 50,
    "index_config": {
        "m": 16,  # Number of connections
        "ef_construction": 200,  # Index build quality
        "ef": 100  # Search quality
    }
}
```

### Community Contributions

#### 1. Content Sharing

**Community Repository**:
- Share homebrew content
- Contribute to spell/monster databases
- Create reusable campaign templates
- Build system-specific rules

#### 2. Feature Development

**Contributing Code**:
```bash
# Fork the repository
git clone https://github.com/yourusername/MDMAI.git
cd MDMAI

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/
black src/
flake8 src/

# Submit pull request
git push origin feature/your-feature-name
```

**Areas for Contribution**:
- New game system support
- Additional AI provider integrations
- Mobile app improvements
- Performance optimizations
- Documentation and tutorials

### Enterprise Features

#### 1. Multi-Tenancy

For gaming stores, clubs, or educational institutions:

```python
# Multi-tenant configuration
tenant_config = {
    "enabled": True,
    "isolation_level": "database",  # or "schema"
    "default_limits": {
        "campaigns": 10,
        "sources": 100,
        "storage_gb": 50
    }
}
```

#### 2. Advanced Analytics

```python
# Usage analytics for organizations
analytics_config = {
    "track_usage": True,
    "metrics": [
        "search_queries_per_day",
        "active_campaigns", 
        "user_engagement",
        "popular_content"
    ],
    "retention_days": 90
}
```

#### 3. Compliance and Security

**GDPR Compliance**:
- Data export tools
- Right to deletion
- Audit logging
- Consent management

**Enterprise Security**:
- LDAP/Active Directory integration
- Multi-factor authentication
- Role-based access control
- SOC 2 compliance tools

---

**Congratulations!** You've completed the comprehensive getting started guide for the TTRPG Assistant. You should now be able to:

✅ Install and configure the system  
✅ Import and search your rulebooks  
✅ Create and manage campaigns  
✅ Run engaging game sessions  
✅ Use advanced features and integrations  
✅ Troubleshoot common issues  
✅ Plan for scaling and growth  

## Quick Reference

### Essential Commands

```bash
# Start/stop service
sudo systemctl start ttrpg-assistant
sudo systemctl stop ttrpg-assistant

# View logs
tail -f /var/log/ttrpg-assistant/app.log

# Health check
curl http://localhost:8000/health

# Backup data
python deploy/backup/backup_manager.py --create
```

### Key API Endpoints

```bash
# Search
POST /tools/call {"tool": "search", "params": {"query": "..."}}

# Add source
POST /tools/call {"tool": "add_source", "params": {"pdf_path": "..."}}

# Create campaign
POST /tools/call {"tool": "create_campaign", "params": {"name": "..."}}

# Start session
POST /tools/call {"tool": "start_session", "params": {"campaign_id": "..."}}
```

### Support Resources

- **Documentation**: `/docs` directory
- **GitHub Issues**: Report bugs and feature requests
- **Community Discord**: Real-time help and discussion
- **Wiki**: Community guides and tutorials

Happy gaming! 🎲✨