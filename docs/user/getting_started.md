# Getting Started Guide

This guide will walk you through setting up and using the TTRPG Assistant for the first time.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

For CPU-only systems:
```bash
pip install -r requirements-cpu.txt
```

### Step 4: Quick Setup (Optional)

Run the quick setup script for automatic configuration:
```bash
./quick_setup.sh  # On Windows: quick_setup.bat
```

## Initial Configuration

### Setting Up ChromaDB

The database initializes automatically on first run, but you can configure the storage location:

```python
# config/settings.py
CHROMADB_PATH = "/path/to/your/storage"
```

### Configuring Logging

Adjust logging levels in `config/logging_config.py`:

```python
LOGGING_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE_PATH = "logs/ttrpg_assistant.log"
```

## First Run

### Starting the Server

```bash
python src/main.py
```

You should see:
```
TTRPG Assistant MCP Server v1.0.0
Initializing ChromaDB...
Server ready on stdin/stdout
```

### Using with MCP Client

The server communicates via the Model Context Protocol. You'll need an MCP client to interact with it:

1. **Claude Desktop** (Recommended)
2. **Custom MCP Client**
3. **Python Script Interface**

### Example: Python Client

```python
import asyncio
from mcp.client import MCPClient

async def main():
    # Initialize client
    client = MCPClient()
    await client.connect_stdio("python src/main.py")
    
    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {[t['name'] for t in tools]}")
    
    # Search for content
    result = await client.call_tool(
        "search",
        {
            "query": "fireball spell",
            "max_results": 3
        }
    )
    print(result)

asyncio.run(main())
```

## Adding Your First Source

### Step 1: Prepare Your PDFs

Organize your PDFs in a dedicated folder:
```
/home/user/rpg_books/
  ├── dnd5e/
  │   ├── players_handbook.pdf
  │   ├── monster_manual.pdf
  │   └── dungeon_masters_guide.pdf
  └── other_systems/
```

### Step 2: Add a Rulebook

```python
result = await add_source(
    pdf_path="/home/user/rpg_books/dnd5e/players_handbook.pdf",
    rulebook_name="Player's Handbook",
    system="D&D 5e",
    source_type="rulebook"
)
```

### Step 3: Monitor Progress

The system will display:
- Extraction progress
- Number of pages processed
- Chunks created
- Indexing status

### Step 4: Verify Addition

```python
sources = await list_sources(system="D&D 5e")
print(f"Added {len(sources['sources'])} sources")
```

## Creating Your First Campaign

### Step 1: Create Campaign

```python
campaign = await create_campaign(
    name="Lost Mine of Phandelver",
    system="D&D 5e",
    description="A starter adventure for new adventurers"
)
campaign_id = campaign["campaign_id"]
```

### Step 2: Add Characters

```python
# Generate a character
character = await generate_character(
    system="D&D 5e",
    level=1,
    class_type="Fighter"
)

# Add to campaign
await update_campaign_data(
    campaign_id=campaign_id,
    data_type="character",
    data=character["character"]
)
```

### Step 3: Add NPCs

```python
# Generate an innkeeper
npc = await generate_npc(
    system="D&D 5e",
    role="innkeeper",
    name="Toblen Stonehill"
)

# Add to campaign
await update_campaign_data(
    campaign_id=campaign_id,
    data_type="npc",
    data=npc["npc"]
)
```

## Running Your First Session

### Step 1: Start Session

```python
session = await start_session(
    campaign_id=campaign_id,
    session_name="Session 1: Goblin Ambush"
)
session_id = session["session_id"]
```

### Step 2: Set Initiative

```python
await set_initiative(
    session_id=session_id,
    initiative_order=[
        {"name": "Goblin 1", "initiative": 15, "type": "monster"},
        {"name": "Fighter", "initiative": 12, "type": "player"},
        {"name": "Wizard", "initiative": 10, "type": "player"},
        {"name": "Goblin 2", "initiative": 8, "type": "monster"}
    ]
)
```

### Step 3: Track Combat

```python
# Add monster
monster = await add_monster(
    session_id=session_id,
    name="Goblin Boss",
    hp=21,
    ac=15
)

# Update HP after damage
await update_monster_hp(
    session_id=session_id,
    monster_id=monster["monster_id"],
    new_hp=14
)
```

### Step 4: Add Notes

```python
await add_session_note(
    session_id=session_id,
    note="Party successfully ambushed the goblins",
    category="combat"
)
```

### Step 5: End Session

```python
await end_session(
    session_id=session_id,
    summary="Party defeated goblin ambush and found a map"
)
```

## Essential Commands

### Search Commands

```python
# Basic search
await search(query="how does stealth work")

# Filtered search
await search(
    query="dragon",
    content_type="monster",
    rulebook="Monster Manual"
)

# Exact match
await search(query='"armor class 15"', use_hybrid=False)
```

### Campaign Commands

```python
# List all campaigns
campaigns = await list_campaigns()

# Get campaign details
data = await get_campaign_data(campaign_id=campaign_id)

# Archive campaign
await delete_campaign(campaign_id=campaign_id, hard_delete=False)
```

### Generation Commands

```python
# Generate with hints
character = await generate_character(
    system="D&D 5e",
    backstory_hints="Grew up in a thieves' guild"
)

# Generate party
party = await generate_party(
    system="D&D 5e",
    size=4,
    level=3,
    composition="balanced"
)
```

## Configuration Tips

### Performance Optimization

For large PDF collections:
```python
# config/settings.py
BATCH_SIZE = 100  # Process more chunks at once
CACHE_SIZE = 2000  # Larger cache for frequent queries
EMBEDDING_BATCH_SIZE = 50  # Parallel embedding generation
```

### Memory Management

For systems with limited RAM:
```python
# config/settings.py
MAX_CACHE_MB = 500  # Limit cache size
CHUNK_SIZE = 512  # Smaller chunks
CLEANUP_INTERVAL = 300  # More frequent cleanup
```

## Next Steps

Now that you're set up:

1. **Add More Sources**: Import all your rulebooks
2. **Customize Personalities**: Create unique voices for your campaigns
3. **Explore Advanced Search**: Master query techniques
4. **Automate Workflows**: Create scripts for common tasks

Continue with:
- [Managing Campaigns](./campaigns.md)
<!-- Note: The following guides are coming soon:
- [Running Sessions](./sessions.md)
- [Search Guide](./search.md)
-->
