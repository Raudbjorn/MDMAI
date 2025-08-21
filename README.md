# TTRPG Assistant MCP Server

A Model Context Protocol (MCP) server designed to assist with Tabletop Role-Playing Games (TTRPGs). This creates a comprehensive assistant for Dungeon Masters and Game Runners that can quickly retrieve relevant rules, spells, monsters, and campaign information during gameplay.

## Features

- **PDF Processing**: Extracts and indexes content from TTRPG rulebooks
- **Hybrid Search**: Combines semantic and keyword search for accurate results
- **Campaign Management**: Store and retrieve campaign-specific data
- **Session Tracking**: Manage initiative, monster health, and session notes
- **Personality System**: Adapts responses to match game system tone
- **Character/NPC Generation**: Create characters with appropriate stats and backstories
- **Adaptive Learning**: Improves PDF processing accuracy over time

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the environment configuration:
```bash
cp .env.example .env
```

5. Edit `.env` to configure your settings (optional)

## Usage

### Running the MCP Server

Start the server in stdio mode (for MCP clients):
```bash
python src/main.py
```

### Development Mode

For development and testing, you can run the server in HTTP mode:
```bash
# Edit .env and set MCP_STDIO_MODE=false
python src/main.py
```

The server will be available at `http://localhost:8000`

## MCP Tools

The server exposes the following MCP tools:

### search
Search across TTRPG content with semantic and keyword matching.

```python
await search(
    query="fireball spell",
    rulebook="Player's Handbook",
    source_type="rulebook",
    max_results=5
)
```

### add_source
Add a new PDF source to the knowledge base.

```python
await add_source(
    pdf_path="/path/to/rulebook.pdf",
    rulebook_name="Player's Handbook",
    system="D&D 5e",
    source_type="rulebook"
)
```

### list_sources
List available sources in the system.

```python
await list_sources(
    system="D&D 5e",
    source_type="rulebook"
)
```

### create_campaign
Create a new campaign.

```python
await create_campaign(
    name="Dragon's Crown",
    system="D&D 5e",
    description="A campaign about recovering an ancient artifact"
)
```

### get_campaign_data
Retrieve campaign-specific data.

```python
await get_campaign_data(
    campaign_id="uuid-here",
    data_type="characters"
)
```

### server_info
Get information about the server status and configuration.

```python
await server_info()
```

## Project Structure

```
MDMAI/
├── src/
│   ├── core/           # Core functionality
│   ├── pdf_processing/ # PDF extraction and parsing
│   ├── search/         # Search engine implementation
│   ├── campaign/       # Campaign management
│   ├── session/        # Session tracking
│   ├── personality/    # Personality system
│   ├── models/         # Data models
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── tests/              # Test suite
├── data/               # Data storage
│   ├── chromadb/       # Vector database
│   └── cache/          # Cache directory
└── logs/               # Log files
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8 .
```

### Type Checking

```bash
mypy src/
```

## Configuration

Key configuration options in `.env`:

- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `CHROMA_DB_PATH`: Path to ChromaDB storage
- `EMBEDDING_MODEL`: Model for generating embeddings
- `ENABLE_HYBRID_SEARCH`: Enable/disable hybrid search
- `MAX_CHUNK_SIZE`: Maximum size for document chunks
- `ENABLE_ADAPTIVE_LEARNING`: Enable learning from processed PDFs

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

See [tasks.md](tasks.md) for the detailed implementation roadmap and progress tracking.

## Documentation

- [Requirements](requirements.md) - Detailed project requirements
- [Design](design.md) - Technical architecture and design
- [Tasks](tasks.md) - Implementation tasks and timeline
- [Claude Context](CLAUDE.md) - Development context and guidelines