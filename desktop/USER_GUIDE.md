# TTRPG Assistant Desktop Application - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [First Launch](#first-launch)
4. [Core Features](#core-features)
5. [Campaign Management](#campaign-management)
6. [PDF Processing](#pdf-processing)
7. [Search and Retrieval](#search-and-retrieval)
8. [Character Generation](#character-generation)
9. [Session Management](#session-management)
10. [Data Management](#data-management)
11. [Keyboard Shortcuts](#keyboard-shortcuts)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Features](#advanced-features)

## Getting Started

Welcome to the TTRPG Assistant Desktop Application! This powerful tool helps Dungeon Masters and Game Runners manage their tabletop role-playing game sessions with AI-powered assistance.

### System Requirements

**Minimum Requirements:**
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 4GB
- **Storage**: 500MB for application + space for your data
- **Display**: 1280x720 resolution

**Recommended Requirements:**
- **RAM**: 8GB or more
- **Storage**: 2GB+ for optimal performance
- **Display**: 1920x1080 resolution or higher
- **Internet**: Required for AI provider features (optional for offline mode)

## Installation

### Windows

1. Download the `.msi` installer from the releases page
2. Double-click the installer file
3. Follow the installation wizard
4. The application will be installed to `C:\Program Files\TTRPG Assistant` by default
5. A desktop shortcut will be created automatically

### macOS

1. Download the `.dmg` file from the [releases page](https://github.com/Raudbjorn/MDMAI/releases)
2. Open the downloaded `.dmg` file
3. Drag the TTRPG Assistant app to your Applications folder
4. The first time you run the app, you may need to right-click and select "Open" due to Gatekeeper

### Linux

**AppImage (Recommended):**
1. Download the `.AppImage` file
2. Make it executable: `chmod +x TTRPG-Assistant-*.AppImage`
3. Run the application: `./TTRPG-Assistant-*.AppImage`

**Debian/Ubuntu (.deb):**
```bash
sudo dpkg -i ttrpg-assistant_*.deb
sudo apt-get install -f  # Install dependencies if needed
```

**Fedora/RHEL (.rpm):**
```bash
sudo rpm -i ttrpg-assistant-*.rpm
```

## First Launch

### Initial Setup

1. **Welcome Screen**: On first launch, you'll see a welcome screen with a quick tutorial
2. **Data Directory**: Choose where to store your campaign data (default: Documents/TTRPG Assistant)
3. **AI Provider Setup** (Optional):
   - Enter API keys for AI providers (Claude, OpenAI, Google)
   - Or skip to use offline features only
4. **Import Existing Data**: If you have data from the web version, you can import it here

### MCP Server Initialization

The application automatically starts the Model Context Protocol (MCP) server in the background. You'll see a status indicator in the bottom-right corner:
- 🟢 Green: Connected and ready
- 🟡 Yellow: Connecting or reconnecting
- 🔴 Red: Disconnected (click to retry)

## Core Features

### Main Interface

The application has a clean, intuitive interface with:

1. **Sidebar Navigation**: Quick access to all major features
2. **Main Content Area**: Where you interact with the current feature
3. **Status Bar**: Shows connection status, current campaign, and quick actions
4. **Quick Search**: Press `Ctrl/Cmd + K` to search anywhere

### Themes

- **Light Mode**: Default bright theme
- **Dark Mode**: Easy on the eyes for long sessions
- **System**: Follows your OS theme preference

To change themes: Settings → Appearance → Theme

## Campaign Management

### Creating a Campaign

1. Click **"Campaigns"** in the sidebar
2. Click **"New Campaign"** button
3. Fill in the details:
   - **Name**: Your campaign's name
   - **System**: D&D 5e, Pathfinder, Call of Cthulhu, etc.
   - **Description**: Brief overview
   - **Players**: Add player names (optional)
4. Click **"Create"**

### Managing Campaigns

- **Edit**: Click the pencil icon next to any campaign
- **Archive**: Right-click → Archive (keeps data but hides from active list)
- **Delete**: Right-click → Delete (confirmation required)
- **Export**: Right-click → Export to share or backup
- **Import**: Use File → Import Campaign

### Campaign Features

Each campaign includes:
- **Overview Dashboard**: Quick stats and recent activity
- **Characters & NPCs**: Manage all characters in your campaign
- **Locations**: Track important places
- **Session Notes**: Organized by date
- **Plot Threads**: Keep track of ongoing storylines
- **Resources**: Party inventory and resources

## PDF Processing

### Importing PDFs

1. Navigate to **"Sources"** in the sidebar
2. Click **"Import PDF"** or drag and drop files
3. Select processing options:
   - **Quick Import**: Fast, basic text extraction
   - **Deep Processing**: Slower but preserves tables and formatting
   - **OCR Mode**: For scanned PDFs (requires more time)
4. Click **"Start Import"**

### Managing Sources

- **Categories**: Automatically categorized (Rulebooks, Adventures, Supplements)
- **Search within PDFs**: Use the search bar to find content
- **Annotations**: Add notes to specific pages
- **Bookmarks**: Save important sections for quick access

### Supported Formats

- PDF files (.pdf)
- Images with text (.png, .jpg) - uses OCR
- Plain text files (.txt, .md)
- EPUB files (.epub) - for digital books

## Search and Retrieval

### Basic Search

1. Click the **Search** icon or press `Ctrl/Cmd + K`
2. Type your query
3. Press Enter or click **Search**

### Advanced Search

Use special operators for precise searches:
- **Quotes**: `"exact phrase"` for exact matches
- **AND**: `sword AND magic` for both terms
- **OR**: `fighter OR warrior` for either term
- **NOT**: `spell NOT wizard` to exclude terms
- **Wildcards**: `drag*` matches dragon, dragonborn, etc.

### Search Filters

- **Source Type**: Rulebooks, Campaign notes, Characters
- **Game System**: Filter by game system
- **Date Range**: For time-sensitive searches
- **Content Type**: Rules, Lore, Mechanics, etc.

### Search Results

Results show:
- **Relevance Score**: How well it matches your query
- **Source**: Where the information comes from
- **Page Number**: For PDF sources
- **Preview**: Snippet with highlighted terms
- **Actions**: View full text, go to source, copy reference

## Character Generation

### Quick Character

1. Go to **"Characters"**
2. Click **"Generate Character"**
3. Choose:
   - **System**: D&D 5e, Pathfinder, etc.
   - **Race/Species**
   - **Class/Profession**
   - **Level**
4. Click **"Generate"**

### Advanced Options

- **Ability Scores**: Standard Array, Point Buy, or Random
- **Background**: Detailed backstory generation
- **Equipment**: Starting gear based on class and level
- **Personality**: Traits, ideals, bonds, and flaws
- **Name**: Use name generator or enter custom

### NPC Generation

1. Click **"Generate NPC"**
2. Select NPC type:
   - Quest Giver
   - Merchant
   - Guard
   - Noble
   - Villain
   - Custom
3. Adjust parameters as needed
4. Click **"Generate"**

### Managing Characters

- **Character Sheets**: Full interactive character sheets
- **Quick Stats**: View key information at a glance
- **Notes**: Add session notes for each character
- **Inventory**: Track items and equipment
- **Progression**: Level up and track changes

## Session Management

### Starting a Session

1. Go to **"Sessions"**
2. Click **"Start New Session"**
3. Select your campaign
4. The session tracker opens with:
   - Initiative tracker
   - Quick dice roller
   - Notes panel
   - Timer

### Initiative Tracker

- **Add Combatant**: Click + or press `Ctrl/Cmd + I`
- **Roll Initiative**: Click dice icon or enter manually
- **Track HP**: Click HP values to modify
- **Conditions**: Right-click to add conditions (stunned, poisoned, etc.)
- **Next Turn**: Space bar or click arrow
- **End Combat**: Click "End Combat" to clear

### Session Notes

- **Auto-save**: Notes save automatically every 30 seconds
- **Timestamps**: Press `Ctrl/Cmd + T` to insert timestamp
- **Categories**: Tag notes as Combat, RP, Lore, etc.
- **Link Entities**: Type @ to link characters, locations, items

### Dice Roller

- **Quick Roll**: Click common dice (d20, d6, etc.)
- **Custom**: Enter formulas like `3d6+2` or `1d20+5`
- **Advantage/Disadvantage**: Hold Shift for advantage, Alt for disadvantage
- **History**: View last 20 rolls
- **Macros**: Save common rolls for quick access

## Data Management

### Backup and Restore

**Creating Backups:**
1. Go to Settings → Data → Backup
2. Click **"Create Backup"**
3. Choose location and filename
4. Backups include all campaigns, characters, and settings

**Restoring:**
1. Settings → Data → Restore
2. Select backup file
3. Choose what to restore (selective or full)
4. Click **"Restore"**

### Export Options

- **Campaign Export**: Share complete campaigns with others
- **Character Export**: Export as PDF, JSON, or Roll20 format
- **Session Logs**: Export as formatted text or markdown
- **Full Data Export**: Everything in a portable format

### Data Sync

If using cloud storage:
1. Settings → Sync
2. Choose provider (Dropbox, Google Drive, OneDrive)
3. Authorize access
4. Enable auto-sync

## Keyboard Shortcuts

### Global Shortcuts

| Action | Windows/Linux | macOS |
|--------|--------------|-------|
| Quick Search | `Ctrl + K` | `Cmd + K` |
| New Campaign | `Ctrl + N` | `Cmd + N` |
| Save | `Ctrl + S` | `Cmd + S` |
| Settings | `Ctrl + ,` | `Cmd + ,` |
| Toggle Sidebar | `Ctrl + B` | `Cmd + B` |
| Full Screen | `F11` | `Cmd + Ctrl + F` |

### Session Shortcuts

| Action | Shortcut |
|--------|----------|
| Next Turn | `Space` |
| Previous Turn | `Shift + Space` |
| Add Combatant | `Ctrl/Cmd + I` |
| Roll Dice | `Ctrl/Cmd + R` |
| Insert Timestamp | `Ctrl/Cmd + T` |

### Navigation

| Action | Shortcut |
|--------|----------|
| Navigate Tabs | `Ctrl/Cmd + 1-9` |
| Previous Tab | `Ctrl/Cmd + Shift + Tab` |
| Next Tab | `Ctrl/Cmd + Tab` |
| Close Tab | `Ctrl/Cmd + W` |

## Troubleshooting

### Common Issues

**Application Won't Start:**
- Check system requirements
- Try running as administrator (Windows)
- Check antivirus isn't blocking the app
- Reinstall the application

**MCP Server Disconnected:**
- Click the status indicator to reconnect
- Check Settings → Advanced → MCP Logs
- Restart the application if needed

**PDF Import Fails:**
- Ensure PDF isn't password protected
- Try different processing mode
- Check available disk space
- For large PDFs, increase timeout in Settings

**Search Not Working:**
- Rebuild search index: Settings → Data → Rebuild Index
- Check if sources are properly imported
- Clear cache: Settings → Data → Clear Cache

### Performance Issues

**Slow Performance:**
1. Close unnecessary tabs
2. Clear cache: Settings → Data → Clear Cache
3. Reduce PDF processing quality for faster imports
4. Disable real-time sync if using cloud storage

**High Memory Usage:**
1. Limit concurrent PDF processing
2. Close unused campaigns
3. Clear old session logs
4. Restart the application periodically

### Getting Help

- **Built-in Help**: Press `F1` or click Help menu
- **Documentation**: [Online docs](https://github.com/Raudbjorn/MDMAI/wiki)
- **Community**: [Discord server](https://discord.gg/ttrpg-assistant)
- **Bug Reports**: [GitHub Issues](https://github.com/Raudbjorn/MDMAI/issues)
- **Email Support**: support@ttrpg-assistant.com

## Advanced Features

### AI Providers

Configure AI providers for enhanced features:

1. Go to Settings → AI Providers
2. Add API keys for:
   - **Claude (Anthropic)**: Best for creative content
   - **GPT (OpenAI)**: Good all-around performance
   - **Gemini (Google)**: Fast and cost-effective
   - **Ollama (Local)**: Privacy-focused, runs locally

### Ollama Integration

For completely offline AI:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama2`
3. In app: Settings → AI Providers → Ollama
4. Select your installed model
5. No internet required!

### Custom Personalities

Create system-specific AI personalities:

1. Settings → Personalities
2. Click "New Personality"
3. Configure:
   - **Name**: e.g., "Gritty Cyberpunk GM"
   - **Tone**: Formal, Casual, Dramatic, etc.
   - **Vocabulary**: System-specific terms
   - **Behavior**: How it responds to queries

### Automation

Create automated workflows:

1. Settings → Automation
2. Examples:
   - Auto-roll initiative at combat start
   - Generate session summary at session end
   - Create NPCs when entering new locations
   - Alert when players level up

### Plugin System

Extend functionality with plugins:

1. Settings → Plugins
2. Browse available plugins
3. Install with one click
4. Popular plugins:
   - **Map Maker**: Visual battle maps
   - **Sound Board**: Ambient music and effects
   - **Character Voice**: AI voice for NPCs
   - **Loot Generator**: Random treasure

### Developer Mode

For advanced users:

1. Settings → Advanced → Developer Mode
2. Features:
   - Console access
   - API endpoint testing
   - Raw data editor
   - Performance profiler
   - Debug logging

## Tips and Best Practices

### Organization Tips

- **Naming Convention**: Use consistent naming (e.g., "Campaign - Session 01")
- **Tags**: Tag everything for easy searching
- **Regular Backups**: Weekly backups recommended
- **Archive Old Campaigns**: Keeps interface clean

### Performance Tips

- **Batch Import**: Import multiple PDFs at once
- **Index Optimization**: Rebuild index monthly
- **Clean Session Logs**: Archive old sessions
- **Selective Sync**: Only sync active campaigns

### GM Tips

- **Prep Templates**: Create templates for common scenarios
- **Quick References**: Bookmark frequently used rules
- **NPC Gallery**: Pre-generate NPCs for improvisation
- **Session Prep**: Use checklists for consistent prep

## Privacy and Security

### Data Privacy

- **Local First**: All data stored locally by default
- **Encryption**: Sensitive data encrypted at rest
- **No Telemetry**: No usage data sent without consent
- **API Keys**: Stored securely in OS keychain

### Security Features

- **Auto-lock**: Lock app after inactivity
- **Backup Encryption**: Optional password protection
- **Secure Delete**: Overwrite deleted data
- **Audit Log**: Track all data changes

## Updates

### Auto-Updates

The app checks for updates automatically:
1. Notification appears when update available
2. Click "Update" to download
3. App restarts with new version
4. Release notes show changes

### Manual Updates

1. Help → Check for Updates
2. Or download from [releases page](https://github.com/Raudbjorn/MDMAI/releases)

## Appendix

### File Locations

**Windows:**
- App: `C:\Program Files\TTRPG Assistant\`
- Data: `%APPDATA%\TTRPG Assistant\`
- Logs: `%APPDATA%\TTRPG Assistant\logs\`

**macOS:**
- App: `/Applications/TTRPG Assistant.app`
- Data: `~/Library/Application Support/TTRPG Assistant/`
- Logs: `~/Library/Logs/TTRPG Assistant/`

**Linux:**
- App: `/opt/ttrpg-assistant/` or AppImage location
- Data: `~/.config/ttrpg-assistant/`
- Logs: `~/.config/ttrpg-assistant/logs/`

### Command Line Options

```bash
ttrpg-assistant [options]

Options:
  --campaign <name>    Open specific campaign
  --minimize          Start minimized to tray
  --portable          Use portable mode (data in app directory)
  --debug             Enable debug logging
  --reset             Reset all settings to default
```

### Uninstallation

**Windows:** Use Add/Remove Programs or uninstaller in Start Menu

**macOS:** Drag app to Trash, then:
```bash
rm -rf ~/Library/Application\ Support/TTRPG\ Assistant/
```

**Linux:**
```bash
# For .deb
sudo apt remove ttrpg-assistant

# For .rpm  
sudo rpm -e ttrpg-assistant

# For AppImage
# Just delete the file

# Remove data
rm -rf ~/.config/ttrpg-assistant/
```

---

Thank you for using TTRPG Assistant! May your campaigns be epic and your dice rolls favorable! 🎲

For the latest updates and community content, visit our [website](https://ttrpg-assistant.com) or join our [Discord](https://discord.gg/ttrpg-assistant).
