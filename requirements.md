# TTRPG Assistant MCP Server - Requirements Document

## Project Overview
A Model Context Protocol (MCP) server designed to assist with Tabletop Role-Playing Games (TTRPGs), creating a comprehensive "side-car" assistant for Dungeon Masters and Game Runners that can quickly retrieve relevant rules, spells, monsters, and campaign information during gameplay.

## Functional Requirements

### REQ-001: Rule and Game Element Search
**User Story:** As a Dungeon Master, I want to quickly look up rules, spells, and monster stats from my TTRPG rulebooks during gameplay, so that I can maintain game flow without manually searching through physical or digital books.

**Acceptance Criteria:**
- WHEN a user queries for a specific rule or game element THEN the system SHALL return relevant information from the parsed rulebook content with semantic similarity matching
- WHEN multiple relevant results exist THEN the system SHALL rank results by relevance and return the top matches with source page references
- IF a query matches content from multiple rulebooks THEN the system SHALL clearly identify which source each result comes from

### REQ-002: Campaign Data Management
**User Story:** As a Game Runner, I want to store and retrieve campaign-specific data (characters, NPCs, locations, plot points), so that I can maintain continuity and quickly access relevant information during sessions.

**Acceptance Criteria:**
- WHEN campaign data is stored THEN the system SHALL organize it by campaign identifier and data type (characters, NPCs, locations, etc.)
- WHEN querying campaign data THEN the system SHALL support both exact matches and semantic search across stored content
- WHEN updating campaign data THEN the system SHALL maintain version history and allow rollback to previous states
- IF campaign data references rulebook content THEN the system SHALL create linkages between campaign elements and relevant rules

### REQ-003: Source Material Integration
**User Story:** As a developer or advanced user, I want to easily add new source material to the system, so that I can expand the knowledge base without technical complexity.

**Acceptance Criteria:**
- WHEN a PDF source is provided THEN the system SHALL extract text content while preserving structure and formatting
- WHEN processing a source THEN the system SHALL use the book's glossary/index to create meaningful content chunks and metadata
- IF a source has already been processed THEN the system SHALL detect duplicates and offer options to update or skip
- WHEN adding a source THEN the user SHALL be able to specify whether it is a "rulebook" or "flavor" source

### REQ-004: MCP Tool Interface
**User Story:** As an LLM or AI assistant, I want to access TTRPG information through standardized MCP tools, so that I can provide accurate and contextual responses about game rules and campaign data.

**Acceptance Criteria:**
- WHEN the MCP server receives a search request THEN it SHALL return structured data including content, source, page numbers, and relevance scores
- WHEN multiple search types are needed THEN the system SHALL support both vector similarity search and traditional keyword search
- WHEN campaign context is relevant THEN the system SHALL cross-reference rulebook content with stored campaign data
- IF search results are ambiguous THEN the system SHALL provide clarifying context and suggest refinement options

### REQ-005: System Personality Adaptation
**User Story:** As a user, I want the LLM to adopt a personality that is appropriate for the game system I am using, so that the interaction feels more immersive.

**Acceptance Criteria:**
- WHEN a source is imported THEN the system SHALL extract a "personality" from the text
- WHEN a user interacts with the LLM THEN the system SHALL use the personality of the relevant source to configure the LLM's voice and style

**Example Personalities:**
- **D&D 5e - "Wise Sage"**: Authoritative, omniscient, academic with magical terminology
- **Blades in the Dark - "Shadowy Informant"**: Mysterious, conspiratorial, Victorian criminal underworld style
- **Delta Green - "Classified Handler"**: Formal, authoritative, government/military briefing style
- **Call of Cthulhu - "Antiquarian Scholar"**: Scholarly, ominous, academic with eldritch undertones

### REQ-006: Character Creation Support
**User Story:** As a player, I want to create a character for a new campaign, so that I can start playing the game.

**Acceptance Criteria:**
- WHEN a user wants to create a character THEN the system SHALL provide the character creation rules from the relevant rulebook
- WHEN a user wants to create a backstory for their character THEN the system SHALL generate a backstory that is consistent with the rulebook's "vibe" and any details the player provides

### REQ-007: NPC Generation
**User Story:** As a Game Master, I want to generate NPCs for my campaign, so that I can quickly populate the world with interesting characters.

**Acceptance Criteria:**
- WHEN a user wants to generate an NPC THEN the system SHALL create an NPC with stats that are appropriate for the player characters' level
- WHEN a user wants to generate an NPC THEN the system SHALL use the rulebook's "vibe" to create a character that is consistent with the game world

### REQ-008: Session Management
**User Story:** As a Game Master, I want to manage my game sessions, so that I can keep track of initiative, monster health, and session notes.

**Acceptance Criteria:**
- WHEN a user wants to start a new session THEN the system SHALL create a new session with an empty initiative tracker, monster list, and notes
- WHEN a user wants to add a note to the session THEN the system SHALL add the note to the session's notes
- WHEN a user wants to set the initiative order THEN the system SHALL set the initiative order for the session
- WHEN a user wants to add a monster to the session THEN the system SHALL add the monster to the session's monster list
- WHEN a user wants to update a monster's health THEN the system SHALL update the monster's health in the session's monster list
- WHEN a user wants to view the session data THEN the system SHALL display the session's notes, initiative order, and monster list

### REQ-009: Flavor Source Support
**User Story:** As a user, I want to add non-rulebook source material to the system, so that I can enhance the narrative and immersive aspects of the game.

**Acceptance Criteria:**
- WHEN a user adds a new source THEN they SHALL be able to specify whether it is a "rulebook" or "flavor" source
- WHEN a user generates a character backstory or an NPC THEN the system SHALL use the "flavor" sources to inform the generation process
- WHEN a user searches for information THEN they SHALL be able to filter the results by source type

### REQ-010: Advanced Search Capabilities
**User Story:** As a user, I want advanced search capabilities with hybrid semantic and keyword search, so that I can find information even with imprecise queries.

**Acceptance Criteria:**
- WHEN a user performs a search THEN the system SHALL use both semantic similarity and keyword matching for comprehensive results
- WHEN a search query is ambiguous THEN the system SHALL provide query suggestions and completion
- WHEN a user wants to explain search results THEN the system SHALL provide detailed explanations of relevance factors
- WHEN a user wants search statistics THEN the system SHALL provide analytics about the search service performance

### REQ-011: Adaptive PDF Processing
**User Story:** As a user, I want adaptive PDF processing that learns content patterns, so that the system becomes more accurate over time.

**Acceptance Criteria:**
- WHEN processing PDFs THEN the system SHALL learn content type patterns and improve classification accuracy
- WHEN processing multiple PDFs from the same system THEN the system SHALL reuse learned patterns for better parsing
- WHEN adaptive learning is enabled THEN the system SHALL cache learned patterns for future use
- WHEN processing content THEN the system SHALL provide statistics about learned patterns

### REQ-012: Web UI Access with AI Provider Integration
**User Story:** As a user, I want to access the TTRPG Assistant through a web interface using my own AI provider account, so that I can use the tools without installing custom desktop applications.

**Acceptance Criteria:**
- WHEN a user accesses the web UI THEN they SHALL be able to authenticate with their own AI provider credentials (Anthropic, OpenAI, Google Gemini)
- WHEN authenticated THEN the system SHALL validate and store the provider credentials securely
- WHEN using the UI THEN the user SHALL be able to switch between different AI providers during a session
- IF a provider fails THEN the system SHALL offer automatic fallback to alternative providers
- WHEN multiple providers are configured THEN the system SHALL optimize for cost and performance based on user preferences

### REQ-013: MCP Bridge Service
**User Story:** As a developer, I want a bridge service that connects the web UI to the stdio-based MCP server, so that we maintain the reliability of local operations while enabling web access.

**Acceptance Criteria:**
- WHEN the bridge service starts THEN it SHALL spawn dedicated MCP server processes per user session
- WHEN receiving HTTP/SSE requests THEN the bridge SHALL translate them to stdio commands for the MCP server
- WHEN the MCP server responds THEN the bridge SHALL stream results back to the UI in real-time
- IF a process crashes THEN the bridge SHALL automatically restart it with context recovery
- WHEN managing sessions THEN the bridge SHALL enforce resource limits and session timeouts

### REQ-014: Multi-User Collaborative Sessions
**User Story:** As a Game Master, I want to run collaborative sessions where multiple players can connect and interact, so that we can play together remotely.

**Acceptance Criteria:**
- WHEN creating a session THEN the GM SHALL be able to invite other players via shareable links
- WHEN players join THEN they SHALL see real-time updates of game state and AI responses
- WHEN multiple users are connected THEN the system SHALL broadcast changes to all participants
- IF a player disconnects THEN their state SHALL be preserved for reconnection
- WHEN players interact THEN the system SHALL maintain proper turn order and permissions

### REQ-015: Context Persistence and Management
**User Story:** As a user, I want my conversation context and game state to persist across sessions, so that I can resume where I left off.

**Acceptance Criteria:**
- WHEN a session ends THEN the system SHALL save all context including conversation history, tool results, and game state
- WHEN resuming a session THEN the system SHALL restore the complete context within 5 seconds
- WHEN switching AI providers THEN the system SHALL translate context to the new provider's format
- IF context grows large THEN the system SHALL implement intelligent compression and pruning
- WHEN accessing old sessions THEN the system SHALL provide search and filtering capabilities

### REQ-016: Tool Result Visualization
**User Story:** As a player, I want rich visualizations for tool results, so that I can better understand game information and state.

**Acceptance Criteria:**
- WHEN displaying character sheets THEN the UI SHALL render them in an interactive, visual format
- WHEN showing dice rolls THEN the UI SHALL provide animated representations
- WHEN presenting maps or locations THEN the UI SHALL offer interactive exploration capabilities
- WHEN displaying tables THEN the UI SHALL provide sortable, filterable data grids
- IF visualization fails THEN the UI SHALL gracefully degrade to text representation

### REQ-017: Security and Authentication
**User Story:** As an administrator, I want robust security for the web interface, so that user data and AI credentials are protected.

**Acceptance Criteria:**
- WHEN users authenticate THEN the system SHALL support multiple methods (API keys, OAuth, JWT)
- WHEN storing credentials THEN the system SHALL use industry-standard encryption
- WHEN creating sessions THEN each user SHALL get an isolated MCP process with restricted permissions
- WHEN accessing tools THEN the system SHALL enforce granular permission controls
- IF suspicious activity is detected THEN the system SHALL implement rate limiting and alerting

### REQ-018: Performance Optimization and Caching
**User Story:** As a user, I want fast response times even with multiple concurrent users, so that gameplay remains fluid.

**Acceptance Criteria:**
- WHEN multiple users access the same data THEN the system SHALL use intelligent caching to reduce redundant operations
- WHEN AI responses are similar THEN the system SHALL implement response caching with context hashing
- WHEN load increases THEN the system SHALL automatically scale MCP server processes
- IF response time exceeds thresholds THEN the system SHALL implement predictive prefetching
- WHEN monitoring performance THEN the system SHALL maintain < 50ms context retrieval for 95th percentile

### REQ-019: Cost Optimization for AI Providers
**User Story:** As a user, I want the system to optimize AI provider usage based on cost and performance, so that I can manage expenses.

**Acceptance Criteria:**
- WHEN multiple providers are available THEN the system SHALL estimate costs for each request
- WHEN cost optimization is enabled THEN the system SHALL route to the most cost-effective provider
- WHEN providers have different strengths THEN the system SHALL match provider to task type
- IF a user sets budget limits THEN the system SHALL track and enforce spending caps
- WHEN providing cost data THEN the system SHALL show real-time usage and projection analytics

### REQ-020: Responsive Web Design
**User Story:** As a player, I want to access the TTRPG Assistant from any device (desktop, tablet, or mobile), so that I can play from anywhere with a consistent experience.

**Acceptance Criteria:**
- WHEN accessing from any device THEN the UI SHALL adapt responsively to the screen size
- WHEN using mobile devices THEN touch interactions SHALL be properly supported
- WHEN offline THEN core functionality SHALL remain available through service workers
- IF on a slow connection THEN the app SHALL provide progressive loading and feedback
- WHEN using touch interfaces THEN the system SHALL provide appropriate touch controls
- WHEN bandwidth is limited THEN the system SHALL implement progressive loading
- IF connection is unstable THEN the system SHALL provide basic offline read-only access to cached data (full offline sync deferred to future release)
- WHEN switching devices THEN the system SHALL maintain session continuity

**Note on Offline Capabilities:** 
- **Initial Release (v1.0)**: Basic read-only access to recently viewed content cached in browser storage
- **Future Release (v2.0)**: Full offline synchronization with conflict resolution will be designed and implemented as a separate major feature, requiring dedicated architecture documentation

## Non-Functional Requirements

### NFR-001: Technology Stack
- The system SHALL use Python as the primary programming language
- The system SHALL use an embeddable database (ChromaDB recommended) rather than external databases
- The system SHALL focus on local MCP operations over stdin/stdout rather than TCP/HTTP

### NFR-002: Content Processing
- The system SHALL preserve tabular information with structure preservation
- The system SHALL display tables in a concise and easy to understand way
- The system SHALL highlight rows that are pertinent to the query

### NFR-003: Metadata Extraction
- The system SHALL extract vernacular terms and system-specific terminology
- The system SHALL identify writing style patterns (formal vs. casual, mysterious vs. straightforward)
- The system SHALL detect tone patterns (authoritative, scholarly, conspiratorial, etc.)
- The system SHALL recognize perspective (first-person, instructional, omniscient)
- The system SHALL collect common expressions and speech patterns

### NFR-004: Performance
- The system SHALL provide search results within reasonable response times for interactive gameplay
- The system SHALL efficiently handle large PDF documents and multiple rulebooks
- The system SHALL cache processed data for faster subsequent access

### NFR-005: Frontend Technology Stack (Updated 2024)
- The system SHALL use SvelteKit as the frontend framework for optimal performance and developer experience
- The system SHALL implement server-side rendering (SSR) for improved SEO and initial load times
- The system SHALL use TypeScript for type safety throughout the frontend codebase
- The system SHALL follow responsive web design principles (no separate mobile app)
- The system SHALL use native Svelte stores for state management

### NFR-006: Python Modernization (Updated 2024)
- The system SHALL use modern, maintained Python packages (pypdf instead of PyPDF2, httpx instead of requests)
- The system SHALL implement error-as-values pattern using Result types for graceful error handling
- The system SHALL use structured logging with correlation IDs for debugging
- The system SHALL implement retry logic with exponential backoff for external services
- The system SHALL use type hints and static type checking with mypy

## Desktop Application Requirements (Phase 23)

### Functional Requirements

#### REQ-021: Desktop Application Core
- The system SHALL provide a standalone desktop application for Windows, macOS, and Linux
- The desktop app SHALL embed the Python MCP server as a subprocess
- The desktop app SHALL reuse the existing SvelteKit frontend in a WebView
- The desktop app SHALL operate fully offline without internet connectivity
- The desktop app SHALL provide native file system access with appropriate sandboxing

#### REQ-022: Installation and Distribution
- The system SHALL provide platform-specific installers (MSI/NSIS for Windows, DMG for macOS, AppImage/deb/rpm for Linux)
- The installer SHALL bundle all required dependencies including Python runtime
- The application SHALL support auto-updates with user consent
- The installation package SHALL be under 70MB total size
- The system SHALL support portable/no-install mode for USB deployment

#### REQ-023: Process Management
- The desktop app SHALL manage the Python MCP server lifecycle (start, stop, restart)
- The system SHALL handle process crashes gracefully with automatic restart
- The app SHALL monitor resource usage and provide alerts for high consumption
- The system SHALL support running multiple isolated sessions
- The app SHALL clean up all resources on exit

#### REQ-024: Native Integration
- The system SHALL integrate with the OS system tray for background operation
- The app SHALL register file associations for .ttrpg project files
- The system SHALL support drag-and-drop for PDF import
- The app SHALL provide native file dialogs for better UX
- The system SHALL integrate with OS notifications

#### REQ-025: Data Management
- The desktop app SHALL store all data locally in the user's data directory
- The system SHALL provide data export/import functionality
- The app SHALL support automatic backups with configurable retention
- The system SHALL migrate data from web version if requested
- The app SHALL encrypt sensitive data using OS keychain services

#### REQ-026: Performance Requirements
- The desktop app SHALL start in under 2 seconds on modern hardware
- The app SHALL use less than 150MB RAM when idle
- The system SHALL maintain sub-5ms IPC latency between frontend and backend
- The app SHALL support PDF processing of 100+ page documents locally
- The system SHALL handle 10,000+ indexed documents efficiently

### Non-Functional Requirements

#### NFR-007: Desktop Security
- The app SHALL run the Python process with minimal required permissions
- The system SHALL validate all IPC messages between components
- The app SHALL implement CSP headers in the WebView
- The system SHALL use OS-native credential storage for API keys
- The app SHALL log security events for audit purposes

#### NFR-008: Desktop User Experience
- The app SHALL provide a native look and feel on each platform
- The system SHALL support keyboard shortcuts for common operations
- The app SHALL remember window size and position between sessions
- The system SHALL provide smooth animations and transitions
- The app SHALL support both light and dark themes with OS sync

#### NFR-009: Desktop Reliability
- The app SHALL handle network disconnections gracefully
- The system SHALL recover from subprocess crashes automatically
- The app SHALL preserve user data during unexpected shutdowns
- The system SHALL validate data integrity on startup
- The app SHALL provide diagnostic tools for troubleshooting

#### NFR-010: Desktop Maintainability
- The app SHALL share 95% of frontend code with web version
- The system SHALL use single codebase for all desktop platforms
- The app SHALL provide debug logging with configurable levels
- The system SHALL support remote diagnostics with user consent
- The app SHALL track usage metrics locally (with opt-out)