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