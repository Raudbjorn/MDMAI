# MCP Tools Specification for TTRPG Assistant

## Overview
This document specifies the comprehensive set of MCP (Model Context Protocol) tools for the TTRPG Assistant server. These tools are designed to provide complete support for Dungeon Masters and Game Runners during live gaming sessions, covering rules lookup, campaign management, character generation, combat automation, and real-time collaboration features.

## Tool Categories

### 1. Rules and Content Search Tools

#### search_rules
**Description**: Advanced semantic and keyword search across all loaded rulebooks with campaign context awareness.

**Parameters**:
```typescript
{
  query: string;                    // Search query (required)
  search_type?: "semantic" | "keyword" | "hybrid" | "exact";  // Default: "hybrid"
  systems?: string[];               // Filter by game systems (e.g., ["D&D 5e", "Pathfinder"])
  source_types?: ("rulebook" | "flavor" | "campaign")[];  // Filter by source type
  campaign_id?: string;             // Include campaign-specific context
  include_tables?: boolean;         // Include table data in results (default: true)
  include_examples?: boolean;       // Include rule examples (default: true)
  max_results?: number;            // Maximum results to return (default: 10)
  page_range?: {start: number, end: number};  // Limit to page range
  relevance_threshold?: number;    // Minimum relevance score (0-1, default: 0.5)
}
```

**Returns**:
```typescript
{
  success: boolean;
  results: Array<{
    content: string;
    source: string;
    page_number: number;
    relevance_score: number;
    context: string;
    tables?: Array<TableData>;
    examples?: string[];
    campaign_references?: Array<CampaignReference>;
  }>;
  query_suggestions?: string[];
  total_matches: number;
}
```

**Example Usage**:
```javascript
await search_rules({
  query: "grappling rules",
  search_type: "hybrid",
  systems: ["D&D 5e"],
  campaign_id: "campaign_123",
  include_examples: true
});
```

#### get_spell
**Description**: Retrieve detailed spell information with cross-referencing and campaign-specific modifications.

**Parameters**:
```typescript
{
  spell_name: string;              // Spell name (required)
  system?: string;                 // Game system (default: current campaign system)
  level?: number;                  // Filter by spell level
  school?: string;                 // Filter by magic school
  class_list?: string[];           // Filter by classes that can cast
  include_variants?: boolean;      // Include homebrew/variant versions
  campaign_id?: string;            // Check for campaign-specific modifications
  format?: "full" | "summary" | "card";  // Output format (default: "full")
}
```

**Returns**:
```typescript
{
  success: boolean;
  spell: {
    name: string;
    level: number;
    school: string;
    casting_time: string;
    range: string;
    components: {
      verbal: boolean;
      somatic: boolean;
      material: string;
    };
    duration: string;
    description: string;
    at_higher_levels?: string;
    classes: string[];
    source: string;
    page: number;
    campaign_modifications?: Array<Modification>;
    similar_spells?: Array<{name: string; similarity: number}>;
  };
}
```

#### get_monster
**Description**: Retrieve complete monster statistics with encounter scaling and tactical suggestions.

**Parameters**:
```typescript
{
  monster_name: string;            // Monster name (required)
  system?: string;                 // Game system
  cr_range?: {min: number, max: number};  // Challenge rating range
  type?: string;                   // Monster type (e.g., "dragon", "undead")
  party_level?: number;            // Party level for scaling suggestions
  party_size?: number;             // Party size for encounter balance
  environment?: string;            // Environment context
  include_tactics?: boolean;       // Include tactical suggestions (default: true)
  include_loot?: boolean;         // Include loot tables (default: true)
  campaign_id?: string;            // Check for campaign-specific versions
}
```

**Returns**:
```typescript
{
  success: boolean;
  monster: {
    name: string;
    size: string;
    type: string;
    alignment: string;
    ac: number;
    hp: {average: number; formula: string};
    speed: Record<string, string>;
    stats: {
      str: number;
      dex: number;
      con: number;
      int: number;
      wis: number;
      cha: number;
    };
    saves?: Record<string, number>;
    skills?: Record<string, number>;
    damage_resistances?: string[];
    damage_immunities?: string[];
    condition_immunities?: string[];
    senses: string[];
    languages: string[];
    cr: string;
    xp: number;
    abilities: Array<{name: string; description: string}>;
    actions: Array<{name: string; description: string}>;
    legendary_actions?: Array<{name: string; description: string}>;
    lair_actions?: Array<{description: string}>;
    tactics?: {
      opening_moves: string[];
      combat_strategy: string;
      retreat_conditions: string;
    };
    loot?: Array<{item: string; chance: number}>;
    encounter_difficulty?: string;
    scaling_suggestions?: string[];
  };
}
```

### 2. Dice and Randomization Tools

#### roll_dice
**Description**: Advanced dice roller with modifiers, advantage/disadvantage, and roll history.

**Parameters**:
```typescript
{
  expression: string;              // Dice expression (e.g., "3d6+2", "1d20+5")
  advantage?: boolean;             // Roll with advantage
  disadvantage?: boolean;          // Roll with disadvantage
  reroll_ones?: boolean;          // Reroll natural 1s
  drop_lowest?: number;           // Drop N lowest dice
  drop_highest?: number;          // Drop N highest dice
  exploding?: boolean;            // Exploding dice (reroll max values)
  target?: number;                // Target number for success counting
  label?: string;                 // Label for the roll
  character_id?: string;          // Associate with character
  save_to_log?: boolean;          // Save to session log (default: true)
}
```

**Returns**:
```typescript
{
  success: boolean;
  result: {
    total: number;
    rolls: number[];
    expression: string;
    modified_total?: number;
    dropped?: number[];
    exploded?: number[];
    successes?: number;
    critical?: boolean;
    fumble?: boolean;
    label?: string;
    timestamp: string;
  };
}
```

#### generate_random_table
**Description**: Generate results from random tables or create custom random tables.

**Parameters**:
```typescript
{
  table_name?: string;             // Predefined table name
  custom_table?: Array<{          // Custom table definition
    weight: number;
    result: string;
    subtable?: string;
  }>;
  rolls?: number;                  // Number of rolls (default: 1)
  unique?: boolean;               // Ensure unique results
  modifiers?: Record<string, number>;  // Dice modifiers
}
```

### 3. Combat Management Tools

#### manage_initiative
**Description**: Comprehensive initiative tracker with conditions and effects management.

**Parameters**:
```typescript
{
  action: "start" | "add" | "remove" | "next" | "previous" | "sort" | "clear" | "delay" | "ready";
  session_id: string;              // Session ID (required)
  combatant?: {
    name: string;
    initiative: number;
    type: "player" | "npc" | "monster" | "ally";
    hp?: {current: number; max: number};
    ac?: number;
    conditions?: string[];
    id?: string;
  };
  target_position?: number;        // For delay/ready actions
}
```

**Returns**:
```typescript
{
  success: boolean;
  initiative_order: Array<{
    position: number;
    name: string;
    initiative: number;
    type: string;
    current_turn: boolean;
    conditions: string[];
    hp_percentage?: number;
    rounds_until_effect_end?: Record<string, number>;
  }>;
  current_round: number;
  current_turn: string;
}
```

#### track_damage
**Description**: Track damage, healing, and temporary hit points with death saves.

**Parameters**:
```typescript
{
  target_id: string;               // Target combatant ID
  amount: number;                  // Damage (positive) or healing (negative)
  damage_type?: string;            // Type of damage
  is_critical?: boolean;           // Critical hit
  resistance?: boolean;            // Target has resistance
  vulnerability?: boolean;         // Target has vulnerability
  temp_hp?: number;               // Temporary hit points to add
  death_save?: "success" | "failure" | "critical";  // Death save result
}
```

#### apply_condition
**Description**: Apply and track conditions with automatic effect reminders.

**Parameters**:
```typescript
{
  target_id: string;               // Target combatant ID
  condition: string;               // Condition name
  duration?: number;               // Duration in rounds
  save_dc?: number;                // Save DC if applicable
  save_type?: string;              // Save type (e.g., "CON", "WIS")
  source?: string;                 // Source of condition
}
```

### 4. Campaign Management Tools

#### create_campaign
**Description**: Create a new campaign with automatic rulebook linking.

**Parameters**:
```typescript
{
  name: string;                    // Campaign name (required)
  system: string;                  // Game system (required)
  description?: string;            // Campaign description
  setting?: string;                // Campaign setting
  players?: Array<{                // Initial player list
    name: string;
    character_id?: string;
  }>;
  homebrew_rules?: string[];       // List of homebrew rules
  safety_tools?: string[];         // X-card, lines/veils, etc.
}
```

#### manage_campaign_timeline
**Description**: Track and manage campaign timeline and events.

**Parameters**:
```typescript
{
  action: "add_event" | "update_event" | "remove_event" | "advance_time" | "get_timeline";
  campaign_id: string;
  event?: {
    date: string;                  // In-game date
    title: string;
    description: string;
    type: "quest" | "combat" | "roleplay" | "milestone" | "downtime";
    participants?: string[];
    location?: string;
  };
  time_advancement?: {
    amount: number;
    unit: "hours" | "days" | "weeks" | "months" | "years";
  };
}
```

#### track_campaign_resources
**Description**: Track party resources, wealth, and inventory.

**Parameters**:
```typescript
{
  campaign_id: string;
  resource_type: "gold" | "items" | "consumables" | "quest_items" | "reputation";
  action: "add" | "remove" | "transfer" | "list";
  resource?: {
    name: string;
    quantity: number;
    owner?: string;                // Character or party
    description?: string;
    value?: number;
  };
}
```

### 5. Character and NPC Tools

#### generate_character
**Enhanced from existing tool with additional parameters**:
```typescript
{
  system: string;
  level: number;
  character_class?: string;
  race?: string;
  background?: string;              // Character background
  alignment?: string;               // Alignment
  ability_scores?: "random" | "standard_array" | "point_buy" | "manual";
  equipment?: "starting" | "random" | "by_wealth" | "custom";
  personality?: {
    traits: string[];
    ideals: string[];
    bonds: string[];
    flaws: string[];
  };
  appearance?: {
    age?: number;
    height?: string;
    weight?: string;
    appearance_notes?: string;
  };
  relationships?: Array<{
    character_id: string;
    relationship_type: string;
  }>;
}
```

#### generate_npc_batch
**Description**: Generate multiple NPCs for a location or encounter.

**Parameters**:
```typescript
{
  count: number;                   // Number of NPCs to generate
  location?: string;               // Location context
  theme?: string;                  // Theme (e.g., "tavern staff", "city guards")
  diversity?: "low" | "medium" | "high";  // Variety in generated NPCs
  include_relationships?: boolean;  // Generate relationships between NPCs
  importance_distribution?: {       // Distribution of importance levels
    minor: number;
    supporting: number;
    major: number;
  };
}
```

### 6. Location and Map Tools

#### generate_location
**Description**: Generate detailed locations with inhabitants and points of interest.

**Parameters**:
```typescript
{
  type: string;                    // Location type (e.g., "tavern", "dungeon", "city")
  size?: "tiny" | "small" | "medium" | "large" | "huge";
  theme?: string;                  // Thematic elements
  danger_level?: "safe" | "low" | "moderate" | "high" | "deadly";
  include_npcs?: boolean;          // Generate NPCs for location
  include_loot?: boolean;          // Generate loot/treasure
  include_secrets?: boolean;       // Include secret areas/information
  campaign_id?: string;            // Link to campaign
}
```

**Returns**:
```typescript
{
  success: boolean;
  location: {
    name: string;
    type: string;
    description: string;
    atmosphere: string;
    notable_features: string[];
    npcs?: Array<NPCReference>;
    rooms?: Array<{
      name: string;
      description: string;
      contents: string[];
      exits: string[];
      secrets?: string[];
    }>;
    loot?: Array<TreasureItem>;
    plot_hooks?: string[];
    random_encounters?: Array<{
      chance: number;
      description: string;
    }>;
  };
}
```

#### manage_map_markers
**Description**: Manage markers and notes on maps.

**Parameters**:
```typescript
{
  action: "add" | "update" | "remove" | "list";
  map_id: string;
  marker?: {
    id?: string;
    type: "location" | "npc" | "event" | "treasure" | "danger" | "note";
    coordinates?: {x: number; y: number};
    label: string;
    description?: string;
    visibility: "all" | "gm_only" | "discovered";
  };
}
```

### 7. Story and Plot Tools

#### generate_plot_hook
**Description**: Generate plot hooks based on campaign context.

**Parameters**:
```typescript
{
  theme?: string;                  // Theme or genre
  urgency?: "low" | "medium" | "high" | "immediate";
  scope?: "personal" | "local" | "regional" | "global";
  involves?: string[];             // NPCs or locations to involve
  campaign_id?: string;            // Use campaign context
  complexity?: "simple" | "moderate" | "complex";
}
```

#### manage_quest_tracker
**Description**: Track quests, objectives, and progress.

**Parameters**:
```typescript
{
  action: "create" | "update" | "complete" | "fail" | "list";
  quest?: {
    id?: string;
    title: string;
    description: string;
    objectives: Array<{
      description: string;
      completed: boolean;
      optional?: boolean;
    }>;
    rewards?: string[];
    deadline?: string;
    quest_giver?: string;
    status?: "active" | "completed" | "failed" | "abandoned";
  };
  campaign_id: string;
}
```

#### generate_plot_twist
**Description**: Generate contextual plot twists for the story.

**Parameters**:
```typescript
{
  current_situation: string;       // Brief description of current situation
  twist_type?: "betrayal" | "revelation" | "reversal" | "complication" | "random";
  involves_npc?: string;           // Specific NPC involvement
  severity?: "minor" | "moderate" | "major";
}
```

### 8. Session Management Tools

#### take_notes
**Enhanced note-taking with AI categorization and tagging**:
```typescript
{
  session_id: string;
  content: string;
  category?: "general" | "combat" | "roleplay" | "loot" | "quest" | "rules" | "npc";
  tags?: string[];
  timestamp?: string;
  is_secret?: boolean;             // GM-only note
  related_entities?: Array<{       // Link to characters, NPCs, locations
    type: string;
    id: string;
  }>;
  auto_categorize?: boolean;       // Use AI to categorize
}
```

#### generate_session_recap
**Description**: Generate a recap of the previous session.

**Parameters**:
```typescript
{
  session_id?: string;             // Specific session or latest
  style?: "narrative" | "bullet_points" | "newspaper" | "bardic_tale";
  perspective?: "neutral" | "character" | "npc";
  include_secrets?: boolean;       // Include GM-only information
  length?: "brief" | "standard" | "detailed";
}
```

#### plan_session
**Description**: Help plan the next session with prep assistance.

**Parameters**:
```typescript
{
  campaign_id: string;
  session_type?: "combat_heavy" | "roleplay_heavy" | "exploration" | "mixed";
  expected_duration?: number;      // Hours
  player_goals?: string[];         // Known player objectives
  gm_goals?: string[];            // GM's intended story beats
  required_prep?: Array<{          // Things to prepare
    type: "npcs" | "locations" | "encounters" | "handouts";
    details: string;
  }>;
}
```

### 9. Real-time Collaboration Tools

#### broadcast_to_players
**Description**: Send information to all connected players.

**Parameters**:
```typescript
{
  session_id: string;
  message_type: "narration" | "description" | "rule" | "image" | "handout";
  content: string;
  target?: "all" | "specific" | string[];  // Target players
  reveal_type?: "immediate" | "gradual" | "on_trigger";
}
```

#### request_player_action
**Description**: Request specific actions from players.

**Parameters**:
```typescript
{
  session_id: string;
  player_id: string;
  action_type: "roll" | "decision" | "roleplay" | "skill_check";
  prompt: string;
  options?: string[];              // For decision types
  timeout?: number;                // Seconds before auto-skip
  is_secret?: boolean;             // Hidden from other players
}
```

#### sync_game_state
**Description**: Synchronize game state across all clients.

**Parameters**:
```typescript
{
  session_id: string;
  state_type: "initiative" | "map" | "resources" | "conditions" | "all";
  data?: any;                      // State-specific data
  force_update?: boolean;          // Override client caches
}
```

### 10. Audio/Visual Enhancement Tools

#### play_ambience
**Description**: Control ambient music and sound effects.

**Parameters**:
```typescript
{
  action: "play" | "pause" | "stop" | "crossfade";
  track?: string;                  // Track identifier or URL
  category?: "combat" | "exploration" | "town" | "tavern" | "mystery" | "boss";
  volume?: number;                 // 0-100
  loop?: boolean;
  fade_duration?: number;          // Seconds for crossfade
}
```

#### display_image
**Description**: Display images, maps, or handouts to players.

**Parameters**:
```typescript
{
  image_url: string;
  display_type: "fullscreen" | "window" | "overlay";
  target?: "all" | string[];       // Target players
  duration?: number;               // Auto-hide after seconds
  allow_zoom?: boolean;
  allow_annotations?: boolean;      // Let players draw on it
}
```

## Integration with Real-time Features

### WebSocket Events
All tools that modify game state will automatically emit WebSocket events to connected clients:

```typescript
// Event structure
{
  event: "tool_execution";
  tool: string;                    // Tool name
  result: any;                     // Tool result
  affects: string[];               // Affected entity IDs
  broadcast: boolean;              // Should broadcast to players
  timestamp: string;
}
```

### State Synchronization
Tools automatically sync state through:
1. **Optimistic Updates**: Immediate local state updates
2. **Server Confirmation**: Authoritative state from server
3. **Conflict Resolution**: Last-write-wins with version vectors
4. **Delta Compression**: Only changed fields transmitted

### Caching Strategy
```typescript
{
  cache_layers: [
    "memory",       // In-memory cache for active session
    "indexeddb",    // Browser storage for offline access
    "server"        // Server-side cache for shared data
  ],
  invalidation: {
    entity_based: true,     // Invalidate by entity ID
    time_based: true,       // TTL for different data types
    event_based: true       // Invalidate on specific events
  }
}
```

## Performance Considerations

### Tool Batching
Multiple tool calls can be batched for efficiency:
```typescript
await batchTools([
  {tool: "roll_dice", params: {expression: "1d20+5"}},
  {tool: "get_monster", params: {monster_name: "Goblin"}},
  {tool: "take_notes", params: {content: "Combat started"}}
]);
```

### Prefetching
Common sequences are prefetched:
- Monster stats → Likely combat actions
- Spell lookup → Spell slot tracking
- Location entry → Inhabitant NPCs

### Progressive Loading
Large results support pagination:
```typescript
{
  page: 1,
  per_page: 20,
  total: 156,
  has_more: true,
  continuation_token: "..."
}
```

## Security and Permissions

### Tool Access Control
```typescript
{
  role_permissions: {
    gm: ["*"],                     // All tools
    player: [
      "search_rules",
      "get_spell",
      "roll_dice",
      "get_character",
      // ... limited set
    ],
    spectator: [
      "search_rules",
      "get_*"                      // Read-only tools
    ]
  }
}
```

### Rate Limiting
```typescript
{
  rate_limits: {
    search_rules: "10/minute",
    generate_*: "5/minute",
    roll_dice: "60/minute",
    broadcast_*: "20/minute"
  }
}
```

## Error Handling

All tools follow consistent error response format:
```typescript
{
  success: false;
  error: {
    code: string;                  // Error code
    message: string;               // Human-readable message
    details?: any;                 // Additional context
    suggestions?: string[];        // Possible fixes
    retry_after?: number;          // For rate limits
  };
}
```

## Extensibility

### Custom Tool Registration
```typescript
mcp_server.register_tool({
  name: "custom_tool",
  description: "Custom tool description",
  parameters: {...},
  handler: async (params) => {...},
  permissions: ["gm"],
  cache: {ttl: 300, key: (p) => `${p.id}`}
});
```

### Tool Middleware
```typescript
mcp_server.use_tool_middleware(async (tool, params, next) => {
  // Pre-processing
  console.log(`Executing ${tool} with params:`, params);
  
  const result = await next();
  
  // Post-processing
  await audit_log(tool, params, result);
  
  return result;
});
```

## Implementation Priority

### Phase 1 (Core - Immediate)
- search_rules (enhanced)
- get_spell (enhanced)
- get_monster (enhanced)
- roll_dice (enhanced)
- manage_initiative
- take_notes (enhanced)

### Phase 2 (Campaign - Week 1)
- create_campaign
- manage_campaign_timeline
- track_campaign_resources
- generate_character (enhanced)
- generate_npc (enhanced)

### Phase 3 (Session - Week 2)
- track_damage
- apply_condition
- generate_session_recap
- plan_session
- manage_quest_tracker

### Phase 4 (Real-time - Week 3)
- broadcast_to_players
- request_player_action
- sync_game_state
- display_image

### Phase 5 (Advanced - Week 4+)
- generate_location
- generate_plot_hook
- generate_plot_twist
- play_ambience
- generate_npc_batch
- manage_map_markers

## Testing Requirements

Each tool must include:
1. **Unit tests**: Parameter validation, core logic
2. **Integration tests**: Database operations, cross-tool interactions
3. **Performance tests**: Response time < 200ms for 95th percentile
4. **Load tests**: Support 100 concurrent sessions
5. **Security tests**: Permission enforcement, input sanitization

## Documentation Requirements

Each tool must document:
1. **Purpose**: Clear description of functionality
2. **Parameters**: All parameters with types and constraints
3. **Returns**: Complete return structure
4. **Examples**: At least 2 usage examples
5. **Errors**: Possible error conditions
6. **Performance**: Expected response times
7. **Permissions**: Required user roles
8. **Side Effects**: Other tools or state affected