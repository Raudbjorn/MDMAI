# Spike 6: Third-party Integration

## Overview
This spike explores integration strategies for connecting the TTRPG MCP Server with external platforms, AI providers, and content services.

## 1. Integration Architecture

### 1.1 MCP Gateway Pattern
```typescript
// MCP Integration Gateway
interface IntegrationGateway {
  provider: string;
  authenticate(): Promise<AuthToken>;
  executeCommand(command: IntegrationCommand): Promise<Result>;
  handleWebhook(payload: WebhookPayload): Promise<void>;
}

// Integration Registry
class IntegrationRegistry {
  private integrations = new Map<string, IntegrationGateway>();
  
  register(provider: string, gateway: IntegrationGateway) {
    this.integrations.set(provider, gateway);
  }
  
  async execute(provider: string, command: IntegrationCommand) {
    const gateway = this.integrations.get(provider);
    if (!gateway) {
      return { ok: false, error: new Error(`Unknown provider: ${provider}`) };
    }
    return gateway.executeCommand(command);
  }
}
```

### 1.2 OAuth2 Flow Manager
```python
from fastapi import FastAPI, HTTPException
from authlib.integrations.fastapi_client import OAuth
import httpx

class OAuth2Manager:
    def __init__(self):
        self.oauth = OAuth()
        self.configure_providers()
    
    def configure_providers(self):
        # D&D Beyond OAuth
        self.oauth.register(
            name='dndbeyond',
            client_id=os.getenv('DDB_CLIENT_ID'),
            client_secret=os.getenv('DDB_CLIENT_SECRET'),
            authorize_url='https://www.dndbeyond.com/oauth/authorize',
            access_token_url='https://www.dndbeyond.com/oauth/token',
            api_base_url='https://api.dndbeyond.com/v1/'
        )
        
        # Roll20 OAuth
        self.oauth.register(
            name='roll20',
            client_id=os.getenv('ROLL20_CLIENT_ID'),
            client_secret=os.getenv('ROLL20_CLIENT_SECRET'),
            authorize_url='https://app.roll20.net/oauth/authorize',
            access_token_url='https://app.roll20.net/oauth/token',
            api_base_url='https://app.roll20.net/api/'
        )
```

## 2. TTRPG Platform Integrations

### 2.1 D&D Beyond Integration
```python
# MCP Tool for D&D Beyond
@mcp.tool()
async def import_dndbeyond_character(
    character_url: str,
    access_token: str
) -> Dict[str, Any]:
    """Import a character from D&D Beyond"""
    # Extract character ID from URL
    character_id = extract_character_id(character_url)
    
    # Fetch character data
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.dndbeyond.com/character/{character_id}",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
    character_data = response.json()
    
    # Convert to universal format
    return {
        "name": character_data["name"],
        "class": character_data["classes"][0]["definition"]["name"],
        "level": character_data["classes"][0]["level"],
        "race": character_data["race"]["fullName"],
        "stats": {
            "str": character_data["stats"][0]["value"],
            "dex": character_data["stats"][1]["value"],
            "con": character_data["stats"][2]["value"],
            "int": character_data["stats"][3]["value"],
            "wis": character_data["stats"][4]["value"],
            "cha": character_data["stats"][5]["value"]
        },
        "hp": {
            "current": character_data["currentHp"],
            "max": character_data["maxHp"]
        },
        "ac": character_data["armorClass"],
        "inventory": character_data["inventory"],
        "spells": character_data["spells"]
    }
```

### 2.2 Roll20 API Integration
```typescript
// Roll20 Integration Service
export class Roll20Integration implements IntegrationGateway {
  provider = 'roll20';
  private api: Roll20API;
  
  async authenticate(): Promise<AuthToken> {
    // OAuth2 flow for Roll20
    return await this.oauth.authenticate('roll20');
  }
  
  async syncCampaign(campaignId: string): Promise<Result<Campaign>> {
    const campaign = await this.api.getCampaign(campaignId);
    
    // Sync to MCP
    const mcpCampaign = {
      name: campaign.name,
      system: campaign.tags.includes('dnd5e') ? 'D&D 5e' : 'Generic',
      players: campaign.players.map(p => ({
        id: p.id,
        name: p.displayname,
        character: p.characters[0]
      })),
      maps: campaign.pages.map(page => ({
        id: page.id,
        name: page.name,
        width: page.width * 70,  // Convert to pixels
        height: page.height * 70,
        background: page.imgsrc,
        tokens: page.graphics.filter(g => g.represents)
      }))
    };
    
    return { ok: true, value: mcpCampaign };
  }
  
  async sendToVTT(action: VTTAction): Promise<Result<void>> {
    switch (action.type) {
      case 'roll':
        await this.api.sendChat(action.expression);
        break;
      case 'move_token':
        await this.api.moveToken(action.tokenId, action.position);
        break;
      case 'update_hp':
        await this.api.updateBar(action.tokenId, 1, action.value);
        break;
    }
    return { ok: true, value: undefined };
  }
}
```

### 2.3 Foundry VTT Module
```javascript
// Foundry VTT Module - mcp-bridge.js
Hooks.once('init', async function() {
  game.settings.register('mcp-bridge', 'serverUrl', {
    name: 'MCP Server URL',
    hint: 'URL of your MCP bridge server',
    scope: 'world',
    config: true,
    type: String,
    default: 'ws://localhost:8765'
  });
  
  // Initialize WebSocket connection
  game.mcp = new MCPBridge();
  await game.mcp.connect();
});

class MCPBridge {
  constructor() {
    this.ws = null;
    this.handlers = new Map();
  }
  
  async connect() {
    const url = game.settings.get('mcp-bridge', 'serverUrl');
    this.ws = new WebSocket(url);
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMCPMessage(message);
    };
    
    // Register Foundry hooks
    Hooks.on('updateActor', this.onActorUpdate.bind(this));
    Hooks.on('createChatMessage', this.onChatMessage.bind(this));
    Hooks.on('canvasReady', this.onCanvasReady.bind(this));
  }
  
  handleMCPMessage(message) {
    switch (message.type) {
      case 'search_rules':
        this.displayRulesResult(message.data);
        break;
      case 'get_monster':
        this.createMonsterActor(message.data);
        break;
      case 'roll_dice':
        this.executeRoll(message.data.expression);
        break;
    }
  }
  
  async createMonsterActor(monsterData) {
    const actorData = {
      name: monsterData.name,
      type: 'npc',
      data: {
        attributes: {
          hp: { value: monsterData.hp, max: monsterData.hp },
          ac: { value: monsterData.ac }
        },
        abilities: monsterData.stats,
        details: {
          cr: monsterData.cr,
          type: monsterData.type
        }
      }
    };
    
    await Actor.create(actorData);
  }
}
```

### 2.4 Fantasy Grounds Integration
```xml
<!-- Fantasy Grounds Extension - extension.xml -->
<?xml version="1.0" encoding="iso-8859-1"?>
<root version="3.0">
  <properties>
    <name>MCP Bridge</name>
    <version>1.0</version>
    <author>TTRPG MCP</author>
    <description>Bridge to MCP Server</description>
  </properties>
  
  <base>
    <script name="MCPBridge" file="scripts/mcp_bridge.lua" />
  </base>
  
  <announcement text="MCP Bridge v1.0 loaded" font="emotefont" />
</root>
```

```lua
-- mcp_bridge.lua
local socket = require("socket")
local json = require("json")

function onInit()
  -- Connect to MCP bridge
  mcpConnection = socket.tcp()
  mcpConnection:connect("localhost", 8765)
  mcpConnection:settimeout(0)
  
  -- Register handlers
  Interface.onDesktopInit = onDesktopInit
  ChatManager.registerSlashHandler("mcp", processMCPCommand)
end

function processMCPCommand(sCommand, sParams)
  local message = {
    type = "command",
    command = sCommand,
    params = sParams,
    campaign = Session.RulesetName
  }
  
  mcpConnection:send(json.encode(message) .. "\n")
  
  -- Get response
  local response = mcpConnection:receive()
  if response then
    local data = json.decode(response)
    ChatManager.SystemMessage(data.result)
  end
end
```

## 3. AI Provider Integrations

### 3.1 Multi-Provider Architecture
```python
from abc import ABC, abstractmethod
from typing import Optional, List
import backoff

class AIProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass

class MultiProviderAI:
    def __init__(self):
        self.providers = {
            'anthropic': AnthropicProvider(),
            'openai': OpenAIProvider(),
            'google': GoogleProvider(),
            'local': LocalLLMProvider()  # Ollama, etc.
        }
        self.primary = 'anthropic'
        self.fallback_order = ['openai', 'google', 'local']
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate with automatic fallback"""
        providers = [self.primary] + self.fallback_order
        
        for provider_name in providers:
            try:
                provider = self.providers[provider_name]
                result = await provider.generate(prompt, **kwargs)
                
                # Track usage for cost optimization
                await self.track_usage(provider_name, prompt, result)
                
                return result
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        raise Exception("All AI providers failed")
    
    async def track_usage(self, provider: str, prompt: str, response: str):
        """Track usage for cost optimization"""
        tokens = self.estimate_tokens(prompt) + self.estimate_tokens(response)
        cost = self.calculate_cost(provider, tokens)
        
        await self.db.record_usage({
            'provider': provider,
            'tokens': tokens,
            'cost': cost,
            'timestamp': datetime.utcnow()
        })
```

### 3.2 Cost Optimization Strategy
```python
class CostOptimizer:
    def __init__(self):
        self.costs = {
            'anthropic': {'input': 0.008, 'output': 0.024},  # per 1K tokens
            'openai': {'input': 0.002, 'output': 0.006},
            'google': {'input': 0.001, 'output': 0.002},
            'local': {'input': 0, 'output': 0}
        }
        
    def select_provider(self, task_type: str, priority: str) -> str:
        """Select provider based on task and priority"""
        if priority == 'quality':
            return 'anthropic'  # Best quality
        elif priority == 'speed':
            return 'openai'  # Fast and reliable
        elif priority == 'cost':
            return 'local' if self.local_available() else 'google'
        else:  # balanced
            # Use tiered approach
            if task_type in ['character_generation', 'plot_creation']:
                return 'anthropic'  # Complex creative tasks
            elif task_type in ['rules_search', 'spell_lookup']:
                return 'openai'  # Standard queries
            else:
                return 'google'  # Simple tasks
```

## 4. Content Integrations

### 4.1 Discord Bot Integration
```python
import discord
from discord.ext import commands

class MCPBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.mcp_client = MCPClient()
    
    async def setup_hook(self):
        await self.load_extension('cogs.dice')
        await self.load_extension('cogs.rules')
        await self.load_extension('cogs.campaign')
    
    @commands.command()
    async def roll(self, ctx, expression: str):
        """Roll dice using MCP server"""
        result = await self.mcp_client.tool('roll_dice', {
            'expression': expression,
            'player': str(ctx.author)
        })
        
        embed = discord.Embed(
            title="🎲 Dice Roll",
            description=f"{ctx.author.mention} rolled {expression}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Result", value=result['total'], inline=True)
        embed.add_field(name="Details", value=result['breakdown'], inline=True)
        
        await ctx.send(embed=embed)
    
    @commands.command()
    async def rule(self, ctx, *, query: str):
        """Search rules"""
        result = await self.mcp_client.tool('search_rules', {
            'query': query,
            'limit': 1
        })
        
        if result['results']:
            rule = result['results'][0]
            embed = discord.Embed(
                title=rule['title'],
                description=rule['content'][:1024],
                color=discord.Color.green()
            )
            embed.set_footer(text=f"Source: {rule['source']} p.{rule['page']}")
            await ctx.send(embed=embed)
```

### 4.2 Twitch Extension
```javascript
// Twitch Extension - viewer.js
const twitch = window.Twitch.ext;

twitch.onAuthorized((auth) => {
  // Connect to MCP through broadcaster's channel
  const ws = new WebSocket(`wss://mcp-bridge.example.com/twitch/${auth.channelId}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
      case 'dice_roll':
        showDiceOverlay(data);
        break;
      case 'initiative_update':
        updateInitiativeTracker(data);
        break;
      case 'character_status':
        updateCharacterPanel(data);
        break;
    }
  };
});

function showDiceOverlay(rollData) {
  const overlay = document.getElementById('dice-overlay');
  overlay.innerHTML = `
    <div class="dice-animation">
      <div class="dice-result">${rollData.total}</div>
      <div class="dice-expression">${rollData.expression}</div>
      <div class="dice-breakdown">${rollData.breakdown}</div>
    </div>
  `;
  
  overlay.classList.add('show');
  setTimeout(() => overlay.classList.remove('show'), 5000);
}
```

### 4.3 OBS Overlay Support
```html
<!-- OBS Browser Source - overlay.html -->
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      font-family: 'Fira Sans', sans-serif;
      background: transparent;
    }
    
    .initiative-tracker {
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.8);
      border: 2px solid #gold;
      border-radius: 8px;
      padding: 10px;
      color: white;
      min-width: 200px;
    }
    
    .character-card {
      display: flex;
      align-items: center;
      padding: 5px;
      margin: 5px 0;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
    }
    
    .current-turn {
      background: rgba(255, 215, 0, 0.3);
      border-left: 3px solid gold;
    }
  </style>
</head>
<body>
  <div id="initiative-tracker" class="initiative-tracker"></div>
  
  <script>
    const ws = new WebSocket('ws://localhost:8765/obs');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'initiative_update') {
        updateInitiativeDisplay(data.order, data.currentTurn);
      }
    };
    
    function updateInitiativeDisplay(order, currentTurn) {
      const tracker = document.getElementById('initiative-tracker');
      
      tracker.innerHTML = `
        <h3>Initiative Order</h3>
        ${order.map((char, index) => `
          <div class="character-card ${index === currentTurn ? 'current-turn' : ''}">
            <span class="init-value">${char.initiative}</span>
            <span class="char-name">${char.name}</span>
            ${char.hp ? `<span class="hp">${char.hp.current}/${char.hp.max}</span>` : ''}
          </div>
        `).join('')}
      `;
    }
  </script>
</body>
</html>
```

## 5. Data Exchange Formats

### 5.1 Universal Character Sheet Format (UCSF)
```typescript
interface UniversalCharacterSheet {
  version: '1.0';
  system: 'dnd5e' | 'pathfinder2e' | 'coc7e' | 'generic';
  
  character: {
    name: string;
    player: string;
    concept: string;  // Class/profession/archetype
    level?: number;
    experience?: number;
  };
  
  attributes: {
    [key: string]: {
      value: number;
      modifier?: number;
      base?: number;
      temporary?: number;
    };
  };
  
  skills: {
    [name: string]: {
      value: number;
      attribute?: string;
      proficiency?: 'untrained' | 'trained' | 'expert' | 'master';
      specialization?: string;
    };
  };
  
  combat?: {
    hp: { current: number; max: number; temp?: number };
    ac?: number;
    initiative?: number;
    speed?: { [type: string]: number };
    attacks?: Array<{
      name: string;
      bonus: number;
      damage: string;
      type: string;
      range?: string;
    }>;
  };
  
  inventory?: Array<{
    name: string;
    quantity: number;
    weight?: number;
    value?: number;
    equipped?: boolean;
    properties?: string[];
  }>;
  
  features?: Array<{
    name: string;
    source: string;
    description: string;
    uses?: { current: number; max: number; recharge: string };
  }>;
  
  spells?: {
    ability: string;
    dc: number;
    attack: number;
    slots: { [level: string]: { current: number; max: number } };
    known: Array<{
      name: string;
      level: number;
      school: string;
      prepared?: boolean;
    }>;
  };
  
  notes?: {
    appearance?: string;
    backstory?: string;
    personality?: string;
    ideals?: string;
    bonds?: string;
    flaws?: string;
    [key: string]: string | undefined;
  };
}
```

### 5.2 Campaign Export Format
```json
{
  "version": "1.0",
  "campaign": {
    "name": "The Lost Mines",
    "system": "dnd5e",
    "created": "2024-01-01T00:00:00Z",
    "modified": "2024-01-15T00:00:00Z",
    "gm": "user123",
    "players": ["user456", "user789"]
  },
  "world": {
    "setting": "Forgotten Realms",
    "currentDate": "1492 DR, Flamerule 15",
    "locations": [
      {
        "id": "loc_001",
        "name": "Phandalin",
        "type": "town",
        "description": "A small frontier town",
        "npcs": ["npc_001", "npc_002"]
      }
    ],
    "npcs": [
      {
        "id": "npc_001",
        "name": "Sildar Hallwinter",
        "role": "ally",
        "location": "loc_001",
        "stats": "noble"
      }
    ]
  },
  "story": {
    "quests": [
      {
        "id": "quest_001",
        "name": "Find Gundren Rockseeker",
        "status": "active",
        "objectives": [
          {
            "description": "Investigate the ambush site",
            "completed": true
          },
          {
            "description": "Track down the goblins",
            "completed": false
          }
        ]
      }
    ],
    "timeline": [
      {
        "session": 1,
        "date": "2024-01-01",
        "events": [
          "Party met in Neverwinter",
          "Accepted job from Gundren",
          "Ambushed on Triboar Trail"
        ]
      }
    ]
  },
  "resources": {
    "maps": [
      {
        "id": "map_001",
        "name": "Cragmaw Hideout",
        "url": "https://example.com/maps/cragmaw.jpg",
        "gridSize": 5,
        "scale": "5ft"
      }
    ],
    "handouts": [
      {
        "id": "handout_001",
        "name": "Gundren's Letter",
        "content": "Meet me in Phandalin...",
        "revealed": true
      }
    ]
  }
}
```

### 5.3 Dice Notation Standard
```typescript
// Extended dice notation parser
interface DiceExpression {
  notation: string;  // Original expression
  dice: Array<{
    count: number;
    sides: number;
    modifiers?: {
      exploding?: number;     // Explode on this value or higher
      reroll?: number;        // Reroll if below this value
      keep?: 'highest' | 'lowest';
      keepCount?: number;
      droplowest?: number;
      drophighest?: number;
    };
  }>;
  modifier?: number;
  advantage?: boolean;
  disadvantage?: boolean;
  critRange?: number;  // Natural rolls >= this are crits
}

// Examples:
// "3d6+2" - Basic roll
// "1d20+5 adv" - Roll with advantage
// "4d6kh3" - Roll 4d6, keep highest 3
// "2d10!10" - Exploding 10s
// "1d20+7 crit19" - Crits on 19-20
```

## Implementation Plan

### Phase 1: Core Integrations (Weeks 1-2)
- [ ] OAuth2 manager setup
- [ ] D&D Beyond character import
- [ ] Discord bot basic commands
- [ ] Universal character format

### Phase 2: VTT Integrations (Weeks 3-4)
- [ ] Roll20 API integration
- [ ] Foundry VTT module
- [ ] Fantasy Grounds extension
- [ ] WebSocket bridge service

### Phase 3: Streaming Support (Week 5)
- [ ] Twitch extension
- [ ] OBS overlay
- [ ] StreamElements integration

### Phase 4: AI Providers (Week 6)
- [ ] Multi-provider architecture
- [ ] Cost optimization
- [ ] Fallback mechanisms
- [ ] Usage tracking

## Security Considerations

1. **API Key Management**
   - Store in environment variables
   - Rotate regularly
   - Use separate keys per integration

2. **OAuth Token Storage**
   - Encrypt refresh tokens
   - Implement token expiry
   - Secure token refresh flow

3. **Rate Limiting**
   - Per-provider limits
   - User-based quotas
   - Backoff strategies

4. **Data Privacy**
   - User consent for data sharing
   - GDPR compliance
   - Data retention policies

## Testing Strategy

1. **Integration Tests**
   - Mock external APIs
   - Test OAuth flows
   - Verify data transformations

2. **Load Testing**
   - Concurrent API calls
   - WebSocket connections
   - Rate limit handling

3. **Error Scenarios**
   - Provider outages
   - Invalid tokens
   - Malformed data

## Conclusion

The integration architecture provides a robust foundation for connecting with external TTRPG platforms while maintaining security, reliability, and performance. The modular design allows for easy addition of new integrations as the ecosystem evolves.