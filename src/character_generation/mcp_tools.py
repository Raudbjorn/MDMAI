"""MCP tools for character and NPC generation."""

import json
import logging
from typing import Any, Dict, List, Optional

from .backstory_generator import BackstoryGenerator
from .character_generator import CharacterGenerator
from .models import NPC, Character
from .npc_generator import NPCGenerator
from .validators import CharacterValidator, ValidationError

logger = logging.getLogger(__name__)

# Module-level instances
_character_generator: Optional[CharacterGenerator] = None
_backstory_generator: Optional[BackstoryGenerator] = None
_npc_generator: Optional[NPCGenerator] = None
_db = None
_personality_manager = None


def initialize_character_tools(db, personality_manager=None):
    """
    Initialize character generation tools with dependencies.

    Args:
        db: Database manager instance
        personality_manager: Optional personality manager for style-aware generation
    """
    global _character_generator, _backstory_generator, _npc_generator, _db, _personality_manager

    _db = db
    _personality_manager = personality_manager

    # Initialize generators
    _character_generator = CharacterGenerator()
    _backstory_generator = BackstoryGenerator(personality_manager=personality_manager)
    _npc_generator = NPCGenerator()

    # Set personality manager if available
    if personality_manager:
        _character_generator.set_personality_manager(personality_manager)
        _npc_generator.set_personality_manager(personality_manager)

    logger.info("Character generation tools initialized")


def register_character_tools(mcp_server):
    """
    Register character generation tools with the MCP server.

    Args:
        mcp_server: The FastMCP server instance
    """

    @mcp_server.tool()
    async def generate_character(
        system: str = "D&D 5e",
        level: int = 1,
        character_class: Optional[str] = None,
        race: Optional[str] = None,
        name: Optional[str] = None,
        backstory_hints: Optional[str] = None,
        backstory_depth: str = "standard",
        stat_generation: str = "standard",
        use_personality: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a complete player character with stats and backstory.

        Args:
            system: Game system (e.g., "D&D 5e", "Pathfinder")
            level: Character level (1-20)
            character_class: Specific class or None for random
            race: Specific race or None for random
            name: Character name or None for generation
            backstory_hints: Hints for backstory generation
            backstory_depth: "simple", "standard", or "detailed"
            stat_generation: "standard", "random", or "point_buy"
            use_personality: Use system personality for backstory style

        Returns:
            Complete character data including stats, equipment, and backstory
        """
        try:
            # Sanitize inputs
            sanitized_params = CharacterValidator.sanitize_input(
                {
                    "system": system,
                    "level": level,
                    "character_class": character_class,
                    "race": race,
                    "name": name,
                    "backstory_hints": backstory_hints,
                },
                Character,
            )

            logger.info(
                f"Generating character for {sanitized_params.get('system', system)} at level {sanitized_params.get('level', level)}"
            )

            if not _character_generator:
                return {"success": False, "error": "Character generator not initialized"}

            # Generate base character with sanitized inputs
            character = _character_generator.generate_character(
                system=sanitized_params.get("system", system),
                level=sanitized_params.get("level", level),
                character_class=sanitized_params.get("character_class", character_class),
                race=sanitized_params.get("race", race),
                name=sanitized_params.get("name", name),
                backstory_hints=sanitized_params.get("backstory_hints", backstory_hints),
                stat_generation=stat_generation,
            )

            # Generate enhanced backstory
            if backstory_depth != "none":
                character.backstory = _backstory_generator.generate_backstory(
                    character,
                    hints=backstory_hints,
                    depth=backstory_depth,
                    use_flavor_sources=use_personality,
                )

            # Store in database if available
            if _db:
                try:
                    doc_dict = character.to_dict()
                    _db.add_document(
                        collection_name="characters",
                        document_id=character.id,
                        content=f"Character: {character.name}, {character.get_race_name()} {character.get_class_name()}",
                        metadata={
                            "type": "player_character",
                            "system": system,
                            "level": level,
                            "class": character.get_class_name(),
                            "race": character.get_race_name(),
                            "name": character.name,
                            "document": json.dumps(doc_dict),
                        },
                    )
                    logger.info(f"Character {character.name} stored in database")
                except Exception as e:
                    logger.warning(f"Failed to store character in database: {e}")

            return {
                "success": True,
                "message": f"Generated character: {character.name}",
                "character": character.to_dict(),
            }

        except ValidationError as e:
            logger.error(f"Validation error during character generation: {str(e)}")
            return {"success": False, "error": f"Validation error: {str(e)}"}
        except Exception as e:
            logger.error(f"Character generation failed: {str(e)}")
            return {"success": False, "error": f"Character generation failed: {str(e)}"}

    @mcp_server.tool()
    async def generate_npc(
        system: str = "D&D 5e",
        role: Optional[str] = None,
        level: Optional[int] = None,
        name: Optional[str] = None,
        personality_traits: Optional[List[str]] = None,
        importance: str = "minor",
        party_level: Optional[int] = None,
        backstory_depth: str = "simple",
        use_personality: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate an NPC with appropriate stats and personality.

        Args:
            system: Game system
            role: NPC role/occupation (e.g., "merchant", "guard", "noble")
            level: Specific level, or auto-calculate from party
            name: NPC name or None for generation
            personality_traits: Specific traits or None for generation
            importance: "minor", "supporting", or "major"
            party_level: Party level for scaling
            backstory_depth: "none", "simple", "standard", or "detailed"
            use_personality: Use system personality for generation

        Returns:
            Complete NPC data including stats, personality, and backstory
        """
        try:
            # Sanitize inputs
            sanitized_params = CharacterValidator.sanitize_input(
                {
                    "system": system,
                    "role": role,
                    "level": level,
                    "name": name,
                    "importance": importance,
                    "party_level": party_level,
                },
                NPC,
            )

            logger.info(
                f"Generating {sanitized_params.get('importance', importance)} NPC with role {sanitized_params.get('role', role)}"
            )

            if not _npc_generator:
                return {"success": False, "error": "NPC generator not initialized"}

            # Generate NPC with sanitized inputs
            npc = _npc_generator.generate_npc(
                system=sanitized_params.get("system", system),
                role=sanitized_params.get("role", role),
                level=sanitized_params.get("level", level),
                name=sanitized_params.get("name", name),
                personality_traits=personality_traits,
                importance=sanitized_params.get("importance", importance),
                party_level=sanitized_params.get("party_level", party_level),
                backstory_depth=backstory_depth,
            )

            # Store in database if available
            if _db:
                try:
                    doc_dict = npc.to_dict()
                    _db.add_document(
                        collection_name="npcs",
                        document_id=npc.id,
                        content=f"NPC: {npc.name}, {npc.get_role_name()}",
                        metadata={
                            "type": "npc",
                            "system": system,
                            "role": npc.get_role_name(),
                            "importance": importance,
                            "level": npc.stats.level,
                            "name": npc.name,
                            "location": npc.location,
                            "document": json.dumps(doc_dict),
                        },
                    )
                    logger.info(f"NPC {npc.name} stored in database")
                except Exception as e:
                    logger.warning(f"Failed to store NPC in database: {e}")

            return {
                "success": True,
                "message": f"Generated NPC: {npc.name} ({npc.get_role_name()})",
                "npc": npc.to_dict(),
            }

        except ValidationError as e:
            logger.error(f"Validation error during NPC generation: {str(e)}")
            return {"success": False, "error": f"Validation error: {str(e)}"}
        except Exception as e:
            logger.error(f"NPC generation failed: {str(e)}")
            return {"success": False, "error": f"NPC generation failed: {str(e)}"}

    @mcp_server.tool()
    async def generate_character_backstory(
        character_data: Dict[str, Any],
        hints: Optional[str] = None,
        depth: str = "standard",
        use_flavor_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate or enhance a backstory for an existing character.

        Args:
            character_data: Character data dictionary
            hints: Optional hints to guide generation
            depth: "simple", "standard", or "detailed"
            use_flavor_sources: Whether to incorporate flavor sources

        Returns:
            Enhanced backstory data
        """
        try:
            logger.info("Generating enhanced backstory for character")

            if not _backstory_generator:
                return {"success": False, "error": "Backstory generator not initialized"}

            # Create character from data
            character = Character.from_dict(character_data)

            # Generate backstory
            backstory = _backstory_generator.generate_backstory(
                character, hints=hints, depth=depth, use_flavor_sources=use_flavor_sources
            )

            # Update character
            character.backstory = backstory

            return {
                "success": True,
                "message": "Backstory generated successfully",
                "backstory": backstory.to_dict(),
                "character": character.to_dict(),
            }

        except Exception as e:
            logger.error(f"Backstory generation failed: {str(e)}")
            return {"success": False, "error": f"Backstory generation failed: {str(e)}"}

    @mcp_server.tool()
    async def list_generated_characters(
        system: Optional[str] = None,
        character_class: Optional[str] = None,
        level_min: Optional[int] = None,
        level_max: Optional[int] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        List previously generated characters from the database.

        Args:
            system: Filter by game system
            character_class: Filter by class
            level_min: Minimum level filter
            level_max: Maximum level filter
            limit: Maximum number of results

        Returns:
            List of generated characters
        """
        try:
            if not _db:
                return {"success": False, "error": "Database not available"}

            # Build query
            where_clause = {"type": "player_character"}
            if system:
                where_clause["system"] = system
            if character_class:
                where_clause["class"] = character_class

            # Query database
            results = await _db.query_collection(
                collection_name="characters", where=where_clause, limit=limit
            )

            # Filter by level if specified
            characters = []
            for result in results.get("documents", []):
                doc = result.get("document", {})
                if isinstance(doc, str):
                    doc = json.loads(doc)

                level = doc.get("stats", {}).get("level", 1)
                if level_min and level < level_min:
                    continue
                if level_max and level > level_max:
                    continue

                characters.append(
                    {
                        "id": doc.get("id"),
                        "name": doc.get("name"),
                        "system": doc.get("system"),
                        "class": doc.get("character_class") or doc.get("custom_class"),
                        "race": doc.get("race") or doc.get("custom_race"),
                        "level": level,
                        "created_at": doc.get("created_at"),
                    }
                )

            return {
                "success": True,
                "message": f"Found {len(characters)} characters",
                "characters": characters,
            }

        except Exception as e:
            logger.error(f"Failed to list characters: {str(e)}")
            return {"success": False, "error": f"Failed to list characters: {str(e)}"}

    @mcp_server.tool()
    async def list_generated_npcs(
        system: Optional[str] = None,
        role: Optional[str] = None,
        importance: Optional[str] = None,
        location: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        List previously generated NPCs from the database.

        Args:
            system: Filter by game system
            role: Filter by NPC role
            importance: Filter by importance level
            location: Filter by location
            limit: Maximum number of results

        Returns:
            List of generated NPCs
        """
        try:
            if not _db:
                return {"success": False, "error": "Database not available"}

            # Build query
            where_clause = {"type": "npc"}
            if system:
                where_clause["system"] = system
            if role:
                where_clause["role"] = role
            if importance:
                where_clause["importance"] = importance
            if location:
                where_clause["location"] = location

            # Query database
            results = await _db.query_collection(
                collection_name="npcs", where=where_clause, limit=limit
            )

            # Format results
            npcs = []
            for result in results.get("documents", []):
                doc = result.get("document", {})
                if isinstance(doc, str):
                    import json

                    doc = json.loads(doc)

                npcs.append(
                    {
                        "id": doc.get("id"),
                        "name": doc.get("name"),
                        "system": doc.get("system"),
                        "role": doc.get("role") or doc.get("custom_role"),
                        "importance": doc.get("importance"),
                        "location": doc.get("location"),
                        "level": doc.get("stats", {}).get("level", 0),
                        "attitude": doc.get("attitude_towards_party"),
                        "created_at": doc.get("created_at"),
                    }
                )

            return {"success": True, "message": f"Found {len(npcs)} NPCs", "npcs": npcs}

        except Exception as e:
            logger.error(f"Failed to list NPCs: {str(e)}")
            return {"success": False, "error": f"Failed to list NPCs: {str(e)}"}

    logger.info("Character generation tools registered with MCP server")
