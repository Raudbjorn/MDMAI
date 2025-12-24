"""MCP tools for character and NPC generation."""

import json
import logging
from typing import Any, Dict, List, Optional

from .backstory_generator import BackstoryGenerator
from .character_generator import CharacterGenerator
from .models import NPC, Character, TTRPGGenre
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
        genre: Optional[str] = None,
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
            genre: Optional genre override (e.g., "fantasy", "sci-fi", "cyberpunk")

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
                genre=genre,
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

    @mcp_server.tool()
    async def list_available_genres() -> Dict[str, Any]:
        """
        List all available TTRPG genres for character and NPC generation.
        
        Returns:
            List of available genres with descriptions
        """
        try:
            genres = []
            for genre in TTRPGGenre:
                if genre != TTRPGGenre.UNKNOWN and genre != TTRPGGenre.CUSTOM:
                    # Create human-readable description
                    description = {
                        TTRPGGenre.FANTASY: "Medieval fantasy with magic, dragons, and mythical creatures",
                        TTRPGGenre.SCI_FI: "Science fiction with space travel, advanced technology, and aliens",
                        TTRPGGenre.CYBERPUNK: "High-tech dystopian future with hacking and corporate control",
                        TTRPGGenre.COSMIC_HORROR: "Lovecraftian horror with unknowable entities and madness",
                        TTRPGGenre.POST_APOCALYPTIC: "Post-nuclear wasteland survival and rebuilding",
                        TTRPGGenre.SUPERHERO: "Comic book heroes with superpowers and secret identities",
                        TTRPGGenre.STEAMPUNK: "Victorian-era technology powered by steam and clockwork",
                        TTRPGGenre.WESTERN: "American Old West with cowboys, outlaws, and frontier towns",
                        TTRPGGenre.MODERN: "Contemporary real-world setting",
                        TTRPGGenre.SPACE_OPERA: "Epic space adventures with galactic empires and alien species",
                        TTRPGGenre.URBAN_FANTASY: "Modern world with hidden supernatural elements",
                        TTRPGGenre.HISTORICAL: "Historical time periods without fantasy elements",
                        TTRPGGenre.NOIR: "Dark crime stories with morally ambiguous characters",
                        TTRPGGenre.PULP: "1930s adventure stories with larger-than-life heroes",
                        TTRPGGenre.MILITARY: "Military campaigns and warfare",
                        TTRPGGenre.HORROR: "General horror themes with monsters and fear",
                        TTRPGGenre.MYSTERY: "Detective stories and puzzle-solving adventures",
                        TTRPGGenre.MYTHOLOGICAL: "Based on real-world mythologies and legends",
                        TTRPGGenre.ANIME: "Japanese animation and manga inspired adventures",
                        TTRPGGenre.GENERIC: "Generic rules suitable for any setting",
                    }.get(genre, f"A {genre.value.replace('_', ' ')} themed setting")
                    
                    genres.append({
                        "name": genre.value,
                        "display_name": genre.value.replace('_', ' ').title(),
                        "description": description
                    })
            
            return {
                "success": True,
                "message": f"Found {len(genres)} available genres",
                "genres": genres
            }
            
        except Exception as e:
            logger.error(f"Failed to list genres: {str(e)}")
            return {"success": False, "error": f"Failed to list genres: {str(e)}"}

    @mcp_server.tool()
    async def get_genre_content(
        genre: str,
        content_type: str = "all"  # "races", "classes", "roles", "all"
    ) -> Dict[str, Any]:
        """
        Get available content for a specific genre.
        
        Args:
            genre: The genre to get content for
            content_type: Type of content to retrieve ("races", "classes", "roles", "all")
            
        Returns:
            Genre-specific content information
        """
        try:
            # Validate genre
            try:
                genre_enum = TTRPGGenre(genre.lower())
            except ValueError:
                return {"success": False, "error": f"Unknown genre: {genre}"}
            
            result = {
                "success": True,
                "genre": genre,
                "content": {}
            }
            
            # Get character classes for this genre
            if content_type in ["classes", "all"]:
                from .models import CharacterClass
                
                # Genre-specific class mappings
                genre_classes = {
                    TTRPGGenre.FANTASY: [
                        CharacterClass.FIGHTER, CharacterClass.WIZARD, CharacterClass.CLERIC, 
                        CharacterClass.ROGUE, CharacterClass.RANGER, CharacterClass.PALADIN,
                        CharacterClass.BARBARIAN, CharacterClass.SORCERER, CharacterClass.WARLOCK,
                        CharacterClass.DRUID, CharacterClass.MONK, CharacterClass.BARD, CharacterClass.ARTIFICER
                    ],
                    TTRPGGenre.SCI_FI: [
                        CharacterClass.ENGINEER, CharacterClass.SCIENTIST, CharacterClass.PILOT,
                        CharacterClass.MARINE, CharacterClass.DIPLOMAT, CharacterClass.XENOBIOLOGIST,
                        CharacterClass.TECH_SPECIALIST, CharacterClass.PSION, CharacterClass.BOUNTY_HUNTER
                    ],
                    TTRPGGenre.CYBERPUNK: [
                        CharacterClass.NETRUNNER, CharacterClass.SOLO, CharacterClass.FIXER,
                        CharacterClass.CORPORATE, CharacterClass.ROCKERBOY, CharacterClass.TECHIE,
                        CharacterClass.MEDIA, CharacterClass.COP, CharacterClass.NOMAD
                    ],
                    TTRPGGenre.WESTERN: [
                        CharacterClass.GUNSLINGER, CharacterClass.LAWMAN, CharacterClass.OUTLAW,
                        CharacterClass.GAMBLER, CharacterClass.PREACHER, CharacterClass.PROSPECTOR,
                        CharacterClass.NATIVE_SCOUT
                    ],
                    TTRPGGenre.COSMIC_HORROR: [
                        CharacterClass.INVESTIGATOR, CharacterClass.SCHOLAR, CharacterClass.ANTIQUARIAN,
                        CharacterClass.OCCULTIST, CharacterClass.ALIENIST, CharacterClass.ARCHAEOLOGIST,
                        CharacterClass.JOURNALIST, CharacterClass.DETECTIVE, CharacterClass.PROFESSOR
                    ],
                    TTRPGGenre.POST_APOCALYPTIC: [
                        CharacterClass.SURVIVOR, CharacterClass.SCAVENGER, CharacterClass.RAIDER,
                        CharacterClass.MEDIC, CharacterClass.MECHANIC, CharacterClass.TRADER,
                        CharacterClass.WARLORD, CharacterClass.MUTANT_HUNTER, CharacterClass.VAULT_DWELLER
                    ],
                    TTRPGGenre.SUPERHERO: [
                        CharacterClass.VIGILANTE, CharacterClass.POWERED, CharacterClass.GENIUS,
                        CharacterClass.MARTIAL_ARTIST, CharacterClass.MYSTIC, CharacterClass.ALIEN_HERO,
                        CharacterClass.TECH_HERO, CharacterClass.SIDEKICK
                    ]
                }
                
                classes = genre_classes.get(genre_enum, [CharacterClass.FIGHTER, CharacterClass.ROGUE])
                result["content"]["classes"] = [
                    {
                        "name": cls.value,
                        "display_name": cls.value.replace('_', ' ').title()
                    }
                    for cls in classes
                ]
            
            # Get character races for this genre
            if content_type in ["races", "all"]:
                from .models import CharacterRace
                
                # Genre-specific race mappings
                genre_races = {
                    TTRPGGenre.FANTASY: [
                        CharacterRace.HUMAN, CharacterRace.ELF, CharacterRace.DWARF,
                        CharacterRace.HALFLING, CharacterRace.ORC, CharacterRace.TIEFLING,
                        CharacterRace.DRAGONBORN, CharacterRace.GNOME, CharacterRace.HALF_ELF, CharacterRace.HALF_ORC
                    ],
                    TTRPGGenre.SCI_FI: [
                        CharacterRace.TERRAN, CharacterRace.MARTIAN, CharacterRace.BELTER,
                        CharacterRace.CYBORG, CharacterRace.ANDROID, CharacterRace.AI_CONSTRUCT,
                        CharacterRace.GREY_ALIEN, CharacterRace.REPTILIAN, CharacterRace.INSECTOID,
                        CharacterRace.ENERGY_BEING, CharacterRace.SILICON_BASED, CharacterRace.UPLIFTED_ANIMAL
                    ],
                    TTRPGGenre.CYBERPUNK: [
                        CharacterRace.AUGMENTED_HUMAN, CharacterRace.FULL_CONVERSION_CYBORG,
                        CharacterRace.BIOENGINEERED, CharacterRace.CLONE, CharacterRace.DIGITAL_CONSCIOUSNESS
                    ],
                    TTRPGGenre.COSMIC_HORROR: [
                        CharacterRace.HUMAN, CharacterRace.DEEP_ONE_HYBRID, CharacterRace.GHOUL,
                        CharacterRace.DREAMLANDS_NATIVE, CharacterRace.TOUCHED
                    ],
                    TTRPGGenre.POST_APOCALYPTIC: [
                        CharacterRace.PURE_STRAIN_HUMAN, CharacterRace.MUTANT, CharacterRace.GHOUL_WASTELANDER,
                        CharacterRace.SYNTHETIC, CharacterRace.HYBRID, CharacterRace.RADIANT
                    ],
                    TTRPGGenre.SUPERHERO: [
                        CharacterRace.HUMAN, CharacterRace.METAHUMAN, CharacterRace.INHUMAN,
                        CharacterRace.ATLANTEAN, CharacterRace.AMAZONIAN, CharacterRace.KRYPTONIAN, CharacterRace.ASGARDIAN
                    ]
                }
                
                races = genre_races.get(genre_enum, [CharacterRace.HUMAN])
                result["content"]["races"] = [
                    {
                        "name": race.value,
                        "display_name": race.value.replace('_', ' ').title()
                    }
                    for race in races
                ]
            
            # Get NPC roles (all roles are available for all genres)
            if content_type in ["roles", "all"]:
                from .models import NPCRole
                
                roles = [role for role in NPCRole if role != NPCRole.CUSTOM]
                result["content"]["roles"] = [
                    {
                        "name": role.value,
                        "display_name": role.value.replace('_', ' ').title()
                    }
                    for role in roles
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get genre content: {str(e)}")
            return {"success": False, "error": f"Failed to get genre content: {str(e)}"}

    @mcp_server.tool() 
    async def search_characters_by_genre(
        genre: str,
        system: Optional[str] = None,
        level_min: Optional[int] = None,
        level_max: Optional[int] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for characters by genre with additional filters.
        
        Args:
            genre: TTRPG genre to filter by
            system: Optional system filter
            level_min: Minimum level filter
            level_max: Maximum level filter
            limit: Maximum number of results
            
        Returns:
            List of characters matching the genre and filters
        """
        try:
            if not _db:
                return {"success": False, "error": "Database not available"}

            # Validate genre
            try:
                genre_enum = TTRPGGenre(genre.lower())
            except ValueError:
                return {"success": False, "error": f"Unknown genre: {genre}"}

            # Build query
            where_clause = {"type": "player_character"}
            if system:
                where_clause["system"] = system

            # Query database
            results = await _db.query_collection(
                collection_name="characters", where=where_clause, limit=limit * 2  # Get more to filter
            )

            # Filter by genre and level
            characters = []
            for result in results.get("documents", []):
                doc = result.get("document", {})
                if isinstance(doc, str):
                    doc = json.loads(doc)

                # Check if character matches genre (you might need to implement genre detection logic)
                character_system = doc.get("system", "").lower()
                character_genre = _determine_system_genre(character_system)
                
                if character_genre != genre_enum:
                    continue

                level = doc.get("stats", {}).get("level", 1)
                if level_min and level < level_min:
                    continue
                if level_max and level > level_max:
                    continue

                characters.append({
                    "id": doc.get("id"),
                    "name": doc.get("name"),
                    "system": doc.get("system"),
                    "genre": genre,
                    "class": doc.get("character_class") or doc.get("custom_class"),
                    "race": doc.get("race") or doc.get("custom_race"),
                    "level": level,
                    "created_at": doc.get("created_at"),
                })
                
                if len(characters) >= limit:
                    break

            return {
                "success": True,
                "message": f"Found {len(characters)} characters for genre {genre}",
                "genre": genre,
                "characters": characters,
            }

        except Exception as e:
            logger.error(f"Failed to search characters by genre: {str(e)}")
            return {"success": False, "error": f"Failed to search characters by genre: {str(e)}"}

    @mcp_server.tool()
    async def search_npcs_by_genre(
        genre: str,
        role: Optional[str] = None,
        importance: Optional[str] = None,
        location: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for NPCs by genre with additional filters.
        
        Args:
            genre: TTRPG genre to filter by
            role: Optional NPC role filter
            importance: Optional importance level filter
            location: Optional location filter
            limit: Maximum number of results
            
        Returns:
            List of NPCs matching the genre and filters
        """
        try:
            if not _db:
                return {"success": False, "error": "Database not available"}

            # Validate genre
            try:
                genre_enum = TTRPGGenre(genre.lower())
            except ValueError:
                return {"success": False, "error": f"Unknown genre: {genre}"}

            # Build query
            where_clause = {"type": "npc"}
            if role:
                where_clause["role"] = role
            if importance:
                where_clause["importance"] = importance
            if location:
                where_clause["location"] = location

            # Query database
            results = await _db.query_collection(
                collection_name="npcs", where=where_clause, limit=limit * 2
            )

            # Filter by genre
            npcs = []
            for result in results.get("documents", []):
                doc = result.get("document", {})
                if isinstance(doc, str):
                    doc = json.loads(doc)

                # Check if NPC matches genre
                npc_system = doc.get("system", "").lower()
                npc_genre = _determine_system_genre(npc_system)
                
                if npc_genre != genre_enum:
                    continue

                npcs.append({
                    "id": doc.get("id"),
                    "name": doc.get("name"),
                    "system": doc.get("system"),
                    "genre": genre,
                    "role": doc.get("role") or doc.get("custom_role"),
                    "importance": doc.get("importance"),
                    "location": doc.get("location"),
                    "level": doc.get("stats", {}).get("level", 0),
                    "attitude": doc.get("attitude_towards_party"),
                    "created_at": doc.get("created_at"),
                })
                
                if len(npcs) >= limit:
                    break

            return {
                "success": True,
                "message": f"Found {len(npcs)} NPCs for genre {genre}",
                "genre": genre,
                "npcs": npcs
            }

        except Exception as e:
            logger.error(f"Failed to search NPCs by genre: {str(e)}")
            return {"success": False, "error": f"Failed to search NPCs by genre: {str(e)}"}

    logger.info("Character generation tools registered with MCP server")


def _determine_system_genre(system: str) -> TTRPGGenre:
    """Helper function to determine genre from system name."""
    system_lower = system.lower()
    
    # Fantasy systems
    if any(keyword in system_lower for keyword in ['d&d', 'pathfinder', 'fantasy', 'dungeon', 'dragon']):
        return TTRPGGenre.FANTASY
    
    # Sci-Fi systems
    elif any(keyword in system_lower for keyword in ['traveller', 'star wars', 'starfinder', 'sci-fi', 'science fiction', 'space']):
        return TTRPGGenre.SCI_FI
    
    # Cyberpunk systems
    elif any(keyword in system_lower for keyword in ['cyberpunk', 'shadowrun', 'cyber']):
        return TTRPGGenre.CYBERPUNK
    
    # Cosmic Horror systems
    elif any(keyword in system_lower for keyword in ['call of cthulhu', 'cthulhu', 'cosmic horror', 'lovecraft']):
        return TTRPGGenre.COSMIC_HORROR
    
    # Post-Apocalyptic systems
    elif any(keyword in system_lower for keyword in ['fallout', 'gamma world', 'apocalypse', 'wasteland']):
        return TTRPGGenre.POST_APOCALYPTIC
    
    # Western systems
    elif any(keyword in system_lower for keyword in ['deadlands', 'western', 'wild west', 'cowboy']):
        return TTRPGGenre.WESTERN
    
    # Superhero systems
    elif any(keyword in system_lower for keyword in ['mutants & masterminds', 'champions', 'superhero', 'hero']):
        return TTRPGGenre.SUPERHERO
    
    # Modern systems
    elif any(keyword in system_lower for keyword in ['modern', 'contemporary', 'urban']):
        return TTRPGGenre.MODERN
    
    # Horror systems
    elif any(keyword in system_lower for keyword in ['horror', 'world of darkness', 'vampire']):
        return TTRPGGenre.HORROR
    
    # Default to generic if no match
    else:
        return TTRPGGenre.GENERIC
