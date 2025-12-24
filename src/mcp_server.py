#!/usr/bin/env python3
"""
Minimal MCP Server for TTRPG Demo
This is a simple MCP server that provides a dice rolling tool for demonstration
"""

import asyncio
import random
import re
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("TTRPG Demo")

@mcp.tool()
async def roll_dice(expression: str) -> Dict[str, Any]:
    """
    Roll dice using standard RPG notation (e.g., '3d6+2', '1d20', '2d10+5')
    
    Args:
        expression: Dice expression like '3d6+2' or '1d20'
        
    Returns:
        Dictionary with total, rolls breakdown, and expression
    """
    # Parse dice expression
    pattern = r'(\d+)d(\d+)([+-]\d+)?'
    match = re.match(pattern, expression.strip())
    
    if not match:
        return {
            "error": "Invalid dice expression. Use format like '3d6+2' or '1d20'",
            "expression": expression
        }
    
    num_dice = int(match.group(1))
    num_sides = int(match.group(2))
    modifier = int(match.group(3) or 0)

    # Validate input to prevent DoS
    MAX_DICE = 1000
    MAX_SIDES = 1000
    if num_dice > MAX_DICE:
        return {"error": f"Cannot roll more than {MAX_DICE} dice.", "expression": expression}
    if num_sides > MAX_SIDES:
        return {"error": f"Dice cannot have more than {MAX_SIDES} sides.", "expression": expression}
    if num_dice < 1 or num_sides < 1:
        return {"error": "Number of dice and sides must be at least 1.", "expression": expression}

    # Roll the dice using list comprehension
    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    
    total = sum(rolls) + modifier
    
    # Build breakdown string
    breakdown_parts = [str(r) for r in rolls]
    if modifier != 0:
        breakdown_parts.append(f"{modifier:+d}")
    breakdown = " + ".join(breakdown_parts).replace("+ -", "- ")
    
    return {
        "expression": expression,
        "total": total,
        "rolls": rolls,
        "modifier": modifier,
        "breakdown": f"{breakdown} = {total}",
        "dice": {
            "count": num_dice,
            "sides": num_sides
        }
    }

@mcp.tool()
async def search_rules(query: str, limit: int = 3) -> Dict[str, Any]:
    """
    Search for game rules (demo version with hardcoded rules)
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    # Demo rules database
    demo_rules = [
        {
            "title": "Advantage and Disadvantage",
            "content": "When you have advantage, roll twice and take the higher result. With disadvantage, roll twice and take the lower.",
            "page": 173,
            "source": "Player's Handbook"
        },
        {
            "title": "Critical Hits",
            "content": "When you score a critical hit (natural 20), you get to roll extra dice for the attack's damage. Roll all damage dice twice and add them together.",
            "page": 196,
            "source": "Player's Handbook"
        },
        {
            "title": "Death Saving Throws",
            "content": "When you start your turn with 0 hit points, you must make a death saving throw. Roll 1d20: 10 or higher is a success, 9 or lower is a failure. Three successes stabilize you, three failures mean death.",
            "page": 197,
            "source": "Player's Handbook"
        },
        {
            "title": "Initiative",
            "content": "At the beginning of combat, every participant rolls initiative (1d20 + Dexterity modifier). Participants act in order from highest to lowest initiative.",
            "page": 189,
            "source": "Player's Handbook"
        },
        {
            "title": "Ability Checks",
            "content": "To make an ability check, roll 1d20 and add the relevant ability modifier. If you're proficient in the skill, add your proficiency bonus.",
            "page": 174,
            "source": "Player's Handbook"
        }
    ]
    
    # Simple search - filter rules that contain the query
    query_lower = query.lower()
    results = []
    
    for rule in demo_rules:
        if query_lower in rule["title"].lower() or query_lower in rule["content"].lower():
            results.append(rule)
            if len(results) >= limit:
                break
    
    return {
        "query": query,
        "results": results,
        "total_found": len(results)
    }

@mcp.tool()
async def get_character_stats() -> Dict[str, Any]:
    """
    Get demo character stats for testing
    
    Returns:
        Dictionary with character information
    """
    return {
        "name": "Thorin Ironforge",
        "class": "Fighter",
        "level": 5,
        "race": "Dwarf",
        "hp": {
            "current": 38,
            "max": 44
        },
        "ac": 18,
        "stats": {
            "strength": 16,
            "dexterity": 12,
            "constitution": 15,
            "intelligence": 10,
            "wisdom": 13,
            "charisma": 8
        },
        "proficiency_bonus": 3,
        "saving_throws": {
            "strength": 6,
            "dexterity": 1,
            "constitution": 5,
            "intelligence": 0,
            "wisdom": 1,
            "charisma": -1
        }
    }

# Run the MCP server
if __name__ == "__main__":
    import sys
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Starting TTRPG Demo MCP Server...", file=sys.stderr)
    print("Available tools: roll_dice, search_rules, get_character_stats", file=sys.stderr)
    
    # Run the server
    mcp.run()