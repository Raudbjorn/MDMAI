"""Source management module for TTRPG Assistant."""

from .flavor_integrator import FlavorIntegrator
from .mcp_tools import initialize_source_tools, register_source_tools
from .models import (
    FlavorSource,
    Source,
    SourceCategory,
    SourceMetadata,
    SourceQuality,
    SourceRelationship,
)
from .source_manager import SourceManager
from .source_organizer import SourceOrganizer
from .source_validator import SourceValidator

__all__ = [
    "SourceManager",
    "SourceValidator",
    "SourceOrganizer",
    "FlavorIntegrator",
    "Source",
    "SourceMetadata",
    "SourceCategory",
    "SourceRelationship",
    "SourceQuality",
    "FlavorSource",
    "initialize_source_tools",
    "register_source_tools" "register_source_tools",
]
