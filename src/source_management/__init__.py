"""Source management module for TTRPG Assistant."""

from .source_manager import SourceManager
from .source_validator import SourceValidator
from .source_organizer import SourceOrganizer
from .flavor_integrator import FlavorIntegrator
from .models import (
    Source,
    SourceMetadata,
    SourceCategory,
    SourceRelationship,
    SourceQuality,
    FlavorSource
)
from .mcp_tools import (
    initialize_source_tools,
    register_source_tools
)

__all__ = [
    'SourceManager',
    'SourceValidator',
    'SourceOrganizer',
    'FlavorIntegrator',
    'Source',
    'SourceMetadata',
    'SourceCategory',
    'SourceRelationship',
    'SourceQuality',
    'FlavorSource',
    'initialize_source_tools',
    'register_source_tools'
]