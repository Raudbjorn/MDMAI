"""Data models for source management."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
import uuid


class SourceType(Enum):
    """Types of source material."""
    RULEBOOK = "rulebook"
    FLAVOR = "flavor"
    SUPPLEMENT = "supplement"
    ADVENTURE = "adventure"
    SETTING = "setting"
    BESTIARY = "bestiary"
    SPELL_COMPENDIUM = "spell_compendium"
    ITEM_CATALOG = "item_catalog"
    CUSTOM = "custom"


class ContentCategory(Enum):
    """Categories of content within sources."""
    RULES = "rules"
    MECHANICS = "mechanics"
    LORE = "lore"
    NARRATIVE = "narrative"
    CHARACTERS = "characters"
    LOCATIONS = "locations"
    ITEMS = "items"
    SPELLS = "spells"
    MONSTERS = "monsters"
    CLASSES = "classes"
    RACES = "races"
    FEATS = "feats"
    SKILLS = "skills"
    ADVENTURES = "adventures"
    MAPS = "maps"
    TABLES = "tables"
    APPENDICES = "appendices"


class QualityLevel(Enum):
    """Quality levels for sources."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNVALIDATED = "unvalidated"


class ProcessingStatus(Enum):
    """Processing status of a source."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class SourceQuality:
    """Quality metrics for a source."""
    overall_score: float = 0.0  # 0.0 to 1.0
    text_quality: float = 0.0
    structure_quality: float = 0.0
    metadata_completeness: float = 0.0
    content_coverage: float = 0.0
    
    level: QualityLevel = QualityLevel.UNVALIDATED
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Detailed metrics
    total_pages: int = 0
    extracted_pages: int = 0
    total_chunks: int = 0
    valid_chunks: int = 0
    avg_chunk_length: float = 0.0
    
    def calculate_overall_score(self):
        """Calculate the overall quality score."""
        self.overall_score = (
            self.text_quality * 0.3 +
            self.structure_quality * 0.3 +
            self.metadata_completeness * 0.2 +
            self.content_coverage * 0.2
        )
        
        # Determine quality level
        if self.overall_score >= 0.8:
            self.level = QualityLevel.EXCELLENT
        elif self.overall_score >= 0.6:
            self.level = QualityLevel.GOOD
        elif self.overall_score >= 0.4:
            self.level = QualityLevel.FAIR
        else:
            self.level = QualityLevel.POOR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['level'] = self.level.value
        return data


@dataclass
class SourceMetadata:
    """Metadata for a source document."""
    title: str
    system: str
    source_type: SourceType
    
    # Publication information
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    edition: Optional[str] = None
    version: Optional[str] = None
    isbn: Optional[str] = None
    
    # Content information
    language: str = "English"
    page_count: int = 0
    categories: List[ContentCategory] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    # File information
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: int = 0
    
    # Processing information
    processed_at: Optional[datetime] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['source_type'] = self.source_type.value
        data['categories'] = [cat.value for cat in self.categories]
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceMetadata':
        """Create from dictionary."""
        if isinstance(data.get('source_type'), str):
            data['source_type'] = SourceType(data['source_type'])
        
        if 'categories' in data:
            data['categories'] = [
                ContentCategory(cat) if isinstance(cat, str) else cat
                for cat in data['categories']
            ]
        
        if isinstance(data.get('processed_at'), str):
            data['processed_at'] = datetime.fromisoformat(data['processed_at'])
        
        return cls(**data)


@dataclass
class SourceCategory:
    """Category organization for sources."""
    name: str
    category_type: ContentCategory
    parent_category: Optional[str] = None
    
    # Content details
    description: str = ""
    source_ids: List[str] = field(default_factory=list)
    subcategories: List[str] = field(default_factory=list)
    
    # Organization
    priority: int = 0
    is_primary: bool = False
    auto_assigned: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['category_type'] = self.category_type.value
        return data


@dataclass
class SourceRelationship:
    """Relationships between sources."""
    source_id: str
    related_source_id: str
    relationship_type: str  # "supplement_to", "requires", "conflicts_with", etc.
    
    description: Optional[str] = None
    strength: float = 1.0  # 0.0 to 1.0
    bidirectional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Source:
    """Complete source document model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: SourceMetadata = field(default_factory=lambda: SourceMetadata(
        title="Unknown",
        system="Unknown",
        source_type=SourceType.CUSTOM
    ))
    
    # Content
    content_chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_count: int = 0
    total_tokens: int = 0
    
    # Quality
    quality: SourceQuality = field(default_factory=SourceQuality)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Organization
    categories: List[SourceCategory] = field(default_factory=list)
    relationships: List[SourceRelationship] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Status
    status: ProcessingStatus = ProcessingStatus.PENDING
    errors: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "metadata": self.metadata.to_dict(),
            "chunk_count": self.chunk_count,
            "total_tokens": self.total_tokens,
            "quality": self.quality.to_dict(),
            "validation_results": self.validation_results,
            "categories": [cat.to_dict() for cat in self.categories],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "dependencies": self.dependencies,
            "status": self.status.value,
            "errors": self.errors,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Source':
        """Create from dictionary."""
        source = cls()
        
        source.id = data.get('id', source.id)
        
        if 'metadata' in data:
            source.metadata = SourceMetadata.from_dict(data['metadata'])
        
        source.chunk_count = data.get('chunk_count', 0)
        source.total_tokens = data.get('total_tokens', 0)
        
        if 'quality' in data:
            quality_data = data['quality']
            source.quality = SourceQuality(**quality_data) if isinstance(quality_data, dict) else source.quality
        
        source.validation_results = data.get('validation_results', {})
        
        if 'categories' in data:
            source.categories = [
                SourceCategory(**cat) if isinstance(cat, dict) else cat
                for cat in data['categories']
            ]
        
        if 'relationships' in data:
            source.relationships = [
                SourceRelationship(**rel) if isinstance(rel, dict) else rel
                for rel in data['relationships']
            ]
        
        source.dependencies = data.get('dependencies', [])
        
        if 'status' in data:
            source.status = ProcessingStatus(data['status'])
        
        source.errors = data.get('errors', [])
        
        if isinstance(data.get('created_at'), str):
            source.created_at = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            source.updated_at = datetime.fromisoformat(data['updated_at'])
        if isinstance(data.get('last_accessed'), str):
            source.last_accessed = datetime.fromisoformat(data['last_accessed'])
        
        source.access_count = data.get('access_count', 0)
        
        return source


@dataclass
class FlavorSource(Source):
    """Flavor source with narrative-specific properties."""
    
    # Narrative elements
    narrative_style: str = ""
    tone: str = ""
    themes: List[str] = field(default_factory=list)
    
    # Extracted elements
    characters: List[Dict[str, str]] = field(default_factory=list)
    locations: List[Dict[str, str]] = field(default_factory=list)
    events: List[Dict[str, str]] = field(default_factory=list)
    quotes: List[Dict[str, str]] = field(default_factory=list)
    
    # Generation parameters
    creativity_modifier: float = 1.0  # How much to influence generation
    canonical: bool = False  # Whether this is official canon
    priority: int = 0  # Priority when multiple sources conflict
    
    # Usage tracking
    times_used: int = 0
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "narrative_style": self.narrative_style,
            "tone": self.tone,
            "themes": self.themes,
            "characters": self.characters,
            "locations": self.locations,
            "events": self.events,
            "quotes": self.quotes,
            "creativity_modifier": self.creativity_modifier,
            "canonical": self.canonical,
            "priority": self.priority,
            "times_used": self.times_used,
            "last_used": self.last_used.isoformat() if self.last_used else None
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlavorSource':
        """Create from dictionary."""
        # First create base Source
        source = super().from_dict(data)
        
        # Create FlavorSource with Source data
        flavor = cls()
        flavor.__dict__.update(source.__dict__)
        
        # Add flavor-specific fields
        flavor.narrative_style = data.get('narrative_style', '')
        flavor.tone = data.get('tone', '')
        flavor.themes = data.get('themes', [])
        flavor.characters = data.get('characters', [])
        flavor.locations = data.get('locations', [])
        flavor.events = data.get('events', [])
        flavor.quotes = data.get('quotes', [])
        flavor.creativity_modifier = data.get('creativity_modifier', 1.0)
        flavor.canonical = data.get('canonical', False)
        flavor.priority = data.get('priority', 0)
        flavor.times_used = data.get('times_used', 0)
        
        if isinstance(data.get('last_used'), str):
            flavor.last_used = datetime.fromisoformat(data['last_used'])
        
        return flavor