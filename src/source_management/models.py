"""Data models for source management."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
            self.text_quality * 0.3
            + self.structure_quality * 0.3
            + self.metadata_completeness * 0.2
            + self.content_coverage * 0.2
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
        data["level"] = self.level.value
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
        data["source_type"] = self.source_type.value
        data["categories"] = [cat.value for cat in self.categories]
        if self.processed_at:
            data["processed_at"] = self.processed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceMetadata":
        """Create from dictionary."""
        # Work with a copy to avoid modifying the input
        data_copy = data.copy()

        # Convert source_type if needed
        if isinstance(data_copy.get("source_type"), str):
            try:
                data_copy["source_type"] = SourceType(data_copy["source_type"])
            except ValueError:
                # Default to CUSTOM if invalid
                data_copy["source_type"] = SourceType.CUSTOM

        # Convert categories if needed
        if "categories" in data_copy:
            categories = []
            for cat in data_copy["categories"]:
                if isinstance(cat, str):
                    try:
                        categories.append(ContentCategory(cat))
                    except ValueError:
                        # Skip invalid categories
                        continue
                else:
                    categories.append(cat)
            data_copy["categories"] = categories

        # Convert datetime if needed
        if isinstance(data_copy.get("processed_at"), str):
            data_copy["processed_at"] = datetime.fromisoformat(data_copy["processed_at"])

        # Filter to known fields to avoid TypeError
        known_fields = {
            "title",
            "system",
            "source_type",
            "author",
            "publisher",
            "publication_date",
            "edition",
            "version",
            "isbn",
            "language",
            "page_count",
            "categories",
            "tags",
            "description",
            "file_path",
            "file_hash",
            "file_size",
            "processed_at",
            "processing_time",
        }

        filtered_data = {k: v for k, v in data_copy.items() if k in known_fields}

        return cls(**filtered_data)


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
        data["category_type"] = self.category_type.value
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
    metadata: SourceMetadata = field(
        default_factory=lambda: SourceMetadata(
            title="Unknown", system="Unknown", source_type=SourceType.CUSTOM
        )
    )

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
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        """Create from dictionary."""
        # Work with a copy to avoid modifying the input
        data_copy = data.copy()

        source = cls()

        source.id = data_copy.get("id", source.id)

        if "metadata" in data_copy:
            source.metadata = SourceMetadata.from_dict(data_copy["metadata"])

        source.chunk_count = data_copy.get("chunk_count", 0)
        source.total_tokens = data_copy.get("total_tokens", 0)

        if "quality" in data_copy:
            quality_data = data_copy["quality"]
            if isinstance(quality_data, dict):
                # Filter to known SourceQuality fields
                quality_fields = {
                    "overall_score",
                    "text_quality",
                    "structure_quality",
                    "metadata_completeness",
                    "content_coverage",
                    "level",
                    "issues",
                    "warnings",
                    "total_pages",
                    "extracted_pages",
                    "total_chunks",
                    "valid_chunks",
                    "avg_chunk_length",
                }
                filtered_quality = {k: v for k, v in quality_data.items() if k in quality_fields}

                # Convert level if it's a string
                if "level" in filtered_quality and isinstance(filtered_quality["level"], str):
                    try:
                        filtered_quality["level"] = QualityLevel(filtered_quality["level"])
                    except ValueError:
                        filtered_quality["level"] = QualityLevel.UNVALIDATED

                source.quality = SourceQuality(**filtered_quality)

        source.validation_results = data_copy.get("validation_results", {})

        if "categories" in data_copy:
            source.categories = []
            for cat in data_copy["categories"]:
                if isinstance(cat, dict):
                    # Filter to known SourceCategory fields
                    cat_fields = {
                        "name",
                        "category_type",
                        "parent_category",
                        "description",
                        "source_ids",
                        "subcategories",
                        "priority",
                        "is_primary",
                        "auto_assigned",
                    }
                    filtered_cat = {k: v for k, v in cat.items() if k in cat_fields}

                    # Convert category_type if it's a string
                    if "category_type" in filtered_cat and isinstance(
                        filtered_cat["category_type"], str
                    ):
                        try:
                            filtered_cat["category_type"] = ContentCategory(
                                filtered_cat["category_type"]
                            )
                        except ValueError:
                            continue  # Skip invalid categories

                    source.categories.append(SourceCategory(**filtered_cat))

        if "relationships" in data_copy:
            source.relationships = []
            for rel in data_copy["relationships"]:
                if isinstance(rel, dict):
                    # Filter to known SourceRelationship fields
                    rel_fields = {
                        "source_id",
                        "related_source_id",
                        "relationship_type",
                        "description",
                        "strength",
                        "bidirectional",
                    }
                    filtered_rel = {k: v for k, v in rel.items() if k in rel_fields}
                    source.relationships.append(SourceRelationship(**filtered_rel))

        source.dependencies = data_copy.get("dependencies", [])

        if "status" in data_copy:
            try:
                source.status = ProcessingStatus(data_copy["status"])
            except ValueError:
                source.status = ProcessingStatus.PENDING

        source.errors = data_copy.get("errors", [])

        if isinstance(data_copy.get("created_at"), str):
            source.created_at = datetime.fromisoformat(data_copy["created_at"])
        if isinstance(data_copy.get("updated_at"), str):
            source.updated_at = datetime.fromisoformat(data_copy["updated_at"])
        if isinstance(data_copy.get("last_accessed"), str):
            source.last_accessed = datetime.fromisoformat(data_copy["last_accessed"])

        source.access_count = data_copy.get("access_count", 0)

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
        data.update(
            {
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
                "last_used": self.last_used.isoformat() if self.last_used else None,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlavorSource":
        """Create from dictionary."""
        # Work with a copy to avoid modifying the input
        data_copy = data.copy()

        # First create base Source
        source = super().from_dict(data_copy)

        # Create FlavorSource with Source data
        flavor = cls()
        flavor.__dict__.update(source.__dict__)

        # Add flavor-specific fields
        flavor.narrative_style = data_copy.get("narrative_style", "")
        flavor.tone = data_copy.get("tone", "")
        flavor.themes = data_copy.get("themes", [])
        flavor.characters = data_copy.get("characters", [])
        flavor.locations = data_copy.get("locations", [])
        flavor.events = data_copy.get("events", [])
        flavor.quotes = data_copy.get("quotes", [])
        flavor.creativity_modifier = data_copy.get("creativity_modifier", 1.0)
        flavor.canonical = data_copy.get("canonical", False)
        flavor.priority = data_copy.get("priority", 0)
        flavor.times_used = data_copy.get("times_used", 0)

        if isinstance(data_copy.get("last_used"), str):
            flavor.last_used = datetime.fromisoformat(data_copy["last_used"])

        return flavor
