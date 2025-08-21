"""Content chunking module for processing extracted PDF text."""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import uuid

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class ContentChunk:
    """Represents a chunk of content from a document."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'rule', 'table', 'narrative', 'stat_block', 'spell', 'monster'
    page_start: int
    page_end: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    char_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        """Calculate counts after initialization."""
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())


class ContentChunker:
    """Handles intelligent content chunking with semantic boundaries."""
    
    def __init__(
        self,
        max_chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize content chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.current_section = None
        self.current_subsection = None
    
    def chunk_document(
        self,
        pdf_content: Dict[str, Any],
        source_metadata: Dict[str, Any]
    ) -> List[ContentChunk]:
        """
        Chunk an entire document intelligently.
        
        Args:
            pdf_content: Extracted PDF content
            source_metadata: Metadata about the source
            
        Returns:
            List of content chunks
        """
        chunks = []
        
        # Process tables separately
        if "tables" in pdf_content:
            table_chunks = self._chunk_tables(pdf_content["tables"], source_metadata)
            chunks.extend(table_chunks)
        
        # Process main text content
        for page in pdf_content["pages"]:
            page_chunks = self._chunk_page(
                page,
                source_metadata,
                pdf_content.get("toc", [])
            )
            chunks.extend(page_chunks)
        
        # Deduplicate chunks
        chunks = self._deduplicate_chunks(chunks)
        
        logger.info(
            f"Document chunked",
            total_chunks=len(chunks),
            source=source_metadata.get("rulebook_name", "Unknown"),
        )
        
        return chunks
    
    def _chunk_page(
        self,
        page: Dict[str, Any],
        source_metadata: Dict[str, Any],
        toc: List[Dict]
    ) -> List[ContentChunk]:
        """
        Chunk a single page of content.
        
        Args:
            page: Page content dictionary
            source_metadata: Source metadata
            toc: Table of contents for section detection
            
        Returns:
            List of chunks from this page
        """
        chunks = []
        text = page["text"]
        page_num = page["page_number"]
        
        # Update current section based on TOC
        self._update_current_section(page_num, toc)
        
        # Detect content blocks
        blocks = self._detect_content_blocks(text)
        
        for block in blocks:
            # Determine chunk type
            chunk_type = self._classify_content_type(block["text"])
            
            # Create chunk if it's substantial enough
            if len(block["text"].strip()) > 50:  # Minimum content threshold
                chunk = ContentChunk(
                    id=str(uuid.uuid4()),
                    content=block["text"],
                    metadata={
                        **source_metadata,
                        "page": page_num,
                        "block_type": block["type"],
                    },
                    chunk_type=chunk_type,
                    page_start=page_num,
                    page_end=page_num,
                    section=self.current_section,
                    subsection=self.current_subsection,
                )
                
                # Split large chunks if necessary
                if chunk.char_count > self.max_chunk_size:
                    sub_chunks = self._split_large_chunk(chunk)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk)
        
        return chunks
    
    def _detect_content_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect semantic content blocks in text.
        
        Args:
            text: Page text
            
        Returns:
            List of content blocks
        """
        blocks = []
        
        # Split by double newlines first (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_block = []
        current_type = None
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect block type
            block_type = self._detect_block_type(para)
            
            # If type changes, save current block and start new one
            if current_type and block_type != current_type:
                if current_block:
                    blocks.append({
                        "text": "\n\n".join(current_block),
                        "type": current_type,
                    })
                current_block = [para]
                current_type = block_type
            else:
                current_block.append(para)
                current_type = block_type
        
        # Don't forget the last block
        if current_block:
            blocks.append({
                "text": "\n\n".join(current_block),
                "type": current_type,
            })
        
        return blocks
    
    def _detect_block_type(self, text: str) -> str:
        """
        Detect the type of content block.
        
        Args:
            text: Block text
            
        Returns:
            Block type identifier
        """
        # Check for stat block patterns
        if re.search(r'\b(AC|HP|Speed|STR|DEX|CON|INT|WIS|CHA)\b.*\d+', text):
            return "stat_block"
        
        # Check for spell format
        if re.search(r'\b(Casting Time|Range|Components|Duration):', text):
            return "spell"
        
        # Check for table-like content
        if text.count('|') > 3 or text.count('\t') > 3:
            return "table"
        
        # Check for numbered rules
        if re.match(r'^\d+\.[\d.]*\s+', text):
            return "rule"
        
        # Check for header
        if re.match(r'^[A-Z][A-Z\s]+$', text.split('\n')[0]) and len(text.split('\n')[0]) < 100:
            return "header"
        
        # Default to narrative
        return "narrative"
    
    def _classify_content_type(self, text: str) -> str:
        """
        Classify the content type for search and retrieval.
        
        Args:
            text: Content text
            
        Returns:
            Content type classification
        """
        text_lower = text.lower()
        
        # Monster/creature detection
        if any(keyword in text_lower for keyword in [
            'challenge rating', 'hit points', 'armor class', 'creature',
            'monster', 'beast', 'dragon', 'undead'
        ]):
            return "monster"
        
        # Spell detection
        if any(keyword in text_lower for keyword in [
            'casting time', 'spell level', 'components', 'duration',
            'spell save', 'spell attack'
        ]):
            return "spell"
        
        # Rule detection
        if any(keyword in text_lower for keyword in [
            'must', 'cannot', 'when you', 'if you', 'rule',
            'action', 'bonus action', 'reaction'
        ]):
            return "rule"
        
        # Stat block detection
        if re.search(r'\b(STR|DEX|CON|INT|WIS|CHA)\s*\d+', text):
            return "stat_block"
        
        # Table detection
        if text.count('|') > 5 or text.count('\t') > 5:
            return "table"
        
        # Default to narrative
        return "narrative"
    
    def _chunk_tables(
        self,
        tables: List[Dict],
        source_metadata: Dict[str, Any]
    ) -> List[ContentChunk]:
        """
        Create chunks from extracted tables.
        
        Args:
            tables: List of extracted tables
            source_metadata: Source metadata
            
        Returns:
            List of table chunks
        """
        chunks = []
        
        for table_info in tables:
            # Convert table to text representation
            table_text = self._table_to_text(table_info["data"])
            
            if len(table_text) > 50:  # Minimum content threshold
                chunk = ContentChunk(
                    id=str(uuid.uuid4()),
                    content=table_text,
                    metadata={
                        **source_metadata,
                        "page": table_info["page"],
                        "table_index": table_info["table_index"],
                        "rows": table_info["rows"],
                        "cols": table_info["cols"],
                    },
                    chunk_type="table",
                    page_start=table_info["page"],
                    page_end=table_info["page"],
                    section=self.current_section,
                )
                chunks.append(chunk)
        
        return chunks
    
    def _table_to_text(self, table_data: List[List]) -> str:
        """
        Convert table data to searchable text format.
        
        Args:
            table_data: 2D list of table cells
            
        Returns:
            Text representation of table
        """
        if not table_data:
            return ""
        
        lines = []
        
        # Add header if present
        if table_data[0]:
            header = " | ".join(str(cell) if cell else "" for cell in table_data[0])
            lines.append(header)
            lines.append("-" * len(header))
        
        # Add data rows
        for row in table_data[1:]:
            if row:
                row_text = " | ".join(str(cell) if cell else "" for cell in row)
                lines.append(row_text)
        
        return "\n".join(lines)
    
    def _split_large_chunk(self, chunk: ContentChunk) -> List[ContentChunk]:
        """
        Split a large chunk into smaller overlapping chunks.
        
        Args:
            chunk: Large content chunk
            
        Returns:
            List of smaller chunks
        """
        sub_chunks = []
        text = chunk.content
        
        # Find natural break points (sentences, paragraphs)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Create chunk
                sub_chunk_text = " ".join(current_chunk)
                sub_chunk = ContentChunk(
                    id=str(uuid.uuid4()),
                    content=sub_chunk_text,
                    metadata=chunk.metadata.copy(),
                    chunk_type=chunk.chunk_type,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    subsection=chunk.subsection,
                )
                sub_chunks.append(sub_chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_size = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        overlap_sentences.insert(0, sent)
                        if overlap_size >= self.chunk_overlap:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            sub_chunk_text = " ".join(current_chunk)
            sub_chunk = ContentChunk(
                id=str(uuid.uuid4()),
                content=sub_chunk_text,
                metadata=chunk.metadata.copy(),
                chunk_type=chunk.chunk_type,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                subsection=chunk.subsection,
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _deduplicate_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """
        Remove duplicate or near-duplicate chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Deduplicated list of chunks
        """
        seen_content = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a normalized version for comparison
            normalized = re.sub(r'\s+', ' ', chunk.content.lower().strip())
            
            # Check if we've seen similar content
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"Duplicate chunk removed: {chunk.id}")
        
        return unique_chunks
    
    def _update_current_section(self, page_num: int, toc: List[Dict]):
        """
        Update current section based on page number and TOC.
        
        Args:
            page_num: Current page number
            toc: Table of contents
        """
        for entry in toc:
            if "page" in entry and entry["page"] <= page_num:
                if entry["level"] == 0:
                    self.current_section = entry["title"]
                    self.current_subsection = None
                elif entry["level"] == 1:
                    self.current_subsection = entry["title"]