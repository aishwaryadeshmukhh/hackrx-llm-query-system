"""
Optimized text-based chunking without GPU dependencies.
Fast, memory-efficient, and reliable chunking for document processing.
"""

import re
import math
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import hashlib
import time

@dataclass
class OptimizedChunkConfig:
    """Configuration for optimized text chunking."""
    chunk_size: int = 800          # Target characters per chunk
    chunk_overlap: int = 150       # Overlap between chunks in characters
    min_chunk_size: int = 200      # Minimum chunk size
    max_chunk_size: int = 1500     # Maximum chunk size
    split_on_sentences: bool = True # Try to split on sentence boundaries
    preserve_paragraphs: bool = True # Try to keep paragraphs intact


class OptimizedTextChunker:
    """
    Fast, CPU-only text chunker optimized for performance and simplicity.
    
    Features:
    - Pure text-based chunking (no embeddings/GPU)
    - Sentence and paragraph boundary awareness
    - Configurable overlap and sizing
    - Memory efficient processing
    - Fast execution for large documents
    """
    
    def __init__(self, config: Optional[OptimizedChunkConfig] = None):
        self.config = config or OptimizedChunkConfig()
        print(f"✅ OptimizedTextChunker initialized (CPU-only, {self.config.chunk_size} chars/chunk)")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 20:  # Filter very short paragraphs
                cleaned_paragraphs.append(paragraph)
        
        return cleaned_paragraphs
    
    def _create_overlapping_chunks(self, text_units: List[str], unit_type: str = "sentence") -> List[str]:
        """Create overlapping chunks from text units."""
        chunks = []
        current_chunk = ""
        current_length = 0
        overlap_buffer = []
        
        for i, unit in enumerate(text_units):
            unit_length = len(unit)
            
            # Check if adding this unit would exceed max chunk size
            if current_length + unit_length > self.config.max_chunk_size and current_chunk:
                # Finalize current chunk if it meets minimum size
                if current_length >= self.config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap for next chunk
                    overlap_text = ""
                    overlap_length = 0
                    
                    # Add units to overlap buffer (from end of current chunk)
                    for j in range(len(overlap_buffer) - 1, -1, -1):
                        if overlap_length + len(overlap_buffer[j]) <= self.config.chunk_overlap:
                            overlap_text = overlap_buffer[j] + " " + overlap_text
                            overlap_length += len(overlap_buffer[j])
                        else:
                            break
                    
                    # Start new chunk with overlap
                    current_chunk = overlap_text.strip()
                    current_length = len(current_chunk)
                    overlap_buffer = []
                else:
                    # Current chunk too small, continue building
                    pass
            
            # Add current unit to chunk
            if current_chunk:
                current_chunk += " " + unit
            else:
                current_chunk = unit
            
            current_length += unit_length + 1  # +1 for space
            overlap_buffer.append(unit)
            
            # Keep overlap buffer reasonable size
            if len(overlap_buffer) > 10:
                overlap_buffer.pop(0)
        
        # Add final chunk if it has content
        if current_chunk.strip() and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _simple_chunk_by_size(self, text: str) -> List[str]:
        """Simple fallback chunking by character count."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.config.chunk_size
            
            # If this is not the last chunk, try to end at a good break point
            if end < text_length:
                # Look for sentence endings within reasonable distance
                search_start = max(start + self.config.chunk_size - 100, start)
                search_end = min(end + 100, text_length)
                
                # Find the best break point (sentence ending)
                best_break = -1
                for i in range(search_end - 1, search_start - 1, -1):
                    if text[i] in '.!?':
                        # Make sure next character is whitespace or end
                        if i + 1 >= text_length or text[i + 1].isspace():
                            best_break = i + 1
                            break
                
                if best_break > start:
                    end = best_break
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if chunk and len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position (with overlap)
            if end >= text_length:
                break
            
            start = max(end - self.config.chunk_overlap, start + 1)
        
        return chunks
    
    def chunk_text(self, text: str, document_name: str = "unknown") -> List[Dict[str, Any]]:
        """
        Chunk a single text document into optimized chunks.
        
        Args:
            text: Input text to chunk
            document_name: Name of the document for metadata
            
        Returns:
            List of chunk dictionaries
        """
        if not text or not text.strip():
            return []
        
        start_time = time.time()
        text = text.strip()
        
        print(f"📄 Chunking {document_name}: {len(text):,} characters")
        
        # Try different chunking strategies based on text characteristics
        chunks = []
        
        if self.config.preserve_paragraphs and '\n\n' in text:
            # Strategy 1: Paragraph-aware chunking
            paragraphs = self._split_into_paragraphs(text)
            
            if paragraphs:
                print(f"📝 Using paragraph-based chunking ({len(paragraphs)} paragraphs)")
                chunks = self._create_overlapping_chunks(paragraphs, "paragraph")
        
        # If paragraph chunking didn't work well, try sentence-based
        if not chunks and self.config.split_on_sentences:
            sentences = self._split_into_sentences(text)
            
            if len(sentences) > 5:  # Only if we have enough sentences
                print(f"📝 Using sentence-based chunking ({len(sentences)} sentences)")
                chunks = self._create_overlapping_chunks(sentences, "sentence")
        
        # Fallback to simple character-based chunking
        if not chunks:
            print(f"📝 Using character-based chunking (fallback)")
            chunks = self._simple_chunk_by_size(text)
        
        # Convert to structured format
        structured_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_name}_{i}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
            
            structured_chunks.append({
                'chunk_id': chunk_id,
                'document_name': document_name,
                'content': chunk_text,
                'chunk_index': i,
                'char_count': len(chunk_text),
                'metadata': {
                    'chunking_method': 'optimized_text',
                    'source_document': document_name,
                    'chunk_size_config': self.config.chunk_size,
                    'overlap_config': self.config.chunk_overlap
                }
            })
        
        processing_time = time.time() - start_time
        print(f"✅ Created {len(structured_chunks)} chunks in {processing_time:.2f}s")
        
        return structured_chunks


def chunk_documents_optimized(parsed_content: List[Dict[str, Any]], 
                            chunk_size: int = 800,
                            chunk_overlap: int = 150,
                            save_parsed_text: bool = False,
                            output_dir: str = "results") -> List[Dict[str, Any]]:
    """
    Optimized document chunking function - CPU only, fast and reliable.
    
    Args:
        parsed_content: List of parsed document dictionaries
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        save_parsed_text: Whether to save text files (unused in optimized version)
        output_dir: Output directory (unused in optimized version)
        
    Returns:
        List of chunk dictionaries
    """
    if not parsed_content:
        return []
    
    print(f"🚀 Starting Optimized Text Chunking...")
    start_time = time.time()
    
    # Create optimized chunker
    config = OptimizedChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=max(50, chunk_size // 8),  # More reasonable minimum
        max_chunk_size=chunk_size * 2,
        split_on_sentences=True,
        preserve_paragraphs=True
    )
    
    chunker = OptimizedTextChunker(config)
    all_chunks = []
    
    for doc_data in parsed_content:
        doc_name = doc_data.get('document_name', 'unknown')
        content = doc_data.get('content', '')
        
        if not content or not content.strip():
            print(f"⚠️ Skipping empty document: {doc_name}")
            continue
        
        # Chunk the document
        doc_chunks = chunker.chunk_text(content, doc_name)
        all_chunks.extend(doc_chunks)
    
    total_time = time.time() - start_time
    total_chars = sum(len(doc.get('content', '')) for doc in parsed_content)
    avg_chunk_size = sum(len(chunk['content']) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
    
    print(f"✅ Optimized Text Chunking complete!")
    print(f"📊 Generated {len(all_chunks)} total chunks")
    if total_time > 0:
        print(f"⚡ Processing speed: {total_chars / total_time:.0f} chars/sec")
    else:
        print(f"⚡ Processing speed: >1M chars/sec (instantaneous)")
    print(f"📏 Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"⏱️ Total time: {total_time:.3f}s")
    
    return all_chunks
