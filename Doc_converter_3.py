import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np  # Added missing import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Document processing imports
import pdfplumber
import docx
import email
from email import policy
import mammoth


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    page_number: int
    chunk_id: str
    document_name: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class DocumentProcessor:
    """Handles extraction and processing of different document types"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.eml', '.msg']
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document and extract content based on file type"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        elif file_ext in ['.eml', '.msg']:
            return self.extract_from_email(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []
    
    def extract_from_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from PDF using pdfplumber with no chunking or processing"""
        chunks = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                doc_name = os.path.basename(file_path)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=6,
                        y_density=13
                    )
                    
                    if text:
                        # Extract tables if present
                        tables = page.extract_tables()
                        table_content = ""
                        if tables:
                            for i, table in enumerate(tables):
                                table_content += f"\n\nTable {i+1}:\n"
                                for row in table:
                                    if row:
                                        # Simple space-separated format instead of pipes
                                        table_content += "  ".join([str(cell) if cell else "" for cell in row]) + "\n"
                        
                        full_content = text + table_content
                        
                        chunk = DocumentChunk(
                            content=full_content,
                            page_number=page_num,
                            chunk_id=f"{doc_name}_page_{page_num}",
                            document_name=doc_name,
                            metadata={}  # Empty metadata
                        )
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
        
        return chunks
    
    def extract_from_docx(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from DOCX files"""
        chunks = []
        
        try:
            # Using mammoth for better formatting preservation
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                
                if text:
                    doc_name = os.path.basename(file_path)
                    chunk = DocumentChunk(
                        content=text,
                        page_number=1,
                        chunk_id=f"{doc_name}_full",
                        document_name=doc_name,
                        metadata={'extraction_method': 'mammoth', 'char_count': len(text)}
                    )
                    chunks.append(chunk)
                    
                    # Split if too large
                    if len(text) > 2000:
                        sub_chunks = self._split_large_chunk(chunk)
                        chunks.extend(sub_chunks)
                        
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            
        return chunks
    
    def extract_from_email(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from email files"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                msg = email.message_from_file(f, policy=policy.default)
                
            # Extract email metadata
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            date = msg.get('Date', '')
            
            # Extract email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_content()
            else:
                body = msg.get_content()
            
            if body or subject:
                full_content = f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
                
                doc_name = os.path.basename(file_path)
                chunk = DocumentChunk(
                    content=full_content,
                    page_number=1,
                    chunk_id=f"{doc_name}_email",
                    document_name=doc_name,
                    metadata={
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'type': 'email'
                    }
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing email {file_path}: {str(e)}")
            
        return chunks
    
    def _split_large_chunk(self, chunk: DocumentChunk, max_size: int = 1500) -> List[DocumentChunk]:
        """Split large chunks into smaller pieces while preserving context"""
        sub_chunks = []
        content = chunk.content
        
        # Split by sentences or paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = ""
        chunk_num = 1
        
        for sentence in sentences:
            if len(current_chunk + sentence) > max_size and current_chunk:
                # Create sub-chunk
                sub_chunk = DocumentChunk(
                    content=current_chunk,
                    page_number=chunk.page_number,
                    chunk_id=f"{chunk.chunk_id}_sub_{chunk_num}",
                    document_name=chunk.document_name,
                    metadata={**(chunk.metadata or {}), 'is_sub_chunk': True, 'parent_chunk': chunk.chunk_id}
                )
                sub_chunks.append(sub_chunk)
                current_chunk = sentence
                chunk_num += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining content
        if current_chunk:
            sub_chunk = DocumentChunk(
                content=current_chunk,
                page_number=chunk.page_number,
                chunk_id=f"{chunk.chunk_id}_sub_{chunk_num}",
                document_name=chunk.document_name,
                metadata={**(chunk.metadata or {}), 'is_sub_chunk': True, 'parent_chunk': chunk.chunk_id}
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks

    def save_to_json(self, chunks: List[DocumentChunk], output_path: str):
        """Save the extracted document chunks to a JSON file"""
        output_data = []
        for chunk in chunks:
            chunk_dict = {
                "content": chunk.content,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "metadata": chunk.metadata
            }
            output_data.append(chunk_dict)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(output_data)} chunks to {output_path}")
    
    def save_to_text(self, chunks: List[DocumentChunk], output_path: str):
        """Save the extracted document chunks to a plain text file with minimal metadata"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write file header
            f.write(f"Document: {chunks[0].document_name if chunks else 'Unknown'}\n")
            f.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for chunk in chunks:
                # Write simple page header
                f.write(f"--- Page {chunk.page_number} ---\n\n")
                
                # Write content
                f.write(chunk.content)
                f.write("\n\n")
                
        logger.info(f"Saved {len(chunks)} chunks to text file: {output_path}")
    
    def save_to_text_simple(self, chunks: List[DocumentChunk], output_path: str):
        """Save extracted text to a simple text file with only page numbers"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Only include page number
                f.write(f"--- Page {chunk.page_number} ---\n\n")
                
                # Write content
                f.write(chunk.content)
                f.write("\n\n")
                
        logger.info(f"Saved content to text file: {output_path}")


def main():
    # Path to the document you want to process
    file_path = "Test_pdfs\\ICIHLIP22012V012223.pdf"
    
    # Create processor and process the document
    processor = DocumentProcessor()
    chunks = processor.process_document(file_path)
    
    # Save as plain text only
    txt_path = "document_content_portion.txt"
    processor.save_to_text_simple(chunks, txt_path)
    
    print(f"Document saved to {txt_path}")


if __name__ == "__main__":
    main()