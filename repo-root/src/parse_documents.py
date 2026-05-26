"""
Module: parse_documents.py
Functionality: Enhanced PDF parsing using PyMuPDF with advanced table extraction and layout understanding.
"""
import os
import re
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF for PDF handling
import pandas as pd

def detect_table_structures(page, min_rows=2, min_cols=2) -> List[Dict]:
    """
    Detect table structures using PyMuPDF's table finding capabilities.
    
    Args:
        page: PyMuPDF page object
        min_rows: Minimum number of rows to consider a table
        min_cols: Minimum number of columns to consider a table
        
    Returns:
        List of detected tables with their content
    """
    tables = []
    
    try:
        # Use PyMuPDF's find_tables method if available
        if hasattr(page, 'find_tables'):
            found_tables = page.find_tables()
            
            for table in found_tables:
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    if table_data and len(table_data) >= min_rows:
                        # Filter out empty rows and columns
                        filtered_data = []
                        for row in table_data:
                            if row and any(cell and str(cell).strip() for cell in row):
                                filtered_data.append([str(cell).strip() if cell else "" for cell in row])
                        
                        if len(filtered_data) >= min_rows and len(filtered_data[0]) >= min_cols:
                            # Create DataFrame and convert to markdown
                            try:
                                df = pd.DataFrame(filtered_data[1:], columns=filtered_data[0])
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                
                                if df.shape[0] >= 1 and df.shape[1] >= min_cols:
                                    table_markdown = df.to_markdown(index=False, tablefmt='pipe')
                                    tables.append({
                                        'content': table_markdown,
                                        'type': 'table',
                                        'rows': len(df) + 1,  # +1 for header
                                        'cols': len(df.columns),
                                        'bbox': table.bbox if hasattr(table, 'bbox') else None
                                    })
                            except Exception as e:
                                print(f"⚠️ Table DataFrame creation error: {e}")
                                continue
                                
                except Exception as e:
                    print(f"⚠️ Table extraction error: {e}")
                    continue
        
    except Exception as e:
        print(f"⚠️ Table detection error: {e}")
    
    return tables

def extract_text_blocks(page) -> List[Dict]:
    """
    Extract text blocks with positioning information for better layout understanding.
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        List of text blocks with metadata
    """
    text_blocks = []
    
    try:
        # Get text blocks with position information
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = ""
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            block_text += text + " "
                
                if block_text.strip():
                    text_blocks.append({
                        'text': block_text.strip(),
                        'bbox': block.get("bbox", [0, 0, 0, 0]),
                        'type': 'text_block'
                    })
    
    except Exception as e:
        print(f"⚠️ Text block extraction error: {e}")
        # Fallback to simple text extraction
        try:
            simple_text = page.get_text()
            if simple_text.strip():
                text_blocks.append({
                    'text': simple_text.strip(),
                    'bbox': [0, 0, 0, 0],
                    'type': 'text_block'
                })
        except Exception as fallback_error:
            print(f"⚠️ Fallback text extraction error: {fallback_error}")
    
    return text_blocks

def group_text_into_paragraphs(text_blocks: List[Dict]) -> List[str]:
    """
    Group text blocks into logical paragraphs based on positioning and content.
    
    Args:
        text_blocks: List of text blocks with bbox information
        
    Returns:
        List of paragraph texts
    """
    if not text_blocks:
        return []
    
    # Sort blocks by vertical position (top to bottom)
    sorted_blocks = sorted(text_blocks, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    
    paragraphs = []
    current_paragraph = []
    last_y = None
    paragraph_threshold = 20  # Pixel threshold for paragraph breaks
    
    for block in sorted_blocks:
        text = block['text'].strip()
        if not text:
            continue
            
        y_pos = block['bbox'][1]  # Top y-coordinate
        
        # Check if this should start a new paragraph
        if last_y is not None and abs(y_pos - last_y) > paragraph_threshold:
            # Save current paragraph and start new one
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                if len(paragraph_text.strip()) > 15:  # Filter very short text
                    paragraphs.append(paragraph_text.strip())
            current_paragraph = [text]
        else:
            current_paragraph.append(text)
        
        last_y = y_pos
    
    # Add final paragraph
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        if len(paragraph_text.strip()) > 15:
            paragraphs.append(paragraph_text.strip())
    
    return paragraphs

def parse_document_enhanced_pymupdf(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """
    Optimized PDF parsing using PyMuPDF with fast, simple extraction.
    Now uses optimized approach for better performance.
    
    Args:
        pdf_path: Path to the PDF file
        save_parsed_text: Whether to save parsed content to text files
        
    Returns:
        Dictionary containing parsed content
    """
    try:
        doc_name = os.path.basename(pdf_path)
        
        print(f"🚀 Starting optimized PyMuPDF parsing for {doc_name}...")
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"📄 Processing {total_pages} pages...")
        
        processing_start = time.time()
        
        all_text = ""
        ordered_content = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Simple, fast text extraction
            page_text = page.get_text()
            
            if page_text.strip():
                # Clean the text
                cleaned_text = clean_text(page_text)
                
                if cleaned_text:
                    all_text += cleaned_text + "\n\n"
                    
                    ordered_content.append({
                        'content': cleaned_text,
                        'type': 'text',
                        'page': page_num + 1,
                        'source': 'optimized_pymupdf'
                    })
            
            if (page_num + 1) % 10 == 0:  # Progress update every 10 pages
                print(f"   📖 Processed {page_num + 1}/{total_pages} pages...")
        
        doc.close()
        
        processing_time = time.time() - processing_start
        
        # Final text cleanup
        final_text = clean_text(all_text)
        
        result = {
            'document_name': doc_name,
            'content': final_text,
            'ordered_content': ordered_content,
            'total_pages': total_pages,
            'parsing_method': 'optimized_pymupdf',
            'processing_time': processing_time,
            'metadata': {
                'total_elements': len(ordered_content),
                'text_elements': len(ordered_content),
                'table_elements': 0,
                'pages_processed': total_pages,
                'characters_extracted': len(final_text)
            }
        }
        
        print(f"✅ Optimized parsing complete: {len(final_text)} characters in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"❌ Enhanced PyMuPDF parsing error: {e}")
        # Fallback to simple extraction
        return parse_document_simple_fallback(pdf_path, save_parsed_text)

def parse_document_simple_fallback(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """
    Simple fallback parsing method using PyMuPDF only.
    
    Args:
        pdf_path: Path to the PDF file
        save_parsed_text: Whether to save parsed content to text files
        
    Returns:
        Dictionary containing parsed content
    """
    try:
        doc = fitz.open(pdf_path)
        doc_name = os.path.basename(pdf_path)
        total_pages = len(doc)
        
        all_text = ""
        ordered_content = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                all_text += text + "\n\n"
                ordered_content.append({
                    'content': text.strip(),
                    'type': 'text',
                    'page': page_num + 1,
                    'source': 'simple_pymupdf'
                })
        
        doc.close()
        
        return {
            'document_name': doc_name,
            'content': all_text,
            'ordered_content': ordered_content,
            'total_pages': total_pages,
            'parsing_method': 'simple_pymupdf',
            'metadata': {
                'total_elements': len(ordered_content),
                'text_elements': len(ordered_content),
                'table_elements': 0,
                'pages_processed': total_pages
            }
        }
        
    except Exception as e:
        print(f"❌ Simple fallback parsing error: {e}")
        raise

# Main parsing function - now uses enhanced PyMuPDF by default
def parse_document_paddleocr(pdf_path: str, save_parsed_text: bool = False, use_gpu: bool = False) -> dict:
    """
    Parse PDF using enhanced PyMuPDF (legacy function name for compatibility).
    
    Args:
        pdf_path: Path to the PDF file
        save_parsed_text: Whether to save parsed content to text files
        use_gpu: Ignored (for compatibility)
        
    Returns:
        Dictionary containing parsed content with improved accuracy
    """
    return parse_document_enhanced_pymupdf(pdf_path, save_parsed_text)

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove excessive whitespace while preserving line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
    
    return text.strip()

def load_and_parse_documents(document_paths: List[str], save_parsed_text: bool = False) -> List[Dict]:
    """
    Parse multiple PDF documents using Enhanced PyMuPDF.
    
    Args:
        document_paths: List of paths to PDF documents
        save_parsed_text: Whether to save parsed content to text files for inspection
    """
    parsed_docs = []
    
    for path in document_paths:
        doc_name = os.path.basename(path)
        
        if not os.path.exists(path):
            parsed_docs.append({
                'document_name': doc_name, 
                'parsed_output': {"error": f"File not found: {path}"}
            })
            continue
        
        print(f"🔄 Processing {doc_name} with Enhanced PyMuPDF...")
        parsed_output = parse_document_enhanced_pymupdf(path, save_parsed_text=save_parsed_text)
        parsed_docs.append({
            'document_name': doc_name, 
            'parsed_output': parsed_output
        })
    
    return parsed_docs

def load_and_parse_from_folder(docs_folder: str, file_filter: Optional[List[str]] = None, save_parsed_text: bool = False) -> List[Dict]:
    """
    Load and parse documents from a folder using Enhanced PyMuPDF, optionally filtering by filenames.
    
    Args:
        docs_folder: Folder containing PDF documents
        file_filter: Optional list of filenames to process. If None, processes all PDFs.
        save_parsed_text: Whether to save parsed content to text files for inspection
    """
    if not os.path.exists(docs_folder):
        return []
    
    # Get all PDF files in folder
    all_files = [f for f in os.listdir(docs_folder) if f.lower().endswith('.pdf')]
    
    # Apply filter if provided
    if file_filter:
        files_to_process = [f for f in all_files if f in file_filter]
    else:
        files_to_process = all_files
    
    print(f"📁 Found {len(files_to_process)} PDF files to process with Enhanced PyMuPDF")
    
    # Create full paths
    document_paths = [os.path.join(docs_folder, filename) for filename in files_to_process]
    
    return load_and_parse_documents(document_paths, save_parsed_text=save_parsed_text)
