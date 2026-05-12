import fitz  # PyMuPDF
import os
import re

def convert_pdf_to_markdown(pdf_path):
    """Convert a PDF file to markdown text with proper column-by-column reading"""
    markdown_content = []
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Process each page
    for page_num, page in enumerate(doc):
        
        # Add page header
        markdown_content.append(f"## Page {page_num + 1}\n")
        
        # Check if page has multi-column layout
        if is_multi_column_page(page):
            page_text = extract_columns_separately(page)
        else:
            # Single column - use regular extraction with proper ordering
            page_text = extract_single_column_ordered(page)
        
        # Process text
        page_text = re.sub(r'\n{3,}', '\n\n', page_text)
        
        # Process headings
        lines = page_text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append(line)
                continue
                
            # Check if line looks like a heading
            if (line.isupper() and len(line) > 3 and len(line) < 50) or re.match(r'^\d+\.', line):
                processed_lines.append(f"### {line}")
            else:
                processed_lines.append(line)
        
        processed_text = '\n'.join(processed_lines)
        markdown_content.append(processed_text + "\n\n")
    
    # Close the document
    doc.close()
    
    return "".join(markdown_content)

def extract_columns_separately(page):
    """Extract left column completely, then right column completely"""
    
    # Get page dimensions
    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height
    
    # Define column boundaries
    # Left column: 0% to 48% of page width
    left_rect = fitz.Rect(0, 0, page_width * 0.48, page_height)
    
    # Right column: 52% to 100% of page width  
    right_rect = fitz.Rect(page_width * 0.52, 0, page_width, page_height)
    
    # Extract left column text with proper ordering
    left_text = extract_text_from_rect_ordered(page, left_rect)
    
    # Extract right column text with proper ordering
    right_text = extract_text_from_rect_ordered(page, right_rect)
    
    # Combine: ENTIRE left column first, then ENTIRE right column
    combined_text = ""
    
    if left_text.strip():
        combined_text += "<!-- LEFT COLUMN -->\n" + left_text.strip()
    
    if right_text.strip():
        if combined_text:
            combined_text += "\n\n<!-- RIGHT COLUMN -->\n" + right_text.strip()
        else:
            combined_text = right_text.strip()
    
    return combined_text

def extract_text_from_rect_ordered(page, rect):
    """Extract text from a specific rectangle with proper top-to-bottom ordering"""
    
    # Get text dictionary for the entire page
    text_dict = page.get_text("dict")
    
    text_blocks = []
    
    # Extract text spans that fall within the rectangle
    for block in text_dict["blocks"]:
        if "lines" in block:  # This is a text block
            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])
                    
                    # Check if this span overlaps with our target rectangle
                    if rect.intersects(span_rect) and span["text"].strip():
                        text_blocks.append({
                            'text': span["text"],
                            'y': span["bbox"][1],  # top y coordinate
                            'x': span["bbox"][0],  # left x coordinate
                            'bbox': span["bbox"]
                        })
    
    if not text_blocks:
        return ""
    
    # Sort by Y position (top to bottom), then by X position (left to right)
    text_blocks.sort(key=lambda block: (block['y'], block['x']))
    
    # Group blocks into lines based on Y position
    lines = []
    current_line = []
    current_y = None
    y_tolerance = 3  # pixels
    
    for block in text_blocks:
        if current_y is None or abs(block['y'] - current_y) <= y_tolerance:
            # Same line
            current_line.append(block)
            current_y = block['y']
        else:
            # New line
            if current_line:
                # Sort current line by X position and combine
                current_line.sort(key=lambda b: b['x'])
                line_text = ''.join([b['text'] for b in current_line])
                if line_text.strip():
                    lines.append(line_text.strip())
            
            current_line = [block]
            current_y = block['y']
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda b: b['x'])
        line_text = ''.join([b['text'] for b in current_line])
        if line_text.strip():
            lines.append(line_text.strip())
    
    return '\n'.join(lines)

def extract_single_column_ordered(page):
    """Extract text from single column with proper ordering"""
    return extract_text_from_rect_ordered(page, page.rect)

def is_multi_column_page(page):
    """Detect if a page has multi-column layout"""
    # Get text blocks with positions
    text_dict = page.get_text("dict")
    
    if not text_dict["blocks"]:
        return False
    
    x_positions = []
    
    # Collect x-positions of all text
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        x_positions.append(span["bbox"][0])
    
    if len(x_positions) < 10:  # Not enough text to determine
        return False
    
    # Analyze distribution
    page_width = page.rect.width
    left_half = [x for x in x_positions if x < page_width * 0.5]
    right_half = [x for x in x_positions if x >= page_width * 0.5]
    
    # If we have significant text in both halves, it's likely multi-column
    return (len(left_half) > len(x_positions) * 0.25 and 
            len(right_half) > len(x_positions) * 0.25)

def convert_pdf_simple_column_extraction(pdf_path):
    """Simplified version using PyMuPDF's get_textbox for cleaner column extraction"""
    markdown_content = []
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Process each page
    for page_num, page in enumerate(doc):
        
        # Add page header
        markdown_content.append(f"## Page {page_num + 1}\n")
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Check if multi-column by testing different extraction methods
        full_text = page.get_text()
        
        # Try column extraction
        left_rect = fitz.Rect(0, 0, page_width * 0.48, page_height)
        right_rect = fitz.Rect(page_width * 0.52, 0, page_width, page_height)
        
        left_text = page.get_textbox(left_rect).strip()
        right_text = page.get_textbox(right_rect).strip()
        
        # If both columns have substantial text, treat as multi-column
        if len(left_text) > 50 and len(right_text) > 50:
            # Multi-column: left column first, then right column
            page_text = ""
            if left_text:
                page_text += f"<!-- LEFT COLUMN -->\n{left_text}"
            if right_text:
                page_text += f"\n\n<!-- RIGHT COLUMN -->\n{right_text}"
        else:
            # Single column
            page_text = full_text
        
        # Process text
        page_text = re.sub(r'\n{3,}', '\n\n', page_text)
        
        # Process headings
        lines = page_text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append(line)
                continue
                
            # Check if line looks like a heading
            if (line.isupper() and len(line) > 3 and len(line) < 50) or re.match(r'^\d+\.', line):
                processed_lines.append(f"### {line}")
            else:
                processed_lines.append(line)
        
        processed_text = '\n'.join(processed_lines)
        markdown_content.append(processed_text + "\n\n")
    
    # Close the document
    doc.close()
    
    return "".join(markdown_content)

# Usage example
if __name__ == "__main__":
    # Path to the PDF
    pdf_path = "Test_pdfs\\BAJHLIP23020V012223.pdf"
    
    print("Converting PDF to Markdown with proper column-by-column reading...")
    
    # Method 1: Advanced column extraction
    try:
        markdown_content = convert_pdf_to_markdown(pdf_path)
        
        with open("converted_document_columns.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print("✓ Advanced method: 'converted_document_columns.md' created.")
        print("  This reads ENTIRE left column first, then ENTIRE right column.")
    except Exception as e:
        print(f"✗ Advanced method failed: {e}")
    
    # Method 2: Simple column extraction (more reliable)
    try:
        markdown_content_simple = convert_pdf_simple_column_extraction(pdf_path)
        
        with open("converted_document_simple_columns.md", "w", encoding="utf-8") as f:
            f.write(markdown_content_simple)
        
        print("✓ Simple method: 'converted_document_simple_columns.md' created.")
        print("  This uses get_textbox for cleaner column separation.")
    except Exception as e:
        print(f"✗ Simple method failed: {e}")
    
    print("\nCheck the generated files. The simple method usually works better for clean column separation.")