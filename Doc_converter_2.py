import pdfplumber
import os

def convert_pdf_to_markdown(pdf_path):
    markdown_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                markdown_content.append(f"## Page {page_num + 1}\n\n{text}\n\n")
                
                # Extract tables if any
                tables = page.extract_tables()
                for table_num, table in enumerate(tables):
                    markdown_content.append(f"### Table {table_num + 1}\n\n")
                    markdown_table = []
                    # Create markdown table header
                    markdown_table.append("| " + " | ".join(str(cell) if cell else "" for cell in table[0]) + " |")
                    markdown_table.append("| " + " | ".join(["---"] * len(table[0])) + " |")
                    
                    # Create markdown table rows
                    for row in table[1:]:
                        markdown_table.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
                    
                    markdown_content.append("\n".join(markdown_table) + "\n\n")
    
    return "\n".join(markdown_content)

# Path to the PDF file
pdf_path = "Test_pdfs\\ICIHLIP22012V012223.pdf"

# Convert PDF to markdown
markdown_text = convert_pdf_to_markdown(pdf_path)

# Save markdown to a file
output_path = "converted_document_2.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(f"Markdown file has been saved as '{output_path}'")