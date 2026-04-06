import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from pathlib import Path
from dotenv import load_dotenv

from extractor import run_extraction
from vector_store import index_document
from validator import validate_dates, format_validation_report

load_dotenv()

# Tell pytesseract where Tesseract is installed on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── DATE PATTERNS ───────────────────────────────────────────
DATE_PATTERNS = [
    # DD/MM/YYYY or MM/DD/YYYY
    (r'\b([0-3]?[0-9]/[01]?[0-9]/20[0-9]{2})\b', 'DD/MM/YYYY or MM/DD/YYYY'),
    
    # ISO YYYY-MM-DD
    (r'\b(20[0-9]{2}-[01][0-9]-[0-3][0-9])\b', 'ISO YYYY-MM-DD'),
    
    # DD-MM-YYYY
    (r'\b([0-3]?[0-9]-[01]?[0-9]-20[0-9]{2})\b', 'DD-MM-YYYY'),
    
    # Full month name + DD + YYYY — e.g. January 15, 2019 or January 15 2019
    (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-3]?[0-9],?\s+20[0-9]{2})\b', 'Full Month DD YYYY'),
    
    # Abbreviated month — e.g. Jan 15, 2019
    (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+[0-3]?[0-9],?\s+20[0-9]{2})\b', 'MMM DD YYYY'),
    
    # DD Full Month YYYY — e.g. 15 January 2019
    (r'\b([0-3]?[0-9]\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20[0-9]{2})\b', 'DD Full Month YYYY'),

    # DD MMM YYYY — e.g. 15 Jan 2019
    (r'\b([0-3]?[0-9]\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+20[0-9]{2})\b', 'DD MMM YYYY'),
    
    # YYYYMMDD compact
    (r'\b(20[0-9]{2}[01][0-9][0-3][0-9])\b', 'YYYYMMDD compact'),
    
    # AMBIGUOUS short — e.g. 01/02/03
    (r'\b([0-9]{2}/[0-9]{2}/[0-9]{2})\b', 'AMBIGUOUS short'),
]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF - tries direct first, falls back to OCR"""
    doc = fitz.open(pdf_path)
    full_text = ""
    page_texts = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        if len(text.strip()) < 50:  # Likely scanned - use OCR
            print(f"  Page {page_num+1}: scanned → using OCR")
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img)
        else:
            print(f"  Page {page_num+1}: digital → direct extraction")
        
        page_texts.append((page_num+1, text))
        full_text += text
    
    doc.close()
    return full_text, page_texts

def extract_dates(text, page_num=None):
    """Extract all dates with their format type"""
    found_dates = []
    for pattern, format_name in DATE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            found_dates.append({
                'raw': match.group(),
                'format': format_name,
                'page': page_num,
                'ambiguous': 'AMBIGUOUS' in format_name or 
                           format_name == 'DD/MM/YYYY or MM/DD/YYYY'
            })
    return found_dates

def run_compliance_scan(pdf_path):
    """Main compliance scan on a single document"""
    print(f"\n{'='*50}")
    print(f"Scanning: {Path(pdf_path).name}")
    print('='*50)
    
    # Extract text
    full_text, page_texts = extract_text_from_pdf(pdf_path)
    
    # Extract dates per page
    result = run_extraction(page_texts, Path(pdf_path).name, full_text)
    print(f"  Sections: {[s['section_type'] for s in result['sections']]}")
    print(f"  Packet summary: {result['packet_summary'][:150]}...")
    index_document(page_texts, result["dates"], Path(pdf_path).name, result.get("sections", []))
    validation = validate_dates(result["dates"])
    print(f"\n  Validation: {validation['critical_count']} critical, "
        f"{validation['warning_count']} warnings")
    if validation["critical"]:
        for f in validation["critical"]:
            print(f"  🔴 {f['message']} (Page {f['page']})") 
    return result
    

# ─── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    docs_folder = Path("data/sample_docs")
    pdf_files = list(docs_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found in data/sample_docs/")
    else:
        for pdf_file in pdf_files:
            run_compliance_scan(pdf_file)