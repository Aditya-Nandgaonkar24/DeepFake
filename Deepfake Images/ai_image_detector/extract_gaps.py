import os
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system(f"{sys.executable} -m pip install PyMuPDF pypdf")
    import fitz

def extract_text(pdf_path):
    print(f"--- Extracting text from: {pdf_path} ---")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Just extract first 3 pages and last 2 pages to get Abstract, Intro, and Conclusion/Future Work (where gaps are usually stated)
        num_pages = len(doc)
        pages_to_extract = list(range(min(3, num_pages)))
        if num_pages > 3:
            pages_to_extract.extend(range(max(3, num_pages-2), num_pages))
            
        for page_num in pages_to_extract:
            page = doc.load_page(page_num)
            text += page.get_text()
            
        print(text[:2000]) # print summary
        print(f"\n... Extraction successful for {pdf_path}. Extracted length: {len(text)}\n")
        
        # Look specifically for keywords related to gaps
        print("Possible Gap Indicators:")
        lines = text.split('\n')
        for i, line in enumerate(lines):
            lower_line = line.lower()
            if any(word in lower_line for word in ['future work', 'limitation', 'gap', 'however', 'fails to', 'struggles with', 'challenge']):
                print(f"  Line: {line.strip()}")
                
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

if __name__ == '__main__':
    folder = r"c:\Users\adity\Downloads\ai_image_detector (2)\ai_image_detector\Research Papers"
    for file in os.listdir(folder):
        if file.endswith('.pdf'):
            extract_text(os.path.join(folder, file))
