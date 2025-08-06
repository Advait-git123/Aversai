from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re, json
import fitz

# ----------- Paths -----------
RAW_PDF_DIR = Path("docs/raw_pdfs")
CLEAN_TEXT_DIR = Path("docs/clean_text")
CHUNK_OUTPUT = CLEAN_TEXT_DIR / "chunks.jsonl"

CLEAN_TEXT_DIR.mkdir(exist_ok=True, parents=True)

# ----------- PDF to Clean Text -----------
def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    full_text = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            print(f"[✓] Page {i+1} extracted with {len(text)} characters")
            full_text.append(text)
        else:
            print(f"[!] Page {i+1} of {pdf_path.name} is empty.")
    return "\n\n".join(full_text)

def clean_text(text: str) -> str:
    # Soft cleaning to preserve paragraph structure
    text = re.sub(r"\n{2,}", "\n", text)  
    text = re.sub(r"[ \t]+", " ", text)   
    return text.strip()

def convert_pdfs_to_text():
    print("📄 Extracting text from PDFs...")
    for pdf_file in RAW_PDF_DIR.glob("*.pdf"):
        raw_text = extract_pdf_text(pdf_file)
        print(f"[DEBUG] Extracted {len(raw_text)} characters from {pdf_file.name}")
        print(f"[DEBUG] First 300 chars:\n{raw_text[:300]}...\n{'-'*60}")
        cleaned = clean_text(raw_text)
        output_path = CLEAN_TEXT_DIR / pdf_file.with_suffix(".txt").name
        output_path.write_text(cleaned, encoding="utf-8")
        print(f"[✓] Saved: {output_path.name} ({len(cleaned)} characters)")



# ----------- Text Chunking -----------
def chunk_text_files():
    print("🔪 Chunking text files...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []
    for txt_file in CLEAN_TEXT_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        print(f"[DEBUG] Text from {txt_file.name}: {text[:200]}...")
        if not text:
            print(f"[!] Skipping {txt_file.name} — empty text.")
            continue

        print(f"→ Chunking {txt_file.name} ({len(text)} chars)")
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{txt_file.stem}_c{i}"
            all_chunks.append({
                "id": chunk_id,
                "source": txt_file.name,
                "text": chunk
            })

    with open(CHUNK_OUTPUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[✓] Saved {len(all_chunks)} chunks to {CHUNK_OUTPUT}")

# ----------- Run All -----------
def run():
    convert_pdfs_to_text()
    chunk_text_files()

if __name__ == "__main__":
    run()
