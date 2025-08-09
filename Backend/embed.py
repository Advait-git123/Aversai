from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import json

CHUNKS_FILE = Path("Backend/docs/clean_text/chunks.jsonl")
CHROMA_DIR = "Backend/docs/chroma"

# Load chunks from file
def load_chunks(path: Path):
    print("📚 Loading chunks from JSONL...")
    chunks, metadatas = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            chunks.append(entry["text"])
            metadatas.append({"id": entry["id"], "source": entry["source"]})
    print(f"[✓] Loaded {len(chunks)} chunks.")
    return chunks, metadatas

def embed_chunks(chunks, metadatas):
    print("🔎 Initializing embedding model...")
    embed_model = HuggingFaceEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")

    print("🧠 Creating Chroma vector store...")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embed_model
    )

    vectordb.add_texts(chunks, metadatas=metadatas)
    vectordb.persist()
    print(f"[✓] Embedded and saved {len(chunks)} chunks to Chroma at '{CHROMA_DIR}'")

def run():
    chunks, metadatas = load_chunks(CHUNKS_FILE)
    embed_chunks(chunks, metadatas)

if __name__ == "__main__":
    run()
