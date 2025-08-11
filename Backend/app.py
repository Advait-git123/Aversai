from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.parsing import parse_query
from utils.rule_engine import apply_rules
from rag_chain import query_rag
from ingest import convert_pdfs_to_text, chunk_text_files
from embed import embed_chunks
from dotenv import load_dotenv
import requests
from pathlib import Path
import os
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Directories
RAW_PDF_DIR = Path("Backend/docs/raw_pdfs")
CLEAN_TEXT_DIR = Path("Backend/docs/clean_text")
CHUNKS_FILE = CLEAN_TEXT_DIR / "chunks.jsonl"

# ---------- Auth Dependency ----------
def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API token")

# ---------- FastAPI Setup ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request Schema ----------
class QueryInput(BaseModel):
    documents: str            # Remote PDF URL
    questions: list[str]      # List of questions

@app.get("/")
def root():
    return {"message": "AVERSAI API is up and running."}

# ---------- Helper: Download PDF ----------
def download_pdf(url: str, filename: str = "remote.pdf") -> Path:
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code} - {response.text[:100]}")
        RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_PDF_DIR / filename
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"[✓] Downloaded: {filename}")
        return path
    except Exception as e:
        print(f"[✗] Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

# ---------- Helper: Load Chunks & Metadata ----------
def load_chunks_and_metadata():
    if not CHUNKS_FILE.exists():
        raise HTTPException(status_code=500, detail="No chunks.jsonl found after chunking.")
    chunks = []
    metadatas = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data["text"])
            metadatas.append(data["metadata"])
    return chunks, metadatas

# ---------- Main HackRx Endpoint ----------
@app.post("/hackrx/run")
async def hackrx_run(input: QueryInput, _: str = Depends(verify_token)):
    try:
        pdf_url = input.documents
        questions = input.questions

        # Step 1: Ingest remote document
        download_pdf(pdf_url)
        convert_pdfs_to_text()
        chunk_text_files()

        # Step 2: Load chunks & embed
        chunks, metadatas = load_chunks_and_metadata()
        embed_chunks(chunks, metadatas)

        answers = []

        # Step 3: Answer each question
        for q in questions:
            parsed = parse_query(q)
            rule_answer = apply_rules(parsed)
            if rule_answer:
                answers.append(rule_answer.get("justification", "Rule matched"))
                continue

            rag_answer = query_rag(q)
            if isinstance(rag_answer, dict):
                answers.append(rag_answer.get("justification", str(rag_answer)))
            else:
                answers.append(rag_answer)

        return {"answers": answers}

    except HTTPException:
        raise  # Let FastAPI handle HTTP errors
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
