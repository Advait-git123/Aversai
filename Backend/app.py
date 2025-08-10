from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Backend.utils.parsing import parse_query
from Backend.utils.rule_engine import apply_rules
from Backend.rag_chain import query_rag
from Backend.ingest import convert_pdfs_to_text, chunk_text_files
from Backend.embed import embed_chunks
from dotenv import load_dotenv
import requests
from pathlib import Path
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Directories
RAW_PDF_DIR = Path("docs/raw_pdfs")

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
def download_pdf(url: str, filename: str = "remote.pdf"):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download PDF")
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_PDF_DIR / filename
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"[✓] Downloaded: {filename}")
    return path

# ---------- Main HackRx Endpoint ----------
@app.post("/hackrx/run")
async def hackrx_run(input: QueryInput):
    pdf_url = input.documents
    questions = input.questions

    # Step 1: Ingest remote document
    download_pdf(pdf_url)
    convert_pdfs_to_text()
    chunk_text_files()
    embed_chunks()

    answers = []

    # Step 2: Answer each question
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

    return { "answers": answers }
