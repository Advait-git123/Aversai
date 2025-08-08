import spacy
import re
from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Example procedures and cities – extend as needed
KNOWN_PROCEDURES = ["knee surgery", "bypass surgery", "cataract surgery"]
KNOWN_CITIES = ["Pune", "Mumbai", "Delhi", "Bangalore"]

def extract_entities_spacy(text: str) -> Dict:
    doc = nlp(text)
    age = None
    gender = None

    # Age
    age_match = re.search(r"(\d{2})[- ]?year[- ]?old", text.lower())
    if age_match:
        age = int(age_match.group(1))

    # Gender
    if "male" in text.lower():
        gender = "male"
    elif "female" in text.lower():
        gender = "female"

    # Procedure
    procedure = next((p for p in KNOWN_PROCEDURES if p.lower() in text.lower()), None)

    # City
    city = next((c for c in KNOWN_CITIES if c.lower() in text.lower()), None)

    # Policy months
    months_match = re.search(r"(issued|started).*?(\d+)\s*(month|mo)", text.lower())
    policy_months = int(months_match.group(2)) if months_match else None

    # If most are missing → fallback
    confidence = sum(bool(v) for v in [age, gender, procedure, city, policy_months]) / 5
    return {
        "age": age,
        "gender": gender,
        "procedure": procedure,
        "city": city,
        "policy_months": policy_months,
        "confidence": confidence
    }

# --------------- Fallback: GPT-4o if spaCy fails ---------------
def extract_entities_llm(text: str) -> Dict:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    parser = JsonOutputParser()

    system_msg = SystemMessage(
        content="Extract structured data from insurance query into this JSON:\n"
                "{ age:int, gender:str, procedure:str, city:str, policy_months:int }"
    )
    human_msg = HumanMessage(content=text)

    result = llm([system_msg, human_msg])
    return parser.parse(result.content)

# ----------- Final wrapper -----------
def parse_query(text: str) -> Dict:
    parsed = extract_entities_spacy(text)
    if parsed["confidence"] >= 0.8:
        return {k: v for k, v in parsed.items() if k != "confidence"}

    print("[!] Low confidence – falling back to LLM parsing")
    return extract_entities_llm(text)
