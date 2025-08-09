import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY, "Missing GEMINI_API_KEY in .env"

# Paths
TEXT_DIR = Path("docs/clean_text")
OUTPUT_PATH = Path("Backend/docs/rules/extracted_rules_llm.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",      # or the model you want, e.g., "gemini-1.5-pro"
    temperature=0,                
    google_api_key=GEMINI_API_KEY  # Note: Use google_api_key param
)

SYSTEM_PROMPT = """You are a compliance assistant. Your job is to extract logical insurance rules from clauses.

For every input clause, return either:
- A JSON rule like this:
{
  "rule_id": "<slug or keyword>",
  "if": {
    "<field>": "<condition>"
  },
  "then": {
    "<decision|cap|reason>": "<value>"
  }
}
Or just return null if no rule is found.

Respond only with a single JSON object.
"""

def sanitize(text: str):
    return re.sub(r"\s+", " ", text).strip()

def extract_rules_from_text(text: str, file_id: str):
    chunks = [sanitize(p) for p in re.split(r"[\.\n]", text) if len(p.strip()) > 30]
    rules = []

    for idx, clause in enumerate(chunks):
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=clause)
            ]
            response = llm(messages).content
            if "null" in response.lower():
                continue
            rule = json.loads(response)
            rule["rule_id"] = f"{file_id}_{idx}"
            rules.append(rule)
            print(f"[✓] Rule found: {rule['rule_id']}")
        except Exception as e:
            print(f"[!] Error in clause {idx}: {e}")

    return rules

def run():
    all_rules = []
    for file_path in TEXT_DIR.glob("*.txt"):
        print(f"📄 Scanning {file_path.name}...")
        text = file_path.read_text(encoding="utf-8")
        rules = extract_rules_from_text(text, file_path.stem)
        all_rules.extend(rules)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rules, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_rules)} rules to {OUTPUT_PATH}")

if __name__ == "__main__":
    run()
