import json
from pathlib import Path

RULE_PATH = Path("Backend/docs/rules/extracted_rules_llm.json")
if RULE_PATH.exists():
    content = RULE_PATH.read_text(encoding="utf-8").strip()
    if content:
        try:
            RULES = json.loads(content)
        except Exception as e:
            print(f"Error parsing rules file: {e}")
            RULES = []
    else:
        print("Rules file is empty. No rules loaded.")
        RULES = []
else:
    print("Rules file does not exist. No rules loaded.")
    RULES = []

def apply_rules(parsed: dict) -> dict | None:
    for rule in RULES:
        conditions = rule.get("if", {})
        matched = True

        for field, expected in conditions.items():
            value = parsed.get(field)
            if value is None:
                matched = False
                break
            if expected.startswith(">"):
                matched &= float(value) > float(expected[1:])
            elif expected.startswith("<"):
                matched &= float(value) < float(expected[1:])
            else:
                matched &= str(value).lower() == str(expected).lower()

        if matched:
            then = rule["then"]
            return {
                "decision": then.get("decision", "manual_review"),
                "justification": then.get("reason", "Rule triggered"),
                "payout": then.get("max_payout", 0),
                "citations": [rule["rule_id"]]
            }
    return None
