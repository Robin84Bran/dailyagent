# .github/scripts/aggregate_to_json.py
import json, os, time
from datetime import datetime, timezone

# If your individual bots already write JSONs, read them here.
# Otherwise we just emit "unavailable" and improve later.
result = {
  "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
  "pi_cycle": None,
  "ahr999": None,
  "eth_btc": None,
  "compound_hf": None,
  "v4q_score": None
}

os.makedirs("reports", exist_ok=True)
with open("reports/daily_report.json", "w") as f:
    json.dump(result, f, indent=2)
print("Wrote reports/daily_report.json")
