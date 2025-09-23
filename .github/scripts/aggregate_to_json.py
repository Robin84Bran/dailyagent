# scripts/aggregate_to_json.py
import json, os, re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Map bot stdout files -> JSON keys in the final report
FILES = {
    "pi_cycle_top_bot.out":           "pi_cycle",
    "ahr_999_bot.out":                "ahr999",
    "morning_brief_bot_v8.out":       "compound_hf",
    "iSunOne_Wealth_Bot_v4o.out":     "v4o_score",
    "iSunOne_Wealth_Bot_v4s.out":     "v4s_score",     # extra, nice to compare v4o vs v4s
}

def read_stdout(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore").strip()
    if not text:
        return None

    # Try to extract the last JSON object if the bot prints JSON
    # This is very forgiving: looks for {...} blocks
    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    if matches:
        for m in reversed(matches):
            try:
                return json.loads(m.group(0))
            except Exception:
                continue

    # Fall back to raw tail of text (keep it short)
    tail = text[-4000:]  # cap size
    return {"raw_stdout": tail}

def main():
    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for fname, key in FILES.items():
        data[key] = read_stdout(LOGS / fname)

    # Write rolling latest
    latest = REPORTS / "daily_report.json"
    latest.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Write dated archive (daily_report_YYYY_MM_DD.json, in UTC)
    d = datetime.now(timezone.utc)
    dated = REPORTS / f"daily_report_{d:%Y_%m_%d}.json"
    dated.write_text(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
