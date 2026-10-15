#!/usr/bin/env python3
"""Create today's (or a back-dated) daily log entry from the template,
and link it at the top of the Daily Log section in the main README.

Usage:
    python scripts/new_log.py
    python scripts/new_log.py --topic "Week 1 lab"
    python scripts/new_log.py --date 2026-10-10 --topic "Catch-up entry"
"""

import argparse
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "templates" / "daily-log.md"
LOGS_DIR = ROOT / "logs"
README_PATH = ROOT / "README.md"
LOG_START = "<!-- LOG-START -->"
LOG_END = "<!-- LOG-END -->"


def parse_args():
    parser = argparse.ArgumentParser(description="Create a new daily log entry.")
    parser.add_argument("--topic", default="", help="Topic for this log entry")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Log date as YYYY-MM-DD (defaults to today)",
    )
    args = parser.parse_args()

    try:
        date.fromisoformat(args.date)
    except ValueError:
        parser.error(f"--date must be in YYYY-MM-DD format, got {args.date!r}")

    return args


def create_log_file(log_date: str, topic: str) -> Path:
    log_path = LOGS_DIR / f"{log_date}.md"

    if log_path.exists():
        return log_path

    lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    lines[0] = f"# {log_date}"

    if topic:
        topic_index = lines.index("## Topic")
        lines.insert(topic_index + 1, f"\n{topic}")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def update_readme(log_date: str, topic: str) -> None:
    readme_lines = README_PATH.read_text(encoding="utf-8").splitlines()
    start = readme_lines.index(LOG_START)
    end = readme_lines.index(LOG_END)

    entry = f"- [{log_date}](logs/{log_date}.md) — {topic}".rstrip(" —")

    # Remove any existing line for this date so we never duplicate it.
    block = readme_lines[start + 1 : end]
    block = [line for line in block if f"logs/{log_date}.md" not in line]
    block.insert(0, entry)

    readme_lines[start + 1 : end] = block
    README_PATH.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    log_path = create_log_file(args.date, args.topic)
    update_readme(args.date, args.topic)
    print(log_path)


if __name__ == "__main__":
    main()
