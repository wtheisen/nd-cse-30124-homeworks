#!/usr/bin/env python3
"""
Convert homework02 JSON data into flat CSVs for easy pandas use.

Outputs are written to: homework_data/homework02/csv/
"""

import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ALIAS_DIR = BASE_DIR / "alias_statements"
SUSPECT_DIR = BASE_DIR / "suspect_statements"
OUTPUT_DIR = BASE_DIR / "csv"
ALIAS_WIDE_DIR = OUTPUT_DIR / "alias_wide"

SAMPLE_ID_RE = re.compile(r"_(\d+)\.json$")


def extract_sample_id(filename: str):
    match = SAMPLE_ID_RE.search(filename)
    if match:
        return int(match.group(1))
    return ""


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned.lower() or "unknown"


def parse_time_to_minutes(time_str: str) -> int:
    match = re.match(r"^(\d{1,2}):(\d{2})", time_str or "")
    if not match:
        return 0
    hours = int(match.group(1))
    minutes = int(match.group(2))
    return hours * 60 + minutes


def main():
    alias_rows = []
    alias_by_name = {}
    for file_path in sorted(ALIAS_DIR.glob("*.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        alias_name = data.get("name", "")
        alias_by_name.setdefault(alias_name, []).append(
            {
                "sample_id": extract_sample_id(file_path.name),
                "statements": data.get("suspect_statement", []),
                "source_file": file_path.name,
            }
        )
        sample_id = extract_sample_id(file_path.name)
        for idx, entry in enumerate(data.get("suspect_statement", [])):
            alias_rows.append(
                {
                    "alias_name": alias_name,
                    "alias_sample_id": sample_id,
                    "statement_index": idx,
                    "time": entry.get("time", ""),
                    "location": entry.get("location", ""),
                    "activity": entry.get("activity", ""),
                    "source_file": file_path.name,
                }
            )

    write_csv(
        OUTPUT_DIR / "alias_statements.csv",
        [
            "alias_name",
            "alias_sample_id",
            "statement_index",
            "time",
            "location",
            "activity",
            "source_file",
        ],
        alias_rows,
    )

    for alias_name, samples in alias_by_name.items():
        time_set = set()
        for sample in samples:
            for entry in sample["statements"]:
                time_set.add(entry.get("time", ""))
        times = sorted(time_set, key=parse_time_to_minutes)

        rows = []
        for sample in sorted(samples, key=lambda s: s["sample_id"]):
            row = {"alias_sample_id": sample["sample_id"]}
            for entry in sample["statements"]:
                time_key = entry.get("time", "")
                location = entry.get("location", "")
                activity = entry.get("activity", "")
                cell = f"({location}, {activity})"
                if row.get(time_key):
                    row[time_key] = f"{row[time_key]} | {cell}"
                else:
                    row[time_key] = cell
            for time_key in times:
                row.setdefault(time_key, "")
            rows.append(row)

        fieldnames = ["alias_sample_id"] + times
        output_name = f"alias_{slugify(alias_name)}.csv"
        write_csv(ALIAS_WIDE_DIR / output_name, fieldnames, rows)

    suspect_statement_rows = []
    gps_rows = []
    witness_rows = []
    witness_observation_rows = []

    for file_path in sorted(SUSPECT_DIR.glob("*.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        suspect_name = data.get("name", "")

        for idx, entry in enumerate(data.get("suspect_statement", [])):
            suspect_statement_rows.append(
                {
                    "suspect_name": suspect_name,
                    "statement_index": idx,
                    "time": entry.get("time", ""),
                    "location": entry.get("location", ""),
                    "activity": entry.get("activity", ""),
                    "source_file": file_path.name,
                }
            )

        for idx, entry in enumerate(data.get("gps_data", [])):
            gps_rows.append(
                {
                    "suspect_name": suspect_name,
                    "gps_index": idx,
                    "time": entry.get("time", ""),
                    "location": entry.get("location", ""),
                    "signal_strength": entry.get("signal_strength", ""),
                    "source_file": file_path.name,
                }
            )

        for witness_name, witness_data in data.get("witness_statements", {}).items():
            witness_rows.append(
                {
                    "suspect_name": suspect_name,
                    "witness_name": witness_name,
                    "witness_reliability": witness_data.get(
                        "witness_reliability", ""
                    ),
                    "source_file": file_path.name,
                }
            )

            for idx, obs in enumerate(witness_data.get("observations", [])):
                witness_observation_rows.append(
                    {
                        "suspect_name": suspect_name,
                        "witness_name": witness_name,
                        "observation_index": idx,
                        "time": obs.get("time", ""),
                        "location": obs.get("location", ""),
                        "activity": obs.get("activity", ""),
                        "source_file": file_path.name,
                    }
                )

    write_csv(
        OUTPUT_DIR / "suspect_statements.csv",
        [
            "suspect_name",
            "statement_index",
            "time",
            "location",
            "activity",
            "source_file",
        ],
        suspect_statement_rows,
    )

    write_csv(
        OUTPUT_DIR / "suspect_gps.csv",
        ["suspect_name", "gps_index", "time", "location", "signal_strength", "source_file"],
        gps_rows,
    )

    write_csv(
        OUTPUT_DIR / "witnesses.csv",
        ["suspect_name", "witness_name", "witness_reliability", "source_file"],
        witness_rows,
    )

    write_csv(
        OUTPUT_DIR / "witness_observations.csv",
        [
            "suspect_name",
            "witness_name",
            "observation_index",
            "time",
            "location",
            "activity",
            "source_file",
        ],
        witness_observation_rows,
    )


if __name__ == "__main__":
    main()
