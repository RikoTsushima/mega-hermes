"""L4 Session Archiver — compress and extract history from hermes sessions.

Reads session_*.json files from ~/.hermes/sessions/, extracts user/assistant
conversation summaries, archives compressed sessions by month, and maintains
an all_histories.txt index for cross-session search.

Usage:
    python -m agent.session_archiver --dry-run   # preview what would happen
    python -m agent.session_archiver --run        # actually archive

Designed to run via cron (every 12h) or manually.
"""

import json
import os
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home


def _sessions_dir() -> Path:
    return get_hermes_home() / "sessions"


def _l4_dir() -> Path:
    d = get_hermes_home() / "memories" / "L4_sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_history(session_data: dict) -> List[str]:
    """Extract [USER]/[Agent] history lines from session messages."""
    lines = []
    messages = session_data.get("messages", [])
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            continue
        summary = content.strip().replace("\n", " ")[:120]
        if role == "user" and not summary.startswith("[Note:"):
            lines.append(f"[USER] {summary}")
        elif role == "assistant" and len(summary) > 10:
            lines.append(f"[Agent] {summary}")
    return lines


def _existing_sessions(l4_dir: Path) -> set:
    """Read session names already in all_histories.txt."""
    hist_path = l4_dir / "all_histories.txt"
    if not hist_path.exists():
        return set()
    text = hist_path.read_text(encoding="utf-8")
    return {line.strip().replace("SESSION: ", "")
            for line in text.splitlines() if line.startswith("SESSION: ")}


def _format_history_block(session_name: str, history_lines: List[str]) -> str:
    sep = "=" * 60
    return f"\n{sep}\nSESSION: {session_name}\n{sep}\n" + "\n".join(history_lines) + "\n"


def _session_name(session_data: dict) -> str:
    """Derive a short name from session metadata."""
    start = session_data.get("session_start", "")
    try:
        dt = datetime.fromisoformat(start)
        return dt.strftime("%m%d_%H%M")
    except (ValueError, TypeError):
        return session_data.get("session_id", "unknown")[:12]


def batch_process(dry_run: bool = True) -> Dict:
    """Archive old sessions: extract history, compress to zip, clean up."""
    sessions_dir = _sessions_dir()
    l4_dir = _l4_dir()

    if not sessions_dir.exists():
        print("No sessions directory found")
        return {"processed": 0, "skipped": 0}

    cutoff = time.time() - 7200  # skip files modified within 2h
    existing = _existing_sessions(l4_dir)

    session_files = sorted(sessions_dir.glob("session_*.json"))
    print(f"Found {len(session_files)} session files, {len(existing)} already archived")

    results = []
    skipped = []
    errors = []

    for fp in session_files:
        fname = fp.name
        if fp.stat().st_mtime > cutoff:
            skipped.append((fname, "recent(<2h)"))
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append((fname, str(e)))
            continue

        msg_count = data.get("message_count", 0)
        if msg_count < 4:
            skipped.append((fname, f"too_short({msg_count} msgs)"))
            continue

        sn = _session_name(data)
        if sn in existing:
            skipped.append((fname, f"dup:{sn}"))
            continue

        history = _extract_history(data)
        if len(history) < 2:
            skipped.append((fname, "no_meaningful_history"))
            continue

        results.append((sn, fp, history, data))

    results.sort(key=lambda x: x[0])
    print(f"\nPhase 1: {len(results)} new, {len(skipped)} skip, {len(errors)} err")
    for f, r in skipped[:5]:
        print(f"  SKIP {f}: {r}")
    for f, e in errors[:3]:
        print(f"  ERR  {f}: {e}")

    if dry_run:
        print("\n[DRY RUN] Pass dry_run=False to execute.")
        return {"processed": len(results), "skipped": len(skipped),
                "errors": len(errors), "sessions": [r[0] for r in results]}

    # Phase 2: Append history
    hist_path = l4_dir / "all_histories.txt"
    with open(hist_path, "a", encoding="utf-8") as f:
        for sn, _, history, _ in results:
            f.write(_format_history_block(sn, history))
    print(f"Appended {len(results)} sessions to all_histories.txt")

    # Phase 3: Archive to monthly zips
    by_month: Dict[str, list] = defaultdict(list)
    for sn, fp, _, data in results:
        start = data.get("session_start", "")
        try:
            dt = datetime.fromisoformat(start)
            month_key = dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            month_key = "unknown"
        by_month[month_key].append((sn, fp))

    for mk, items in sorted(by_month.items()):
        zpath = l4_dir / f"{mk}.zip"
        mode = "a" if zpath.exists() else "w"
        with zipfile.ZipFile(zpath, mode, zipfile.ZIP_DEFLATED) as zf:
            names = set(zf.namelist()) if mode == "a" else set()
            for sn, fp in items:
                arc_name = f"{sn}.json"
                if arc_name not in names:
                    zf.write(fp, arc_name)
        print(f"  {mk}.zip: +{len(items)}")

    report = {"processed": len(results), "skipped": len(skipped),
              "errors": len(errors), "new_sessions": len(results)}
    print(f"\nDone: {report}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="L4 session archiver")
    ap.add_argument("--run", action="store_true", help="execute (default: dry run)")
    args = ap.parse_args()
    batch_process(dry_run=not args.run)
