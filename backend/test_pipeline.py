"""
test_pipeline.py  –  End-to-end pipeline test (no HTTP server needed)

Run from project root:
    conda run -n bctproject python backend/test_pipeline.py
Or from backend/:
    conda run -n bctproject python test_pipeline.py

Each TEST_CONVERSATION is a list of messages that share ONE session, so you
can verify multi-turn context, slot carry-over, and result tracking.
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Path setup – makes  `from app.xxx import ...`  work when run from either
# backend/ or the project root.
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # backend/
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from app.dialogue_manager import dialogue_manager
from app.schemas import ChatRequest


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bold":    "\033[1m",
    "reset":   "\033[0m",
    "cyan":    "\033[96m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "red":     "\033[91m",
    "magenta": "\033[95m",
    "blue":    "\033[94m",
}

def c(color: str, text: str) -> str:
    return f"{C[color]}{text}{C['reset']}"


# ─────────────────────────────────────────────────────────────────────────────
# Test conversations
# Each entry is a dict:
#   label   – short name shown in the header
#   session – shared session_id for all turns (None → auto-generate once)
#   turns   – list of user messages
# ─────────────────────────────────────────────────────────────────────────────
TEST_CONVERSATIONS = [
    {
        "label": "Greeting + clarify",
        "turns": [
            "Hello",
            "What can you help me with?",
        ],
    },
    {
        "label": "College location lookup",
        "turns": [
            "Where is Kathmandu University located?",
        ],
    },
    {
        "label": "Fee info with follow-up",
        "turns": [
            "What is the fee structure?",
            "Pulchowk Campus",
        ],
    },
    {
        "label": "Search by location",
        "turns": [
            "Show me engineering colleges in Lalitpur",
        ],
    },
    {
        "label": "Search by program",
        "turns": [
            "Which colleges offer computer engineering?",
        ],
    },
    {
        "label": "Search by fee budget",
        "turns": [
            "Find colleges where the fee is under 6 lakhs",
        ],
    },
    {
        "label": "Hostel availability",
        "turns": [
            "Does Thapathali Campus have hostel?",
        ],
    },
    {
        "label": "Scholarship info",
        "turns": [
            "Tell me about scholarships at Kathford",
        ],
    },
    {
        "label": "Pass percentage",
        "turns": [
            "What is the pass rate at Pulchowk?",
        ],
    },
    {
        "label": "Admission process",
        "turns": [
            "How do I apply to Kathford International College?",
        ],
    },
    {
        "label": "Contact info",
        "turns": [
            "What is the contact number of Thapathali Campus?",
        ],
    },
    {
        "label": "Recommendation (full 3-turn slot filling)",
        "turns": [
            "Recommend me a college",          # triggers slot-fill question 1
            "I am interested in computer engineering",   # fills major
            "My IOE rank is 2800",             # fills rank
            "My budget is 7 lakhs",            # fills budget → fires query
        ],
    },
    {
        "label": "Recommendation (single-turn: all info given)",
        "turns": [
            "Recommend a college for civil engineering, rank 1500, budget 8 lakhs",
        ],
    },
    {
        "label": "Recommendation with scholarship filter",
        "turns": [
            "Suggest colleges for electrical engineering with scholarship",
            "rank 3000",
            "6 lakh budget",
        ],
    },
    {
        "label": "Context carry-over: location then type",
        "turns": [
            "Show private colleges in Kathmandu",
            "What about public ones?",         # should carry location=Kathmandu
        ],
    },
    {
        "label": "Full college info",
        "turns": [
            "Tell me about Sagarmatha Engineering College",
        ],
    },
    {
        "label": "Goodbye",
        "turns": [
            "Thanks, that's all!",
            "Bye",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_divider(char: str = "═", width: int = 72):
    print(c("bold", char * width))


def _print_context_snapshot(session_id: str):
    """Print a compact view of the current session context."""
    debug = dialogue_manager.slot_manager.get_debug_info(session_id)
    if "error" in debug:
        return

    print(c("blue", "\n  ┌─ CONTEXT SNAPSHOT ─────────────────────────────────────────"))
    print(c("blue", f"  │  turn={debug['turn_count']}  intent={debug['current_intent']}  family={debug['intent_family']}"))

    if debug["slots"]:
        print(c("blue",  "  │  DB slots   : ") + str(debug["slots"]))
    if debug["missing_slots"]:
        print(c("yellow","  │  missing    : ") + str(debug["missing_slots"]))

    rec = debug.get("rec_context", {})
    filled_rec = {k: v for k, v in rec.items() if v and k != "last_recommended"}
    if filled_rec:
        print(c("blue",  "  │  rec slots  : ") + str(filled_rec))
    print(c("blue", f"  │  ready_rec  : {debug['ready_to_query']}"))

    if debug.get("last_result_count"):
        names = debug["last_results"]
        preview = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        print(c("green", f"  │  last fetch : {debug['last_result_count']} results  [{preview}]"))

    print(c("blue", "  └────────────────────────────────────────────────────────────"))


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_conversation(label: str, turns: list[str], session_id: str):
    """Run one multi-turn conversation and print results."""
    _print_divider()
    print(c("magenta", f"  CONVERSATION: {label}"))
    print(c("magenta", f"  session: {session_id}"))
    _print_divider("─")

    for turn_num, message in enumerate(turns, 1):
        print(f"\n{c('bold', f'  Turn {turn_num}')}  {c('cyan', f'> {message}')}")

        req = ChatRequest(message=message, session_id=session_id)
        t0  = datetime.now()

        try:
            resp = await dialogue_manager.process_message(req)
        except Exception as exc:
            print(c("red", f"  ✗ ERROR: {exc}"))
            import traceback; traceback.print_exc()
            continue

        elapsed = (datetime.now() - t0).total_seconds()

        # Bot reply
        print(f"\n{c('green', '  Bot →')} {resp.message}")
        print(
            c("yellow",
              f"  intent={resp.intent}  conf={resp.confidence:.0%}  "
              f"time={elapsed:.2f}s")
        )

        # Context snapshot
        _print_context_snapshot(session_id)
        print()


async def main():
    _print_divider()
    print(c("bold", "  BCT COLLEGE RECOMMENDATION SYSTEM  –  PIPELINE TEST"))
    print(c("bold", f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    _print_divider()

    # ── Init ──────────────────────────────────────────────────────────────────
    print(f"\n{c('yellow', 'Initialising dialogue manager…')}")
    try:
        await dialogue_manager.initialize()
        print(c("green", "✓ Ready\n"))
    except Exception as exc:
        print(c("red", f"✗ Init failed: {exc}"))
        raise

    # ── Which conversations to run? ───────────────────────────────────────────
    # Change `SELECTED` to a list of indices (0-based) to run specific ones,
    # or leave as None to run all.
    SELECTED: list[int] | None = None          # e.g. [0, 11] for greeting + rec

    conversations = (
        [TEST_CONVERSATIONS[i] for i in SELECTED]
        if SELECTED is not None
        else TEST_CONVERSATIONS
    )

    # ── Run each conversation with its own unique session ─────────────────────
    session_ids = {}
    for i, conv in enumerate(conversations):
        # stable per-run session ID for each conversation
        sid = f"test_sess_{i:02d}"
        session_ids[i] = sid
        await run_conversation(
            label=conv["label"],
            turns=conv["turns"],
            session_id=sid,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_divider()
    print(c("bold", "  SUMMARY"))
    _print_divider("─")
    for i, conv in enumerate(conversations):
        sid = session_ids[i]
        debug = dialogue_manager.slot_manager.get_debug_info(sid)
        turns = debug.get("turn_count", "?")
        last_count = debug.get("last_result_count", 0)
        print(
            f"  [{i:02d}] {conv['label']:<45}  "
            f"turns={turns}  last_db={last_count} results"
        )

    _print_divider()
    print(c("green", "  ✓ All conversations complete"))
    _print_divider()

    await dialogue_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
