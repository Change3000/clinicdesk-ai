"""Evaluates the red-flag safety rules on every safety set.
Goal: catch every genuine emergency while not flagging routine messages that
merely mention scary words (e.g. 'chest x-ray results')."""
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rules_file = sys.argv[1] if len(sys.argv) > 1 else "safety_rules.json"
rules = json.loads((ROOT / "data" / rules_file).read_text())["rules"]


def red_flags(text):
    t = text.lower()
    return [r["label"] for r in rules
            if re.search(r["pattern"], t) and not (r["exclude"] and re.search(r["exclude"], t))]


def evaluate(fname, verbose=True):
    tp = fn = fp = tn = 0
    for r in csv.DictReader(open(ROOT / "data" / fname), delimiter="\t"):
        hit, urgent = red_flags(r["message"]), r["expected"] == "urgent"
        if urgent and hit: tp += 1
        elif urgent:
            fn += 1
            if verbose: print("   MISSED EMERGENCY:", r["message"])
        elif hit:
            fp += 1
            if verbose: print("   FALSE ALARM:", r["message"], hit)
        else: tn += 1
    print(f"{fname:<28} emergencies caught {tp}/{tp+fn}   routine wrongly flagged {fp}/{fp+tn}")


if __name__ == "__main__":
    print(f"Rules: {rules_file}")
    for f in ["safety_test.tsv", "safety_holdout_fresh.tsv", "safety_final_check.tsv", "safety_unseen_v3.tsv"]:
        if (ROOT / "data" / f).exists():
            evaluate(f)
    n = 0
    for f in ["train_v1.tsv", "train_v2_additions.tsv", "test.tsv", "holdout_fresh.tsv"]:
        for r in csv.DictReader(open(ROOT / "data" / f), delimiter="\t"):
            if red_flags(r["message"]):
                n += 1; print("   routine message flagged:", r["message"])
    print(f"Routine messages in the intent datasets wrongly flagged: {n}")
