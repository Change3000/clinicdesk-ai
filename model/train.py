"""
ClinicDesk AI - training & evaluation script.

Trains a TF-IDF + Logistic Regression intent classifier on synthetic GP-clinic
patient messages, evaluates it on a held-out test set written in a different
(messier, SMS-style) voice, and exports the model as JSON so the web app can run
it entirely in the browser (no server, no API key, no patient data leaves the device).

Usage:
    python model/train.py --version v1     # baseline experiment
    python model/train.py --version v3 --export   # final model -> app/model.json
"""
import argparse, csv, json, math, re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELS = ["book_appointment", "change_appointment", "repeat_prescription", "test_results",
          "referral", "medical_certificate", "billing_medicare", "general_enquiry", "clinical_concern"]


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["label"], r["message"]) for r in csv.DictReader(f, delimiter="\t")]


def load_norm():
    return json.loads((DATA / "normalise.json").read_text())


def normalise(text, norm_map):
    """Lower-case and expand SMS shorthand / common misspellings (same logic as app.js)."""
    t = text.lower()
    t = re.sub(r"[’']", "", t)
    toks = re.findall(r"[a-z0-9]+|[^a-z0-9\s]", t)
    out = []
    for tok in toks:
        out.append(norm_map.get(tok, tok))
    return " ".join(out)


def stem(tok):
    """Very light suffix stripping (identical in app.js): parking->park, results->result."""
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def prep_v4(text, norm_map):
    return " ".join(stem(t) for t in normalise(text, norm_map).split())


# Keyword cues: explicit requests ("book", "cancel") are strong signals that a
# message is admin, even when a symptom is mentioned. Added to the model's logits.
CUES = {
    "book_appointment": r"\b(book|appointment|come in|get in|see (a|the|someone|any) ?(doctor|gp)?|spot|slot)\b",
    "change_appointment": r"\b(cancel|reschedul|resched|postpone|move|rebook|swap|shift|cannot make|can not make|running late)\w*",
    "repeat_prescription": r"\b(prescription|repeat|refill|renew)\w*",
    "medical_certificate": r"\b(certificate|sick note|doctor note|note from)\w*",
    "billing_medicare": r"\b(fee|charge|invoice|bulk bill|receipt|cost|pay|gap|rebate|refund)\w*",
}
CUE_BOOST = 1.5


CONFIGS = {
    # v1: clean, formal training data only; raw text; unigrams
    "v1": dict(train=["train_v1.tsv"], normalise=False, ngram=(1, 1), class_weight=None),
    # v2: + messy SMS-style training data (prompt revision 2) + shorthand normalisation
    "v2": dict(train=["train_v1.tsv", "train_v2_additions.tsv"], normalise=True, ngram=(1, 1), class_weight=None),
    # v3: + bigrams, sublinear tf, balanced class weights, stronger C
    "v3": dict(train=["train_v1.tsv", "train_v2_additions.tsv"], normalise=True, ngram=(1, 2), class_weight="balanced"),
    # v4: unigrams again + light stemming + keyword-cue boosts (hybrid rules + ML)
    "v4": dict(train=["train_v1.tsv", "train_v2_additions.tsv"], normalise=True, ngram=(1, 1), class_weight=None,
               stem=True, cues=True),
}


def run(version, export=False):
    cfg = CONFIGS[version]
    norm_map = load_norm() if cfg["normalise"] else {}
    train = [r for f in cfg["train"] for r in read_tsv(DATA / f)]
    test = read_tsv(DATA / "test.tsv")
    prep = (lambda s: normalise(s, norm_map)) if cfg["normalise"] else (lambda s: s.lower())
    if cfg.get("stem"):
        prep = lambda s: prep_v4(s, norm_map)

    vec = TfidfVectorizer(ngram_range=cfg["ngram"], sublinear_tf=True, lowercase=False,
                          token_pattern=r"(?u)\b\w\w+\b")
    Xtr = vec.fit_transform([prep(m) for _, m in train])
    ytr = [l for l, _ in train]
    clf = LogisticRegression(max_iter=3000, C=10.0 if version == "v3" else (3.0 if version == "v4" else 1.0),
                             class_weight=cfg["class_weight"])
    clf.fit(Xtr, ytr)

    def predict(msgs):
        X = vec.transform([prep(m) for m in msgs])
        if not cfg.get("cues"):
            return clf.predict(X)
        logits = clf.decision_function(X)
        classes = list(clf.classes_)
        for i, m in enumerate(msgs):
            nm_text = normalise(m, norm_map)
            hit = {lab for lab, rx in CUES.items() if re.search(rx, nm_text)}
            # a change request ("cancel/move my appointment") should not also boost "book"
            if "change_appointment" in hit:
                hit.discard("book_appointment")
            for lab in hit:
                logits[i][classes.index(lab)] += CUE_BOOST
        return np.array([classes[j] for j in logits.argmax(1)])

    yte = [l for l, _ in test]
    pred = predict([m for _, m in test])
    acc = accuracy_score(yte, pred)
    holdout = read_tsv(DATA / "holdout_fresh.tsv")
    hpred = predict([m for _, m in holdout])
    hacc = accuracy_score([l for l, _ in holdout], hpred)
    print(f"\n=== {version}: train={len(train)} msgs, vocab={len(vec.vocabulary_)}, test={len(test)} ===")
    print(f"Test accuracy: {acc:.1%}   |   Fresh holdout accuracy (never used for tuning): {hacc:.1%}")
    print(classification_report(yte, pred, labels=LABELS, digits=2, zero_division=0))
    errors = [(t, p, m) for (t, m), p in zip(test + holdout, list(pred) + list(hpred)) if t != p]
    for t, p, m in errors:
        print(f"  MISS  true={t:<20} pred={p:<20} | {m}")

    if export:
        cm = confusion_matrix(yte, pred, labels=LABELS).tolist()
        rep = classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)
        vocab = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
        classes = list(clf.classes_)
        model = {
            "version": version,
            "labels": classes,
            "vocab": [w for w, _ in vocab],
            "idf": [round(float(x), 4) for x in vec.idf_],
            "ngram": list(cfg["ngram"]),
            "coef": [[round(float(x), 4) for x in row] for row in clf.coef_],
            "intercept": [round(float(x), 4) for x in clf.intercept_],
            "normalise": load_norm(),
            "stem": bool(cfg.get("stem")),
            "cues": CUES if cfg.get("cues") else {},
            "cue_boost": CUE_BOOST,
            "metrics": {
                "train_size": len(train), "test_size": len(test), "accuracy": round(acc, 4),
                "holdout_size": len(holdout), "holdout_accuracy": round(hacc, 4),
                "per_class": {k: {"precision": round(v["precision"], 2), "recall": round(v["recall"], 2),
                                  "f1": round(v["f1-score"], 2)} for k, v in rep.items() if k in LABELS},
                "confusion": {"labels": LABELS, "matrix": cm},
            },
        }
        out = ROOT / "app" / "model.json"
        out.write_text(json.dumps(model, separators=(",", ":")))
        print(f"Exported {out} ({out.stat().st_size/1024:.0f} KB)")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # v2 is the shipped model: it generalised best to the fresh holdout set (see docs).
    ap.add_argument("--version", default="v2", choices=CONFIGS)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    run(a.version, a.export)
