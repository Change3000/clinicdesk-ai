"""Urgency model: a second, learned safety net next to the red-flag rules.
Keyword rules caught only 2 of 10 emergencies written in new words, so this
model learns from character fragments ("breath", "bleed", "unresp") as well as
words. Trained on earlier safety sets + routine messages; evaluated on the set
that was never used for any rule or model change (safety_unseen_v3.tsv)."""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
rd = lambda f: list(csv.DictReader(open(D / f), delimiter="\t"))

pos, neg = [], []
for f in ["safety_test.tsv", "safety_holdout_fresh.tsv", "safety_final_check.tsv"]:
    for r in rd(f):
        (pos if r["expected"] == "urgent" else neg).append(r["message"])
for f in ["train_v1.tsv", "train_v2_additions.tsv"]:
    neg += [r["message"] for r in rd(f)]
X_txt = pos + neg
y = [1] * len(pos) + [0] * len(neg)

word = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")
char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
X = hstack([word.fit_transform(X_txt), char.fit_transform(X_txt)]).tocsr()
clf = LogisticRegression(max_iter=5000, C=4.0, class_weight="balanced").fit(X, y)

def score(msgs):
    Z = hstack([word.transform(msgs), char.transform(msgs)]).tocsr()
    return clf.predict_proba(Z)[:, 1]

th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
# false-alarm rate on routine messages the model never trained on
routine = [r["message"] for f in ["test.tsv", "holdout_fresh.tsv"] for r in rd(f)]
fa = (score(routine) >= th).sum()
un = rd("safety_unseen_v3.tsv")
s = score([r["message"] for r in un])
caught = sum(1 for r, p in zip(un, s) if r["expected"] == "urgent" and p >= th)
falarm = sum(1 for r, p in zip(un, s) if r["expected"] != "urgent" and p >= th)
print(f"threshold {th}: unseen emergencies caught {caught}/10, unseen routine flagged {falarm}/6, routine test+holdout flagged {fa}/{len(routine)}")
for r, p in zip(un, s):
    print(f"   {p:.2f} {r['expected']:<10} {r['message']}")

if "--export" in sys.argv:
    out = {
        "threshold": th,
        "word_vocab": sorted(word.vocabulary_, key=word.vocabulary_.get),
        "word_idf": [round(float(x), 4) for x in word.idf_],
        "char_vocab": sorted(char.vocabulary_, key=char.vocabulary_.get),
        "char_idf": [round(float(x), 4) for x in char.idf_],
        "coef": [round(float(x), 4) for x in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 4),
    }
    (ROOT / "app" / "urgency.json").write_text(json.dumps(out, separators=(",", ":")))
    print("exported app/urgency.json", round((ROOT / "app" / "urgency.json").stat().st_size / 1024), "KB")
