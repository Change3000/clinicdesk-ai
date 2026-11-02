"""Builds the single-file app: inlines the trained models, safety rules and engine
into app/template.html.  Outputs:
  dist/index.html          standalone page (open in any browser, or host on GitHub Pages)
  dist/artifact_body.html  same page without the <html>/<head> wrapper (for hosts that add one)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
t = (ROOT / "app" / "template.html").read_text()
t = (t.replace("/*MODEL_JSON*/", (ROOT / "app" / "model.json").read_text())
      .replace("/*RULES_JSON*/", (ROOT / "data" / "safety_rules.json").read_text())
      .replace("/*URGENCY_JSON*/", (ROOT / "app" / "urgency.json").read_text())
      .replace("/*ENGINE_JS*/", (ROOT / "app" / "engine.js").read_text()))
assert "/*" + "MODEL_JSON*/" not in t
dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
(dist / "artifact_body.html").write_text(t)
standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
              '<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + t + '\n</body>\n</html>\n')
(dist / "index.html").write_text(standalone)
print("built", (dist / "index.html").stat().st_size // 1024, "KB")
