"""Inline twin.js, demo data and study results into one self-contained page (web/dist/index.html)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tpl = (ROOT / "web" / "index.template.html").read_text()
out = (tpl.replace("/*TWINJS*/", (ROOT / "web" / "twin.js").read_text())
          .replace("/*DEMO*/", (ROOT / "web" / "demo_data.json").read_text())
          .replace("/*CLINIC*/", (ROOT / "web" / "clinic_data.json").read_text())
          .replace("/*STUDY*/", (ROOT / "results" / "study.json").read_text()))
dist = ROOT / "web" / "dist"
dist.mkdir(exist_ok=True)
(dist / "index.html").write_text(out)
# standalone copy with a full document (head = title, fonts, styles; body = app)
cut = out.index('<header class="appbar">')
standalone = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
              + out[:cut] + '</head><body>' + out[cut:] + '</body></html>')
(dist / "standalone.html").write_text(standalone)
# GitHub Pages serves docs/ (main branch); the dashboard is the site's index page
(ROOT / "docs" / "index.html").write_text(standalone)
(ROOT / "docs" / ".nojekyll").write_text("")
print("wrote", dist / "index.html", "and docs/index.html", len(out) // 1024, "KB")
