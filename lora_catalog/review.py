"""
A review page for the cards of one folder: a self-contained HTML file (thumbnails embedded), filterable by kind and
scope, for judging the classification by eye. Written to the tool folder; it is a local file, not published.

    python run.py review --only Krea2/graphic-art

(c) Eric Hiss. Dual license, see LICENSE.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
from collections import Counter

from .cards import preview_images
from .config import DATA_DIR, Settings
from .store import Catalog

THUMB_SIDE = 220


def _thumb(path: str) -> str:
    try:
        from PIL import Image
        from .cards import VIDEO_EXTENSIONS, _video_frame
        if path.lower().endswith(VIDEO_EXTENSIONS):
            image = _video_frame(path)
            if image is None:
                return ""
        else:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        if True:
            image.thumbnail((THUMB_SIDE, THUMB_SIDE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=78)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return ""


def _list(value) -> list:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def build_review(settings: Settings, only: str) -> str:
    catalog = Catalog(settings.catalog_path)
    rows = catalog.query("""select l.*, c.kind, c.scope, c.effect, c.triggers as card_triggers, c.weight_typical,
                                   c.weight_low, c.weight_high, c.overrides, c.nsfw_level as card_nsfw, c.confidence, c.evidence,
                                   c.artists, c.medium, c.movement_era, c.palette, c.line_quality, c.suited_subjects,
                                   c.refined_by
                            from loras l join cards c on c.path = l.path where l.rel_dir like ?
                            order by c.kind, c.confidence desc""", (only.replace("\\", "/") + "%",))
    catalog.close()
    kinds = Counter(row["kind"] for row in rows)
    cards_html = []
    for row in rows:
        images = preview_images(row["path"], 1)
        thumb = _thumb(images[0]) if images else ""
        triggers = ", ".join(f"<code>{html.escape(t.get('text', ''))}</code> <small>{t.get('use', '')}</small>"
                             for t in _list(row["card_triggers"])) or "<small>none</small>"
        overrides = ", ".join(_list(row["overrides"])) or "-"
        basis_label = {"named_in_evidence": "named", "name_match": "name match", "recognised_style": "recognised"}
        artists = ", ".join(f"<b>{html.escape(a.get('name', ''))}</b> <small>{basis_label.get(a.get('basis'), '')}</small>"
                            for a in _list(row["artists"]))
        subjects = ", ".join(html.escape(str(s)) for s in _list(row["suited_subjects"]))
        style_rows = "".join(f"<dt>{label}</dt><dd>{html.escape(str(value))}</dd>" for label, value in (
            ("medium", row["medium"]), ("era", row["movement_era"]), ("palette", row["palette"]),
            ("lines", row["line_quality"])) if value)
        if subjects:
            style_rows += f"<dt>good for</dt><dd>{subjects}</dd>"
        name = row["model_name"] or row["stem"]
        confidence = row["confidence"] or 0
        cards_html.append(f"""
<article data-kind="{row['kind']}" data-scope="{row['scope']}" data-low="{1 if confidence < 0.6 else 0}" data-artist="{1 if artists else 0}">
  <div class="thumb">{f'<img src="{thumb}" alt="">' if thumb else '<div class="none">no preview</div>'}</div>
  <div class="body">
    <div class="tags"><span class="kind">{row['kind']}</span><span>{row['scope']}</span>
      <span class="conf{' low' if confidence < 0.6 else ''}">conf {confidence:.2f}</span>
      {'<span class="nsfw">nsfw ' + str(row['card_nsfw']) + '</span>' if (row['card_nsfw'] or 0) >= 2 else ''}</div>
    <h3>{html.escape(name)}</h3>
    {f'<div class="artist">{artists}</div>' if artists else ''}
    <p>{html.escape(row['effect'] or '')}</p>
    <dl>{style_rows}<dt>triggers</dt><dd>{triggers}</dd>
        <dt>weight</dt><dd>{row['weight_typical']} ({row['weight_low']}-{row['weight_high']})</dd>
        <dt>overrides</dt><dd>{html.escape(overrides)}</dd>
        <dt>file</dt><dd><small>{html.escape(row['rel_dir'])}/{html.escape(row['file_name'])}</small></dd></dl>
  </div>
</article>""")
    buttons = "".join(f'<button data-kind="{kind}">{kind} <small>{count}</small></button>' for kind, count in kinds.most_common())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>LoRA cards - {html.escape(only)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#faf9f6; --fg:#222; --muted:#777; --card:#fff; --line:#e3e0d8; --accent:#3a5a8c; --warn:#b5651d; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17181a; --fg:#e6e4df; --muted:#9a978f; --card:#212326; --line:#34363a; --accent:#8fb0e0; --warn:#e0a060; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin-bottom:12px; }}
nav {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; position:sticky; top:0; background:var(--bg); padding:8px 0; }}
button {{ border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:14px; padding:4px 10px; cursor:pointer; }}
button.on {{ border-color:var(--accent); color:var(--accent); }}
main {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(420px, 1fr)); gap:12px; }}
article {{ display:flex; gap:12px; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; }}
.thumb img {{ width:{THUMB_SIDE // 1.4:.0f}px; border-radius:4px; display:block; }} .none {{ width:150px; color:var(--muted); }}
.body {{ min-width:0; }} h3 {{ margin:4px 0; font-size:15px; }} p {{ margin:4px 0 8px; }}
.tags span {{ font-size:11px; border:1px solid var(--line); border-radius:10px; padding:1px 7px; margin-right:4px; color:var(--muted); }}
.tags .kind {{ color:var(--accent); border-color:var(--accent); }} .tags .low, .tags .nsfw {{ color:var(--warn); border-color:var(--warn); }}
dl {{ display:grid; grid-template-columns:70px 1fr; gap:2px 8px; margin:0; font-size:12px; }} dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
code {{ background:var(--line); padding:0 4px; border-radius:3px; }}
.artist {{ color:var(--accent); margin:2px 0 4px; }} .artist small {{ color:var(--muted); }}
@media (max-width:480px) {{ main {{ grid-template-columns:1fr; }} article {{ flex-direction:column; }} }}
</style></head><body>
<h1>LoRA cards: {html.escape(only)}</h1>
<div class="sub">{len(rows)} LoRAs, written by {html.escape(settings.card_model)}. Click a kind to filter; "low confidence" shows the cards to check first.</div>
<nav><button data-kind="*" class="on">all <small>{len(rows)}</small></button>{buttons}<button data-artist="1">has artist</button><button data-low="1">low confidence</button></nav>
<main>{''.join(cards_html)}</main>
<script>
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('on')); b.classList.add('on');
  document.querySelectorAll('article').forEach(a => {{
    const show = b.dataset.low ? a.dataset.low === '1' : b.dataset.artist ? a.dataset.artist === '1'
               : (b.dataset.kind === '*' || a.dataset.kind === b.dataset.kind);
    a.style.display = show ? '' : 'none'; }});
}});
</script></body></html>"""


def write_review(settings: Settings, only: str) -> str:
    page = build_review(settings, only)
    name = "review_" + "".join(ch if ch.isalnum() else "_" for ch in only).strip("_") + ".html"
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return path
