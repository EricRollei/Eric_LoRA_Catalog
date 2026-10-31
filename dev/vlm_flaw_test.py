# Reliability test (2026-10-30): can the local vision model find the flaws Eric tags by hand?
# Samples rated sweep cells from the catalog (positives per flaw tag + clean cells), asks the card model one JSON
# question set per image, and writes dev/vlm_flaw_test.jsonl. Then: python dev/vlm_flaw_test.py --report
import base64
import io
import json
import os
import random
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lora_catalog import config, llm  # noqa: E402

OUT = os.path.join(ROOT, "dev", "vlm_flaw_test.jsonl")
FLAWS = {"signature or watermark": "signature", "unwanted text or lettering": "text", "unwanted NSFW": "nudity",
         "added frame or border": "frame", "faces or hands broken": "hands_faces", "anatomy wrong": "anatomy"}
PER_FLAW, CLEAN = 15, 40

SYS = """You inspect AI-generated images for specific flaws. Answer only from what is visible in the image. The prompt
that made the image is given so you know what was asked for. Look carefully at the whole image, including corners and
edges.
Questions:
- signature: is there a signature, artist's initials, monogram, logo or watermark anywhere? where?
- text: is there any text, letters or writing anywhere (signs, captions, labels, gibberish lettering)? copy up to 8 words
- text_requested: did the prompt ask for text or lettering to appear?
- nudity: is there nudity (bare breasts, genitals, bare buttocks) or sexual content?
- nudity_requested: did the prompt ask for nudity?
- frame: is the picture surrounded by a drawn frame, border, matte or blank margin?
- hands_faces: is any face or hand visibly malformed (extra, fused or missing fingers; melted or distorted face)?
- anatomy: is any body visibly wrong (extra or missing limbs, impossible joints, merged bodies)?
- prompt_score: 0-100, how well the image matches the prompt
Reply only with JSON: {"signature": "yes|no", "signature_where": "", "text": "yes|no", "text_reads": "",
"text_requested": "yes|no", "nudity": "yes|no", "nudity_requested": "yes|no", "frame": "yes|no",
"hands_faces": "yes|no|no people", "anatomy": "yes|no|no people", "prompt_score": 0}"""


def sample(con):
    rng = random.Random(7)
    rows = [dict(r) for r in con.execute("select id, image_path, prompt, words, stars, tags from stacks "
                                         "where image_path is not null and image_path != ''")]
    rows = [r for r in rows if os.path.isfile(r["image_path"])]
    for r in rows:
        r["tags"] = [t if isinstance(t, str) else t.get("tag") for t in json.loads(r["tags"] or "[]")]
    picked, seen = [], set()
    for tag in FLAWS:
        pos = [r for r in rows if tag in r["tags"] and r["image_path"] not in seen]
        for r in rng.sample(pos, min(PER_FLAW, len(pos))):
            seen.add(r["image_path"])
            picked.append(r)
    clean = [r for r in rows if r["stars"] and not any(t in FLAWS for t in r["tags"]) and r["image_path"] not in seen]
    picked += rng.sample(clean, min(CLEAN, len(clean)))
    return picked


def ask(settings, r):
    from PIL import Image
    im = Image.open(r["image_path"]).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    t0 = time.time()
    out = llm.chat(settings, settings.card_model, [
        {"role": "system", "content": SYS},
        {"role": "user", "content": [{"type": "text", "text": "Prompt: " + (r["prompt"] or "(none)")},
                                     {"type": "image_url", "image_url": {"url": url}}]}],
        temperature=0.0, max_tokens=500)
    try:
        ans = llm.parse_json(out)
    except Exception:
        ans = {"raw": out[:300]}
    return ans, round(time.time() - t0, 1)


def run():
    settings = config.load_settings()
    con = sqlite3.connect(f"file:{settings.catalog_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cells = sample(con)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()}
    print(f"{len(cells)} cells, {len(done)} already done", flush=True)
    with open(OUT, "a", encoding="utf-8") as fh:
        for i, r in enumerate(cells):
            if r["id"] in done:
                continue
            try:
                ans, sec = ask(settings, r)
            except Exception as error:
                ans, sec = {"error": f"{type(error).__name__}: {error}"}, 0
            fh.write(json.dumps({"id": r["id"], "image": r["image_path"], "tags": r["tags"], "stars": r["stars"],
                                 "words": r["words"], "answer": ans, "seconds": sec}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{i + 1}/{len(cells)} {sec}s", flush=True)
    print("DONE", flush=True)


def yes(v):
    return str(v or "").strip().lower().startswith("y")


def report():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if isinstance(r["answer"], dict) and "error" not in r["answer"] and "raw" not in r["answer"]]
    print(f"{len(rows)} answered; median {sorted(r['seconds'] for r in rows)[len(rows) // 2]} s per image\n")
    print(f"{'flaw':28s} {'you tagged':>10s} {'model found':>11s} {'caught':>7s} {'false alarms':>13s}")
    for tag, key in FLAWS.items():
        def model_says(a):
            if key == "text":
                return yes(a.get("text")) and not yes(a.get("text_requested"))
            if key == "nudity":
                return yes(a.get("nudity")) and not yes(a.get("nudity_requested"))
            return yes(a.get(key))
        pos = [r for r in rows if tag in r["tags"]]
        neg = [r for r in rows if not any(t in FLAWS for t in r["tags"])]
        tp = sum(model_says(r["answer"]) for r in pos)
        fp = sum(model_says(r["answer"]) for r in neg)
        print(f"{tag:28s} {len(pos):>10d} {tp + fp:>11d} {f'{tp}/{len(pos)}':>7s} {f'{fp}/{len(neg)}':>13s}")
    # prompt score vs stars
    pts = [(r["stars"], float(r["answer"].get("prompt_score") or 0)) for r in rows if r["stars"]]
    if len(pts) > 5:
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out = [0.0] * len(v)
            for k, i in enumerate(order):
                out[i] = k
            return out
        a, b = ranks([p[0] for p in pts]), ranks([p[1] for p in pts])
        n = len(pts)
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        sd = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        print(f"\nprompt_score vs your stars: rank correlation {cov / sd:+.2f} over {n} cells")
    leaks = [r for r in rows if r["answer"].get("text_reads") and r.get("words") and any(
        w.strip().lower() in str(r["answer"]["text_reads"]).lower() for w in str(r["words"]).split(",")
        if len(w.strip()) > 3)]
    print(f"trigger words read as text in the image: {len(leaks)} cells")
    for r in leaks[:5]:
        print("   ", r["answer"]["text_reads"], "<-", r["words"][:60])


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
