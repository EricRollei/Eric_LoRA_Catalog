"""
Checks for style.py (CSD style vectors), 2026-10-12.

    python dev/verify_style.py [--weights path] [--reference csd_cache.npz]

1. model: the plain-torch ViT in style.py gives the same vectors as the open_clip build the pilot used
   (cosine > 0.999 on images the pilot embedded; --reference is the pilot's csd_cache.npz)
2. pack/unpack and the signature maths
3. stored vectors (when `run.py style` has run): known look-alikes find each other
"""

from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lora_catalog import style  # noqa: E402
from lora_catalog.config import load_settings  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{('  - ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--reference", default=r"A:\LoRA_Catalog\pilot_effect_vectors\csd_cache.npz")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    settings = load_settings()
    settings.style_enabled = True
    if args.weights:
        settings.style_model = args.weights

    # 2. maths
    v = [random.uniform(-1, 1) for _ in range(style.DIM)]
    back = style.unpack(style.pack(v))
    check("pack/unpack float16", max(abs(a - b) for a, b in zip(v, back)) < 1e-3)
    unit = lambda x: [a / (sum(b * b for b in x) ** 0.5) for a in x]  # noqa: E731
    vs = [unit([random.gauss(0, 1) for _ in range(style.DIM)]) for _ in range(5)]
    sig, agree = style._signature(vs)
    pairs = [sum(a * b for a, b in zip(vs[i], vs[j])) for i in range(5) for j in range(i + 1, 5)]
    check("agreement = mean pairwise cos", abs(agree - sum(pairs) / len(pairs)) < 1e-9)
    check("signature unit length", abs(sum(a * a for a in sig) - 1) < 1e-9)

    # 1. model vs the pilot's open_clip vectors
    reason = style.available(settings)
    if reason:
        check("model available", False, reason)
    elif os.path.isfile(args.reference):
        import numpy as np
        data = np.load(args.reference, allow_pickle=True)
        keys = [k for k in data["keys"].tolist() if os.path.isfile(k.split("|")[0]) and "\\images\\" in k]
        random.seed(1)
        picks = random.sample(keys, min(args.n, len(keys)))
        ref = dict(zip(data["keys"].tolist(), data["vecs"]))
        model = style.StyleModel(settings)
        tensors = [model.load_file(k.split("|")[0]) for k in picks]
        got = model.embed_tensors([t for t in tensors if t is not None])
        cos = [float(np.dot(np.asarray(g), ref[k]) / np.linalg.norm(ref[k])) for g, k in
               zip(got, [k for k, t in zip(picks, tensors) if t is not None])]
        check(f"model matches the pilot on {len(cos)} preview images", min(cos) > 0.999,
              f"min cos {min(cos):.5f}, mean {sum(cos) / len(cos):.5f}")
        model.close()
    else:
        print(f"skip model check: {args.reference} not found")

    # 3. stored vectors
    index = style.StyleIndex(settings)
    index.ensure()
    if not index.items:
        print("skip neighbour checks: no stored vectors yet (python run.py style)")
    else:
        def nearest(words):
            hits = [i for i in index.items if words in i["name"].lower()]
            if not hits:
                return None, []
            return hits[0]["name"], [s["name"] for s in index.similar(hits[0]["path"], n=3, kinds="all", nsfw=True)]
        for a, b in (("mike mignola style", "mignola"), ("@gustav klimt", "klimt"), ("araminta frazetta", "frazetta")):
            name, near = nearest(a)
            if name is None:
                print(f"skip: no '{a}'")
                continue
            check(f"{name} -> a '{b}' LoRA in its top 3", any(b in n.lower() for n in near), "; ".join(near))
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
