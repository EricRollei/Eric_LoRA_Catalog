# Sweep Ratings: letter keys for the flaw AND praise buttons, highlighted in the button label (2026-10-30).
import os
import shutil
import sys
import py_compile

C = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_LoRA_Catalog"
L = os.path.join(C, "lora_catalog")
TAG = "_pre_tagkeys_2026-10-30"
P = {}

P[os.path.join(L, "stacks.py")] = [
    ('''FLAWS = [
    ("signature or watermark", "training", "s"), ("unwanted text or lettering", "training", "t"),
    ("added frame or border", "training", ""), ("changes the character or face", "training", "c"),
    ("changes the subject or scene", "fit", ""), ("changes pose or composition", "fit", ""),
    ("ignores the prompt", "fit", "i"), ("same face every time", "training", ""),
    ("faces or hands broken", "training", "f"), ("anatomy wrong", "training", ""),
    ("artifacts, noise or banding", "training", "a"), ("overcooked or too strong", "strength", ""),
    ("too weak", "strength", ""), ("style didn't show", "fit", "w"), ("colour cast", "training", ""),
    ("oversaturated", "training", ""), ("washed out or flat", "training", ""), ("soft or blurry", "training", ""),
    ("unwanted NSFW", "training", "n"),
]
PRAISE = ["great colour", "great detail", "great composition", "great lighting", "great texture or brushwork",
          "true to its style", "keeps the subject", "follows the prompt", "strong mood", "clean and coherent"]''',
     '''# Keys (2026-10-30, Eric's set): one letter per button in the Sweep Ratings zoom; the letter is highlighted in the label.
FLAWS = [
    ("signature or watermark", "training", "s"), ("unwanted text or lettering", "training", "t"),
    ("added frame or border", "training", "r"), ("changes the character or face", "training", "f"),
    ("changes the subject or scene", "fit", ""), ("changes pose or composition", "fit", ""),
    ("ignores the prompt", "fit", "i"), ("same face every time", "training", "e"),
    ("faces or hands broken", "training", "h"), ("anatomy wrong", "training", "a"),
    ("artifacts, noise or banding", "training", "x"), ("overcooked or too strong", "strength", "o"),
    ("too weak", "strength", ""), ("style didn't show", "fit", "w"), ("colour cast", "training", ""),
    ("oversaturated", "training", "v"), ("washed out or flat", "training", ""), ("soft or blurry", "training", "b"),
    ("unwanted NSFW", "training", "n"),
]
PRAISE = ["great colour", "great detail", "great composition", "great lighting", "great texture or brushwork",
          "true to its style", "keeps the subject", "follows the prompt", "strong mood", "clean and coherent"]
PRAISE_KEYS = {"great colour": "c", "great detail": "d", "great composition": "g", "great lighting": "l",
               "great texture or brushwork": "u", "keeps the subject": "k", "follows the prompt": "p",
               "strong mood": "m"}''', 1),
    ('''"flaws": [{"tag": f[0], "cause": f[1], "key": f[2]} for f in FLAWS], "praise": PRAISE,''',
     '''"flaws": [{"tag": f[0], "cause": f[1], "key": f[2]} for f in FLAWS], "praise": PRAISE,
            "praise_keys": PRAISE_KEYS,''', 1),
]

P[os.path.join(L, "sweeps.html")] = [
    ('''function keyed(tag, key) {
  if (!key) return esc(tag);
  const at = tag.toLowerCase().indexOf(key);
  return at >= 0 ? esc(tag.slice(0, at)) + `<u>${esc(tag[at])}</u>` + esc(tag.slice(at + 1)) : `${esc(tag)} <small>(${key.toUpperCase()})</small>`;
}''', '''function keyed(tag, key) {
  // 2026-10-30: the key letter is big, bold and coloured; a word that STARTS with it is preferred
  // ("unwanted NSFW" -> the N of NSFW), else its first appearance, else "(X)" after the label.
  if (!key) return esc(tag);
  const low = tag.toLowerCase();
  let at = -1;
  for (let i = 0; i < low.length; i++) if (low[i] === key && (i === 0 || /[\\s,]/.test(low[i - 1]))) { at = i; break; }
  if (at < 0) at = low.indexOf(key);
  const hk = ch => `<b class="hk">${esc(ch.toUpperCase())}</b>`;
  return at >= 0 ? esc(tag.slice(0, at)) + hk(tag[at]) + esc(tag.slice(at + 1)) : `${esc(tag)} (${hk(key)})`;
}''', 1),
    ('''      `<button class="praise${on(t)}" data-t="${esc(t)}">${esc(t)}</button>`).join("");''',
     '''      `<button class="praise${on(t)}" data-t="${esc(t)}">${keyed(t, (SWEEP.praise_keys || {})[t])}</button>`).join("");''', 1),
    ('''  const flawKey = (SWEEP.flaws || []).find(f => f.key && f.key === k);
  if (flawKey) { ev.preventDefault(); toggleTag(VIEW[POS], flawKey.tag); }''',
     '''  const flawKey = (SWEEP.flaws || []).find(f => f.key && f.key === k);
  const praiseKey = Object.entries(SWEEP.praise_keys || {}).find(([_t, key]) => key === k);
  if (flawKey) { ev.preventDefault(); toggleTag(VIEW[POS], flawKey.tag); }
  else if (praiseKey) { ev.preventDefault(); toggleTag(VIEW[POS], praiseKey[0]); }''', 1),
    ('''.overlay .tags button.praise.on { background:#2e6b47; border-color:#6cc28f; }''',
     '''.overlay .tags button.praise.on { background:#2e6b47; border-color:#6cc28f; }
.overlay .tags button b.hk { font-size:1.3em; font-weight:800; color:#ffcc33; text-decoration:underline; line-height:1; }''', 1),
]


def main():
    staged, ok = {}, True
    for path, edits in P.items():
        text = open(path, encoding="utf-8", newline="").read()
        nl = "\r\n" if "\r\n" in text else "\n"
        for k, (old, new, want) in enumerate(edits):
            o, n = old.replace("\n", nl), new.replace("\n", nl)
            got = text.count(o)
            if got != want:
                print(f"MISMATCH {os.path.basename(path)} edit#{k}: found {got}, expected {want}: {old[:70]!r}")
                ok = False
                continue
            text = text.replace(o, n)
        staged[path] = text
    if not ok:
        print("ABORTED - nothing written")
        return 1
    # the header hint: replace whatever <span class="keys" ...>...</span> line holds with the new key list
    sw = os.path.join(L, "sweeps.html")
    t = staged[sw]
    a = t.find('<span class="keys" title="Flaw keys:')
    b = t.find("</span>", a)
    if a < 0 or b < 0:
        print("MISMATCH header hint")
        return 1
    hint = ('<span class="keys" title="Flaws: S signature, T text, R frame, F changes character/face, I ignores prompt, '
            'E same face, H faces/hands, A anatomy, X artifacts, O overcooked, W style didn\'t show, V oversaturated, '
            'B soft/blurry, N NSFW. Praise: C colour, D detail, G composition, L lighting, U texture, K keeps subject, '
            'P follows prompt, M mood. Rate the baseline cells too: they show how hard the prompt is.">1-5 stars '
            '&middot; 0 clear &middot; + / - stronger / weaker &middot; letters = flaw / praise (highlighted on the '
            'buttons) &middot; arrows &middot; Space zoom')
    staged[sw] = t[:a] + hint + t[b:]
    for path, text in staged.items():
        base, ext = os.path.splitext(os.path.basename(path))
        bk = os.path.join(C, "docs", "backup", f"{base}{TAG}{ext}")
        if not os.path.exists(bk):
            shutil.copy2(path, bk)
        open(path, "w", encoding="utf-8", newline="").write(text)
        if path.endswith(".py"):
            py_compile.compile(path, doraise=True)
        print("patched", os.path.basename(path))
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
