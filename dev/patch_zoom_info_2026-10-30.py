# Sweep Ratings zoom: prompt + settings strip with an on/off toggle (button + key Q), remembered (2026-10-30).
import os
import shutil
import sys
import py_compile

C = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_LoRA_Catalog"
L = os.path.join(C, "lora_catalog")
TAG = "_pre_zoominfo_2026-10-30"
P = {}

HELPER = '''

def _cell_settings(cell: Dict[str, Any], varying: set) -> List[Dict[str, Any]]:
    """The settings line under a zoomed sweep cell (2026-10-30): per stage that ran - sampler, schedule, steps, the
    hybrid handoff, eta, cfg - then size, churn form, shift and seed. vary = differs between cells of this sweep."""
    r = cell.get("resolved") or {}
    if not r:
        return []
    out: List[Dict[str, Any]] = []

    def add(key: str, text: str) -> None:
        out.append({"t": text, "vary": key in varying})

    def num(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    stages = ["s1"]
    if not cell.get("quick_screen"):
        ran = cell.get("trajectory") or {}
        stages += [f"s{n}" for n in (2, 3) if num(r.get(f"upscale_to_stage{n}")) > 0 or f"s{n}" in ran]
    any_eta = False
    for st in stages:
        out.append({"t": st.upper(), "stage": True})
        sampler = str(r.get(f"{st}_sampler", ""))
        add(f"{st}_sampler", sampler)
        add(f"{st}_schedule", str(r.get(f"{st}_schedule", "")))
        add(f"{st}_steps", f"{r.get(f'{st}_steps')} steps")
        if sampler.startswith("lcm_hybrid"):
            handoff = num(r.get(f"{st}_hybrid_handoff"))
            if handoff > 0:
                add(f"{st}_hybrid_handoff", f"handoff \\u03c3{handoff:.2f}")
            else:
                add(f"{st}_hybrid_steps", f"hybrid {r.get(f'{st}_hybrid_steps') or 'auto'}")
        eta = num(r.get(f"{st}_eta"))
        if eta > 0:
            any_eta = True
            add(f"{st}_eta", f"eta {eta:g}")
        cfg = num(r.get(f"{st}_cfg"))
        if cfg:
            add(f"{st}_cfg", f"cfg {cfg:g}")
    out.append({"t": "", "stage": True})
    add("s1_megapixels", f"{r.get('s1_megapixels')} MP" + (" (quick screen)" if cell.get("quick_screen") else ""))
    if any_eta and r.get("churn_form"):
        add("churn_form", f"churn {r.get('churn_form')}")
    if r.get("distilled_shift"):
        add("distilled_shift", f"shift {r.get('distilled_shift')}")
    add("seed", f"seed {cell.get('seed') if cell.get('seed') is not None else r.get('seed')}")
    return out
'''

P[os.path.join(L, "stacks.py")] = [
    ('''def ensure(connection: sqlite3.Connection) -> None:''', HELPER.lstrip("\n") + '''

def ensure(connection: sqlite3.Connection) -> None:''', 1),
    ('''        cells = []
        for cell in manifest.get("cells") or []:
            overrides = cell.get("overrides") or {}''',
     '''        cells = []
        # settings that differ between this sweep's cells (2026-10-30) - highlighted in the zoom's settings line
        seen_values: Dict[str, set] = {}
        for cell in manifest.get("cells") or []:
            for key, value in (cell.get("resolved") or {}).items():
                seen_values.setdefault(key, set()).add(json.dumps(value, sort_keys=True, default=str))
            seen_values.setdefault("seed", set()).add(str(cell.get("seed")))
        varying = {k for k, v in seen_values.items() if len(v) > 1}
        for cell in manifest.get("cells") or []:
            overrides = cell.get("overrides") or {}''', 1),
    ('''                          "marked": bool(mark),''',
     '''                          "marked": bool(mark),
                          "info": {"prompt": overrides.get("prompt", ""), "words": cell.get("words") or "",
                                   "settings": _cell_settings(cell, varying)},''', 1),
]

JS = '''
// Prompt & settings strip in the zoom (2026-10-30): on / off with the button or key Q, remembered.
var INFO_ON = recall("info") !== "0";
function drawInfo() {
  const box = $("#ov_info"), btn = $("#ov_infobtn"), c = VIEW[POS];
  if (btn) btn.classList.toggle("on", INFO_ON);
  if (!box) return;
  if (!INFO_ON || !c || !c.info) { box.style.display = "none"; return; }
  const i = c.info;
  const set = (i.settings || []).map(s => s.stage ? (s.t ? `<span class="stg">${esc(s.t)}</span>` : `<span class="sep"></span>`)
    : s.vary ? `<b class="vary" title="differs between the cells of this sweep">${esc(s.t)}</b>` : `<span>${esc(s.t)}</span>`).join(" ");
  box.innerHTML = (set ? `<div class="set">${set}</div>` : "")
    + (i.words ? `<div class="words" title="the trigger words added for this cell">${esc(i.words)}</div>` : "")
    + `<div class="pr">${i.prompt ? esc(i.prompt) : "<i>the Ultra panel's prompt (not stored in this sweep)</i>"}</div>`;
  box.style.display = "";
}
function toggleInfo() { INFO_ON = !INFO_ON; remember("info", INFO_ON ? "1" : "0"); drawInfo(); }
$("#ov_infobtn").onclick = toggleInfo;
'''

P[os.path.join(L, "sweeps.html")] = [
    ('''  <div class="strip" id="strip"></div>
''', '''  <div class="strip" id="strip"></div>
  <div class="info" id="ov_info"></div>
''', 1),
    ('''    <button data-v="">0 clear</button><button id="ov_close">Esc close</button></div>''',
     '''    <button id="ov_infobtn" title="Show or hide the prompt and settings of this cell (key Q). Yellow = differs between the cells of this sweep.">Q prompt &amp; settings</button>
    <button data-v="">0 clear</button><button id="ov_close">Esc close</button></div>''', 1),
    ('''.overlay .bar .v { font-weight:600; }''', '''.overlay .bar .v { font-weight:600; }
.overlay .info { padding:2px 16px 0; font-size:13px; line-height:1.45; color:#ddd; max-height:24vh; overflow:auto; }
.overlay .info .set { color:#bbb; margin-bottom:3px; } .overlay .info .set > * { margin-right:6px; }
.overlay .info .stg { color:#8fc3ff; font-weight:700; } .overlay .info .sep { margin-left:8px; }
.overlay .info b.vary { color:#ffcc33; font-weight:700; }
.overlay .info .words { color:#f0a060; margin-bottom:2px; } .overlay .info .pr { color:#eee; }
#ov_infobtn.on { background:#2f4a68; border-color:#8fc3ff; }''', 1),
    ('''  drawTags();
}
function closeZoom()''', '''  drawTags();
  drawInfo();
}
function closeZoom()''', 1),
    ('''$("#ov_close").onclick = closeZoom;
''', '''$("#ov_close").onclick = closeZoom;
''' + JS, 1),
    ('''  const flawKey = (SWEEP.flaws || []).find(f => f.key && f.key === k);''',
     '''  if (k === "q") { ev.preventDefault(); toggleInfo(); return; }      // prompt & settings strip (2026-10-30)
  const flawKey = (SWEEP.flaws || []).find(f => f.key && f.key === k);''', 1),
    ('''  return `${c.kind ? `<span class="kind ${c.kind}">${c.kind}</span>` : ""}${stack}<div class="meta">${extra}${clip}</div>`;''',
     '''  const full = (c.info && c.info.prompt) || c.prompt || "";       // hover: the whole prompt (2026-10-30)
  return `${c.kind ? `<span class="kind ${c.kind}">${c.kind}</span>` : ""}${stack}<div class="meta"${full ? ` title="${esc(full)}"` : ""}>${extra}${clip}</div>`;''', 1),
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
