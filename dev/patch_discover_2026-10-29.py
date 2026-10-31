# Discover stage 1 wiring (2026-10-29): API routes, CLI wishlist taste, editor screen. All-or-nothing, backups.
import os
import shutil
import sys
import py_compile

C = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\Eric_LoRA_Catalog"
L = os.path.join(C, "lora_catalog")
TAG = "_pre_discover_2026-10-29"
P = {}

P[os.path.join(L, "server.py")] = [
    ('''                elif url.path == "/api/style_status":''',
     '''                elif url.path == "/api/discover":                 # 2026-10-29, discover.py
                    from . import discover
                    self._json(discover.candidates(editor.settings, query.get("shelf", ""), query.get("all") == "1",
                                                   int(query.get("top", "100"))))
                elif url.path == "/api/style_status":''', 1),
    ('''                elif url.path == "/api/shutdown":''',
     '''                elif url.path == "/api/discover/dismiss":         # 2026-10-29: Not interested / undo
                    from . import discover
                    self._json(discover.dismiss(editor.settings, str(body.get("key") or ""), str(body.get("name") or ""),
                                                bool(body.get("undo"))))
                elif url.path == "/api/shutdown":''', 1),
]

P[os.path.join(L, "harvest.py")] = [
    ('''def wishlist(settings: Settings, top: int = 40, base: str = "", plain: bool = False) -> str:''',
     '''def wishlist(settings: Settings, top: int = 40, base: str = "", plain: bool = False, use_taste: bool = True) -> str:''', 1),
    ('''    for r in connection.execute("select * from lora_pairs where owned_a + owned_b = 1 and score > 0"):
        if base and base.lower() not in (r["base_model"] or "").lower():
            continue
        theirs, ours = (r["b"], r["a"]) if r["owned_a"] else (r["a"], r["b"])
        total[theirs] += r["score"]
        partners[theirs].append(ours)''',
     '''    # 2026-10-29 (Discover stage 1): each pair counts by how you rate YOUR side of it (discover.taste), and LoRAs
    # marked "Not interested" in Discover stay out. use_taste=False (--no-taste) = the earlier ranking.
    from .discover import _qualities, _tables, dismissed, taste as _taste
    quality = _qualities(connection, _tables(connection)) if use_taste else {}
    gone = dismissed(connection)
    for r in connection.execute("select * from lora_pairs where owned_a + owned_b = 1 and score > 0"):
        if base and base.lower() not in (r["base_model"] or "").lower():
            continue
        theirs, ours = (r["b"], r["a"]) if r["owned_a"] else (r["a"], r["b"])
        if theirs in gone:
            continue
        weight = 1.0
        if use_taste:
            row = local.owned(ours)
            weight = _taste(quality.get(str(row["sha256"] or "").lower())) if row is not None else 1.0
        total[theirs] += r["score"] * weight
        partners[theirs].append(ours)''', 1),
    ('''    head = "sum of pair scores" if plain else "pair sum x (0.5 + Civitai score/100)"''',
     '''    head = "sum of pair scores" if plain else "pair sum x (0.5 + Civitai score/100)"
    head = ("taste-weighted " if use_taste else "") + head''', 1),
]

P[os.path.join(L, "__main__.py")] = [
    ('''    pair.add_argument("--plain", action="store_true", help="wishlist by pair scores only (no Civitai calls)")''',
     '''    pair.add_argument("--plain", action="store_true", help="wishlist by pair scores only (no Civitai calls)")
    pair.add_argument("--no-taste", action="store_true", help="wishlist: ignore your ratings of your own LoRAs "
                      "(the ranking before 2026-10-29)")''', 1),
    ('''        print(wishlist(settings, top=args.top, base=args.base, plain=args.plain))''',
     '''        print(wishlist(settings, top=args.top, base=args.base, plain=args.plain, use_taste=not args.no_taste))''', 1),
]

JS = r'''
// ---------- Discover (2026-10-29, docs/design/DISCOVER_STAGE1.md): LoRAs you don't own ------------------------------
// Ranked by how often Civitai users put them in the same image as YOUR LoRAs, weighted by how you rate yours, times
// Civitai's community score; for the base model of the selected shelf. var, not let: loadQueue may run first.
var DISC_ON = false, DISC = null, DISC_POS = 0;
function discShelf() {
  const v = $("#shelf").value;
  if (v === "__folder__") return FOLDER && FOLDER.rel ? FOLDER.rel.split(/[\\/]/)[0] : "";
  return v;
}
function discoverOff(silent) {
  DISC_ON = false; DISC = null; $("#discbtn").classList.remove("on"); $("body").classList.remove("discover");
  if (!silent) guarded(() => loadQueue(CARD ? CARD.path : undefined));
}
async function toggleDiscover(on) {
  if (!on) { discoverOff(false); return; }
  if (DUPES) { DUPES = false; $("#dupbtn").classList.remove("on"); }
  if (LOOKS) { LOOKS = false; $("#looksbtn").classList.remove("on"); }
  $("body").classList.remove("dupes");
  DISC_ON = true; $("#discbtn").classList.add("on"); $("body").classList.add("discover");
  await loadDiscover();
}
async function loadDiscover() {
  const shelf = discShelf();
  $("#queue").innerHTML = `<div class="found">Discover: finding LoRAs you don't own${shelf ? " for " + esc(shelf) : ""}... (the first time can take a few seconds - Civitai numbers)</div>`;
  $("#main").innerHTML = "";
  DISC = await api("/api/discover?" + new URLSearchParams({shelf, top: "100"}));
  DISC_POS = 0; drawDiscover(); discOpen(0);
}
function drawDiscover() {
  const d = DISC || {items: []};
  const h = d.harvested || {};
  const head = d.note ? `<div class="found">${esc(d.note)} &middot; <a id="discoff">done</a></div>`
    : `<div class="found" title="Ranked by how often Civitai users put each LoRA in the same image as yours (distinct people), weighted by your ratings of yours, times Civitai's community score. Harvest: ${h.loras || 0} of your LoRAs, last ${esc(h.last || "?")} - refresh with python run.py harvest --only <folder>, then python run.py pairs.">Discover: <b>${d.items.length}</b> LoRAs you don't own${d.bases && d.bases.length ? " (" + esc(d.bases.join(", ")) + ")" : ""}${d.dismissed ? ` &middot; ${d.dismissed} hidden` : ""} &middot; <a id="discoff">done</a></div>`;
  $("#queue").innerHTML = head + (d.items.map((it, i) => `<div class="item${i === DISC_POS ? " on" : ""}" data-di="${i}">
      <span class="n" title="${esc(it.name)}">${esc(it.name)}</span>${it.other_version ? `<small class="ov" title="you have another version of this model">have v.</small>` : ""}
      <small title="Discover rank">${Math.round(it.rank)}</small></div>`).join("") || "");
  $("#queue").querySelectorAll(".item").forEach(el => el.onclick = () => discOpen(+el.dataset.di));
  const off = $("#discoff"); if (off) off.onclick = () => discoverOff(false);
}
function discOpen(i) {
  if (!DISC || !DISC.items.length) {
    $("#main").innerHTML = `<div class="empty">${esc((DISC && DISC.note) || "Nothing to suggest yet for this shelf.")}</div>`;
    return;
  }
  DISC_POS = Math.max(0, Math.min(DISC.items.length - 1, i));
  $("#queue").querySelectorAll(".item").forEach(el => el.classList.toggle("on", +el.dataset.di === DISC_POS));
  const sel = $("#queue").querySelector(".item.on"); if (sel) sel.scrollIntoView({block: "nearest"});
  const it = DISC.items[DISC_POS], c = it.civitai || {};
  const yours = p => p.rating ? "&#9733;".repeat(p.rating)
    : p.quality != null ? `${p.quality >= 0 ? "+" : ""}${p.quality.toFixed(1)} vs your average` : `<span class="muted">not rated</span>`;
  const rows = it.partners.map(p => `<tr><td>${esc(p.name)}</td><td>${yours(p)}</td><td class="n">${p.users} people</td>
      <td class="n muted" title="how much more often the two meet than chance">lift ${(+p.lift).toFixed(1)}</td>
      <td class="n muted" title="how much this one adds to the rank: pair score x your taste (${p.taste})">${p.contribution.toFixed(1)}</td></tr>`).join("");
  const w = it.weights ? `<b>${it.weights.median}</b> <span class="muted">(people used ${it.weights.low} to ${it.weights.high}, ${it.weights.n} times)</span>` : "unknown";
  $("#main").innerHTML = `<div class="disc">
    <h2>${esc(it.name)} <small>${esc(it.version || "")}</small></h2>
    <div class="badges"><span class="badge">${esc(it.base_model || "?")}</span>${it.nsfw ? `<span class="badge warn">NSFW</span>` : ""}
      ${c.score != null ? `<span class="badge" title="Civitai community score 0-100: approval, likes per download, downloads per day - each a percentile among your LoRAs of the same base">Civitai ${Math.round(c.score)}</span>` : ""}
      ${c.downloads != null ? `<span class="badge">${(+c.downloads).toLocaleString()} downloads</span>` : ""}
      ${c.up != null ? `<span class="badge">${c.up} up / ${c.down || 0} down</span>` : ""}
      ${it.other_version ? `<span class="badge warn" title="${esc(it.other_version)}">you have another version</span>` : ""}</div>
    <p class="why">Used together with <b>${it.n_partners}</b> of your LoRAs in Civitai images${it.n_partners > it.partners.length ? ` (the ${it.partners.length} that count most)` : ""}:</p>
    <table class="disctbl"><tr><th>your LoRA</th><th>your rating</th><th>used together by</th><th></th><th>adds</th></tr>${rows}</table>
    <p class="why">Weight people give it: ${w}</p>
    ${it.examples.length ? `<p class="why">Example images on Civitai: ${it.examples.map((u, k) => `<a href="${esc(u)}" target="_blank" rel="noopener">${k + 1}</a>`).join(" &middot; ")}</p>` : ""}
    <div class="discbtns">
      ${it.url ? `<a class="btn primary" href="${esc(it.url)}" target="_blank" rel="noopener">Open on Civitai</a>` : ""}
      ${it.version_id ? `<button id="disccopy" title="for the Civitai downloader">Copy version id ${it.version_id}</button>` : ""}
      <button id="discno" title="Hide it from Discover and from run.py pairs --wishlist (undo appears below)">Not interested</button>
    </div>
    <p class="muted small">J / K or arrows: next / previous &middot; Esc: back to your cards</p></div>`;
  const cp = $("#disccopy");
  if (cp) cp.onclick = () => { navigator.clipboard.writeText(String(it.version_id)).then(() => msg("version id copied"), () => msg("copy failed", true)); };
  $("#discno").onclick = () => discDismiss(it).catch(e => msg(e.message, true));
}
async function discDismiss(it) {
  await api("/api/discover/dismiss", {key: it.key, name: it.name});
  const i = DISC.items.indexOf(it);
  DISC.items.splice(i, 1); DISC.dismissed = (DISC.dismissed || 0) + 1;
  drawDiscover(); discOpen(i);
  $("#msg").className = "msg";
  $("#msg").innerHTML = `hid ${esc(it.name)} &middot; <a id="discundo" style="cursor:pointer;color:var(--accent)">undo</a>`;
  $("#discundo").onclick = async () => {
    await api("/api/discover/dismiss", {key: it.key, undo: true});
    DISC.items.splice(i, 0, it); DISC.dismissed = Math.max(0, DISC.dismissed - 1);
    drawDiscover(); discOpen(i); msg("restored " + it.name);
  };
}
$("#discbtn").onclick = () => guarded(() => toggleDiscover(!DISC_ON));
'''

CSS = '''
/* Discover (2026-10-29) */
body.discover footer #prev, body.discover footer #save, body.discover footer #unsure, body.discover footer #skip { display:none; }
aside .item small.ov { color:var(--warn); }
.disc { padding:18px 22px; max-width:1100px; }
.disc h2 { margin:0 0 8px; } .disc h2 small { color:var(--muted); font-weight:400; font-size:14px; }
.disc .badges { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }
.disc .badge.warn { color:var(--warn); border-color:var(--warn); }
.disc .why { margin:14px 0 6px; }
.disctbl { border-collapse:collapse; font-size:13px; }
.disctbl td, .disctbl th { padding:4px 10px; border-bottom:1px solid var(--soft); text-align:left; }
.disctbl th { color:var(--muted); font-weight:500; } .disctbl td.n { text-align:right; white-space:nowrap; }
.disc .discbtns { display:flex; gap:8px; margin:18px 0 8px; }
.disc .discbtns a.btn, .disc .discbtns button { border:1px solid var(--line); background:var(--soft); border-radius:6px; padding:6px 13px; cursor:pointer; color:inherit; text-decoration:none; font-size:14px; }
.disc .discbtns a.btn.primary { background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }
.disc .small { font-size:12px; }'''

P[os.path.join(L, "editor.html")] = [
    ('''  <button class="fold" id="foldbtn" title="Show the library folders''',
     '''  <button class="fold" id="discbtn" title="Discover: LoRAs you don't own that Civitai users stack with the LoRAs you rate highly, for the base model of the selected shelf">Discover</button>
  <button class="fold" id="foldbtn" title="Show the library folders''', 1),
    ('''footer .msg.err { color:var(--warn); }''', '''footer .msg.err { color:var(--warn); }''' + CSS, 1),
    ('''async function loadQueue(keepPath) {
''', '''async function loadQueue(keepPath) {
  if (DISC_ON) {                                     // Discover (2026-10-29): a search or image search leaves it
    if ($("#q").value.trim() || IMGQ) discoverOff(true); else return loadDiscover();
  }
''', 1),
    ('''function toggleDupes(on) {
''', '''function toggleDupes(on) {
  if (on && DISC_ON) discoverOff(true);
''', 1),
    ('''function toggleLooks(on) {
''', '''function toggleLooks(on) {
  if (on && DISC_ON) discoverOff(true);
''', 1),
    ('''  const zoomed = $("#overlay").classList.contains("on");
  if (ev.key === "Escape") { unzoom(); if (typing) document.activeElement.blur(); return; }''',
     '''  const zoomed = $("#overlay").classList.contains("on");
  if (DISC_ON) {                                     // Discover (2026-10-29): its own keys
    if (ev.key === "Escape") { if (typing) document.activeElement.blur(); else discoverOff(false); return; }
    if (typing || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (/^(j|J|ArrowRight|ArrowDown)$/.test(ev.key)) { ev.preventDefault(); discOpen(DISC_POS + 1); }
    else if (/^(k|K|ArrowLeft|ArrowUp)$/.test(ev.key)) { ev.preventDefault(); discOpen(DISC_POS - 1); }
    return;
  }
  if (ev.key === "Escape") { unzoom(); if (typing) document.activeElement.blur(); return; }''', 1),
    ('''$("#looksbtn").onclick = () => guarded(() => toggleLooks(!LOOKS));
''', '''$("#looksbtn").onclick = () => guarded(() => toggleLooks(!LOOKS));
''' + JS, 1),
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
    py_compile.compile(os.path.join(L, "discover.py"), doraise=True)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
