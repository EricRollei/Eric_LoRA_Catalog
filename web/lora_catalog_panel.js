// LoRA Catalog sidebar tab: status, refresh / card jobs, the card editor, settings.
// Talks to comfy/routes.py (/lora_catalog/*). (c) Eric Hiss. Dual license, see LICENSE.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const JOBS = [
  { command: "refresh", label: "Refresh catalog", tip: "Every step after adding LoRAs: index, hash, Civitai data, cards for new LoRAs, genre and mood, review pages, facets. Each step skips what is done." },
  { command: "cards", label: "Card new LoRAs", tip: "The vision model writes a card for every LoRA without one." },
  { command: "genre-mood", label: "Genre & mood", tip: "Genre and mood for cards that have none yet (text model)." },
  { command: "facets", label: "Rebuild facets", tip: "The table the LoRA Suggester reads. Run after editing cards outside the editor." },
  { command: "export-sidecars", label: "Forge/A1111 sidecars", tip: "Write <name>.json next to each LoRA (trigger words, preferred weight) for Forge and AUTOMATIC1111. Never touches a sidecar this tool did not write.", options: { write: true } },
];

const css = `
.lcp { display:flex; flex-direction:column; gap:10px; padding:10px; height:100%; box-sizing:border-box; overflow:auto;
       font-size:13px; color:var(--fg-color,#ddd); }
.lcp h3 { margin:4px 0 0; font-size:12px; text-transform:uppercase; letter-spacing:.06em; opacity:.7; font-weight:600; }
.lcp .stats { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; }
.lcp .stat { background:var(--comfy-input-bg,#222); border:1px solid var(--border-color,#444); border-radius:6px; padding:6px 8px; }
.lcp .stat b { display:block; font-size:18px; font-variant-numeric:tabular-nums; }
.lcp .stat span { opacity:.7; font-size:11px; }
.lcp .row { display:flex; flex-wrap:wrap; gap:6px; }
.lcp button { background:var(--comfy-input-bg,#222); color:inherit; border:1px solid var(--border-color,#555); border-radius:6px;
              padding:6px 10px; cursor:pointer; font-size:12px; }
.lcp button:hover { border-color:var(--p-primary-color,#6aa0ff); }
.lcp button.primary { border-color:var(--p-primary-color,#6aa0ff); }
.lcp button:disabled { opacity:.45; cursor:default; }
.lcp .problem { color:#f0a35a; }
.lcp .running { border-left:3px solid var(--p-primary-color,#6aa0ff); padding:4px 8px; background:var(--comfy-input-bg,#222); border-radius:4px; }
.lcp pre { margin:0; max-height:260px; overflow:auto; background:#0008; border-radius:6px; padding:6px; font-size:11px;
           white-space:pre-wrap; word-break:break-word; }
.lcp label { display:flex; flex-direction:column; gap:3px; font-size:11px; opacity:.9; }
.lcp input, .lcp textarea, .lcp select { background:var(--comfy-input-bg,#222); color:inherit; border:1px solid var(--border-color,#555);
           border-radius:4px; padding:4px 6px; font-size:12px; font-family:inherit; }
.lcp details summary { cursor:pointer; opacity:.85; }
.lcp .muted { opacity:.6; font-size:11px; }
`;

async function getJSON(path, options) {
  const response = await api.fetchApi(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}
const post = (path, body) => getJSON(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "onclick") node.onclick = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function toast(severity, summary, detail) {
  try { app.extensionManager.toast.add({ severity, summary, detail, life: 4000 }); }
  catch { console.log(`[LoRA Catalog] ${summary}: ${detail}`); }
}

function buildPanel(root) {
  root.innerHTML = "";
  if (!document.getElementById("lcp-style")) document.head.append(el("style", { id: "lcp-style", text: css }));
  const panel = el("div", { class: "lcp" });
  const stats = el("div", { class: "stats" });
  const problems = el("div");
  const running = el("div");
  const jobRow = el("div", { class: "row" });
  const logName = el("select");
  const logBox = el("pre", { text: "" });
  let lastLog = "refresh";

  for (const job of JOBS) {
    const button = el("button", { title: job.tip, class: job.command === "refresh" ? "primary" : "", text: job.label });
    button.onclick = async () => {
      try {
        const result = await post("/lora_catalog/job", { command: job.command, options: job.options || {} });
        toast("info", "LoRA Catalog", `${job.label} started (pid ${result.pid})`);
        lastLog = result.log; logName.value = lastLog; refresh();
      } catch (error) { toast("error", job.label, error.message); }
    };
    jobRow.append(button);
  }
  const editorButton = el("button", { text: "Open card editor", title: "The card editor in a new browser tab (started if it is not running)." });
  editorButton.onclick = async () => {
    editorButton.disabled = true;
    try {
      const result = await post("/lora_catalog/editor");
      if (result.up) window.open(result.url, "_blank"); else toast("error", "Card editor", "did not start - see the editor log below");
      lastLog = "editor"; logName.value = lastLog;
    } catch (error) { toast("error", "Card editor", error.message); }
    editorButton.disabled = false;
  };
  for (const name of ["refresh", "cards", "genre-mood", "facets", "index", "export-sidecars", "editor"]) logName.append(el("option", { value: name, text: name }));
  logName.onchange = () => { lastLog = logName.value; loadLog(); };

  const settingsBox = el("details");
  settingsBox.append(el("summary", { text: "Settings" }));
  const form = el("div", { style: "display:flex;flex-direction:column;gap:6px;margin-top:6px" });
  settingsBox.append(form);
  settingsBox.addEventListener("toggle", () => { if (settingsBox.open) loadSettings(); });

  panel.append(
    el("h3", { text: "LoRA Catalog" }), stats, problems, running,
    el("h3", { text: "Run" }), jobRow, el("div", { class: "row" }, editorButton),
    el("h3", { text: "Log" }), el("div", { class: "row" }, logName), logBox,
    settingsBox,
    el("div", { class: "muted", text: "Jobs run in their own process: ComfyUI keeps working, and restarting it does not stop them. On one GPU, a card pass and image generation share VRAM." }),
  );
  root.append(panel);

  function stat(value, label) { return el("div", { class: "stat" }, el("b", { text: value ?? "–" }), el("span", { text: label })); }

  async function loadLog() {
    try { const data = await getJSON(`/lora_catalog/log?name=${encodeURIComponent(lastLog)}&lines=150`); logBox.textContent = data.text || "(no log yet)"; logBox.scrollTop = logBox.scrollHeight; }
    catch (error) { logBox.textContent = error.message; }
  }

  async function refresh() {
    if (!root.isConnected) return;
    try {
      const data = await getJSON("/lora_catalog/status");
      const counts = data.counts || {};
      stats.replaceChildren(
        stat(counts.loras ?? counts.catalog, "LoRAs indexed"), stat(counts.cards, "with a card"),
        stat(counts.uncarded, "without a card"), stat(counts.no_genre, "cards without genre/mood"));
      problems.replaceChildren(...(data.problems || []).map(p => el("div", { class: "problem", text: "⚠ " + p })));
      const items = (data.running || []).filter(r => r.command !== "serve");
      const status = data.refresh || {};
      running.replaceChildren(...items.map(job => {
        const step = job.command === "refresh" && status.state === "running" ? ` – step ${status.step}/${status.of} ${status.name}` : "";
        const stop = el("button", { text: "Stop", title: "Stop this run (what it finished is saved)" });
        stop.onclick = async () => { const r = await post("/lora_catalog/job/stop", { pid: job.pid }); toast("info", "LoRA Catalog", r.result); refresh(); };
        return el("div", { class: "running" }, el("div", { text: `${job.command}${step}` }), el("div", { class: "muted", text: `since ${job.started}, pid ${job.pid}` }), stop);
      }));
      if (!items.length && status.state === "failed") running.append(el("div", { class: "problem", text: `Last refresh stopped at step ${status.step} (${status.name}) – see the log.` }));
      for (const button of jobRow.querySelectorAll("button")) button.disabled = items.some(i => ["refresh", "cards", "genre-mood", "index", "facets"].includes(i.command));
      await loadLog();
    } catch (error) {
      problems.replaceChildren(el("div", { class: "problem", text: `Panel cannot reach the catalog: ${error.message}` }));
    }
  }

  async function loadSettings() {
    form.replaceChildren(el("div", { class: "muted", text: "Loading…" }));
    let s;
    try { s = await getJSON("/lora_catalog/settings"); } catch (error) { form.replaceChildren(el("div", { class: "problem", text: error.message })); return; }
    const roots = el("textarea", { rows: "3" }); roots.value = (s.library_roots || []).join("\n");
    const base = el("input", { value: s.llm_base_url || "" });
    const backend = el("select"); for (const b of ["auto", "lmstudio", "ollama", "other"]) backend.append(el("option", { value: b, text: b })); backend.value = s.llm_backend || "auto";
    const card = el("input", { value: s.card_model || "" });
    const text = el("input", { value: s.genre_model || "" });
    const host = el("input", { value: s.civitai_host || "https://civitai.com" });
    const key = el("input", { type: "password", placeholder: s.civitai_key_set ? "set (leave blank to keep)" : "not set" });
    const result = el("pre", { text: "" });
    const save = el("button", { class: "primary", text: "Save" });
    save.onclick = async () => {
      try {
        const r = await post("/lora_catalog/settings", { library_roots: roots.value.split(/\r?\n/), llm_base_url: base.value, llm_backend: backend.value,
          card_model: card.value, genre_model: text.value, civitai_host: host.value, civitai_api_key: key.value });
        key.value = ""; result.textContent = `Saved to ${r.saved}` + (r.problems.length ? "\n" + r.problems.join("\n") : "\nReady.");
        refresh();
      } catch (error) { result.textContent = error.message; }
    };
    const test = el("button", { text: "Test LLM" });
    test.onclick = async () => {
      result.textContent = "Testing (a text call and an image call)…";
      try { const r = await post("/lora_catalog/llm_test"); result.textContent = [r.server, r.text, r.vision, r.json].filter(Boolean).join("\n"); }
      catch (error) { result.textContent = error.message; }
    };
    const presets = el("div", { class: "row" });
    for (const [name, url] of [["LM Studio", "http://localhost:1234/v1"], ["Ollama", "http://localhost:11434/v1"], ["llama.cpp", "http://localhost:8080/v1"]]) {
      const b = el("button", { text: name }); b.onclick = () => { base.value = url; }; presets.append(b);
    }
    form.replaceChildren(
      el("label", {}, "LoRA folders (one per line)", roots),
      el("label", {}, "LLM server (OpenAI-compatible)", base), presets,
      el("label", {}, "Server type (for unloading models)", backend),
      el("label", {}, "Vision model (writes the cards)", card),
      el("label", {}, "Text model (genre & mood; blank = vision model)", text),
      el("label", {}, "Civitai site", host),
      el("label", {}, "Civitai API key", key),
      el("div", { class: "row" }, save, test), result,
      el("div", { class: "muted", text: `Settings file: ${s.settings_file}\nData folder: ${s.data_dir}` }),
    );
  }

  refresh();
  const timer = setInterval(() => { if (!root.isConnected) clearInterval(timer); else refresh(); }, 4000);
}

app.registerExtension({
  name: "Eric.LoRACatalog.Panel",
  setup() {
    const manager = app.extensionManager;
    if (manager && typeof manager.registerSidebarTab === "function") {
      manager.registerSidebarTab({
        id: "eric-lora-catalog", icon: "pi pi-images", title: "LoRA Catalog",
        tooltip: "LoRA Catalog: refresh, card new LoRAs, open the card editor", type: "custom",
        render: (element) => buildPanel(element),
      });
    } else {
      console.warn("[LoRA Catalog] this ComfyUI frontend has no sidebar tabs; use launch.bat / refresh.bat instead");
    }
  },
});
