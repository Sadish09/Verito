const { App, Modal, Notice, Plugin, PluginSettingTab, Setting, requestUrl } =
  require("obsidian");

//Defaults

const DEFAULTS = {
  backendUrl:       "http://localhost:8765",
  autoLaunchBinary: true,
  maxResults:       10,
  minScore:         0.4,
  debounceMs:       400,
  showScorePill:    true,
  showChunkBadge:   true,
  previewLength:    240,
  watchForChanges:  true,
  excludePatterns:  "Templates/**, .trash/**",
  binaryVersion:    null,
};

//BinaryManager

class BinaryManager {
  constructor(plugin) { this.plugin = plugin; this.proc = null; }

  async ensureReady() {
    if (await this.healthy()) return;
    if (!this.exists() || !await this.versionOk()) await this.download();
    await this.launch();
    await this.waitHealthy();
  }

  kill() { try { this.proc?.kill(); } catch {} this.proc = null; }

  name() {
    return process.platform === "darwin" ? "verito-mac"
         : process.platform === "win32"  ? "verito-win.exe"
         :                                 "verito-linux";
  }

  path() {
    const path = require("path"), base = this.plugin.app.vault.adapter.basePath;
    return path.join(base, this.plugin.manifest.dir, this.name());
  }

  exists() { try { return require("fs").existsSync(this.path()); } catch { return false; } }

  async versionOk() {
    return (await this.plugin.loadData())?.binaryVersion === this.plugin.manifest.version;
  }

  async healthy() {
    try {
      const r = await requestUrl({ url: `${this.plugin.settings.backendUrl}/health`, throw: false });
      return r.status === 200 && r.json?.status === "ok";
    } catch { return false; }
  }

  async download() {
    const https = require("https"), fs = require("fs");
    const { version } = this.plugin.manifest;
    const rel   = await requestUrl({
      url: `https://api.github.com/repos/Sadish09/verito/releases/tags/v${version}` });
    const asset = rel.json.assets?.find(a => a.name === this.name());
    if (!asset) throw new Error(`Asset ${this.name()} not in release v${version}`);

    await new Promise((res, rej) => {
      const file = fs.createWriteStream(this.path());
      https.get(asset.browser_download_url, { headers: { "User-Agent": "verito" } }, r => {
        const total = parseInt(r.headers["content-length"] ?? "0", 10);
        let got = 0;
        r.on("data", c => {
          got += c.length;
          this.plugin.broadcast("downloading", { percent: total ? Math.round(got/total*100) : 0 });
        });
        r.pipe(file);
        file.on("finish", res);
        file.on("error", rej);
      });
    });

    if (process.platform !== "win32") fs.chmodSync(this.path(), 0o755);
    await this.plugin.saveData({ ...await this.plugin.loadData(), binaryVersion: version });
  }

  async launch() {
    this.proc = require("child_process").spawn(this.path(), [], { detached: false, stdio: "ignore" });
    this.proc.on("error", e => new Notice(`Verito: ${e.message}`));
    this.proc.on("exit",  () => { this.proc = null; });
  }

  async waitHealthy() {
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 500));
      if (await this.healthy()) return;
    }
    throw new Error("Backend did not start in 15 s");
  }
}

//Plugin

class VeritoPlugin extends Plugin {
  async onload() {
    this.settings  = Object.assign({}, DEFAULTS, await this.loadData());
    this.lastStatus = null;
    this.subs       = new Set();
    this.mgr        = new BinaryManager(this);

    this.addSettingTab(new VeritoSettingTab(this.app, this));
    this.addRibbonIcon("search", "Verito", () => new SearchModal(this.app, this).open());
    this.addCommand({
      id: "open-search", name: "Open semantic search",
      hotkeys: [{ modifiers: ["Ctrl","Shift"], key: "s" }],
      callback: () => new SearchModal(this.app, this).open(),
    });
    this.addCommand({ id: "index-vault", name: "Index vault",
      callback: () => this.index() });

    this.boot();
  }

  onunload() { this.mgr.kill(); }

  broadcast(state, meta = {}) { this.subs.forEach(cb => cb(state, meta)); }
  subscribe(cb) { this.subs.add(cb); return () => this.subs.delete(cb); }

  async boot() {
    try {
      if (this.settings.autoLaunchBinary) await this.mgr.ensureReady();
      this.applyStatus(await this.getStatus());
    } catch(e) { this.broadcast("error", { msg: String(e) }); }
  }

  applyStatus(s) {
    this.lastStatus = s;
    if (!s.model_configured) return this.broadcast("setup");
    if (s.is_indexing)       return this.broadcast("indexing");
    if (!s.ollama_reachable) return this.broadcast("error", { msg: "Ollama unreachable" });
    this.broadcast("idle", { chunks: s.total_chunks, files: s.indexed_files });
  }

  async healthy() {
    try {
      return (await requestUrl({ url: `${this.settings.backendUrl}/health`, throw: false })).status === 200;
    } catch { return false; }
  }

  async getStatus()       { return (await requestUrl({ url: `${this.settings.backendUrl}/status` })).json; }
  async getModels()       { return (await requestUrl({ url: `${this.settings.backendUrl}/models` })).json; }
  async patchConfig(body) {
    await requestUrl({ url: `${this.settings.backendUrl}/config`, method: "PATCH",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }

  async index() {
    const r = await requestUrl({ url: `${this.settings.backendUrl}/index`, method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vault_path: this.app.vault.adapter.basePath,
                             watch: this.settings.watchForChanges }), throw: false });
    if (r.status === 409) { this.broadcast("indexing"); return; }
    if (r.status === 422) { new Notice("Verito: select a model first."); return; }
    this.broadcast("indexing");
    const poll = setInterval(async () => {
      const s = await this.getStatus().catch(() => null);
      if (s && !s.is_indexing) { clearInterval(poll); this.applyStatus(s); }
    }, 2000);
    setTimeout(() => clearInterval(poll), 600_000);
  }

  async search(query, topK, model) {
    const r = await requestUrl({ url: `${this.settings.backendUrl}/search`, method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK, ...(model ? { embedding_model: model } : {}) }) });
    return r.json.results;
  }

  async saveSettings() { await this.saveData(this.settings); }
}

//Search Modal

class SearchModal extends Modal {
  constructor(app, plugin) {
    super(app);
    this.plugin = plugin;
    this.timer  = null;
    this.query  = "";
    this.selIdx = -1;
    this.unsub  = null;
  }

  onOpen() {
    this.contentEl.addClass("ss-modal");
    this.contentEl.innerHTML = `
      <div class="ss-search-row">
        <svg class="ss-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="6.5" cy="6.5" r="4.5"/><path d="M10.5 10.5l3 3"/>
        </svg>
        <input class="ss-input" placeholder="Search your vault…" type="text"/>
        <kbd class="ss-kbd">esc</kbd>
      </div>
      <div class="ss-bar">
        <div class="ss-dot"></div>
        <span class="ss-bar-label">Ready</span>
        <div class="ss-track"><div class="ss-fill"></div></div>
        <span class="ss-bar-count"></span>
      </div>
      <div class="ss-results"></div>`;

    const q       = s => this.contentEl.querySelector(s);
    this.$input   = q(".ss-input");
    this.$dot     = q(".ss-dot");
    this.$label   = q(".ss-bar-label");
    this.$fill    = q(".ss-fill");
    this.$count   = q(".ss-bar-count");
    this.$results = q(".ss-results");

    this.$input.addEventListener("input", () => {
      clearTimeout(this.timer);
      this.query = this.$input.value.trim();
      if (this.query.length < 2) { this.$results.innerHTML = ""; this.setBar("idle"); return; }
      this.setBar("searching", { label: "Waiting…" });
      this.timer = setTimeout(() => this.runSearch(), this.plugin.settings.debounceMs);
    });

    this.$input.addEventListener("keydown", e => {
      const items = [...this.$results.querySelectorAll(".ss-result")];
      if      (e.key === "ArrowDown") { e.preventDefault(); this.moveSel(items,  1); }
      else if (e.key === "ArrowUp")   { e.preventDefault(); this.moveSel(items, -1); }
      else if (e.key === "Enter")     { e.preventDefault(); items[this.selIdx]?.click(); }
      else if (e.key === "Escape")    { this.close(); }
    });

    this.unsub = this.plugin.subscribe((state, meta) => this.setBar(state, meta));
    this.syncStatus();
    this.$input.focus();
  }

  onClose() {
    clearTimeout(this.timer);
    this.unsub?.();
    this.contentEl.empty();
  }

  setBar(state, meta = {}) {
    const GREEN = "var(--ss-green)", AMBER = "var(--ss-amber)", RED = "var(--ss-red)";
    const color = state === "idle" ? GREEN : state === "error" ? RED : AMBER;
    const pulse = ["searching","embedding","indexing"].includes(state);

    this.$dot.style.setProperty("--dot-color", color);
    this.$dot.classList.toggle("ss-dot--pulse", pulse);
    this.$fill.style.setProperty("--fill-color", color);
    this.$fill.classList.toggle("ss-fill--indeterminate", pulse);

    if (state === "downloading") {
      this.$fill.classList.remove("ss-fill--indeterminate");
      this.$fill.style.width    = `${meta.percent ?? 0}%`;
      this.$label.textContent   = `Downloading · ${meta.percent ?? 0}%`;
      this.$count.textContent   = "";
      return;
    }

    if (state === "idle" || state === "error") this.$fill.style.width = "100%";

    this.$label.textContent = {
      idle:      meta.label ?? "Ready",
      setup:     "Select a model to begin",
      searching: meta.label ?? "Searching…",
      embedding: "Invoking Ollama…",
      indexing:  "Indexing vault…",
      error:     meta.msg ?? "Backend unreachable",
    }[state] ?? "Ready";

    this.$count.textContent = {
      idle:      meta.chunks ? `${Number(meta.chunks).toLocaleString()} chunks · ${meta.files} files` : "",
      setup:     "Open settings →",
      embedding: meta.model ?? "",
      error:     "Check settings",
    }[state] ?? "";
  }

  syncStatus() {
    const s = this.plugin.lastStatus;
    if (!s)                  return this.setBar("idle");
    if (!s.model_configured) return this.setBar("setup");
    if (s.is_indexing)       return this.setBar("indexing");
    if (!s.ollama_reachable) return this.setBar("error", { msg: "Ollama unreachable" });
    this.setBar("idle", { chunks: s.total_chunks, files: s.indexed_files });
  }

  async runSearch() {
    if (!this.query) return;
    const { maxResults, minScore, showScorePill, showChunkBadge, previewLength } = this.plugin.settings;
    const model = this.plugin.lastStatus?.embedding_model ?? null;

    this.setBar("embedding", { model: model ?? "" });
    if (!await this.plugin.healthy()) { this.setBar("error", { msg: "Backend unreachable" }); return; }
    this.setBar("searching");

    try {
      const t0  = performance.now();
      const raw = await this.plugin.search(this.query, maxResults, model);
      const res = raw.filter(r => r.score >= minScore);
      this.setBar("idle", { label: `${res.length} result${res.length !== 1 ? "s" : ""} · ${Math.round(performance.now()-t0)}ms` });
      this.renderResults(res, showScorePill, showChunkBadge, previewLength);
    } catch(e) {
      console.error("[Verito]", e);
      this.setBar("error", { msg: "Search failed" });
    }
  }

  moveSel(items, d) {
    if (!items.length) return;
    items[this.selIdx]?.classList.remove("ss-result--selected");
    this.selIdx = ((this.selIdx + d) % items.length + items.length) % items.length;
    items[this.selIdx].classList.add("ss-result--selected");
    items[this.selIdx].scrollIntoView({ block: "nearest" });
  }

  renderResults(res, showScore, showChunk, prevLen) {
    this.selIdx = -1;

    if (!res.length) {
      this.$results.innerHTML = `<div class="ss-empty">No results above the score threshold.</div>`;
      return;
    }

    const words = this.query.toLowerCase().split(/\s+/).filter(w => w.length > 2)
                    .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re    = words.length ? new RegExp(`(${words.join("|")})`, "gi") : null;
    const hl    = text => re ? text.replace(re, "<mark class='ss-mark'>$1</mark>") : text;

    this.$results.innerHTML = res.map((r, i) => {
      const pct = Math.round(r.score * 100);
      const sc  = pct >= 80 ? "ss-score--green" : pct >= 60 ? "ss-score--amber" : "ss-score--red";
      const pre = r.chunk_text.slice(0, prevLen) + (r.chunk_text.length > prevLen ? "…" : "");
      return `
        <div class="ss-result${i === 0 ? " ss-result--top" : ""}" data-path="${r.file_path}">
          <div class="ss-result-header">
            <div class="ss-result-meta">
              <svg class="ss-file-icon" viewBox="0 0 13 16" fill="none" stroke="currentColor" stroke-width="1.2">
                <path d="M2 1h7l3 3v11H2z"/><path d="M9 1v3h3"/>
              </svg>
              <span class="ss-result-name">${r.file_name}</span>
            </div>
            ${showScore ? `<span class="ss-score ${sc}">${pct}%</span>` : ""}
          </div>
          ${r.heading_path ? `<div class="ss-breadcrumb">${r.heading_path}</div>` : ""}
          <div class="ss-preview">${hl(pre)}</div>
          ${showChunk && r.chunk_total > 1
            ? `<span class="ss-chunk-badge">chunk ${r.chunk_index + 1} of ${r.chunk_total}</span>`
            : ""}
        </div>`;
    }).join("");

    this.$results.querySelectorAll(".ss-result").forEach(card => {
      card.addEventListener("click", () => {
        const base = this.app.vault.adapter.basePath;
        const fp   = card.dataset.path ?? "";
        const rel  = fp.startsWith(base) ? fp.slice(base.length).replace(/^[\\/]/, "") : fp;
        this.app.workspace.openLinkText(rel, "", false);
        this.close();
      });
    });
  }
}

//Settings

class VeritoSettingTab extends PluginSettingTab {
  constructor(app, plugin) { super(app, plugin); this.plugin = plugin; }

  async display() {
    const { containerEl: el } = this;
    el.empty();
    const s    = this.plugin.settings;
    const save = () => this.plugin.saveSettings();

    el.createEl("h3", { text: "Backend" });
    new Setting(el).setName("Backend URL")
      .addText(t => t.setValue(s.backendUrl).onChange(v => { s.backendUrl = v; save(); }));
    new Setting(el).setName("Auto-launch binary")
      .addToggle(t => t.setValue(s.autoLaunchBinary).onChange(v => { s.autoLaunchBinary = v; save(); }));
    new Setting(el).setName("Backend status")
      .addButton(b => b.setButtonText("Ping").onClick(async () =>
        new Notice(await this.plugin.healthy() ? "Running" : "Unreachable")));

    el.createEl("h3", { text: "Embedding model" });
    const modelRow = new Setting(el).setName("Model");
    let md = { models: [], selected: null, ollama_reachable: false };
    try { md = await this.plugin.getModels(); } catch {}
    modelRow.addDropdown(dd => {
      if (!md.ollama_reachable) {
        dd.addOption("", "Ollama not running"); dd.setDisabled(true);
      } else if (!md.models.length) {
        dd.addOption("", "No models — run: ollama pull nomic-embed-text"); dd.setDisabled(true);
      } else {
        dd.addOption("", "Select a model…");
        md.models.forEach(m => dd.addOption(m, m));
        if (md.selected) dd.setValue(md.selected);
        dd.onChange(async v => {
          if (!v) return;
          try {
            await this.plugin.patchConfig({ embedding_model: v });
            this.plugin.applyStatus(await this.plugin.getStatus());
            new Notice(`Model set to ${v}`);
          } catch(e) {
            new Notice(e?.status === 404 ? `Run: ollama pull ${v}` : `Failed: ${e}`);
          }
        });
      }
    });

    el.createEl("h3", { text: "Search" });
    new Setting(el).setName("Max results")
      .addSlider(sl => sl.setLimits(1,50,1).setValue(s.maxResults).setDynamicTooltip()
        .onChange(v => { s.maxResults = v; save(); }));
    new Setting(el).setName("Min score (%)")
      .addSlider(sl => sl.setLimits(0,100,5).setValue(Math.round(s.minScore*100)).setDynamicTooltip()
        .onChange(v => { s.minScore = v/100; save(); }));
    new Setting(el).setName("Debounce (ms)")
      .addText(t => t.setValue(String(s.debounceMs))
        .onChange(v => { const n = parseInt(v); if (!isNaN(n)) { s.debounceMs = n; save(); } }));
    new Setting(el).setName("Preview length")
      .addSlider(sl => sl.setLimits(80,500,20).setValue(s.previewLength).setDynamicTooltip()
        .onChange(v => { s.previewLength = v; save(); }));
    new Setting(el).setName("Show score badge")
      .addToggle(t => t.setValue(s.showScorePill).onChange(v => { s.showScorePill = v; save(); }));
    new Setting(el).setName("Show chunk position")
      .addToggle(t => t.setValue(s.showChunkBadge).onChange(v => { s.showChunkBadge = v; save(); }));

    el.createEl("h3", { text: "Indexing" });
    new Setting(el).setName("Watch for changes")
      .addToggle(t => t.setValue(s.watchForChanges).onChange(v => { s.watchForChanges = v; save(); }));
    new Setting(el).setName("Exclude patterns")
      .addText(t => t.setValue(s.excludePatterns).onChange(v => { s.excludePatterns = v; save(); }));
    new Setting(el).setName("Index vault")
      .addButton(b => b.setButtonText("Index now").setCta().onClick(() => this.plugin.index()));
  }
}

module.exports = VeritoPlugin;