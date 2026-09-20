/* Браузерное конфиг-приложение (main).
   Схема — маленькая константа: какие поля показываем и как их крутить.
   Всё, чего нет в схеме, всё равно уходит на сервер в /config/apply как есть
   (редактируем копию эффективного конфига, схема лишь делает часть полей крутибельной). */

// SCHEMA: секция -> список полей. Поддерживаемые типы:
//   range  {path, label, min, max, step, hint?}   -> слайдер
//   number {path, label, step?, hint?}            -> числовое поле
//   toggle {path, label, hint?}                   -> вкл/выкл
//   range2 {path, label, min, max, step, hint?}   -> пара min/max для [a, b]
// path — путь внутри секции (напр. "roi.x" -> cfg[section].roi.x).
const SCHEMA = {
  preprocessing: [
    { type: "range2", path: "roi.x", label: "ROI X (м)", min: -1.0, max: 1.0, step: 0.01 },
    { type: "range2", path: "roi.y", label: "ROI Y (м)", min: -1.0, max: 1.0, step: 0.01 },
    { type: "range2", path: "roi.z", label: "ROI Z (м)", min: 0.0, max: 1.5, step: 0.01 },
    { type: "range",  path: "voxel_size", label: "Voxel size (м)", min: 0.001, max: 0.02, step: 0.0005 },
    { type: "range",  path: "nb_neighbors", label: "SOR: соседей", min: 5, max: 60, step: 1 },
    { type: "range",  path: "std_ratio", label: "SOR: std ratio", min: 0.5, max: 4.0, step: 0.1 },
  ],
  plane_removal: [
    { type: "toggle", path: "enabled", label: "Удалять плоскость" },
    { type: "range",  path: "distance_threshold", label: "Порог расстояния (м)", min: 0.001, max: 0.03, step: 0.001 },
    { type: "range",  path: "num_iterations", label: "RANSAC итераций", min: 100, max: 3000, step: 100 },
  ],
  dbscan: [
    { type: "range", path: "eps", label: "eps (м)", min: 0.005, max: 0.1, step: 0.001 },
    { type: "range", path: "min_points", label: "min_points", min: 5, max: 200, step: 1 },
    { type: "range", path: "min_extent", label: "min_extent (м)", min: 0.005, max: 0.1, step: 0.005 },
    { type: "range", path: "max_extent", label: "max_extent (м)", min: 0.05, max: 0.6, step: 0.01 },
  ],
  global_registration: [
    { type: "toggle", path: "enabled", label: "Global registration" },
    { type: "range",  path: "voxel_size", label: "Voxel size (м)", min: 0.001, max: 0.02, step: 0.0005 },
    { type: "range",  path: "min_fitness", label: "min_fitness", min: 0.0, max: 1.0, step: 0.01 },
  ],
  icp: [
    { type: "range", path: "voxel_size", label: "Voxel size (м)", min: 0.001, max: 0.02, step: 0.0005 },
    { type: "range", path: "max_correspondence_distance", label: "Max corr. dist (м)", min: 0.001, max: 0.05, step: 0.001 },
    { type: "range", path: "fitness_threshold", label: "fitness_threshold", min: 0.0, max: 1.0, step: 0.01 },
  ],
};

let CFG = null; // рабочая копия эффективного конфига

// ---- утилиты пути ----
function getPath(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setPath(obj, path, val) {
  const keys = path.split(".");
  let o = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    if (o[keys[i]] == null || typeof o[keys[i]] !== "object") o[keys[i]] = {};
    o = o[keys[i]];
  }
  o[keys[keys.length - 1]] = val;
}

// ---- toast ----
let toastTimer = null;
function toast(msg, isErr) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = "toast"; }, 2200);
}

// ---- рендер одного поля ----
function fieldRow(section, f) {
  const cur = getPath(CFG[section], f.path);
  const row = document.createElement("div");
  row.className = "row";

  const lab = document.createElement("label");
  lab.innerHTML = f.label + (f.hint ? `<span class="hint">${f.hint}</span>` : "");
  row.appendChild(lab);

  if (f.type === "toggle") {
    const mid = document.createElement("div");
    const tg = document.createElement("div");
    tg.className = "toggle" + (cur ? " on" : "");
    tg.onclick = () => {
      const nv = !tg.classList.contains("on");
      tg.classList.toggle("on", nv);
      setPath(CFG[section], f.path, nv);
    };
    mid.appendChild(tg);
    row.appendChild(mid);
    row.appendChild(document.createElement("span"));
    return row;
  }

  if (f.type === "number") {
    const inp = document.createElement("input");
    inp.type = "number"; inp.step = f.step ?? "any"; inp.value = cur;
    inp.oninput = () => setPath(CFG[section], f.path, parseFloat(inp.value));
    row.appendChild(inp);
    row.appendChild(document.createElement("span"));
    return row;
  }

  if (f.type === "range2") {
    const arr = Array.isArray(cur) ? cur.slice() : [f.min, f.max];
    const wrap = document.createElement("div");
    wrap.className = "pair";
    const mk = (idx) => {
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = f.step; inp.value = arr[idx];
      inp.min = f.min; inp.max = f.max;
      inp.oninput = () => {
        arr[idx] = parseFloat(inp.value);
        setPath(CFG[section], f.path, arr.slice());
      };
      return inp;
    };
    wrap.appendChild(mk(0));
    const dash = document.createElement("span"); dash.textContent = "—";
    wrap.appendChild(dash);
    wrap.appendChild(mk(1));
    row.appendChild(wrap);
    row.appendChild(document.createElement("span"));
    return row;
  }

  // range (default)
  const rng = document.createElement("input");
  rng.type = "range"; rng.min = f.min; rng.max = f.max; rng.step = f.step;
  rng.value = cur;
  const val = document.createElement("span");
  val.className = "val";
  const fmt = (v) => (f.step < 1 ? Number(v).toFixed(String(f.step).split(".")[1]?.length || 3) : String(v));
  val.textContent = fmt(cur);
  rng.oninput = () => {
    const nv = parseFloat(rng.value);
    val.textContent = fmt(nv);
    setPath(CFG[section], f.path, nv);
  };
  row.appendChild(rng);
  row.appendChild(val);
  return row;
}

// ---- рендер всех секций ----
function render() {
  const root = document.getElementById("root");
  root.innerHTML = "";
  for (const [section, fields] of Object.entries(SCHEMA)) {
    if (CFG[section] == null) continue; // секции нет в конфиге — пропускаем
    const sec = document.createElement("div");
    sec.className = "section";
    const h = document.createElement("h2");
    h.textContent = section;
    sec.appendChild(h);
    const rows = document.createElement("div");
    rows.className = "rows";
    for (const f of fields) rows.appendChild(fieldRow(section, f));
    sec.appendChild(rows);
    root.appendChild(sec);
  }
}

// ---- сеть ----
async function loadCfg() {
  const r = await fetch("/config/effective");
  if (!r.ok) throw new Error("не удалось загрузить конфиг");
  CFG = await r.json();
  render();
  document.getElementById("status").textContent = "эффективный конфиг загружен";
}

async function apply() {
  const btn = document.getElementById("btn-apply");
  btn.disabled = true;
  try {
    const r = await fetch("/config/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: CFG }),
    });
    if (!r.ok) throw new Error(await r.text());
    toast("Применено — записано в local");
  } catch (e) {
    toast("Ошибка: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function resetDefault() {
  if (!confirm("Удалить local и вернуться к config.default.yaml?")) return;
  const btn = document.getElementById("btn-reset");
  btn.disabled = true;
  try {
    const r = await fetch("/config/reset", { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    CFG = data.config;
    render();
    toast("Сброшено к default");
  } catch (e) {
    toast("Ошибка: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function snapshot() {
  const note = document.getElementById("note").value || "";
  const btn = document.getElementById("btn-snap");
  btn.disabled = true;
  try {
    const r = await fetch("/config/snapshot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: CFG, note }),
    });
    if (!r.ok) throw new Error(await r.text());
    const blob = await r.blob();
    const cd = r.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="?([^"]+)"?/);
    const fname = m ? m[1] : "config_snapshot.yaml";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = fname;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast("Снимок скачан: " + fname);
  } catch (e) {
    toast("Ошибка: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("btn-apply").onclick = apply;
document.getElementById("btn-reset").onclick = resetDefault;
document.getElementById("btn-snap").onclick = snapshot;

loadCfg().catch((e) => {
  document.getElementById("status").textContent = "ошибка: " + e.message;
  toast(e.message, true);
});
