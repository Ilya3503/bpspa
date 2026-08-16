// Схема: какие поля показывать и как. path — путь в конфиге через точку.
// type: range | number | bool | select. hot:false → нужен рестарт.
const SCHEMA = [
  ["Preprocessing", [
    { path:"preprocessing.voxel_size", type:"range", min:0.001, max:0.01, step:0.001 },
    { path:"preprocessing.nb_neighbors", type:"number" },
    { path:"preprocessing.std_ratio", type:"number", step:0.1 },
  ]],
  ["Plane removal", [
    { path:"plane_removal.enabled", type:"bool" },
    { path:"plane_removal.distance_threshold", type:"range", min:0.002, max:0.03, step:0.001 },
  ]],
  ["DBSCAN", [
    { path:"dbscan.eps", type:"range", min:0.005, max:0.06, step:0.001 },
    { path:"dbscan.min_points", type:"number" },
    { path:"dbscan.min_extent", type:"range", min:0.005, max:0.1, step:0.005 },
    { path:"dbscan.max_extent", type:"range", min:0.05, max:0.5, step:0.01 },
  ]],
  ["Global registration", [
    { path:"global_registration.enabled", type:"bool" },
    { path:"global_registration.min_fitness", type:"range", min:0.02, max:0.5, step:0.01 },
    { path:"global_registration.voxel_size", type:"range", min:0.001, max:0.01, step:0.001 },
  ]],
  ["ICP", [
    { path:"icp.fitness_threshold", type:"range", min:0.05, max:0.6, step:0.01 },
    { path:"icp.max_correspondence_distance", type:"range", min:0.005, max:0.05, step:0.001 },
    { path:"icp.max_iterations", type:"number" },
  ]],
];

let cfg = {};
const $ = s => document.querySelector(s);
const get = p => p.split(".").reduce((o,k)=>o?.[k], cfg);
const set = (p,v) => { const ks=p.split("."); let o=cfg; for(let i=0;i<ks.length-1;i++)o=o[ks[i]]; o[ks[ks.length-1]]=v; };

function render() {
  const form = $("#form"); form.innerHTML = "";
  for (const [section, fields] of SCHEMA) {
    form.insertAdjacentHTML("beforeend", `<h2>${section}</h2>`);
    for (const f of fields) {
      const v = get(f.path);
      if (v === undefined) continue;               // ключа нет в конфиге — пропускаем
      const row = document.createElement("div"); row.className = "row";
      const name = f.path.split(".").pop();
      let control;
      if (f.type === "bool") {
        control = `<input type="checkbox" ${v?"checked":""} data-p="${f.path}" data-t="bool">`;
      } else if (f.type === "range") {
        control = `<input type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${v}" data-p="${f.path}" data-t="num"><span class="val">${v}</span>`;
      } else {
        control = `<input type="number" step="${f.step||1}" value="${v}" data-p="${f.path}" data-t="num">`;
      }
      row.innerHTML = `<label>${name}</label>${control}`;
      form.appendChild(row);
    }
  }
  // обработчики
  form.querySelectorAll("[data-p]").forEach(el => {
    el.addEventListener("input", e => {
      const p = e.target.dataset.p, t = e.target.dataset.t;
      const val = t==="bool" ? e.target.checked : parseFloat(e.target.value);
      set(p, val);
      const span = e.target.parentElement.querySelector(".val");
      if (span) span.textContent = val;
    });
  });
}

async function load() {
  cfg = await (await fetch("/config/effective")).json();
  render();
}
$("#apply").onclick = async () => {
  await fetch("/config/apply", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({config:cfg})});
  $("#msg").textContent = "применено ✓ (следующий цикл)";
};
$("#reset").onclick = async () => {
  if (!confirm("Сбросить к default? Локальные правки удалятся.")) return;
  const r = await (await fetch("/config/reset", {method:"POST"})).json();
  cfg = r.config; render(); $("#msg").textContent = "сброшено к default";
};
$("#download").onclick = async () => {
  const note = $("#note").value;
  const r = await fetch("/config/snapshot", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({config:cfg, note})});
  const blob = await r.blob();
  const name = r.headers.get("Content-Disposition").match(/filename="(.+)"/)[1];
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
};
load();