import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// ---------- DOM ----------
const $ = id => document.getElementById(id);
const stateEl = $('state-value');
const logEl = $('log');
const videoEl = $('video-feed');
const cadSelect = $('cad-select');
const plySelect = $('ply-select');
const showCadCb = $('show-cad');
const cadOpacity = $('cad-opacity');

// ---------- Three.js ----------
const viewer = $('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0d12);

const camera = new THREE.PerspectiveCamera(45, viewer.clientWidth / viewer.clientHeight, 0.01, 100);
camera.position.set(0.5, 0.5, 1.0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.6);
controls.update();

const grid = new THREE.GridHelper(2, 20, 0x303030, 0x202020);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const axes = new THREE.AxesHelper(0.2);
scene.add(axes);

let currentPly = null;
let clusterCloud = null;
let cadCloud = null;

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = viewer.clientWidth / viewer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
});

// ---------- логирование ----------
function log(msg, level='info') {
  const div = document.createElement('div');
  div.className = 'l-' + (level === 'error' ? 'err' : level === 'ok' ? 'ok' : 'info');
  div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.appendChild(div);
  while (logEl.children.length > 100) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

// ---------- WebSocket ----------
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/events`);
  ws.onopen = () => log('WebSocket подключён', 'ok');
  ws.onclose = () => { log('WebSocket отключён, переподключение через 2с...', 'error'); setTimeout(connectWS, 2000); };
  ws.onmessage = e => handleEvent(JSON.parse(e.data));
  return ws;
}
let ws = connectWS();

function handleEvent(ev) {
  switch (ev.event) {
    case 'state_changed':
      stateEl.textContent = ev.state;
      log(`STATE → ${ev.state}`);
      break;

    case 'capture_start':
      log(`Захват view ${ev.view}...`);
      break;

    case 'capture_done':
      log(`Захвачено view ${ev.view}: ${ev.points} точек`, 'ok');
      refreshPlyList();
      break;

    case 'waiting_for_next_view':
      log(ev.message || 'Ожидание NEXT', 'ok');
      break;

    case 'merging_start':
      log('Объединение облаков...');
      break;

    case 'merging_done':
      log(`Merged: ${ev.points} точек`, 'ok');
      if (ev.calibration_is_stub) log('⚠ Используется заглушка калибровки', 'error');
      refreshPlyList();
      break;

    case 'processing_step':
      log(`${ev.step}: ${ev.points_before} → ${ev.points_after}`);
      break;

    case 'clusters_found':
      $('r-num').textContent = ev.num_clusters;
      log(`Найдено кластеров: ${ev.num_clusters}`, 'ok');
      break;

    case 'pose_estimation_start':
      log(`ICP для кластера ${ev.cluster_id} (${ev.cad_model || 'OBB'})`);
      break;

    case 'pose_estimated':
      $('r-fit').textContent = ev.fitness ? ev.fitness.toFixed(3) : 'N/A';
      $('r-pos').textContent = ev.position.map(v => v.toFixed(3)).join(', ');
      $('r-quat').textContent = ev.orientation.map(v => v.toFixed(2)).join(', ');
      log(`Поза кластера ${ev.cluster_id}: ${ev.method} fit=${ev.fitness ?? 'N/A'}`, 'ok');
      break;

    case 'icp_visualization':
      showIcpVisualization(ev);
      break;

    case 'video_frame':
      videoEl.src = ev.image_base64;
      break;

    case 'done':
      log('Цикл завершён', 'ok');
      break;

    case 'error':
      log(`ERROR: ${ev.message}`, 'error');
      break;

    case 'cad_selected':
      log(`CAD выбрана: ${ev.name}`, 'ok');
      break;
  }
}

// ---------- ICP визуализация ----------
function clearIcp() {
  if (clusterCloud) { scene.remove(clusterCloud); clusterCloud.geometry.dispose(); clusterCloud = null; }
  if (cadCloud)     { scene.remove(cadCloud);     cadCloud.geometry.dispose();     cadCloud = null; }
}

function makePointCloud(points, color, size=0.003, opacity=1.0) {
  const geom = new THREE.BufferGeometry();
  const flat = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) {
    flat[i*3+0] = points[i][0];
    flat[i*3+1] = points[i][1];
    flat[i*3+2] = points[i][2];
  }
  geom.setAttribute('position', new THREE.BufferAttribute(flat, 3));
  const mat = new THREE.PointsMaterial({
    color, size, sizeAttenuation: true,
    transparent: opacity < 1.0, opacity,
  });
  return new THREE.Points(geom, mat);
}

function showIcpVisualization(ev) {
  clearIcp();
  clusterCloud = makePointCloud(ev.cluster_points, 0xff3333, 0.003);
  scene.add(clusterCloud);

  if (showCadCb.checked) {
    cadCloud = makePointCloud(ev.cad_points, 0x3399ff, 0.003, parseFloat(cadOpacity.value));
    scene.add(cadCloud);
  }

  // центрируем камеру на bounding box кластера
  const box = new THREE.Box3().setFromObject(clusterCloud);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  camera.position.set(center.x + size, center.y + size, center.z + size);
  controls.update();
}

showCadCb.addEventListener('change', () => {
  if (cadCloud) cadCloud.visible = showCadCb.checked;
});
cadOpacity.addEventListener('input', () => {
  if (cadCloud) {
    cadCloud.material.opacity = parseFloat(cadOpacity.value);
    cadCloud.material.transparent = cadCloud.material.opacity < 1.0;
  }
});

// ---------- команды ----------
async function sendCommand(action) {
  try {
    const r = await fetch('/command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action})
    });
    if (!r.ok) {
      const t = await r.text();
      log(`Команда отклонена: ${t}`, 'error');
    }
  } catch (e) {
    log(`Ошибка: ${e}`, 'error');
  }
}

$('btn-start').onclick = () => sendCommand('start');
$('btn-next').onclick  = () => sendCommand('next_view');
$('btn-reset').onclick = () => sendCommand('reset');
$('btn-stop').onclick  = () => sendCommand('stop');

// ---------- CAD модели ----------
async function refreshCadList() {
  const r = await fetch('/cad_models');
  const j = await r.json();
  cadSelect.innerHTML = '';
  for (const name of j.models) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    if (name === j.selected) opt.selected = true;
    cadSelect.appendChild(opt);
  }
}
$('cad-apply').onclick = async () => {
  const name = cadSelect.value;
  if (!name) return;
  await fetch('/cad_models/select', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name})
  });
};
refreshCadList();

// ---------- PLY-вьювер: открыть произвольный файл ----------
async function refreshPlyList() {
  const folders = ['data', 'results'];
  plySelect.innerHTML = '';
  for (const f of folders) {
    const r = await fetch(`/files/${f}`);
    const j = await r.json();
    for (const name of j.files) {
      const opt = document.createElement('option');
      opt.value = `${f}/${name}`;
      opt.textContent = opt.value;
      plySelect.appendChild(opt);
    }
  }
}
refreshPlyList();

$('ply-load').onclick = () => {
  const path = plySelect.value;
  if (!path) return;
  const loader = new PLYLoader();
  loader.load(`/file/${path}`, geometry => {
    if (currentPly) { scene.remove(currentPly); currentPly.geometry.dispose(); }
    const hasColor = !!geometry.getAttribute('color');
    const mat = new THREE.PointsMaterial({
      size: 0.003, sizeAttenuation: true,
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x66ccff,
    });
    currentPly = new THREE.Points(geometry, mat);
    scene.add(currentPly);

    const box = new THREE.Box3().setFromObject(currentPly);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    controls.target.copy(center);
    camera.position.set(center.x + size, center.y + size, center.z + size);
    controls.update();
    log(`Загружено ${path}`, 'ok');
  }, undefined, err => log(`Ошибка PLY: ${err}`, 'error'));
};