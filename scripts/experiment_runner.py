"""
scripts/experiment_runner.py
============================
Batch-прогон perception pipeline по сетке параметров и/или по набору сцен.

Назначение: собрать данные для исследований по гранту.
  • Исследование №1: как метрики зависят от параметров фильтрации
      → один файл сцены, много комбинаций параметров
  • Исследование №2: как метрики зависят от условий съёмки
      → много файлов сцен, одна (или несколько) комбинаций параметров

Запуск из корня проекта:

    # Быстрый прогон одной сцены (~30 комбинаций)
    python scripts/experiment_runner.py --scenes data/merged_2026-05-11.ply --grid quick

    # Полный прогон одной сцены (~250 комбинаций)
    python scripts/experiment_runner.py --scenes data/merged_2026-05-11.ply --grid full

    # Несколько сцен из папки, baseline-параметры (исследование №2)
    python scripts/experiment_runner.py --scenes data/scenes/*.ply --grid baseline

    # Свои параметры в JSON-файле
    python scripts/experiment_runner.py --scenes data/scene.ply --grid scripts/grids/my_grid.json

    # С CAD-моделью для ICP
    python scripts/experiment_runner.py --scenes data/scene.ply --grid quick --cad Cube_30х30х30.ply

Выход: одна папка results/experiments/<timestamp>/ с файлами:
  • results.csv   — все прогоны, по одному на строку. Готово для pandas.read_csv()
  • config.json   — параметры запуска (для воспроизводимости)
  • errors.log    — полные traceback ошибок, если были
  • summary.json  — агрегированная статистика

Прерывание через Ctrl+C сохраняет всё что успели накопить.
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# --- путь к корню проекта ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from processing import pipeline as pl  # noqa: E402

log = logging.getLogger("runner")


# ==============================================================================
# СЕТКИ ПАРАМЕТРОВ
# ==============================================================================
# Базовая конфигурация — те же значения, что в config.yaml.
# Любая сетка переопределяет только указанные ключи; остальные берутся отсюда.
# Один источник истины — никакого дублирования между runner-ом и продакшеном.

BASELINE: dict[str, Any] = {
    # ROI (метры)
    "roi_x": [-0.5, 0.5],
    "roi_y": [-0.25, 0.25],
    "roi_z": [0.50, 0.75],

    # предобработка
    "voxel_size": 0.005,
    "nb_neighbors": 20,
    "std_ratio": 2.0,

    # удаление плоскости
    "remove_plane": True,
    "plane_distance_threshold": 0.01,
    "plane_ransac_n": 3,
    "plane_num_iterations": 1000,

    # кластеризация
    "eps": 0.025,
    "min_points": 50,
    "min_extent": 0.02,
    "max_extent": 0.30,

    # ICP
    "icp_voxel_size": 0.003,
    "icp_max_correspondence_distance": 0.015,
    "icp_max_iterations": 50,
    "icp_fitness_threshold": 0.24,
}

# Предопределённые сетки. Каждая указывает только те ключи, что варьируются.
GRIDS: dict[str, dict[str, list]] = {
    # один прогон с базовыми значениями (для исследования №2)
    "baseline": {},

    # быстро: ключевые параметры pipeline
    "quick": {
        "voxel_size": [0.003, 0.005, 0.01],
        "eps":        [0.02, 0.025, 0.03],
        "min_points": [30, 50],
    },

    # полно: систематическое исследование
    "full": {
        "voxel_size":              [0.003, 0.005, 0.008, 0.01, 0.015],
        "eps":                     [0.015, 0.02, 0.025, 0.03, 0.04],
        "min_points":              [20, 30, 50],
        "plane_distance_threshold":[0.005, 0.01, 0.02],
        "std_ratio":               [1.5, 2.0, 2.5],
    },

    # для исследования влияния ICP-параметров отдельно
    "icp_only": {
        "icp_voxel_size":                  [0.002, 0.003, 0.005, 0.008],
        "icp_max_correspondence_distance": [0.005, 0.01, 0.015, 0.02, 0.03],
        "icp_max_iterations":              [20, 50, 100],
    },
}


# ==============================================================================
# КОЛОНКИ CSV
# ==============================================================================
# Плоская структура: одна строка = один эксперимент. Удобно для pandas.

CSV_COLUMNS = [
    # идентификация
    "run_idx",
    "scene_file",
    "cad_file",

    # параметры pipeline (все, что в BASELINE + ROI как числа)
    "roi_x_min", "roi_x_max", "roi_y_min", "roi_y_max", "roi_z_min", "roi_z_max",
    "voxel_size", "nb_neighbors", "std_ratio",
    "remove_plane", "plane_distance_threshold", "plane_ransac_n", "plane_num_iterations",
    "eps", "min_points", "min_extent", "max_extent",
    "icp_voxel_size", "icp_max_correspondence_distance",
    "icp_max_iterations", "icp_fitness_threshold",

    # метрики pipeline — счётчики точек
    "points_raw",
    "points_after_clean",
    "points_after_roi",
    "points_after_voxel",
    "points_after_noise",
    "points_after_plane",

    # кластеры
    "n_clusters",
    "best_cluster_id",
    "best_cluster_points",
    "best_cluster_extent_x",
    "best_cluster_extent_y",
    "best_cluster_extent_z",

    # ICP / поза
    "pose_method",         # icp | obb_fallback
    "icp_fitness",
    "icp_rmse",
    "pose_x", "pose_y", "pose_z",
    "quat_x", "quat_y", "quat_z", "quat_w",

    # тайминги (мс)
    "t_load_ms",
    "t_clean_ms",
    "t_roi_ms",
    "t_voxel_ms",
    "t_noise_ms",
    "t_plane_ms",
    "t_dbscan_ms",
    "t_icp_ms",
    "t_total_ms",

    # статус
    "status",   # ok | empty_after_roi | empty_after_noise | no_clusters | error
    "error_short",
]


# ==============================================================================
# ОДИН ЭКСПЕРИМЕНТ
# ==============================================================================

class _StepTimer:
    """Простой контекстный менеджер для замера времени шага в мс."""
    def __init__(self):
        self.ms = 0.0
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self._t) * 1000.0


def run_one(
    scene_file: str,
    params: dict,
    cad_pcd,   # уже загруженный CAD или None
    cad_name: str,
    run_idx: int,
) -> dict:
    """
    Один прогон pipeline над одной сценой с одним набором параметров.
    Возвращает плоский dict со всеми колонками CSV_COLUMNS.
    """
    row: dict = {k: None for k in CSV_COLUMNS}
    row["run_idx"] = run_idx
    row["scene_file"] = Path(scene_file).name
    row["cad_file"] = cad_name

    # параметры → в строку (плоские поля)
    row["roi_x_min"], row["roi_x_max"] = params["roi_x"]
    row["roi_y_min"], row["roi_y_max"] = params["roi_y"]
    row["roi_z_min"], row["roi_z_max"] = params["roi_z"]
    for k in ("voxel_size", "nb_neighbors", "std_ratio",
              "remove_plane", "plane_distance_threshold", "plane_ransac_n", "plane_num_iterations",
              "eps", "min_points", "min_extent", "max_extent",
              "icp_voxel_size", "icp_max_correspondence_distance",
              "icp_max_iterations", "icp_fitness_threshold"):
        row[k] = params[k]

    t_total = time.perf_counter()

    try:
        # ── load ──
        with _StepTimer() as t:
            pcd = pl.load_pcd(scene_file)
        row["t_load_ms"] = round(t.ms, 2)
        row["points_raw"] = len(pcd.points)

        # ── clean NaN ──
        with _StepTimer() as t:
            pcd = pl.clean_nan(pcd)
        row["t_clean_ms"] = round(t.ms, 2)
        row["points_after_clean"] = len(pcd.points)

        # ── ROI ──
        with _StepTimer() as t:
            pcd = pl.crop_roi(pcd, params["roi_x"], params["roi_y"], params["roi_z"])
        row["t_roi_ms"] = round(t.ms, 2)
        row["points_after_roi"] = len(pcd.points)
        if len(pcd.points) == 0:
            row["status"] = "empty_after_roi"
            row["t_total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
            return row

        # ── voxel ──
        with _StepTimer() as t:
            pcd = pl.voxel_downsample(pcd, params["voxel_size"])
        row["t_voxel_ms"] = round(t.ms, 2)
        row["points_after_voxel"] = len(pcd.points)

        # ── statistical filter ──
        with _StepTimer() as t:
            pcd = pl.statistical_filter(pcd, params["nb_neighbors"], params["std_ratio"])
        row["t_noise_ms"] = round(t.ms, 2)
        row["points_after_noise"] = len(pcd.points)
        if len(pcd.points) == 0:
            row["status"] = "empty_after_noise"
            row["t_total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
            return row

        # ── plane removal ──
        if params["remove_plane"]:
            with _StepTimer() as t:
                pcd, _ = pl.remove_plane(
                    pcd,
                    distance_threshold=params["plane_distance_threshold"],
                    ransac_n=params["plane_ransac_n"],
                    num_iterations=params["plane_num_iterations"],
                )
            row["t_plane_ms"] = round(t.ms, 2)
        else:
            row["t_plane_ms"] = 0.0
        row["points_after_plane"] = len(pcd.points)

        # ── DBSCAN ──
        with _StepTimer() as t:
            clusters = pl.cluster_dbscan(
                pcd,
                eps=params["eps"],
                min_points=params["min_points"],
                min_extent=params["min_extent"],
                max_extent=params["max_extent"],
            )
        row["t_dbscan_ms"] = round(t.ms, 2)
        row["n_clusters"] = len(clusters)

        if not clusters:
            row["status"] = "no_clusters"
            row["t_total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
            return row

        # ── pose estimation для каждого кластера, выбираем лучший по fitness ──
        with _StepTimer() as t:
            best_idx = -1
            best_pose = None
            best_cluster = None
            best_fitness = -1.0

            for i, cluster in enumerate(clusters):
                if cad_pcd is not None:
                    pose = pl.run_icp(
                        cluster, cad_pcd,
                        voxel_size=params["icp_voxel_size"],
                        max_correspondence_distance=params["icp_max_correspondence_distance"],
                        max_iterations=params["icp_max_iterations"],
                        fitness_threshold=params["icp_fitness_threshold"],
                    )
                else:
                    pose = pl._obb_fallback(cluster, reason="no CAD")
                fit = pose.get("fitness") or 0.0
                if fit > best_fitness:
                    best_fitness = fit
                    best_idx = i
                    best_pose = pose
                    best_cluster = cluster
        row["t_icp_ms"] = round(t.ms, 2)

        # ── метрики лучшего кластера ──
        info = pl.cluster_info(best_cluster, best_idx)
        row["best_cluster_id"] = best_idx
        row["best_cluster_points"] = info["points_count"]
        ext = info["extent"]
        row["best_cluster_extent_x"] = round(float(ext[0]), 5)
        row["best_cluster_extent_y"] = round(float(ext[1]), 5)
        row["best_cluster_extent_z"] = round(float(ext[2]), 5)

        row["pose_method"] = best_pose["method"]
        if best_pose.get("fitness") is not None:
            row["icp_fitness"] = round(float(best_pose["fitness"]), 5)
        if best_pose.get("inlier_rmse") is not None:
            row["icp_rmse"] = round(float(best_pose["inlier_rmse"]), 6)

        pos = best_pose["position"]
        row["pose_x"], row["pose_y"], row["pose_z"] = (round(float(v), 5) for v in pos)
        q = best_pose["orientation"]
        row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"] = (round(float(v), 5) for v in q)

        row["status"] = "ok"

    except Exception as e:
        row["status"] = "error"
        row["error_short"] = (type(e).__name__ + ": " + str(e))[:200]
        # подробный traceback пишется снаружи в errors.log

    row["t_total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
    return row


# ==============================================================================
# СЕТКА → СПИСОК ПАРАМЕТРОВ
# ==============================================================================

def expand_grid(grid: dict[str, list]) -> list[dict]:
    """
    Декартово произведение значений из grid, наложенное на BASELINE.
    Пустая сетка → один прогон с baseline.
    """
    if not grid:
        return [dict(BASELINE)]

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        p = dict(BASELINE)
        p.update(dict(zip(keys, combo)))
        out.append(p)
    return out


def load_grid(name_or_path: str) -> dict[str, list]:
    """
    Загружает сетку: либо по имени из GRIDS, либо как JSON-файл.
    Валидирует, что все ключи известны и значения — списки.
    """
    if name_or_path in GRIDS:
        return GRIDS[name_or_path]

    p = Path(name_or_path)
    if p.exists() and p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Сетка в {p} должна быть JSON-объектом")
        unknown = set(data) - set(BASELINE)
        if unknown:
            raise ValueError(f"Неизвестные ключи в сетке: {unknown}")
        for k, v in data.items():
            if not isinstance(v, list):
                raise ValueError(f"Значение '{k}' должно быть списком, получено {type(v).__name__}")
        return data

    raise ValueError(
        f"Сетка '{name_or_path}' не найдена. "
        f"Доступные имена: {list(GRIDS)}. Или укажите путь к .json"
    )


# ==============================================================================
# ВВОД/ВЫВОД
# ==============================================================================

def expand_scenes(patterns: list[str]) -> list[Path]:
    """Разворачивает glob-паттерны в конкретные файлы. Сохраняет порядок и удаляет дубли."""
    seen = set()
    out = []
    for pat in patterns:
        # поддерживаем как glob, так и конкретный файл
        matches = sorted(Path(p) for p in glob.glob(pat))
        if not matches and Path(pat).exists():
            matches = [Path(pat)]
        for m in matches:
            if m not in seen and m.is_file():
                seen.add(m)
                out.append(m)
    return out


def resolve_cad(cad_arg: Optional[str]) -> tuple[Optional[Any], str]:
    """
    Возвращает (loaded_pcd_or_None, display_name).
    Грузит CAD один раз перед всеми прогонами (он не меняется).
    """
    if not cad_arg:
        return None, ""
    p = Path(cad_arg)
    if not p.exists():
        # пробуем в cad_models/
        alt = ROOT / "cad_models" / p.name
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"CAD не найден: {cad_arg}")
    pcd = pl.load_pcd(str(p))
    log.info(f"CAD загружен: {p.name} ({len(pcd.points)} точек)")
    return pcd, p.name


# ==============================================================================
# СБОРКА И ЗАПУСК
# ==============================================================================

def summarize(rows: list[dict]) -> dict:
    """Краткая статистика по результатам — для summary.json."""
    if not rows:
        return {}
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1

    ok = [r for r in rows if r["status"] == "ok"]
    fitnesses = [r["icp_fitness"] for r in ok if r["icp_fitness"] is not None]
    times = [r["t_total_ms"] for r in rows if r["t_total_ms"] is not None]

    out: dict[str, Any] = {
        "total_runs": len(rows),
        "by_status": statuses,
        "total_time_sec": round(sum(times) / 1000.0, 1) if times else 0.0,
        "mean_time_ms":  round(sum(times) / len(times), 1) if times else 0.0,
    }
    if fitnesses:
        out["fitness_min"]  = round(min(fitnesses), 4)
        out["fitness_max"]  = round(max(fitnesses), 4)
        out["fitness_mean"] = round(sum(fitnesses) / len(fitnesses), 4)
        best = max(ok, key=lambda r: r["icp_fitness"] or 0)
        out["best_run"] = {
            "run_idx": best["run_idx"],
            "scene_file": best["scene_file"],
            "icp_fitness": best["icp_fitness"],
            "icp_rmse": best["icp_rmse"],
            "voxel_size": best["voxel_size"],
            "eps": best["eps"],
            "min_points": best["min_points"],
        }
    return out


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Batch experiment runner для perception pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="Один или несколько PLY-файлов (можно glob: 'data/*.ply')")
    ap.add_argument("--grid", default="quick",
                    help=f"Имя ({', '.join(GRIDS)}) или путь к JSON-файлу. По умолчанию: quick")
    ap.add_argument("--cad", default=None,
                    help="CAD-модель (имя файла из cad_models/ или полный путь). Без неё ICP не работает.")
    ap.add_argument("--out", default=None,
                    help="Папка для результатов. По умолчанию: results/experiments/<timestamp>/")
    ap.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── входные данные ──
    scenes = expand_scenes(args.scenes)
    if not scenes:
        log.error(f"Не найдено ни одного файла по: {args.scenes}")
        sys.exit(2)

    try:
        grid = load_grid(args.grid)
    except Exception as e:
        log.error(f"Ошибка загрузки сетки: {e}")
        sys.exit(2)

    param_sets = expand_grid(grid)

    try:
        cad_pcd, cad_name = resolve_cad(args.cad)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(2)

    total = len(scenes) * len(param_sets)

    # ── выходная папка ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "results" / "experiments" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    cfg_path = out_dir / "config.json"
    err_path = out_dir / "errors.log"
    sum_path = out_dir / "summary.json"

    # сохраняем конфиг запуска для воспроизводимости
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "scenes": [str(s) for s in scenes],
            "grid_source": args.grid,
            "grid_expanded": grid,
            "param_sets_count": len(param_sets),
            "cad_file": cad_name or None,
            "baseline": BASELINE,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"Сцен: {len(scenes)}, комбинаций параметров: {len(param_sets)}, всего: {total}")
    log.info(f"Выход: {out_dir}")
    if not cad_pcd:
        log.warning("CAD не задан → ICP fitness будет пустым, поза через OBB fallback")

    # ── обработка прерывания: сохраняем что есть ──
    rows: list[dict] = []
    interrupted = False

    def _save_all():
        write_csv(csv_path, rows)
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summarize(rows), f, indent=2, ensure_ascii=False)

    def _on_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        log.warning("Прерывание получено, сохраняю результаты...")

    signal.signal(signal.SIGINT, _on_sigint)

    # ── главный цикл ──
    t_start = time.perf_counter()
    err_file = None
    run_idx = 0

    try:
        for scene in scenes:
            if interrupted:
                break
            log.info(f"=== Сцена: {scene.name} ===")
            for params in param_sets:
                if interrupted:
                    break
                run_idx += 1

                elapsed = time.perf_counter() - t_start
                eta = (elapsed / run_idx) * (total - run_idx) if run_idx else 0

                short_p = (f"vox={params['voxel_size']} eps={params['eps']} "
                           f"min_pts={params['min_points']}")
                print(f"  [{run_idx:4d}/{total}] {short_p}  ETA {eta:5.0f}s ", end="", flush=True)

                row = run_one(str(scene), params, cad_pcd, cad_name, run_idx)
                rows.append(row)

                # короткий отчёт в консоль
                if row["status"] == "ok":
                    fit = row["icp_fitness"]
                    fit_str = f"fit={fit:.3f}" if fit is not None else "no_icp"
                    print(f"✓ clusters={row['n_clusters']} pts_plane={row['points_after_plane']} "
                          f"{fit_str} t={row['t_total_ms']:.0f}ms")
                else:
                    print(f"✗ {row['status']}  {row['error_short'] or ''}")
                    # подробный traceback в errors.log только для status=error
                    if row["status"] == "error":
                        if err_file is None:
                            err_file = open(err_path, "w", encoding="utf-8")
                        err_file.write(f"\n=== run_idx={row['run_idx']} scene={row['scene_file']} ===\n")
                        err_file.write(f"params: {json.dumps(params, ensure_ascii=False)}\n")
                        err_file.write(traceback.format_exc())
                        err_file.flush()

                # промежуточное сохранение каждые 25 прогонов на случай падения
                if run_idx % 25 == 0:
                    _save_all()

    finally:
        if err_file is not None:
            err_file.close()
        _save_all()

    # ── итог ──
    elapsed = time.perf_counter() - t_start
    summary = summarize(rows)
    log.info("=" * 60)
    log.info(f"Готово за {elapsed:.1f}с. Прогонов: {len(rows)}/{total}")
    log.info(f"Статусы: {summary.get('by_status', {})}")
    if "fitness_mean" in summary:
        log.info(f"Fitness: min={summary['fitness_min']} max={summary['fitness_max']} mean={summary['fitness_mean']}")
        log.info(f"Лучший прогон: run_idx={summary['best_run']['run_idx']}")
    log.info(f"Результаты: {csv_path}")
    if err_file is not None:
        log.info(f"Ошибки: {err_path}")


if __name__ == "__main__":
    main()