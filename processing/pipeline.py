"""
Фасад обработки одного облака: preprocessing → detection → pose → сохранение.
Не знает про камеру, состояния и WebSocket. Принимает путь к облаку и конфиг,
возвращает result-dict. События в UI прокидываются колбэком emit.
"""
import logging
from pathlib import Path

import numpy as np

from processing import preprocessing as pre
from processing import detection as det
from processing import pose as pose_est
from processing import io_results as io

log = logging.getLogger(__name__)


def _noop(_event: dict) -> None:
    pass


def _load_cad(cfg: dict):
    """Грузит CAD-облако по имени из конфига → (cad_model, cad_name).
    cad_model=None, если CAD не задан/не найден — тогда поза вернёт failed."""
    cad_name = cfg.get("icp", {}).get("cad_file")
    if not cad_name:
        return None, None
    cad_path = Path("cad_models") / cad_name
    if not cad_path.exists():
        log.warning(f"CAD не найден: {cad_path}")
        return None, cad_name
    return pre.load_pcd(str(cad_path)), cad_name


def _viz_downsample(arr, max_n: int = 5000):
    """Прореживание точек только для отправки в UI (не для вычислений)."""
    if len(arr) > max_n:
        return arr[::len(arr) // max_n][:max_n]
    return arr


def _estimate_candidate(cluster, cad_model, cad_name, cfg, cluster_id, emit) -> dict:
    """Оценка позы одного кандидата (FPFH+RANSAC → ICP) + события в UI.
    Нет CAD / global выключен / ошибка → честный failed (не OBB-костыль)."""
    emit({"event": "pose_estimation_start", "cluster_id": cluster_id,
          "cad_model": cad_name if cad_model is not None else None})

    # выбор пути — единственный: FPFH+RANSAC→ICP
    if cad_model is None:
        pose = pose_est._failed_result(reason="no CAD model")
    elif not cfg.get("global_registration", {}).get("enabled", False):
        pose = pose_est._failed_result(reason="global_registration disabled")
    else:
        try:
            pose = pose_est.run_global_then_icp(
                cluster, cad_model, cad_name,
                global_cfg=cfg["global_registration"], icp_cfg=cfg["icp"])
        except Exception as e:
            log.warning(f"pose failed for cluster {cluster_id}: {e}")
            pose = pose_est._failed_result(reason=f"exception: {e}")

    if pose.get("status") == "failed":
        emit({"event": "pose_failed", "cluster_id": cluster_id,
              "reason": pose.get("reason")})
        return pose

    emit({"event": "pose_estimated", "cluster_id": cluster_id,
          "method": pose["method"], "fitness": pose.get("fitness"),
          "inlier_rmse": pose.get("inlier_rmse"),
          "position": pose["position"], "orientation": pose["orientation"],
          "global_fitness": pose.get("global_fitness"),
          "global_rmse": pose.get("global_rmse")})

    if pose.get("cad_points_transformed") is not None:
        emit({"event": "icp_visualization", "cluster_id": cluster_id,
              "cluster_points": _viz_downsample(np.asarray(cluster.points)).tolist(),
              "cad_points": _viz_downsample(pose["cad_points_transformed"]).tolist(),
              "cad_model_name": cad_name})

    return pose


def run_pipeline(input_file: str, cfg: dict, run_dir: str, emit=_noop) -> dict:
    """Полный прогон по одному облаку. Возвращает result-dict."""
    pre_cfg = cfg["preprocessing"]

    # --- слой A: preprocessing ---
    pcd = pre.clean_nan(pre.load_pcd(input_file))
    n0 = len(pcd.points)

    roi = pre_cfg["roi"]
    pcd = pre.crop_roi(pcd, roi["x"], roi["y"], roi["z"])
    emit({"event": "processing_step", "step": "crop_roi",
          "points_before": n0, "points_after": len(pcd.points)})
    if len(pcd.points) == 0:
        return {"status": "empty", "num_clusters": 0, "clusters": []}

    n = len(pcd.points)
    pcd = pre.voxel_downsample(pcd, pre_cfg["voxel_size"])
    emit({"event": "processing_step", "step": "voxel_downsample",
          "points_before": n, "points_after": len(pcd.points)})

    n = len(pcd.points)
    pcd = pre.statistical_filter(pcd, pre_cfg["nb_neighbors"], pre_cfg["std_ratio"])
    emit({"event": "processing_step", "step": "statistical_filter",
          "points_before": n, "points_after": len(pcd.points)})

    plane_model = None
    if cfg["plane_removal"].get("enabled", True):
        n = len(pcd.points)
        pcd, plane_model = pre.remove_plane(
            pcd,
            distance_threshold=cfg["plane_removal"]["distance_threshold"],
            ransac_n=cfg["plane_removal"]["ransac_n"],
            num_iterations=cfg["plane_removal"]["num_iterations"])
        emit({"event": "processing_step", "step": "ransac_plane",
              "points_before": n, "points_after": len(pcd.points)})

    # --- слой B: detection (кандидаты) ---
    db = cfg["dbscan"]
    clusters = det.cluster_dbscan(pcd, eps=db["eps"], min_points=db["min_points"],
                                  min_extent=db["min_extent"], max_extent=db["max_extent"])
    emit({"event": "clusters_found", "num_clusters": len(clusters),
          "clusters": [{"id": i, "points": len(c.points)} for i, c in enumerate(clusters)]})

    Path(run_dir, "clusters").mkdir(parents=True, exist_ok=True)
    io.save_clusters(clusters, str(Path(run_dir, "clusters")))

    # --- слой C: pose (по каждому кандидату) ---
    cad_model, cad_name = _load_cad(cfg)
    clusters_info = []
    for i, cluster in enumerate(clusters):
        info = det.cluster_info(cluster, i)
        pose = _estimate_candidate(cluster, cad_model, cad_name, cfg, i, emit)
        info["pose"] = {k: v for k, v in pose.items() if k != "cad_points_transformed"}
        clusters_info.append(info)

    # --- слой D: сохранение ---
    result = {"status": "ok", "input_file": input_file,
              "num_clusters": len(clusters), "clusters": clusters_info,
              "plane_model": plane_model}
    if clusters:
        result["annotated_ply"] = io.make_annotated_ply(pcd, clusters, run_dir)
    io.save_position_json(result, run_dir)
    return result