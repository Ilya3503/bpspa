
# ==============================================================================
# КЛАСТЕРИЗАЦИЯ
# ==============================================================================

def cluster_dbscan(pcd, eps=0.025, min_points=50,
                   min_extent=0.02, max_extent=0.30) -> List[o3d.geometry.PointCloud]:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return []
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters = []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        cluster = pcd.select_by_index(idx.tolist())
        cluster_pts = np.asarray(cluster.points)
        if cluster_pts[:, 2].max() - cluster_pts[:, 2].min() < 0.003:
            continue
        extent = cluster.get_axis_aligned_bounding_box().get_extent()
        max_dim = float(np.max(extent))
        if max_dim < min_extent or max_dim > max_extent:
            continue
        clusters.append(cluster)
    return clusters


def cluster_info(cluster: o3d.geometry.PointCloud, cluster_id: int) -> dict:
    try:
        obb = cluster.get_oriented_bounding_box()
        center = list(map(float, obb.center))
        extent = list(map(float, obb.extent))
        R = np.asarray(obb.R)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    except Exception:
        aabb = cluster.get_axis_aligned_bounding_box()
        center = list(map(float, aabb.get_center()))
        extent = list(map(float, aabb.get_extent()))
        R = np.eye(3)
        yaw = 0.0
    return {
        "id": cluster_id,
        "center": center,
        "extent": extent,
        "yaw": yaw,
        "rotation_matrix": R.tolist(),
        "points_count": int(len(cluster.points)),
    }
