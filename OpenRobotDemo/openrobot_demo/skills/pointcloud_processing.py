"""Point cloud processing skills: segmentation, clustering, filtering.

These skills operate on point cloud data (N×3 arrays) and produce
semantic structures (planes, clusters, bounding boxes).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class RANSACPlaneSegmentationSkill(SkillInterface):
    """Segment the dominant plane from a point cloud using RANSAC."""

    name = "ransac_plane_segmentation"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Segment the largest planar surface from a point cloud (e.g. table, floor) using RANSAC.",
            parameters=[
                ParamSchema(name="points", type="list", description="Point cloud as list of [x, y, z] or N×3 ndarray.", required=True),
                ParamSchema(name="distance_threshold", type="float", description="Max distance from plane to inlier (meters).", required=False, default=0.01),
                ParamSchema(name="max_iterations", type="int", description="RANSAC iterations.", required=False, default=100),
                ParamSchema(name="min_inliers", type="int", description="Minimum inliers to accept plane.", required=False, default=100),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether segmentation succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="plane_model", type="list", description="Plane coefficients [a, b, c, d] where ax+by+cz+d=0."),
                ResultSchema(name="inliers", type="list", description="Inlier point indices."),
                ResultSchema(name="outliers", type="list", description="Outlier point indices."),
            ],
            dependencies=[],
        )

    def execute(self, points, distance_threshold: float = 0.01,
                max_iterations: int = 100, min_inliers: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            pts = np.array(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                return {"success": False, "message": f"Expected N×3 point cloud, got shape {pts.shape}."}
            if len(pts) < min_inliers:
                return {"success": False, "message": f"Not enough points ({len(pts)} < {min_inliers})."}

            best_plane = None
            best_inliers = []

            for _ in range(max_iterations):
                # Sample 3 random points
                idx = np.random.choice(len(pts), 3, replace=False)
                p1, p2, p3 = pts[idx]

                # Compute plane normal
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-6:
                    continue
                normal = normal / norm_len
                d = -np.dot(normal, p1)

                # Count inliers
                distances = np.abs(pts @ normal + d)
                inliers = np.where(distances < distance_threshold)[0]

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers.tolist()
                    best_plane = np.concatenate([normal, [d]]).tolist()

            if len(best_inliers) < min_inliers:
                return {"success": False, "message": f"No dominant plane found (max inliers: {len(best_inliers)})."}

            all_indices = set(range(len(pts)))
            outlier_indices = list(all_indices - set(best_inliers))

            return {
                "success": True,
                "message": f"Plane found with {len(best_inliers)} inliers.",
                "plane_model": best_plane,
                "inliers": best_inliers,
                "outliers": outlier_indices,
            }
        except Exception as exc:
            logger.exception("[RANSACPlaneSegmentationSkill] Failed")
            return {"success": False, "message": str(exc)}


class EuclideanClusteringSkill(SkillInterface):
    """Cluster a point cloud using Euclidean distance (DBSCAN-like)."""

    name = "euclidean_clustering"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Cluster a point cloud into groups of nearby points using Euclidean distance thresholding.",
            parameters=[
                ParamSchema(name="points", type="list", description="Point cloud as list of [x, y, z] or N×3 ndarray.", required=True),
                ParamSchema(name="cluster_tolerance", type="float", description="Max distance between points in same cluster (meters).", required=False, default=0.02),
                ParamSchema(name="min_cluster_size", type="int", description="Minimum points per cluster.", required=False, default=10),
                ParamSchema(name="max_cluster_size", type="int", description="Maximum points per cluster.", required=False, default=100000),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether clustering succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="clusters", type="list", description="List of clusters, each is a dict with {'indices': [...], 'center': [x,y,z], 'bbox': [min, max]}"),
                ResultSchema(name="num_clusters", type="int", description="Number of clusters found."),
            ],
            dependencies=[],
        )

    def execute(self, points, cluster_tolerance: float = 0.02,
                min_cluster_size: int = 10, max_cluster_size: int = 100000, **kwargs) -> Dict[str, Any]:
        try:
            pts = np.array(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                return {"success": False, "message": f"Expected N×3 point cloud, got shape {pts.shape}."}

            n = len(pts)
            visited = np.zeros(n, dtype=bool)
            clusters = []

            for i in range(n):
                if visited[i]:
                    continue

                # BFS/region growing from point i
                cluster_indices = [i]
                visited[i] = True
                queue = [i]

                while queue:
                    current = queue.pop(0)
                    # Find neighbors within tolerance
                    dists = np.linalg.norm(pts - pts[current], axis=1)
                    neighbors = np.where((dists < cluster_tolerance) & (~visited))[0]
                    for nb in neighbors:
                        visited[nb] = True
                        queue.append(nb)
                        cluster_indices.append(nb)

                if min_cluster_size <= len(cluster_indices) <= max_cluster_size:
                    cluster_pts = pts[cluster_indices]
                    center = cluster_pts.mean(axis=0).tolist()
                    bbox_min = cluster_pts.min(axis=0).tolist()
                    bbox_max = cluster_pts.max(axis=0).tolist()
                    clusters.append({
                        "indices": cluster_indices,
                        "center": center,
                        "bbox": {"min": bbox_min, "max": bbox_max},
                    })

            return {
                "success": True,
                "message": f"Found {len(clusters)} clusters.",
                "clusters": clusters,
                "num_clusters": len(clusters),
            }
        except Exception as exc:
            logger.exception("[EuclideanClusteringSkill] Failed")
            return {"success": False, "message": str(exc)}


class StatisticalOutlierRemovalSkill(SkillInterface):
    """Remove statistical outliers from a point cloud."""

    name = "statistical_outlier_removal"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Remove points that are far from their k-nearest neighbors (statistical outlier removal).",
            parameters=[
                ParamSchema(name="points", type="list", description="Point cloud as list of [x, y, z] or N×3 ndarray.", required=True),
                ParamSchema(name="k_neighbors", type="int", description="Number of neighbors to consider.", required=False, default=10),
                ParamSchema(name="std_ratio", type="float", description="Points with mean distance > μ + std_ratio·σ are removed.", required=False, default=1.0),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether filtering succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="filtered_points", type="list", description="Point cloud with outliers removed."),
                ResultSchema(name="num_removed", type="int", description="Number of points removed."),
            ],
            dependencies=[],
        )

    def execute(self, points, k_neighbors: int = 10, std_ratio: float = 1.0, **kwargs) -> Dict[str, Any]:
        try:
            pts = np.array(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                return {"success": False, "message": f"Expected N×3 point cloud, got shape {pts.shape}."}

            n = len(pts)
            if n <= k_neighbors:
                return {"success": True, "message": "Too few points for SOR, returning original.", "filtered_points": pts.tolist(), "num_removed": 0}

            # Compute mean distance to k nearest neighbors for each point
            mean_distances = np.zeros(n)
            for i in range(n):
                dists = np.linalg.norm(pts - pts[i], axis=1)
                dists[i] = np.inf  # exclude self
                k_dists = np.partition(dists, k_neighbors)[:k_neighbors]
                mean_distances[i] = k_dists.mean()

            mean = mean_distances.mean()
            std = mean_distances.std()
            threshold = mean + std_ratio * std
            mask = mean_distances <= threshold
            filtered = pts[mask]

            return {
                "success": True,
                "message": f"SOR removed {n - len(filtered)} outliers.",
                "filtered_points": filtered.tolist(),
                "num_removed": int(n - len(filtered)),
            }
        except Exception as exc:
            logger.exception("[StatisticalOutlierRemovalSkill] Failed")
            return {"success": False, "message": str(exc)}
