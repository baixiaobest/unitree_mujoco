"""Local floor and rear obstacle extraction from a PCD map."""
import math
import numpy as np
from scipy.spatial import cKDTree


def load_pcd_xyz(path):
    """Read x/y/z from ASCII or uncompressed binary PCD (including FAST-LIO PCD)."""
    with open(path, "rb") as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD has no DATA field")
            parts = line.decode("ascii").strip().split()
            if parts and not parts[0].startswith("#"):
                header[parts[0].upper()] = parts[1:]
            if parts and parts[0].upper() == "DATA":
                break
        names = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(names))))
        if any(c != 1 for c in counts):
            raise ValueError("PCD fields with COUNT != 1 are unsupported")
        formats = []
        for kind, size in zip(types, sizes):
            code = {"F": "f", "I": "i", "U": "u"}.get(kind)
            if code is None:
                raise ValueError("Unsupported PCD field type")
            formats.append("<" + code + str(size))
        dtype = np.dtype({"names": names, "formats": formats})
        count = int(header["POINTS"][0])
        mode = header["DATA"][0]
        if mode == "binary":
            data = np.fromfile(stream, dtype=dtype, count=count)
            if len(data) != count:
                raise ValueError("Truncated PCD")
            xyz = np.column_stack([data[k] for k in ("x", "y", "z")])
        elif mode == "ascii":
            values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
            xyz = values[:, [names.index(k) for k in ("x", "y", "z")]]
        else:
            raise ValueError("Only ASCII and uncompressed binary PCD are supported")
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def voxel_downsample(points, size_m):
    """Keep one representative per 3D voxel for bounded local processing."""
    if not np.isfinite(size_m) or size_m <= 0:
        raise ValueError("Voxel size must be positive and finite")
    _, indices = np.unique(np.floor(points / size_m).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def supported_obstacles(points, radius_m=0.2, min_points=3):
    """Separate isolated returns from 3D clusters after floor removal."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(radius_m) or radius_m <= 0 or min_points < 1:
        raise ValueError("Invalid obstacle support settings")
    if not len(points):
        return points, points
    counts = cKDTree(points).query_ball_point(points, radius_m, return_length=True)
    keep = counts >= min_points  # Count includes the point itself.
    return points[keep], points[~keep]


def floor_and_rear(points_base, *, floor_radius=2.0, floor_tolerance=0.08,
                   min_obstacle_height=0.12, max_obstacle_height=1.4,
                   max_range=8.0, bins=64, support_radius_m=0.2,
                   min_support_points=3):
    """Return floor support, binned rear hits, plane, retained and rejected points.

    A low-point RANSAC plane is required so a wall cannot silently become floor.
    """
    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30:
        return None
    xy = np.linalg.norm(points[:, :2], axis=1)
    local = points[(xy < floor_radius) & (points[:, 2] > -1.2) & (points[:, 2] < 0.6)]
    if len(local) < 30:
        return None
    low = local[local[:, 2] <= np.quantile(local[:, 2], 0.35)]
    if len(low) < 20:
        return None
    rng = np.random.default_rng(42)
    best = None
    for _ in range(80):
        sample = low[rng.choice(len(low), 3, replace=False)]
        matrix = np.column_stack((sample[:, :2], np.ones(3)))
        if abs(np.linalg.det(matrix)) < 1e-4:
            continue
        plane = np.linalg.solve(matrix, sample[:, 2])
        if np.linalg.norm(plane[:2]) > math.tan(math.radians(25)):
            continue
        residual = np.abs(local[:, 2] - (local[:, :2] @ plane[:2] + plane[2]))
        inliers = np.flatnonzero(residual < floor_tolerance)
        if best is None or len(inliers) > len(best[0]):
            best = (inliers, plane)
    if best is None or len(best[0]) < 20:
        return None
    inliers, _ = best
    plane = np.linalg.lstsq(np.column_stack((local[inliers, :2], np.ones(len(inliers)))), local[inliers, 2], rcond=None)[0]
    if np.linalg.norm(plane[:2]) > math.tan(math.radians(25)):
        return None
    height = points[:, 2] - (points[:, :2] @ plane[:2] + plane[2])
    valid = (points[:, 0] < 0) & (xy <= max_range) & (height >= min_obstacle_height) & (height <= max_obstacle_height)
    rear, rejected = supported_obstacles(points[valid], support_radius_m, min_support_points)
    ranges = np.linalg.norm(rear[:, :2], axis=1)
    indices = np.minimum((np.arctan2(rear[:, 1], -rear[:, 0]) / math.pi + 0.5) * bins, bins - 1).astype(int)
    chosen = []
    for index in range(bins):
        members = np.flatnonzero(indices == index)
        if len(members):
            chosen.append(rear[members[np.argmin(ranges[members])]])
    return local[inliers], np.asarray(chosen, dtype=np.float64).reshape(-1, 3), plane, rear, rejected
