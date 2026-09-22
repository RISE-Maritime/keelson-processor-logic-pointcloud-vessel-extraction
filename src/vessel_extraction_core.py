"""Transport-neutral geometry and tracking for point-cloud vessel extraction."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


NUMERIC_DTYPES = {
    1: np.dtype("u1"), 2: np.dtype("i1"), 3: np.dtype("<u2"),
    4: np.dtype("<i2"), 5: np.dtype("<u4"), 6: np.dtype("<i4"),
    7: np.dtype("<f4"), 8: np.dtype("<f8"),
}


@dataclass(frozen=True)
class PackedField:
    name: str
    offset: int
    numeric_type: int


@dataclass
class Candidate:
    points: np.ndarray
    center: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    length: float
    width: float
    z_min: float
    z_max: float
    score: float


@dataclass
class Track:
    id: int
    candidate: Candidate
    missed_frames: int = 0
    history: list[tuple[float, float, float]] = field(default_factory=list)


def quaternion_matrix(
    x: float, y: float, z: float, w: float,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        x = y = z = 0.0
        w = 1.0
    else:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = translation
    return matrix


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def decode_xyz(data: bytes, point_stride: int, fields: list[PackedField]) -> np.ndarray:
    if point_stride <= 0 or len(data) % point_stride:
        raise ValueError("data length is not a multiple of point_stride")
    count = len(data) // point_stride
    by_name = {item.name.lower(): item for item in fields}
    output = np.empty((count, 3), dtype=np.float64)
    for column, name in enumerate(("x", "y", "z")):
        item = by_name.get(name)
        if item is None:
            raise ValueError(f"point cloud is missing {name!r}")
        dtype = NUMERIC_DTYPES.get(item.numeric_type)
        if dtype is None or item.offset + dtype.itemsize > point_stride:
            raise ValueError(f"invalid packed field {name!r}")
        if count:
            output[:, column] = np.ndarray(
                (count,), dtype=dtype, buffer=data, offset=item.offset,
                strides=(point_stride,),
            )
    return output[np.isfinite(output).all(axis=1)]


def filter_roi(points: np.ndarray, roi: dict[str, Any]) -> np.ndarray:
    if not roi.get("enabled", True) or len(points) == 0:
        return points
    mask = (
        (points[:, 0] >= roi["min_x_m"]) & (points[:, 0] <= roi["max_x_m"]) &
        (points[:, 1] >= roi["min_y_m"]) & (points[:, 1] <= roi["max_y_m"]) &
        (points[:, 2] >= roi["min_z_m"]) & (points[:, 2] <= roi["max_z_m"])
    )
    return points[mask]


def deterministic_cap(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    if len(points) == 0 or size <= 0:
        return points
    cells = np.floor(points / size).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(first)]


def connected_components_xy(
    points: np.ndarray, cell_size: float, minimum_points: int,
) -> list[np.ndarray]:
    if not len(points):
        return []
    cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    members: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        members[(int(cell[0]), int(cell[1]))].append(index)
    remaining = set(members)
    components: list[np.ndarray] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        occupied = [seed]
        while queue:
            cx, cy = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (cx + dx, cy + dy)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        queue.append(neighbour)
                        occupied.append(neighbour)
        indices = [index for cell in occupied for index in members[cell]]
        if len(indices) >= minimum_points:
            components.append(points[np.asarray(indices)])
    return components


def convex_hull_xy(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points[:, :2], dtype=np.float64), axis=0)
    if len(unique) < 3:
        raise ValueError("at least three footprint points are required")
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ao, bo = a - origin, b - origin
        return float(ao[0] * bo[1] - ao[1] * bo[0])

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        raise ValueError("footprint points are collinear")
    return hull


def minimum_area_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = convex_hull_xy(points)
    best_area = math.inf
    best: tuple[np.ndarray, np.ndarray, float, float] | None = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        norm = float(np.linalg.norm(edge))
        if norm < 1e-9:
            continue
        first = edge / norm
        second = np.array([-first[1], first[0]])
        first_extent = float(np.ptp(hull @ first))
        second_extent = float(np.ptp(hull @ second))
        area = first_extent * second_extent
        if area < best_area:
            best_area = area
            best = first, second, first_extent, second_extent
    if best is None:
        raise ValueError("could not estimate footprint axes")
    first, second, first_extent, second_extent = best
    return (first, second) if first_extent >= second_extent else (second, first)


def evaluate_candidate(points: np.ndarray, limits: dict[str, Any]) -> Candidate | None:
    try:
        major, minor = minimum_area_axes(points)
    except ValueError:
        return None
    lower_q, upper_q = limits["lower_quantile"], limits["upper_quantile"]
    major_projection = points[:, :2] @ major
    minor_projection = points[:, :2] @ minor
    major_min, major_max = np.quantile(major_projection, [lower_q, upper_q])
    minor_min, minor_max = np.quantile(minor_projection, [lower_q, upper_q])
    z_min, z_max = np.quantile(points[:, 2], [lower_q, upper_q])
    length = float(major_max - major_min)
    width = float(minor_max - minor_min)
    height = float(z_max - z_min)
    if not (
        limits["min_length_m"] <= length <= limits["max_length_m"] and
        limits["min_width_m"] <= width <= limits["max_width_m"] and
        limits["min_height_m"] <= height <= limits["max_height_m"] and
        length / max(width, 1e-6) >= limits["min_elongation"]
    ):
        return None
    inliers = (
        (major_projection >= major_min) & (major_projection <= major_max) &
        (minor_projection >= minor_min) & (minor_projection <= minor_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    kept = points[inliers]
    center_xy = major * ((major_min + major_max) / 2) + minor * ((minor_min + minor_max) / 2)
    center = np.array([center_xy[0], center_xy[1], (z_min + z_max) / 2])
    score = len(kept) * min(length / max(width, 1e-6), 8.0)
    return Candidate(kept, center, major, minor, length, width,
                     float(z_min), float(z_max), score)


def extract_candidates(points: np.ndarray, config: dict[str, Any]) -> list[Candidate]:
    processing = config["processing"]
    selected = filter_roi(points, config["roi"])
    selected = voxel_downsample(selected, processing["extraction_voxel_m"])
    selected = deterministic_cap(selected, processing["maximum_points"])
    components = connected_components_xy(
        selected, processing["component_cell_m"], processing["minimum_component_points"],
    )
    candidates = [candidate for component in components
                  if (candidate := evaluate_candidate(component, config["shape_limits"])) is not None]
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


class Tracker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(
        self, candidates: list[Candidate], timestamp_s: float,
    ) -> tuple[list[Track], list[int]]:
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for candidate_index, candidate in enumerate(candidates):
                distance = float(np.linalg.norm(track.candidate.center[:2] - candidate.center[:2]))
                if distance <= self.config["max_track_jump_m"]:
                    size_penalty = self.config["size_change_penalty"] * (
                        abs(track.candidate.length - candidate.length) +
                        abs(track.candidate.width - candidate.width)
                    )
                    pairs.append((distance + size_penalty, track_id, candidate_index))
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        active: list[Track] = []
        for _, track_id, candidate_index in sorted(pairs):
            if track_id in used_tracks or candidate_index in used_candidates:
                continue
            used_tracks.add(track_id)
            used_candidates.add(candidate_index)
            track = self.tracks[track_id]
            track.candidate = candidates[candidate_index]
            track.missed_frames = 0
            self._append_history(track, timestamp_s)
            active.append(track)
        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track.missed_frames += 1
        deleted = [track_id for track_id, track in self.tracks.items()
                   if track.missed_frames >= self.config["max_missed_frames"]]
        for track_id in deleted:
            del self.tracks[track_id]
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates or len(self.tracks) >= self.config["max_active_tracks"]:
                continue
            track = Track(self.next_id, candidate)
            self.next_id += 1
            self._append_history(track, timestamp_s)
            self.tracks[track.id] = track
            active.append(track)
        return active, deleted

    def _append_history(self, track: Track, timestamp_s: float) -> None:
        track.history.append((timestamp_s, float(track.candidate.center[0]),
                              float(track.candidate.center[1])))
        oldest = timestamp_s - self.config["history_seconds"]
        track.history = [sample for sample in track.history if sample[0] >= oldest]

    def velocity(self, track: Track) -> tuple[float, float, float] | None:
        if len(track.history) < 3:
            return None
        samples = np.asarray(track.history)
        duration = float(samples[-1, 0] - samples[0, 0])
        if duration < self.config["minimum_course_baseline_s"]:
            return None
        time_axis = samples[:, 0] - np.mean(samples[:, 0])
        denominator = float(time_axis @ time_axis)
        if denominator < 1e-9:
            return None
        vx = float(time_axis @ (samples[:, 1] - np.mean(samples[:, 1])) / denominator)
        vy = float(time_axis @ (samples[:, 2] - np.mean(samples[:, 2])) / denominator)
        return vx, vy, math.hypot(vx, vy)


def hull_mesh(candidate: Candidate, maximum_vertices: int = 256) -> tuple[np.ndarray, np.ndarray]:
    hull = convex_hull_xy(candidate.points)
    if len(hull) > maximum_vertices:
        hull = hull[np.linspace(0, len(hull) - 1, maximum_vertices, dtype=np.int64)]
    relative_xy = hull - candidate.center[:2]
    low = candidate.z_min - candidate.center[2]
    high = candidate.z_max - candidate.center[2]
    count = len(hull)
    vertices = np.vstack((
        np.column_stack((relative_xy, np.full(count, low))),
        np.column_stack((relative_xy, np.full(count, high))),
    ))
    triangles: list[tuple[int, int, int]] = []
    for index in range(1, count - 1):
        triangles.extend(((0, index + 1, index),
                          (count, count + index, count + index + 1)))
    for index in range(count):
        following = (index + 1) % count
        triangles.extend(((index, following, count + following),
                          (index, count + following, count + index)))
    return vertices, np.asarray(triangles, dtype=np.uint32)
