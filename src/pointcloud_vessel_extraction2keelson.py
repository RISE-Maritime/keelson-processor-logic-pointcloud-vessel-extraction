#!/usr/bin/env python3
"""Extract and track vessel geometry from a Keelson point-cloud stream."""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zenoh
from jsonschema import Draft202012Validator

import keelson
from keelson.payloads.foxglove.PointCloud_pb2 import PointCloud
from keelson.payloads.foxglove.SceneUpdate_pb2 import SceneUpdate
from keelson.scaffolding import (
    add_common_arguments, create_zenoh_config, declare_liveliness_token,
    setup_logging,
)

from vessel_extraction_core import (
    PackedField, Tracker, apply_transform, decode_xyz, extract_candidates,
    hull_mesh, quaternion_matrix,
)


logger = logging.getLogger("pointcloud-vessel-extraction2keelson")
OUTPUT_SUBJECT = "vessel_shape_3d"


CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["inputs", "output"],
    "additionalProperties": False,
    "properties": {
        "inputs": {
            "type": "object", "required": ["point_cloud_key"],
            "additionalProperties": False,
            "properties": {
                "point_cloud_key": {"type": "string", "minLength": 1},
                "expected_frame_id": {"type": "string"},
            },
        },
        "output": {
            "type": "object", "required": ["realm", "entity_id", "source_id"],
            "additionalProperties": False,
            "properties": {name: {"type": "string", "minLength": 1}
                           for name in ("realm", "entity_id", "source_id")},
        },
        "processing": {"type": "object"},
        "roi": {"type": "object"},
        "shape_limits": {"type": "object"},
        "tracking": {"type": "object"},
        "visualization": {"type": "object"},
    },
}


def default_config() -> dict[str, Any]:
    return {
        "processing": {
            "process_rate_hz": 5.0,
            "extraction_voxel_m": 0.15,
            "maximum_points": 80000,
            "component_cell_m": 1.25,
            "minimum_component_points": 40,
        },
        "roi": {
            "enabled": True,
            "min_x_m": 1.0, "max_x_m": 45.0,
            "min_y_m": -200.0, "max_y_m": 45.0,
            "min_z_m": -5.0, "max_z_m": 50.0,
        },
        "shape_limits": {
            "min_length_m": 4.0, "max_length_m": 200.0,
            "min_width_m": 1.0, "max_width_m": 45.0,
            "min_height_m": 0.75, "max_height_m": 45.0,
            "min_elongation": 1.2,
            "lower_quantile": 0.03, "upper_quantile": 0.97,
        },
        "tracking": {
            "max_track_jump_m": 25.0,
            "size_change_penalty": 0.1,
            "max_missed_frames": 5,
            "max_active_tracks": 32,
            "history_seconds": 12.0,
            "minimum_course_baseline_s": 1.0,
            "minimum_course_speed_mps": 0.25,
            "course_projection_seconds": 15.0,
            "maximum_projected_length_m": 150.0,
        },
        "visualization": {
            "entity_lifetime_s": 3.0,
            "hull_color_rgba": [0.1, 0.65, 1.0, 0.38],
            "centreline_color_rgba": [1.0, 0.8, 0.05, 1.0],
            "course_color_rgba": [1.0, 0.2, 0.1, 1.0],
            "centreline_thickness_m": 0.22,
            "maximum_hull_vertices": 256,
        },
    }


def deep_update(base: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict[str, Any]:
    supplied = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator(CONFIG_SCHEMA).validate(supplied)
    config = deep_update(default_config(), supplied)
    roi = config["roi"]
    for low, high in (("min_x_m", "max_x_m"), ("min_y_m", "max_y_m"),
                      ("min_z_m", "max_z_m")):
        if roi[low] > roi[high]:
            raise ValueError(f"roi.{low} must not exceed roi.{high}")
    limits = config["shape_limits"]
    if limits["lower_quantile"] >= limits["upper_quantile"]:
        raise ValueError("shape_limits.lower_quantile must be below upper_quantile")
    if config["processing"]["process_rate_hz"] <= 0:
        raise ValueError("processing.process_rate_hz must be positive")
    if config["processing"]["component_cell_m"] <= 0:
        raise ValueError("processing.component_cell_m must be positive")
    return config


def sample_bytes(sample: Any) -> bytes:
    payload = sample.payload if hasattr(sample, "payload") else sample.value
    return payload.to_bytes()


@dataclass
class CloudState:
    points: np.ndarray
    timestamp_ns: int
    frame_id: str
    generation: int


def decode_cloud(payload: bytes) -> tuple[np.ndarray, int, str]:
    cloud = PointCloud.FromString(payload)
    fields = [PackedField(item.name, item.offset, item.type) for item in cloud.fields]
    points = decode_xyz(cloud.data, cloud.point_stride, fields)
    embedded_pose = quaternion_matrix(
        cloud.pose.orientation.x, cloud.pose.orientation.y,
        cloud.pose.orientation.z, cloud.pose.orientation.w,
        (cloud.pose.position.x, cloud.pose.position.y, cloud.pose.position.z),
    )
    return apply_transform(points, embedded_pose), cloud.timestamp.ToNanoseconds(), cloud.frame_id


def set_color(target: Any, rgba: list[float]) -> None:
    target.r, target.g, target.b, target.a = map(float, rgba)


def set_lifetime(entity: Any, seconds: float) -> None:
    entity.lifetime.seconds = int(seconds)
    entity.lifetime.nanos = int(round((seconds % 1.0) * 1e9))


def add_metadata(entity: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        pair = entity.metadata.add()
        pair.key = str(key)
        pair.value = str(value)


def initialise_entity(
    scene: SceneUpdate, entity_id: str, frame_id: str, timestamp_ns: int,
    lifetime_s: float, metadata: dict[str, Any],
) -> Any:
    entity = scene.entities.add()
    entity.timestamp.FromNanoseconds(timestamp_ns)
    entity.frame_id = frame_id
    entity.id = entity_id
    entity.frame_locked = False
    set_lifetime(entity, lifetime_s)
    add_metadata(entity, metadata)
    return entity


def add_deletion(scene: SceneUpdate, entity_id: str, timestamp_ns: int) -> None:
    deletion = scene.deletions.add()
    deletion.timestamp.FromNanoseconds(timestamp_ns)
    deletion.type = 1  # MATCHING_ID
    deletion.id = entity_id


def build_scene(
    active_tracks: list[Any], deleted_track_ids: list[int], tracker: Tracker,
    frame_id: str, timestamp_ns: int, config: dict[str, Any],
) -> SceneUpdate:
    scene = SceneUpdate()
    display = config["visualization"]
    tracking = config["tracking"]
    lifetime = display["entity_lifetime_s"]
    for track_id in deleted_track_ids:
        prefix = f"shape-extracted-vessel-{track_id}"
        for suffix in ("hull", "centreline", "course"):
            add_deletion(scene, f"{prefix}-{suffix}", timestamp_ns)

    for track in active_tracks:
        candidate = track.candidate
        center = candidate.center
        prefix = f"shape-extracted-vessel-{track.id}"
        metadata = {
            "track_id": track.id,
            "geometry_source": "point_cloud_only",
            "anchor_x_m": f"{center[0]:.3f}",
            "anchor_y_m": f"{center[1]:.3f}",
            "anchor_z_m": f"{center[2]:.3f}",
            "length_m": f"{candidate.length:.3f}",
            "width_m": f"{candidate.width:.3f}",
            "height_m": f"{candidate.z_max-candidate.z_min:.3f}",
            "point_count": len(candidate.points),
        }

        hull_entity = initialise_entity(
            scene, f"{prefix}-hull", frame_id, timestamp_ns, lifetime, metadata,
        )
        vertices, triangles = hull_mesh(
            candidate, int(display["maximum_hull_vertices"]),
        )
        mesh = hull_entity.triangles.add()
        mesh.pose.position.x, mesh.pose.position.y, mesh.pose.position.z = map(float, center)
        mesh.pose.orientation.w = 1.0
        set_color(mesh.color, display["hull_color_rgba"])
        for vertex in vertices:
            point = mesh.points.add()
            point.x, point.y, point.z = map(float, vertex)
        mesh.indices.extend(triangles.reshape(-1).tolist())

        line_entity = initialise_entity(
            scene, f"{prefix}-centreline", frame_id, timestamp_ns, lifetime, metadata,
        )
        line = line_entity.lines.add()
        line.type = 0  # LINE_STRIP
        line.pose.position.x, line.pose.position.y, line.pose.position.z = map(float, center)
        line.pose.orientation.w = 1.0
        line.thickness = float(display["centreline_thickness_m"])
        line.scale_invariant = False
        set_color(line.color, display["centreline_color_rgba"])
        half = candidate.major * candidate.length / 2
        for x, y in ((-half[0], -half[1]), (half[0], half[1])):
            point = line.points.add()
            point.x, point.y, point.z = float(x), float(y), 0.0

        velocity = tracker.velocity(track)
        if velocity is None or velocity[2] < tracking["minimum_course_speed_mps"]:
            add_deletion(scene, f"{prefix}-course", timestamp_ns)
            continue
        vx, vy, speed = velocity
        projected_length = min(
            speed * tracking["course_projection_seconds"],
            tracking["maximum_projected_length_m"],
        )
        course_metadata = dict(metadata)
        course_metadata["speed_mps"] = f"{speed:.3f}"
        course_metadata["projection_seconds"] = tracking["course_projection_seconds"]
        course_entity = initialise_entity(
            scene, f"{prefix}-course", frame_id, timestamp_ns, lifetime,
            course_metadata,
        )
        arrow = course_entity.arrows.add()
        arrow.pose.position.x, arrow.pose.position.y, arrow.pose.position.z = map(float, center)
        yaw = math.atan2(vy, vx)
        arrow.pose.orientation.z = math.sin(yaw / 2)
        arrow.pose.orientation.w = math.cos(yaw / 2)
        arrow.head_length = min(1.5, projected_length)
        arrow.shaft_length = max(0.0, projected_length - arrow.head_length)
        arrow.shaft_diameter = 0.35
        arrow.head_diameter = 0.8
        set_color(arrow.color, display["course_color_rgba"])
    return scene


def run(session: Any, config: dict[str, Any], stop: threading.Event) -> None:
    inputs, output = config["inputs"], config["output"]
    expected_frame = inputs.get("expected_frame_id", "")
    tracker = Tracker(config["tracking"])
    lock = threading.Lock()
    newest_cloud: CloudState | None = None
    generation = 0
    last_processed = -1
    previous_counts: tuple[int, int] | None = None

    output_key = keelson.construct_pubsub_key(
        output["realm"], output["entity_id"], OUTPUT_SUBJECT,
        output["source_id"],
    )
    publisher = session.declare_publisher(output_key)
    logger.info("Subscribing to point cloud: %s", inputs["point_cloud_key"])
    logger.info("Publishing vessel geometry: %s", output_key)

    def on_cloud(sample: Any) -> None:
        nonlocal newest_cloud, generation
        try:
            _, _, payload = keelson.uncover(sample_bytes(sample))
            points, timestamp_ns, frame_id = decode_cloud(payload)
            if expected_frame and frame_id != expected_frame:
                raise ValueError(
                    f"cloud frame {frame_id!r} does not match expected frame "
                    f"{expected_frame!r}"
                )
            with lock:
                generation += 1
                newest_cloud = CloudState(
                    points, timestamp_ns or time.time_ns(), frame_id or expected_frame,
                    generation,
                )
        except Exception:
            logger.exception("Rejected point cloud")

    subscriber = session.declare_subscriber(inputs["point_cloud_key"], on_cloud)
    period = 1.0 / config["processing"]["process_rate_hz"]
    try:
        while not stop.wait(period):
            with lock:
                cloud = newest_cloud
            if cloud is None or cloud.generation == last_processed:
                continue
            last_processed = cloud.generation
            started = time.monotonic()
            try:
                candidates = extract_candidates(cloud.points, config)
                timestamp_s = cloud.timestamp_ns / 1e9
                active, deleted = tracker.update(candidates, timestamp_s)
                scene = build_scene(
                    active, deleted, tracker, cloud.frame_id, cloud.timestamp_ns,
                    config,
                )
                if scene.entities or scene.deletions:
                    publisher.put(keelson.enclose(scene.SerializeToString()))
                counts = (len(candidates), len(tracker.tracks))
                if counts != previous_counts:
                    logger.info(
                        "Detected %d candidate(s); tracking %d vessel(s)", *counts,
                    )
                    previous_counts = counts
                logger.debug(
                    "Processed cloud generation=%d input_points=%d candidates=%d "
                    "entities=%d elapsed_ms=%.1f",
                    cloud.generation, len(cloud.points), len(candidates),
                    len(scene.entities), (time.monotonic() - started) * 1000,
                )
            except Exception:
                logger.exception("Failed to process point cloud generation %d", cloud.generation)
    finally:
        subscriber.undeclare()
        publisher.undeclare()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    setup_logging(level=args.log_level)
    zenoh.init_log_from_env_or("error")
    try:
        config = load_config(args.config)
    except Exception:
        logger.exception("Invalid configuration: %s", args.config)
        raise SystemExit(2)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    session = zenoh.open(create_zenoh_config(args.mode, args.connect, args.listen))
    output = config["output"]
    try:
        with declare_liveliness_token(
            session, output["realm"], output["entity_id"], output["source_id"],
        ):
            run(session, config, stop)
    finally:
        session.close()


if __name__ == "__main__":
    main()
