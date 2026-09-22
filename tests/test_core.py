import numpy as np

from vessel_extraction_core import (
    Tracker, connected_components_xy, evaluate_candidate, extract_candidates,
    filter_roi, hull_mesh,
)


def config():
    return {
        "processing": {
            "extraction_voxel_m": 0.05,
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
            "max_track_jump_m": 25.0, "size_change_penalty": 0.1,
            "max_missed_frames": 5, "max_active_tracks": 32,
            "history_seconds": 12.0, "minimum_course_baseline_s": 1.0,
        },
    }


def vessel(center_x=20.0, center_y=-50.0, seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(center_x - 12, center_x + 12, 4000),
        rng.uniform(center_y - 3, center_y + 3, 4000),
        rng.uniform(0, 4, 4000),
    ))


def test_roi_excludes_points_outside_search_region():
    points = np.array([[20, -50, 2], [100, -50, 2], [20, 100, 2]], dtype=float)
    selected = filter_roi(points, config()["roi"])
    assert selected.shape == (1, 3)
    np.testing.assert_array_equal(selected[0], points[0])


def test_minimum_area_candidate_and_mesh():
    candidate = evaluate_candidate(vessel(), config()["shape_limits"])
    assert candidate is not None
    assert 22 < candidate.length < 24
    assert 5 < candidate.width < 6
    vertices, triangles = hull_mesh(candidate)
    assert len(vertices) >= 6
    assert triangles.shape[1] == 3


def test_two_vessels_are_extracted():
    points = np.concatenate((vessel(20, -50, 2), vessel(30, -120, 3)))
    candidates = extract_candidates(points, config())
    assert len(candidates) == 2


def test_tracker_estimates_course_from_motion():
    cfg = config()
    tracker = Tracker(cfg["tracking"])
    for index in range(4):
        candidate = evaluate_candidate(vessel(20 + index, -50, 10 + index),
                                       cfg["shape_limits"])
        assert candidate is not None
        active, deleted = tracker.update([candidate], float(index))
        assert not deleted
    assert len(active) == 1
    velocity = tracker.velocity(active[0])
    assert velocity is not None
    assert 0.7 < velocity[0] < 1.3
    assert abs(velocity[1]) < 0.3


def test_connected_components_rejects_sparse_noise():
    coherent = vessel(seed=7)[:200]
    noise = np.array([[200 + 5*i, 200, 0] for i in range(20)], dtype=float)
    components = connected_components_xy(
        np.concatenate((coherent, noise)), 1.25, 40,
    )
    assert len(components) == 1


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
