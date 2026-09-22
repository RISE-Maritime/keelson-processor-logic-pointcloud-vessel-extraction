# keelson-pointcloud-vessel-extraction

Live Keelson processor that detects and tracks vessel-like geometry directly
from a `foxglove.PointCloud`. It publishes `foxglove.SceneUpdate` entities for
each active geometric track:

- a watertight 3D hull envelope;
- a minimum-area-footprint centreline;
- a projected-course arrow once enough centroid motion is available.

AIS is not subscribed to and no reported vessel dimensions are used.

## Pipeline position

This processor consumes the output of the separate point-cloud merge service:

```text
LiDAR clouds -> transform/merge/thin -> this processor -> vessel_shape_3d
```

The input cloud is expected to be fully transformed into one Cartesian frame.
Its embedded `PointCloud.pose` is applied before extraction. The configured ROI
affects extraction only; this processor does not republish or crop the input
cloud.

## Output

The output key is:

```text
<realm>/@v0/<entity_id>/pubsub/vessel_shape_3d/<source_id>
```

Every track uses stable entity IDs such as:

```text
shape-extracted-vessel-3-hull
shape-extracted-vessel-3-centreline
shape-extracted-vessel-3-course
```

The hull's `TriangleListPrimitive.pose.position` is the geometry-derived anchor
point in the point-cloud frame. Mesh vertices are relative to that anchor.
Metadata contains the anchor XYZ, measured length/width/height, point count, and
track ID.

The model is a conservative above-water envelope inferred from visible returns,
not a naval-architecture reconstruction of hidden or underwater surfaces.

## Configure

Edit `config/example.json`:

- `inputs.point_cloud_key`: exact Keelson key of the merged/thinned cloud.
- `inputs.expected_frame_id`: required frame ID, or an empty string to accept
  the frame carried by each cloud.
- `roi`: the XYZ search region used only for extraction.
- `shape_limits`: admissible vessel geometry.
- `tracking`: association, track expiry, and course estimation.
- `output`: standard Keelson realm/entity/source components.

The supplied defaults contain the current search ROI:

```text
X:    1 to 45 m
Y: -200 to 45 m
Z:   -5 to 50 m
```

## Run with Docker Compose

```bash
docker compose up --build
```

The default compose file connects to `tcp/127.0.0.1:7447` using host networking.
Adjust `--connect` and the configured point-cloud key for the deployment.

## Foxglove

The output subject is custom, so pass the bundled type mapping to
`keelson2foxglove`:

```bash
keelson2foxglove \
  --connect tcp/127.0.0.1:7447 \
  -k 'rise/@v0/**/pubsub/vessel_shape_3d/**' \
  --extra-subjects-types=/config/extra-subjects.yaml,
```

In the Foxglove 3D panel, select the `vessel_shape_3d` topic and use the input
point cloud's frame as the fixed frame.

## Processing details

1. Decode XYZ and apply the embedded point-cloud pose.
2. Select points inside the configured extraction ROI.
3. Voxel-downsample and deterministically cap the segmentation workload.
4. Find occupied-grid connected components in XY.
5. Reject components outside the configured length, width, height, or
   elongation limits.
6. Estimate the centreline from the minimum-area footprint rectangle. It does
   not assume or search for a pointed bow.
7. Associate candidates between frames using centroid distance and size change.
8. Estimate course by linear regression over recent track centroids.
9. Publish hull, centreline, course, and explicit deletion updates.

## Test locally

The geometry tests do not need a running Zenoh router:

```bash
python3 -m pip install numpy pytest
PYTHONPATH=src python3 -m pytest -q
```

## Operational limits

- Coordinates must be metric and Z-up for the configured limits to be
  meaningful.
- Track IDs are process-local and restart from one when the container restarts.
- Neighbouring returns connected at the configured cell size can merge into one
  component. Reduce `component_cell_m` if nearby vessels or quay structures
  bridge together.
- Increase `minimum_component_points` or `min_elongation` if sparse clutter is
  promoted to vessel tracks.
- This is a perception aid and research processor, not a collision-avoidance or
  safety-certified navigation source.
