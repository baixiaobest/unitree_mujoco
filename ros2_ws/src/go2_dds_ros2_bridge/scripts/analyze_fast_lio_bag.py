#!/usr/bin/env python3
"""Create a selectable FAST-LIO and robot body IMU report from a ROS 2 bag.

Run in a sourced ROS 2 workspace containing fast_lio and rosbag2_py.
Install the report dependency with ``python3 -m pip install plotly``:
    python3 scripts/analyze_fast_lio_bag.py BAG_DIR --output report.html

Record: ros2 bag record /Odometry /fast_lio/estimator_diagnostics /imu/data_fastlio_raw
The default IMU is the uncalibrated Go2 body IMU (frame_id=body), not the LiDAR IMU.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path


DIAGNOSTICS_TOPIC = "/fast_lio/estimator_diagnostics"
BODY_IMU_TOPIC = "/imu/data_fastlio_raw"
ODOMETRY_TOPIC = "/Odometry"


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def read_bag(bag: Path, diagnostics_topic: str, imu_topic: str, odometry_topic: str):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import yaml

    metadata = bag / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError(f"No metadata.yaml in {bag}")
    info = yaml.safe_load(metadata.read_text())["rosbag2_bagfile_information"]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=info["storage_identifier"]),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if diagnostics_topic not in topic_types:
        raise ValueError(f"Missing {diagnostics_topic}; available topics: {', '.join(sorted(topic_types))}")
    if topic_types[diagnostics_topic] != "fast_lio/msg/EstimatorDiagnostics":
        raise ValueError(f"{diagnostics_topic} has unexpected type {topic_types[diagnostics_topic]}")
    if imu_topic in topic_types and topic_types[imu_topic] != "sensor_msgs/msg/Imu":
        raise ValueError(f"{imu_topic} has unexpected type {topic_types[imu_topic]}")
    if odometry_topic in topic_types and topic_types[odometry_topic] != "nav_msgs/msg/Odometry":
        raise ValueError(f"{odometry_topic} has unexpected type {topic_types[odometry_topic]}")
    types = {name: get_message(topic_types[name]) for name in
             (diagnostics_topic, imu_topic, odometry_topic) if name in topic_types}
    messages = {name: [] for name in types}
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic not in types:
            continue
        message = deserialize_message(serialized, types[topic])
        messages[topic].append(message)
    for items in messages.values():
        items.sort(key=lambda item: stamp_seconds(item.header.stamp))
    return (messages[diagnostics_topic], messages.get(imu_topic, []),
            imu_topic in topic_types, messages.get(odometry_topic, []),
            odometry_topic in topic_types)


def quaternion_to_rpy(q):
    norm = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    if norm == 0:
        return (math.nan, math.nan, math.nan)
    w, x, y, z = (value / norm for value in (q.w, q.x, q.y, q.z))
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1, min(1, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def make_report(diagnostics, imu, imu_present: bool, imu_topic: str, output: Path,
                odometry=None, odometry_present: bool = False) -> None:
    import plotly.graph_objects as go
    import plotly.io as pio

    if not diagnostics:
        raise ValueError("Diagnostics topic exists but contains no messages")
    first = stamp_seconds(diagnostics[0].header.stamp)
    x = [stamp_seconds(m.header.stamp) - first for m in diagnostics]
    valid = [bool(m.update_valid) for m in diagnostics]
    groups = []

    def add_group(key, title, traces, y_title):
        figure = go.Figure()
        for name, times, values in traces:
            figure.add_trace(go.Scattergl(x=times, y=values, mode="lines", name=name))
        figure.update_layout(title=title, xaxis_title="Seconds from first FAST-LIO scan", yaxis_title=y_title,
                             hovermode="x unified", height=430, margin=dict(l=65, r=25, t=60, b=55))
        groups.append((key, title, figure))

    for field, title, unit in (("position_stddev", "Position standard deviation", "m"),
                               ("rotation_stddev", "Rotation standard deviation", "rad")):
        add_group(field, title, [(axis, x, [getattr(m, field)[i] for m in diagnostics])
                                 for i, axis in enumerate("xyz")], unit)
    odometry = odometry or []
    odometry_note = "Odometry topic absent; pose plots omitted."
    if odometry:
        ox = [stamp_seconds(m.header.stamp) - first for m in odometry]
        add_group("position", "Estimated position", [
            (axis, ox, [getattr(m.pose.pose.position, axis) for m in odometry])
            for axis in "xyz"], "m")
        rpy = [quaternion_to_rpy(m.pose.pose.orientation) for m in odometry]
        add_group("rotation", "Estimated rotation (roll, pitch, yaw)", [
            (axis, ox, [angles[i] for angles in rpy])
            for i, axis in enumerate(("roll", "pitch", "yaw"))], "rad")
        odometry_note = f"{len(odometry)} odometry poses."
    elif odometry_present:
        odometry_note = "Odometry topic exists but contains no poses."
    add_group("matches", "Match availability", [
        ("downsampled", x, [m.downsampled_points for m in diagnostics]),
        ("matched", x, [m.matched_points for m in diagnostics]),
    ], "points")
    add_group("match_ratio", "Match ratio", [("matched / downsampled", x, [
        m.matched_points / m.downsampled_points if m.update_valid and m.downsampled_points else math.nan
        for m in diagnostics])], "fraction")
    add_group("residual", "Mean absolute point-to-plane residual", [("residual", x, [
        m.mean_abs_residual if m.update_valid else math.nan for m in diagnostics])], "m")
    add_group("position_correction", "LiDAR position correction", [
        (axis, x, [m.lidar_position_correction[i] if m.update_valid else math.nan for m in diagnostics])
        for i, axis in enumerate("xyz")
    ] + [("magnitude", x, [math.sqrt(sum(v * v for v in m.lidar_position_correction))
                            if m.update_valid else math.nan for m in diagnostics])], "m")
    add_group("rotation_correction", "LiDAR rotation correction", [("angle", x, [
        m.lidar_rotation_correction if m.update_valid else math.nan for m in diagnostics])], "rad")
    add_group("update_valid", "LiDAR update applied", [("valid", x, [int(value) for value in valid])], "0 / 1")

    imu_note = "Body IMU topic absent; IMU plots omitted."
    if imu:
        wrong_frames = sorted({m.header.frame_id for m in imu if m.header.frame_id != "body"})
        if wrong_frames:
            raise ValueError(f"Refusing non-body IMU on {imu_topic}; frame_id values: {wrong_frames}")
        ix = [stamp_seconds(m.header.stamp) - first for m in imu]
        for key, attr, title, unit in (("accel", "linear_acceleration", "Body accelerometer", "m/s²"),
                                       ("gyro", "angular_velocity", "Body gyroscope", "rad/s")):
            add_group(key, title, [(axis, ix, [getattr(getattr(m, attr), axis) for m in imu])
                                   for axis in "xyz"] + [("magnitude", ix, [math.sqrt(sum(
                                       getattr(getattr(m, attr), axis) ** 2 for axis in "xyz")) for m in imu])], unit)
        imu_note = f"{len(imu)} raw body IMU samples from {imu_topic}. Saturation requires sensor range limits."
    elif imu_present:
        imu_note = f"{imu_topic} exists but contains no samples."

    controls = "".join(
        f'<label><input type="checkbox" checked onchange="document.getElementById(\'{key}\').hidden=!this.checked">{html.escape(title)}</label>'
        for key, title, _ in groups)
    plots = "".join(
        f'<section id="{key}">'
        + (f'<label class="plot-control">Smoothing window (samples): '
           f'<input type="number" min="1" step="1" value="1" '
           f'aria-label="Smoothing window for {html.escape(title, quote=True)}"></label>'
           if key != "update_valid" else "")
        + pio.to_html(fig, full_html=False, include_plotlyjs="inline" if i == 0 else False)
        + '</section>'
        for i, (key, title, fig) in enumerate(groups))
    total = len(diagnostics)
    summary = (f"{total} FAST-LIO scans; {sum(valid)} valid LiDAR updates; "
               f"{total - sum(valid)} invalid updates. {odometry_note} {imu_note}")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>FAST-LIO estimator diagnostics</title><style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#182333}}
.controls{{display:flex;flex-wrap:wrap;gap:.7rem 1.5rem;padding:1rem;background:#eef3f8;border-radius:.6rem}}
label{{white-space:nowrap}} section{{margin:1.2rem 0;border:1px solid #dae1e8;border-radius:.6rem}}
.plot-control{{display:block;margin:1rem 1rem 0}} .plot-control input{{width:5em}}
[hidden]{{display:none!important}}
</style></head><body><h1>FAST-LIO estimator diagnostics</h1>
<p>{html.escape(summary)}</p><p>Times use message timestamps relative to the first scan. These plots show
estimator health and consistency; absolute pose accuracy requires ground truth.</p>
<div class="controls">{controls}</div>{plots}</body></html>"""
    smoothing = """<script>
const rawPlots = [...document.querySelectorAll('section:not(#update_valid)')]
  .map(section => {
    const plot = section.querySelector('.plotly-graph-div');
    return {plot, input: section.querySelector('input'), raw: plot.data.map(trace => [...trace.y])};
  });
function smooth(values, windowSize) {
  const result = new Array(values.length);
  let sum = 0, count = 0;
  const queue = [];
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (value === null || !Number.isFinite(Number(value))) {
      sum = 0; count = 0; queue.length = 0; result[i] = null;
      continue;
    }
    queue.push(Number(value)); sum += Number(value); count++;
    if (queue.length > windowSize) { sum -= queue.shift(); count--; }
    result[i] = sum / count;
  }
  return result;
}
function updateSmoothing({plot, input, raw}) {
  const size = Math.max(1, Math.floor(Number(input.value) || 1));
  input.value = size;
  Plotly.restyle(plot, {y: raw.map(values => size === 1 ? values : smooth(values, size))});
}
for (const item of rawPlots) {
  item.input.addEventListener('change', () => updateSmoothing(item));
}
</script>"""
    document = document.replace('</body></html>',
        '<p>Smoothing uses a trailing moving average within each valid data segment. '
        'Set the window to 1 for raw data. LiDAR update status always shows raw values.</p>'
        + smoothing + '</body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(summary)
    print(f"Report: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path, help="ROS 2 bag directory containing metadata.yaml")
    parser.add_argument("--output", type=Path, help="HTML output path; defaults to BAG_DIR/fast_lio_report.html")
    parser.add_argument("--diagnostics-topic", default=DIAGNOSTICS_TOPIC)
    parser.add_argument("--imu-topic", default=BODY_IMU_TOPIC, help="Must carry sensor_msgs/Imu with frame_id=body")
    parser.add_argument("--odometry-topic", default=ODOMETRY_TOPIC, help="FAST-LIO nav_msgs/Odometry topic")
    args = parser.parse_args()
    try:
        diagnostics, imu, imu_present, odometry, odometry_present = read_bag(
            args.bag, args.diagnostics_topic, args.imu_topic, args.odometry_topic)
        make_report(diagnostics, imu, imu_present, args.imu_topic,
                    args.output or args.bag / "fast_lio_report.html", odometry, odometry_present)
    except (ValueError, ImportError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
