from __future__ import annotations

import math
from pathlib import Path


STATE_ORDER = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
    "vz",
    "roll_dt",
    "pitch_dt",
    "yaw_dt",
)

STATE_ALIASES = {
    "x_dt": "vx",
    "y_dt": "vy",
    "z_dt": "vz",
    "vroll": "roll_dt",
    "vpitch": "pitch_dt",
    "vyaw": "yaw_dt",
}

POSE_STATE_TO_INDEX = {
    "x": 0,
    "y": 7,
    "z": 14,
    "roll": 21,
    "pitch": 28,
    "yaw": 35,
}

TWIST_STATE_TO_INDEX = {
    "vx": 0,
    "vy": 7,
    "vz": 14,
    "roll_dt": 21,
    "pitch_dt": 28,
    "yaw_dt": 35,
}

VARIANCE_SECTION_KEYS = ("variances", "variance", "covariances", "covariance")


def _import_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise SystemExit(
            "The ROS2 bridge runtime cannot import 'yaml'.\n"
            "Install python3-yaml into the same Python interpreter that runs ROS2."
        ) from error
    return yaml


def _coerce_variance_mapping(data: object) -> dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError("Covariance YAML must contain a mapping of state names to variances.")

    section = data
    for section_key in VARIANCE_SECTION_KEYS:
        candidate = data.get(section_key)
        if candidate is not None:
            section = candidate
            break

    if not isinstance(section, dict):
        raise ValueError("Covariance YAML variance section must be a mapping.")

    normalized: dict[str, float] = {}
    for key, value in section.items():
        if not isinstance(key, str):
            raise ValueError("Covariance YAML state keys must be strings.")
        canonical_key = STATE_ALIASES.get(key, key)
        if canonical_key not in STATE_ORDER:
            valid_names = ", ".join(STATE_ORDER + tuple(STATE_ALIASES.keys()))
            raise ValueError(f"Unsupported covariance state '{key}'. Supported names: {valid_names}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Covariance value for '{key}' must be numeric.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f"Covariance value for '{key}' must be finite and non-negative.")
        normalized[canonical_key] = numeric_value
    return normalized


def load_state_variances(covariance_file: Path | None, default_variances: dict[str, float]) -> tuple[dict[str, float], str]:
    variances = dict(default_variances)
    if covariance_file is None:
        return variances, "built-in defaults"

    yaml = _import_yaml_module()
    file_path = Path(covariance_file)
    loaded_data = yaml.safe_load(file_path.read_text())
    if loaded_data is None:
        return variances, str(file_path)

    variances.update(_coerce_variance_mapping(loaded_data))
    return variances, str(file_path)


def build_pose_covariance(variances: dict[str, float]) -> list[float]:
    covariance = [0.0] * 36
    for state_name, index in POSE_STATE_TO_INDEX.items():
        covariance[index] = float(variances[state_name])
    return covariance


def build_twist_covariance(variances: dict[str, float]) -> list[float]:
    covariance = [0.0] * 36
    for state_name, index in TWIST_STATE_TO_INDEX.items():
        covariance[index] = float(variances[state_name])
    return covariance