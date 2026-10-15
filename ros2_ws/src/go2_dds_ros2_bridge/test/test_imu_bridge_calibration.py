from pathlib import Path

import pytest
import yaml

from go2_dds_ros2_bridge.nodes.imu_bridge_node import (
    calibrate_vector, load_imu_calibration, scale_covariance,
)


CONFIG = Path(__file__).resolve().parents[1] / 'config/imu_bridge_fastlio.yaml'


def test_measured_body_calibration_and_covariance():
    acc_bias, acc_scale, gyro_bias, gyro_scale = load_imu_calibration(CONFIG)
    assert len(acc_bias) == len(gyro_bias) == 3
    assert calibrate_vector(acc_bias, acc_bias, acc_scale) == (0.0, 0.0, 0.0)
    assert calibrate_vector(gyro_bias, gyro_bias, gyro_scale) == (0.0, 0.0, 0.0)
    covariance = [2.0, 0.5, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 4.0]
    scaled = scale_covariance(covariance, acc_scale)
    assert scaled[0] == pytest.approx(2.0 * acc_scale[0] ** 2)
    assert scaled[1] == pytest.approx(0.5 * acc_scale[0] * acc_scale[1])
    assert scale_covariance([-1.0] + [0.0] * 8, gyro_scale)[0] == -1.0


def test_rejects_wrong_frame_and_missing_scale(tmp_path):
    report = yaml.safe_load(CONFIG.read_text())
    report['calibration']['frame_id'] = 'lidar'
    path = tmp_path / 'calibration.yaml'
    path.write_text(yaml.safe_dump(report))
    with pytest.raises(SystemExit, match='body'):
        load_imu_calibration(path)
    report['calibration']['frame_id'] = 'body'
    report['calibration']['gyroscope']['scale'] = [1.0, 0.0, 1.0]
    path.write_text(yaml.safe_dump(report))
    with pytest.raises(SystemExit, match='greater than zero'):
        load_imu_calibration(path)
