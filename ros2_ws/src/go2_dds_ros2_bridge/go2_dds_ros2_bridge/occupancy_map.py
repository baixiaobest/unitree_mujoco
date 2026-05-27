from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator


UNKNOWN_CELL = -1
POINT_FIELD_DTYPES = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}


@dataclass(frozen=True)
class OccupancyMapSnapshot:
    resolution_m: float
    width: int
    height: int
    origin_x_m: float
    origin_y_m: float
    data: np.ndarray


def dtype_from_fields(fields: list[PointField], point_step: int, is_bigendian: bool) -> np.dtype:
    names: list[str] = []
    formats: list[np.dtype] = []
    offsets: list[int] = []
    byte_order = ">" if is_bigendian else "<"

    for index, field in enumerate(fields):
        base_dtype = POINT_FIELD_DTYPES.get(field.datatype)
        if base_dtype is None:
            raise ValueError(f"Unsupported PointField datatype: {field.datatype}")
        field_dtype = base_dtype.newbyteorder(byte_order)
        if field.count > 1:
            field_dtype = np.dtype((field_dtype, field.count))
        names.append(field.name or f"unnamed_field_{index}")
        formats.append(field_dtype)
        offsets.append(int(field.offset))

    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(point_step),
        }
    )


def extract_xyz_points(cloud_msg: PointCloud2) -> np.ndarray:
    if cloud_msg.width <= 0 or cloud_msg.height <= 0 or not cloud_msg.data:
        return np.empty((0, 3), dtype=np.float64)

    cloud_dtype = dtype_from_fields(cloud_msg.fields, cloud_msg.point_step, cloud_msg.is_bigendian)
    packed_row_step = int(cloud_msg.width) * int(cloud_msg.point_step)
    row_step = int(cloud_msg.row_step) if cloud_msg.row_step > 0 else packed_row_step
    if row_step < packed_row_step:
        raise ValueError(f"row_step={row_step} is smaller than width * point_step={packed_row_step}")

    expected_buffer_size = row_step * int(cloud_msg.height)
    if len(cloud_msg.data) < expected_buffer_size:
        raise ValueError(
            f"PointCloud2 data buffer is too small: len(data)={len(cloud_msg.data)} expected>={expected_buffer_size}"
        )

    cloud = np.ndarray(
        shape=(int(cloud_msg.height), int(cloud_msg.width)),
        dtype=cloud_dtype,
        buffer=cloud_msg.data,
        strides=(row_step, int(cloud_msg.point_step)),
    )
    packed_cloud = np.array(cloud.reshape(-1), copy=False)
    field_names = packed_cloud.dtype.names or ()
    for field_name in ("x", "y", "z"):
        if field_name not in field_names:
            raise ValueError(f"Point cloud is missing required '{field_name}' field")

    x_values = np.asarray(packed_cloud["x"], dtype=np.float64)
    y_values = np.asarray(packed_cloud["y"], dtype=np.float64)
    z_values = np.asarray(packed_cloud["z"], dtype=np.float64)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
    if not np.any(finite_mask):
        return np.empty((0, 3), dtype=np.float64)

    return np.ascontiguousarray(
        np.column_stack((x_values[finite_mask], y_values[finite_mask], z_values[finite_mask])), dtype=np.float64
    )


@njit(cache=True)
def _clear_mask(mask: np.ndarray) -> None:
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            mask[row, col] = 0


@njit(cache=True)
def _shift_grid(grid: np.ndarray, shift_x_cells: int, shift_y_cells: int, fill_value: float) -> np.ndarray:
    shifted = np.empty_like(grid)
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            shifted[row, col] = fill_value

    for row in range(height):
        source_row = row + shift_y_cells
        if source_row < 0 or source_row >= height:
            continue
        for col in range(width):
            source_col = col + shift_x_cells
            if source_col < 0 or source_col >= width:
                continue
            shifted[row, col] = grid[source_row, source_col]
    return shifted


@njit(cache=True)
def _rasterize_scan(
    points_xy: np.ndarray,
    ray_origin_x_m: float,
    ray_origin_y_m: float,
    map_origin_x_m: float,
    map_origin_y_m: float,
    resolution_m: float,
    width: int,
    height: int,
    free_mask: np.ndarray,
    hit_mask: np.ndarray,
) -> None:
    start_col = int(math.floor((ray_origin_x_m - map_origin_x_m) / resolution_m))
    start_row = int(math.floor((ray_origin_y_m - map_origin_y_m) / resolution_m))
    if start_col < 0 or start_col >= width or start_row < 0 or start_row >= height:
        return

    for point_index in range(points_xy.shape[0]):
        end_col = int(math.floor((points_xy[point_index, 0] - map_origin_x_m) / resolution_m))
        end_row = int(math.floor((points_xy[point_index, 1] - map_origin_y_m) / resolution_m))
        if end_col < 0 or end_col >= width or end_row < 0 or end_row >= height:
            continue

        delta_col = abs(end_col - start_col)
        step_col = 1 if start_col < end_col else -1
        delta_row = -abs(end_row - start_row)
        step_row = 1 if start_row < end_row else -1
        error = delta_col + delta_row

        current_col = start_col
        current_row = start_row
        while current_col != end_col or current_row != end_row:
            free_mask[current_row, current_col] = 1
            doubled_error = 2 * error
            if doubled_error >= delta_row:
                error += delta_row
                current_col += step_col
            if doubled_error <= delta_col:
                error += delta_col
                current_row += step_row

        hit_mask[end_row, end_col] = 1


@njit(cache=True)
def _decay_log_odds(log_odds_grid: np.ndarray, observed_mask: np.ndarray, decay_factor: float) -> None:
    for row in range(log_odds_grid.shape[0]):
        for col in range(log_odds_grid.shape[1]):
            if observed_mask[row, col] != 0:
                log_odds_grid[row, col] = log_odds_grid[row, col] * decay_factor


@njit(cache=True)
def _apply_scan(
    log_odds_grid: np.ndarray,
    observed_mask: np.ndarray,
    free_mask: np.ndarray,
    hit_mask: np.ndarray,
    hit_log_odds_increment: float,
    miss_log_odds_decrement: float,
    min_log_odds: float,
    max_log_odds: float,
) -> None:
    for row in range(log_odds_grid.shape[0]):
        for col in range(log_odds_grid.shape[1]):
            if hit_mask[row, col] != 0:
                updated_value = log_odds_grid[row, col] + hit_log_odds_increment
                if updated_value > max_log_odds:
                    updated_value = max_log_odds
                log_odds_grid[row, col] = updated_value
                observed_mask[row, col] = 1
            elif free_mask[row, col] != 0:
                updated_value = log_odds_grid[row, col] - miss_log_odds_decrement
                if updated_value < min_log_odds:
                    updated_value = min_log_odds
                log_odds_grid[row, col] = updated_value
                observed_mask[row, col] = 1


class RollingOccupancyMap:
    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        resolution_m: float,
        hit_log_odds_increment: float,
        miss_log_odds_decrement: float,
        decay_factor: float,
        min_log_odds: float,
        max_log_odds: float,
    ) -> None:
        if width_m <= 0.0 or height_m <= 0.0 or resolution_m <= 0.0:
            raise ValueError("width_m, height_m, and resolution_m must be positive")
        if hit_log_odds_increment <= 0.0:
            raise ValueError("hit_log_odds_increment must be positive")
        if miss_log_odds_decrement <= 0.0:
            raise ValueError("miss_log_odds_decrement must be positive")
        if decay_factor <= 0.0 or decay_factor > 1.0:
            raise ValueError("decay_factor must be in the interval (0, 1]")
        if min_log_odds >= max_log_odds:
            raise ValueError("min_log_odds must be smaller than max_log_odds")

        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.resolution_m = float(resolution_m)
        self.hit_log_odds_increment = float(hit_log_odds_increment)
        self.miss_log_odds_decrement = float(miss_log_odds_decrement)
        self.decay_factor = float(decay_factor)
        self.min_log_odds = float(min_log_odds)
        self.max_log_odds = float(max_log_odds)
        self.width = int(round(self.width_m / self.resolution_m))
        self.height = int(round(self.height_m / self.resolution_m))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width_m and height_m must produce at least one cell")

        self._log_odds_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self._observed_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self._free_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self._hit_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self._center_cell_x: int | None = None
        self._center_cell_y: int | None = None
        self._center_x_m = 0.0
        self._center_y_m = 0.0
        self._origin_x_m = 0.0
        self._origin_y_m = 0.0

    @staticmethod
    def _snap_center_to_cell(value_m: float, resolution_m: float) -> tuple[int, float]:
        center_cell = int(math.floor(value_m / resolution_m))
        center_value_m = (center_cell + 0.5) * resolution_m
        return center_cell, center_value_m

    def _set_center(self, center_x_m: float, center_y_m: float) -> None:
        self._center_cell_x, self._center_x_m = self._snap_center_to_cell(center_x_m, self.resolution_m)
        self._center_cell_y, self._center_y_m = self._snap_center_to_cell(center_y_m, self.resolution_m)
        self._origin_x_m = self._center_x_m - 0.5 * self.width * self.resolution_m
        self._origin_y_m = self._center_y_m - 0.5 * self.height * self.resolution_m

    def update_window(self, *, center_x_m: float, center_y_m: float) -> None:
        if self._center_cell_x is None or self._center_cell_y is None:
            self._set_center(center_x_m, center_y_m)
            return

        new_center_cell_x, new_center_x_m = self._snap_center_to_cell(center_x_m, self.resolution_m)
        new_center_cell_y, new_center_y_m = self._snap_center_to_cell(center_y_m, self.resolution_m)
        shift_x_cells = new_center_cell_x - self._center_cell_x
        shift_y_cells = new_center_cell_y - self._center_cell_y
        if shift_x_cells == 0 and shift_y_cells == 0:
            return

        self._log_odds_grid = _shift_grid(self._log_odds_grid, shift_x_cells, shift_y_cells, 0.0)
        self._observed_mask = _shift_grid(self._observed_mask, shift_x_cells, shift_y_cells, 0)
        self._center_cell_x = new_center_cell_x
        self._center_cell_y = new_center_cell_y
        self._center_x_m = new_center_x_m
        self._center_y_m = new_center_y_m
        self._origin_x_m = self._center_x_m - 0.5 * self.width * self.resolution_m
        self._origin_y_m = self._center_y_m - 0.5 * self.height * self.resolution_m

    def integrate_point_cloud(
        self,
        *,
        points_xyz_m: np.ndarray,
        ray_origin_xy_m: np.ndarray,
        map_center_xy_m: np.ndarray,
    ) -> None:
        self.update_window(center_x_m=float(map_center_xy_m[0]), center_y_m=float(map_center_xy_m[1]))
        _clear_mask(self._free_mask)
        _clear_mask(self._hit_mask)
        _decay_log_odds(self._log_odds_grid, self._observed_mask, self.decay_factor)
        if points_xyz_m.size == 0:
            return

        points_xy = np.ascontiguousarray(points_xyz_m[:, :2], dtype=np.float64)
        _rasterize_scan(
            points_xy,
            float(ray_origin_xy_m[0]),
            float(ray_origin_xy_m[1]),
            self._origin_x_m,
            self._origin_y_m,
            self.resolution_m,
            self.width,
            self.height,
            self._free_mask,
            self._hit_mask,
        )
        _apply_scan(
            self._log_odds_grid,
            self._observed_mask,
            self._free_mask,
            self._hit_mask,
            self.hit_log_odds_increment,
            self.miss_log_odds_decrement,
            self.min_log_odds,
            self.max_log_odds,
        )

    def _occupancy_data(self) -> np.ndarray:
        occupancy = np.full((self.height, self.width), UNKNOWN_CELL, dtype=np.int8)
        observed = self._observed_mask != 0
        if not np.any(observed):
            return occupancy

        probabilities = 1.0 / (1.0 + np.exp(-self._log_odds_grid[observed]))
        occupancy[observed] = np.clip(np.rint(probabilities * 100.0), 0, 100).astype(np.int8)
        return occupancy

    def snapshot(self) -> OccupancyMapSnapshot:
        return OccupancyMapSnapshot(
            resolution_m=self.resolution_m,
            width=self.width,
            height=self.height,
            origin_x_m=self._origin_x_m,
            origin_y_m=self._origin_y_m,
            data=self._occupancy_data(),
        )