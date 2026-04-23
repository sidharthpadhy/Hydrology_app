from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

# D8 neighbor offsets ordered clockwise from north
D8_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


@dataclass
class HydroOutputs:
    watershed_mask: np.ndarray
    stream_mask: np.ndarray
    longest_stream_mask: np.ndarray
    flood_depths: Dict[int, np.ndarray]
    flow_accumulation: np.ndarray


def _neighbors(r: int, c: int, shape: Tuple[int, int]) -> Iterable[Tuple[int, int, int]]:
    rows, cols = shape
    for idx, (dr, dc) in enumerate(D8_OFFSETS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield idx, nr, nc


def fill_sinks(dem: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Lightweight pit filling for DEM preprocessing."""
    out = dem.copy().astype(np.float32)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        min_neighbors = np.full_like(out, np.inf)
        for dr, dc in D8_OFFSETS:
            window = padded[1 + dr : 1 + dr + out.shape[0], 1 + dc : 1 + dc + out.shape[1]]
            min_neighbors = np.minimum(min_neighbors, window)
        out = np.maximum(out, min_neighbors)
    return out


def flow_direction_d8(dem: np.ndarray) -> np.ndarray:
    """Return downstream neighbor index [0..7], -1 for pits/outside."""
    rows, cols = dem.shape
    direction = np.full((rows, cols), -1, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            current = dem[r, c]
            best_idx = -1
            best_drop = 0.0
            for idx, nr, nc in _neighbors(r, c, dem.shape):
                drop = current - dem[nr, nc]
                if drop > best_drop:
                    best_drop = drop
                    best_idx = idx
            direction[r, c] = best_idx
    return direction


def flow_accumulation(direction: np.ndarray) -> np.ndarray:
    """Simple D8 accumulation using elevation-order proxy."""
    rows, cols = direction.shape
    indegree = np.zeros((rows, cols), dtype=np.int32)
    downstream = np.full((rows, cols, 2), -1, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            idx = direction[r, c]
            if idx >= 0:
                dr, dc = D8_OFFSETS[idx]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    downstream[r, c] = (nr, nc)
                    indegree[nr, nc] += 1

    q = [(r, c) for r in range(rows) for c in range(cols) if indegree[r, c] == 0]
    acc = np.ones((rows, cols), dtype=np.float32)

    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        nr, nc = downstream[r, c]
        if nr >= 0:
            acc[nr, nc] += acc[r, c]
            indegree[nr, nc] -= 1
            if indegree[nr, nc] == 0:
                q.append((nr, nc))

    return acc


def stream_network(accumulation: np.ndarray, threshold: float | None = None) -> np.ndarray:
    if threshold is None:
        threshold = np.percentile(accumulation, 92)
    return accumulation >= threshold


def watershed_from_outlet(direction: np.ndarray, outlet: Tuple[int, int]) -> np.ndarray:
    rows, cols = direction.shape
    upstream_graph = [[[] for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            idx = direction[r, c]
            if idx >= 0:
                dr, dc = D8_OFFSETS[idx]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    upstream_graph[nr][nc].append((r, c))

    mask = np.zeros((rows, cols), dtype=bool)
    stack = [outlet]
    while stack:
        r, c = stack.pop()
        if mask[r, c]:
            continue
        mask[r, c] = True
        stack.extend(upstream_graph[r][c])
    return mask


def longest_stream_path(direction: np.ndarray, stream_mask: np.ndarray) -> np.ndarray:
    rows, cols = direction.shape
    upstream_count = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            if not stream_mask[r, c]:
                continue
            idx = direction[r, c]
            if idx >= 0:
                dr, dc = D8_OFFSETS[idx]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and stream_mask[nr, nc]:
                    upstream_count[nr, nc] += 1

    sources = [(r, c) for r in range(rows) for c in range(cols) if stream_mask[r, c] and upstream_count[r, c] == 0]
    best_path: list[Tuple[int, int]] = []

    for src in sources:
        path = [src]
        r, c = src
        while True:
            idx = direction[r, c]
            if idx < 0:
                break
            dr, dc = D8_OFFSETS[idx]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or not stream_mask[nr, nc]:
                break
            path.append((nr, nc))
            r, c = nr, nc
        if len(path) > len(best_path):
            best_path = path

    mask = np.zeros((rows, cols), dtype=bool)
    for r, c in best_path:
        mask[r, c] = True
    return mask


def flood_depth_for_return_periods(
    dem: np.ndarray,
    slope_proxy: np.ndarray,
    rainfall_depths_mm: Dict[int, float],
    lulc: np.ndarray | None = None,
    soil: np.ndarray | None = None,
) -> Dict[int, np.ndarray]:
    lulc_coeff = 0.55 if lulc is None else np.clip(0.3 + (lulc.astype(np.float32) / max(1.0, float(lulc.max()))) * 0.5, 0.2, 0.9)
    soil_infil = 0.25 if soil is None else np.clip((soil.astype(np.float32) / max(1.0, float(soil.max()))) * 0.45, 0.05, 0.45)

    depths: Dict[int, np.ndarray] = {}
    for rp, rain_mm in rainfall_depths_mm.items():
        runoff_mm = rain_mm * lulc_coeff * (1.0 - soil_infil)
        terrain_loss = np.clip(slope_proxy * 120.0, 0.0, 40.0)
        flood_mm = np.maximum(runoff_mm - terrain_loss, 0.0)
        depths[rp] = (flood_mm / 1000.0).astype(np.float32)  # meters
    return depths


def run_hydrology(
    dem: np.ndarray,
    rainfall_depths_mm: Dict[int, float],
    lulc: np.ndarray | None = None,
    soil: np.ndarray | None = None,
) -> HydroOutputs:
    filled = fill_sinks(dem)
    direction = flow_direction_d8(filled)
    accumulation = flow_accumulation(direction)
    streams = stream_network(accumulation)

    outlet = np.unravel_index(np.argmax(accumulation), accumulation.shape)
    watershed = watershed_from_outlet(direction, outlet)
    longest = longest_stream_path(direction, streams)

    gy, gx = np.gradient(filled)
    slope = np.sqrt(gx**2 + gy**2)
    flood_depths = flood_depth_for_return_periods(filled, slope, rainfall_depths_mm, lulc, soil)

    return HydroOutputs(
        watershed_mask=watershed,
        stream_mask=streams,
        longest_stream_mask=longest,
        flood_depths=flood_depths,
        flow_accumulation=accumulation,
    )
