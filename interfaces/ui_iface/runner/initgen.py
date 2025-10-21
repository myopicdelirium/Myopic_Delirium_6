import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, distance_transform_edt
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
def _fgauss(h: int, w: int, scale: float, g: np.random.Generator) -> NDArray[np.float32]:
    x = g.standard_normal((h, w)).astype(np.float32)
    s = max(1.0, scale / 8.0)
    return gaussian_filter(x, s, mode="wrap").astype(np.float32)
def elevation(h: int, w: int, params: dict, seed: int) -> NDArray[np.float32]:
    g = _rng(seed)
    octaves = int(params.get("octaves", 4))
    base_scale = float(params.get("elevation_scale", 96.0))
    ridge_strength = float(params.get("ridge_strength", 0.4))
    e = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for i in range(octaves):
        scale = base_scale / (2 ** i)
        e += amp * _fgauss(h, w, scale, g)
        amp *= 0.5
    e = (e - e.min()) / (e.max() - e.min() + 1e-8)
    r = 1.0 - np.abs(2.0 * e - 1.0)
    e = (1.0 - ridge_strength) * e + ridge_strength * r
    ky = int(max(1, base_scale / 6))
    kx = int(max(1, base_scale / 6))
    e = gaussian_filter(e, (ky, kx), mode="wrap").astype(np.float32)
    e = (e - e.min()) / (e.max() - e.min() + 1e-8)
    return e
def precipitation(h: int, w: int, params: dict, seed: int, elevation_raster: NDArray[np.float32]) -> NDArray[np.float32]:
    g = _rng(seed)
    scale = float(params.get("precipitation_scale", 128.0))
    p = _fgauss(h, w, scale, g)
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    elev_norm = (elevation_raster - elevation_raster.min()) / (elevation_raster.max() - elevation_raster.min() + 1e-8)
    wind = np.linspace(0.2, 1.0, w, dtype=np.float32)[None, :]
    orog = (1.0 - elev_norm) * 0.4 + wind * 0.6
    p = 0.6 * p + 0.4 * orog
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    return p.astype(np.float32)
def flow_accumulation(E: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    h, w = E.shape
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    flow_to = np.zeros((h, w, 2), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            min_e = E[y, x]
            ty, tx = y, x
            for dy, dx in offsets:
                ny = (y + dy) % h
                nx = (x + dx) % w
                if E[ny, nx] < min_e:
                    min_e = E[ny, nx]
                    ty, tx = ny, nx
            flow_to[y, x, 0] = ty
            flow_to[y, x, 1] = tx
    indeg = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            ty, tx = flow_to[y, x]
            if ty != y or tx != x:
                indeg[ty, tx] += 1
    from collections import deque
    q = deque()
    for y in range(h):
        for x in range(w):
            if indeg[y, x] == 0:
                q.append((y, x))
    acc = np.ones((h, w), dtype=np.float32)
    visited = np.zeros((h, w), dtype=np.bool_)
    while q:
        y, x = q.popleft()
        visited[y, x] = True
        ty, tx = flow_to[y, x]
        if ty == y and tx == x:
            continue
        acc[ty, tx] += acc[y, x]
        indeg[ty, tx] -= 1
        if indeg[ty, tx] == 0:
            q.append((ty, tx))
    closed = ~visited
    return acc, closed
def lakes(E: NDArray[np.float32], A: NDArray[np.float32], threshold: float) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    h, w = E.shape
    mask = np.zeros((h, w), dtype=np.bool_)
    filled = E.copy()
    border = []
    for y in range(h):
        for x in range(w):
            if y in (0, h-1) or x in (0, w-1):
                border.append((E[y, x], y, x))
    import heapq
    heapq.heapify(border)
    water = np.full((h, w), np.inf, dtype=np.float32)
    while border:
        e, y, x = heapq.heappop(border)
        if water[y, x] <= e:
            continue
        water[y, x] = e
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny = (y + dy) % h
            nx = (x + dx) % w
            we = max(e, E[ny, nx])
            if we < water[ny, nx]:
                heapq.heappush(border, (we, ny, nx))
    lake_level = water
    lake_mask = lake_level > E
    inc = np.percentile(A, 100.0 * (1.0 - threshold))
    lake_mask |= A >= inc
    filled = np.where(lake_mask, lake_level, E).astype(np.float32)
    return lake_mask, filled
def hydration_from_hydrology(E: NDArray[np.float32], A: NDArray[np.float32], lake_mask: NDArray[np.bool_], params: dict) -> NDArray[np.float32]:
    h, w = E.shape
    river_thresh = float(params.get("river_percentile", 0.88))
    lake_thresh = float(params.get("lake_fill_threshold", 0.2))
    base_moisture = float(params.get("base_moisture", 0.3))
    river_depth = float(params.get("river_depth", 0.9))
    lake_depth = float(params.get("lake_depth", 1.0))
    
    river_thr = np.percentile(A, 100.0 * river_thresh)
    rivers = A >= river_thr
    
    lake_thr = np.percentile(A, 100.0 * (1.0 - lake_thresh))
    lakes_major = A >= lake_thr
    
    h2o = np.full((h, w), base_moisture, dtype=np.float32)
    
    river_dist = distance_transform_edt(~rivers)
    river_influence = np.exp(-river_dist / 12.0)
    h2o += river_influence * (river_depth - base_moisture)
    
    lake_dist = distance_transform_edt(~lakes_major)
    lake_influence = np.exp(-lake_dist / 20.0)
    h2o += lake_influence * (lake_depth - base_moisture)
    
    elev_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)
    lowland_bonus = (1.0 - elev_norm) * 0.15
    h2o += lowland_bonus
    
    h2o = gaussian_filter(h2o, sigma=3.0, mode='wrap')
    
    h2o = np.clip(h2o, 0.0, 1.0).astype(np.float32)
    
    return h2o
def temperature_meridional(h: int, w: int, params: dict, seed: int) -> NDArray[np.float32]:
    g = _rng(seed)
    amp = float(params.get("amplitude", 0.7))
    noise_amp = float(params.get("noise_amp", 0.05))
    y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
    distance_from_equator = np.abs(y_coords - 0.5) * 2.0
    base_temp = 1.0 - distance_from_equator
    base_temp = 0.5 + amp * (base_temp - 0.5)
    noise = gaussian_filter(g.standard_normal((h, w)).astype(np.float32), 4.0, mode="wrap") * noise_amp
    grad = np.tile(base_temp[:, None], (1, w))
    t = grad + noise
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    return t
def vegetation_init(H2O: NDArray[np.float32], T: NDArray[np.float32], params: dict, seed: int) -> NDArray[np.float32]:
    k = float(params.get("k", 0.08))
    water_half = float(params.get("water_half", 0.35))
    opt = float(params.get("heat_optimum", 0.65))
    sigma = float(params.get("heat_sigma", 0.18))
    K = float(params.get("carrying_capacity", 1.0))
    sw = H2O / (H2O + water_half + 1e-8)
    st = np.exp(-0.5 * ((T - opt) / (sigma + 1e-8)) ** 2)
    g = _rng(seed)
    noise = gaussian_filter(g.standard_normal(H2O.shape).astype(np.float32), 2.0, mode="wrap") * 0.01
    v0 = K * sw * st
    v0 = np.clip(v0 + noise, 0.0, 1.0).astype(np.float32)
    return v0
