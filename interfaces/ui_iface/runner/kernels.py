import numpy as np
from numba import njit
@njit(cache=True, fastmath=True)
def laplacian5(arr, wrapx, wrapy):
    h, w = arr.shape
    out = np.empty_like(arr)
    for y in range(h):
        ym1 = (y - 1 + h) % h if wrapy else max(y - 1, 0)
        yp1 = (y + 1) % h if wrapy else min(y + 1, h - 1)
        for x in range(w):
            xm1 = (x - 1 + w) % w if wrapx else max(x - 1, 0)
            xp1 = (x + 1) % w if wrapx else min(x + 1, w - 1)
            c = arr[y, x]
            out[y, x] = arr[ym1, x] + arr[yp1, x] + arr[y, xm1] + arr[y, xp1] - 4.0 * c
    return out
@njit(cache=True, fastmath=True)
def advect(arr, vx, vy, wrapx, wrapy):
    h, w = arr.shape
    out = np.empty_like(arr)
    for y in range(h):
        for x in range(w):
            fx = x - vx
            fy = y - vy
            if wrapx:
                fx = fx % w
            else:
                fx = min(max(fx, 0.0), w - 1.001)
            if wrapy:
                fy = fy % h
            else:
                fy = min(max(fy, 0.0), h - 1.001)
            x0 = int(np.floor(fx))
            y0 = int(np.floor(fy))
            x1 = (x0 + 1) % w if wrapx else min(x0 + 1, w - 1)
            y1 = (y0 + 1) % h if wrapy else min(y0 + 1, h - 1)
            sx = fx - x0
            sy = fy - y0
            v00 = arr[y0, x0]
            v10 = arr[y0, x1]
            v01 = arr[y1, x0]
            v11 = arr[y1, x1]
            out[y, x] = (1 - sx) * (1 - sy) * v00 + sx * (1 - sy) * v10 + (1 - sx) * sy * v01 + sx * sy * v11
    return out
def step_kernels(tensor, cfg, registry, wrapx, wrapy, noise_rng):
    names = registry["names"]
    coeffs = registry["coeffs"]
    derived = registry["derived"]
    h, w, f = tensor.shape
    new = tensor.copy()
    for i in range(f):
        if derived[i]:
            continue
        c = coeffs[i]
        arr = new[:, :, i]
        d = float(c.get("diffusion", 0.0))
        if d != 0.0:
            arr = arr + d * laplacian5(arr, wrapx, wrapy)
        adv = c.get("advection", {})
        vx = float(adv.get("vx", 0.0))
        vy = float(adv.get("vy", 0.0))
        if vx != 0.0 or vy != 0.0:
            arr = advect(arr, vx, vy, wrapx, wrapy)
        new[:, :, i] = arr
    t_idx = registry["indices"].get("temperature", None)
    h_idx = registry["indices"].get("hydration", None)
    v_idx = registry["indices"].get("vegetation", None)
    if t_idx is not None and h_idx is not None:
        evap = 0.005
        new[:, :, h_idx] = np.clip(new[:, :, h_idx] - evap * np.clip(new[:, :, t_idx], 0.0, 1.0), 0.0, 1.0)
    if v_idx is not None and h_idx is not None and t_idx is not None:
        k = float(cfg["vegetation_profile"].get("k", 0.08))
        water_half = float(cfg["vegetation_profile"].get("water_half", 0.35))
        opt = float(cfg["vegetation_profile"].get("heat_optimum", 0.65))
        sigma = float(cfg["vegetation_profile"].get("heat_sigma", 0.18))
        K = float(cfg["vegetation_profile"].get("carrying_capacity", 1.0))
        H = new[:, :, h_idx]
        T = new[:, :, t_idx]
        V = new[:, :, v_idx]
        sw = H / (H + water_half + 1e-8)
        st = np.exp(-0.5 * ((T - opt) / (sigma + 1e-8)) ** 2)
        growth = k * V * (1.0 - V / (K + 1e-8)) * sw * st
        consume = 0.5 * growth
        new[:, :, v_idx] = np.clip(V + growth, 0.0, 1.0)
        new[:, :, h_idx] = np.clip(H - consume, 0.0, 1.0)
    for i in range(f):
        if derived[i]:
            continue
        c = registry["coeffs"][i]
        dec = float(c.get("decay", 0.0))
        rep = float(c.get("replenish", 0.0))
        if dec != 0.0:
            new[:, :, i] = new[:, :, i] * (1.0 - dec)
        if rep != 0.0:
            new[:, :, i] = np.clip(new[:, :, i] + rep, 0.0, 1.0)
        lo, hi = registry["bounds"][i]
        new[:, :, i] = np.clip(new[:, :, i], lo, hi)
    if "movement_cost" in names:
        idx = registry["indices"]["movement_cost"]
        hi = new[:, :, registry["indices"]["hydration"]] if "hydration" in names else np.zeros((h, w))
        ve = new[:, :, registry["indices"]["vegetation"]] if "vegetation" in names else np.zeros((h, w))
        mc = np.clip(0.3 + 0.5 * ve + 0.2 * (1.0 - hi), 0.0, 1.0)
        new[:, :, idx] = mc
    return new
