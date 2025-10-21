from typing import Dict, Any, List, Tuple
def build_registry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    fields = cfg["fields"]
    names: List[str] = [f["name"] for f in fields]
    indices: Dict[str, int] = {n: i for i, n in enumerate(names)}
    bounds: List[Tuple[float, float]] = [tuple(f["bounds"]) for f in fields]
    coeffs: List[Dict[str, Any]] = [f.get("coeffs", {}) for f in fields]
    derived: List[bool] = [bool(f.get("derived", False)) for f in fields]
    return {"names": names, "indices": indices, "bounds": bounds, "coeffs": coeffs, "derived": derived}
