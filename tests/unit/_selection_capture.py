"""Deterministic snapshot of an identification run, for #61 regressions."""

import dataclasses
import json

import numpy as np


def _plain(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.dtype.kind in "fiu" and arr.size > 12:
            flat = arr.astype(float).ravel()
            return {
                "shape": list(arr.shape),
                "sum": float(flat.sum()),
                "sumsq": float(np.sum(flat**2)),
                "first": float(flat[0]),
                "last": float(flat[-1]),
            }
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    return repr(type(x).__name__)  # objects: only their type is compared


def capture(ident):
    verdict = dataclasses.asdict(ident.verify(scope="execution"))
    verdict.pop("metadata", None)
    result = {k: v for k, v in ident.result.items() if k != "identification config"}
    return json.loads(
        json.dumps(
            _plain({"result": result, "verdict": verdict}), allow_nan=True
        )
    )
