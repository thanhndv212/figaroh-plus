"""Private issue #30 confirmation run of the revised log-Cholesky protocol (D5).

Imports the frozen issue #22 spike unchanged and reruns it with the revised
protocol of docs/decisions/log-cholesky-convergence.md: only the evaluation
budget changes, and convergence is judged by the solver's termination status.
The exploration seed set is a replication; only the confirmation sets, whose
seeds were declared before any run, count towards the go/no-go gates.

Run (figaroh-dev, one BLAS thread), once per Pinocchio profile:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
        docs/development/spikes/log_cholesky_convergence.py --output results.json
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "log_cholesky_convergence_diagnosis",
    _HERE / "log_cholesky_convergence_diagnosis.py",
)
diagnosis = importlib.util.module_from_spec(_spec)
sys.modules["log_cholesky_convergence_diagnosis"] = diagnosis
_spec.loader.exec_module(diagnosis)
base = diagnosis.base

# Frozen before any confirmation run. Everything not listed keeps the #22 value.
PROTOCOL = {
    "max_nfev": 2000,
    "converged_status": [1, 2, 4],
    "gated_starts": ["nominal", "repaired_ols"],
    "bound_activity_tolerance": 1e-6,
}
SEED_SETS = {
    "exploration": [2201, 2202, 2203, 2204, 2205, 2206],
    "confirmation_a": [3001, 3002, 3003, 3004, 3005, 3006],
    "confirmation_b": [3101, 3102, 3103, 3104, 3105, 3106],
    "confirmation_c": [3201, 3202, 3203, 3204, 3205, 3206],
}
CONFIRMATION = [name for name in SEED_SETS if name.startswith("confirmation")]


def seeds(values):
    model, train, validation, noise, *starts = values
    return {
        "seed_model": model,
        "seed_train": train,
        "seed_validation": validation,
        "seed_noise": noise,
        "seed_starts": starts,
    }


def bound_active(p):
    """True when any decoded block sits on the experiment's coordinate bounds."""
    limit = base.CONFIG["coordinate_bounds"][1] - PROTOCOL["bound_activity_tolerance"]
    try:
        z = np.concatenate([base.encode(x) for x in p.reshape(-1, 10)])
    except np.linalg.LinAlgError:
        return None
    return bool(np.max(np.abs(z)) >= limit)


def annotate(record, fixture, ridge):
    """Add convergence, bound activity and ridge-optimum diagnostics."""
    p = np.asarray(record["parameters"])
    train = (fixture["Y"] @ p - fixture["tau"]).reshape(fixture["nv"], -1)
    # Same data as base.run(): the fixture must reproduce the recorded error.
    np.testing.assert_allclose(
        np.sqrt(np.mean(train**2)), record["train_rmse"], rtol=1e-9, atol=1e-14
    )
    sp = np.sqrt(base.CONFIG["prior_strength"]) / fixture["scale"]
    r = np.r_[fixture["Y"] @ p - fixture["tau"], sp * (p - fixture["prior"])]
    cost = 0.5 * float(r @ r)
    record["ridge_relative_cost_excess"] = (cost - ridge["cost"]) / ridge["cost"]
    if record["method"] == "log_cholesky":
        record["converged"] = record.get("status") in PROTOCOL["converged_status"]
        record["bound_active"] = bound_active(p)
    return record


def gates(records):
    def find(case, method, start=None):
        for record in records:
            if (record["case"], record["method"], record.get("initialization")) == (
                case,
                method,
                start,
            ):
                return record
        raise KeyError((case, method, start))

    failures = []
    for record in records:
        if record.get("initialization") not in PROTOCOL["gated_starts"]:
            continue
        label = f"{record['case']}/{record['initialization']}"
        if not record["converged"]:
            failures.append(f"{label}: not converged (status {record.get('status')})")
        if not all(link["feasible"] for link in record["physical"]):
            failures.append(f"{label}: infeasible")
        if record["runtime_seconds"] > base.CONFIG["go_runtime_seconds"]:
            failures.append(f"{label}: runtime {record['runtime_seconds']:.2f} s")
    sdp = find("noisy", "ols_sdp")
    for start in PROTOCOL["gated_starts"]:
        clean = find("clean", "log_cholesky", start)["heldout_rmse"]
        if clean > base.CONFIG["go_clean_rmse_Nm"]:
            failures.append(f"clean/{start}: held-out {clean:.3e} Nm")
        noisy = find("noisy", "log_cholesky", start)["heldout_rmse"]
        limit = base.CONFIG["go_heldout_vs_sdp_factor"] * sdp["heldout_rmse"]
        if not sdp["success"] or noisy > limit:
            failures.append(f"noisy/{start}: held-out {noisy:.6f} > {limit:.6f} Nm")
    return {"pass": not failures, "failures": failures}


def run_set(name):
    frozen = dict(base.CONFIG)
    base.CONFIG.update(seeds(SEED_SETS[name]), max_nfev=PROTOCOL["max_nfev"])
    try:
        result = base.run()
        fixtures = diagnosis.build_fixtures(seeds(SEED_SETS[name]))
        for case, fixture in fixtures.items():
            ridge = diagnosis.reference_optimum(fixture, base.CONFIG["prior_strength"])
            result["fixtures"][case]["ridge_optimum"] = {
                "cost": ridge["cost"],
                "min_pseudo_eigenvalue": ridge["min_eig"],
                "infeasible_links": ridge["n_infeasible_links"],
            }
            for record in result["records"]:
                if record["case"] == case:
                    annotate(record, fixture, ridge)
    finally:
        base.CONFIG.clear()
        base.CONFIG.update(frozen)
    result["gates"] = gates(result["records"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sets", nargs="+", choices=list(SEED_SETS), default=list(SEED_SETS)
    )
    args = parser.parse_args()
    sets = {name: run_set(name) for name in args.sets}
    confirmed = [name for name in CONFIRMATION if name in sets]
    output = {
        "protocol": PROTOCOL,
        "seed_sets": SEED_SETS,
        "scripts_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                _HERE / "log_cholesky_convergence_diagnosis.py",
                _HERE / "log_cholesky_feasibility.py",
            ]
        },
        "go": len(confirmed) == len(CONFIRMATION)
        and all(sets[name]["gates"]["pass"] for name in confirmed),
        "sets": sets,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for name, result in sets.items():
        gate = result["gates"]
        print(f"{name}: {'pass' if gate['pass'] else 'FAIL'}")
        for failure in gate["failures"]:
            print(f"  {failure}")
    print(f"go: {output['go']}")


if __name__ == "__main__":
    main()
