"""Private issue #155 experiment: boundary-aware log-Cholesky fit (LC1 go gate).

Imports the frozen #22 spike and #30 diagnosis unchanged and changes what the
solver is asked to do (docs/decisions/log-cholesky-boundary-aware.md):

1. Detect: solve the #22 objective without the physical constraint in closed
   form. Its minimiser is the exact optimum of the log-Cholesky problem when it
   is physically feasible with a margin; otherwise the infimum is on the
   boundary and log-Cholesky coordinates cannot attain it.
2. Interior: fit in log-Cholesky coordinates with the regressor null-space
   component pinned to the prior (same minimiser, removes the drift), and stop
   on the identifiable residual distance to the detected optimum.
3. Boundary: solve the same torque objective as a convex SDP with an
   eigenvalue margin on every pseudo-inertia, instead of letting log
   coordinates diverge.

Not an installable solver or supported API. Run (figaroh-dev, one BLAS thread),
once per Pinocchio profile:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
        docs/development/spikes/log_cholesky_boundary_aware.py --output out.json
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import picos as pc
from scipy.optimize import least_squares

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "log_cholesky_convergence_diagnosis",
    _HERE / "log_cholesky_convergence_diagnosis.py",
)
diagnosis = importlib.util.module_from_spec(_spec)
sys.modules["log_cholesky_convergence_diagnosis"] = diagnosis
_spec.loader.exec_module(diagnosis)
base = diagnosis.base

# Method and stopping rule. Everything not listed keeps the #22/#30 value
# (objective, prior 1e-6, scales, fixtures, oracle, 2000-evaluation budget).
PROTOCOL = {
    "max_nfev": 2000,
    "method": "dogbox",
    "rank_cutoff": 1e-10,
    "eigenvalue_margin": 1e-6,
    "stop_identifiable_rms_Nm": 1e-7,
    "sdp_solver": "cvxopt",
    "sdp_tolerance": 1e-9,
    "gated_starts": ["nominal", "repaired_ols"],
}
SEED_SETS = {
    # Seen before this protocol was frozen: #22 exploration and #30 confirmation.
    "development_2201": [2201, 2202, 2203, 2204, 2205, 2206],
    "development_3001": [3001, 3002, 3003, 3004, 3005, 3006],
    "development_3101": [3101, 3102, 3103, 3104, 3105, 3106],
    "development_3201": [3201, 3202, 3203, 3204, 3205, 3206],
    # Declared with the frozen protocol; never run before it was committed.
    "confirmation_a": [4001, 4002, 4003, 4004, 4005, 4006],
    "confirmation_b": [4101, 4102, 4103, 4104, 4105, 4106],
    "confirmation_c": [4201, 4202, 4203, 4204, 4205, 4206],
}
CONFIRMATION = [name for name in SEED_SETS if name.startswith("confirmation")]
CASES = diagnosis.CASES


# --------------------------------------------------------------------- data
def build(values):
    """Rebuild the #22 fixtures, baselines and all five starts for one seed set."""
    seed_model, seed_train, seed_validation, seed_noise, *seed_starts = values
    cfg = base.CONFIG
    model = base.pin.buildSampleModelManipulator()
    rng = np.random.default_rng(seed_model)
    n = model.njoints - 1
    truth_z = np.zeros((n, 10))
    truth_z[:, 0] = rng.uniform(-0.3, 0.5, n)
    truth_z[:, 1:4] = rng.uniform(-1.8, -0.8, (n, 3))
    truth_z[:, 4:] = rng.normal(0, 0.08, (n, 6))
    prior = base.decode((truth_z + rng.normal(0, 0.12, truth_z.shape)).ravel())
    scale = np.tile([1] + [0.2] * 9, n)
    reference = diagnosis.build_fixtures(
        {
            "seed_model": seed_model,
            "seed_train": seed_train,
            "seed_validation": seed_validation,
            "seed_noise": seed_noise,
        }
    )
    fixtures = {}
    for case in CASES:
        case_z = truth_z.copy()
        if case == "near_boundary_truth":
            case_z[:, 1] -= 4
        truth = base.decode(case_z.ravel())
        for j, p in enumerate(truth.reshape(-1, 10), 1):
            model.inertias[j] = base.pin.Inertia.FromDynamicParameters(p)
        Y, clean, states = base.trajectory(
            model, seed_train, cfg["samples_train"], weak=case == "weak_excitation"
        )
        Yv, tauv, statesv = base.trajectory(
            model, seed_validation, cfg["samples_validation"]
        )
        tau = clean.copy()
        if case != "clean":
            tau += np.random.default_rng(seed_noise).normal(
                0, cfg["noise_std_Nm"], tau.shape
            )
        ols = prior + np.linalg.lstsq(Y, tau - Y @ prior, rcond=1e-10)[0]
        repaired, changes = base.repair(ols)
        boundary_z = np.concatenate([base.encode(x) for x in prior.reshape(-1, 10)])
        boundary_z = boundary_z.reshape(-1, 10)
        boundary_z[:, 1:4] -= 3
        starts = {
            "nominal": prior,
            "repaired_ols": repaired,
            "near_boundary": base.decode(boundary_z.ravel()),
        }
        for seed in seed_starts:
            perturbed = truth_z + np.random.default_rng(seed).normal(
                0, 0.3, truth_z.shape
            )
            starts[f"perturbed_{seed}"] = base.decode(perturbed.ravel())
        # Same data as the #22 spike and the #30 diagnosis fixtures.
        mirror = reference[case]
        for mine, key in [(Y, "Y"), (tau, "tau"), (Yv, "Yv"), (tauv, "tauv")]:
            np.testing.assert_array_equal(mine, mirror[key])
        for mine, key in [(prior, "prior"), (ols, "ols")]:
            np.testing.assert_array_equal(mine, mirror[key])
        np.testing.assert_array_equal(repaired, mirror["starts"]["repaired_ols"])
        fixtures[case] = {
            "Y": Y,
            "tau": tau,
            "Yv": Yv,
            "tauv": tauv,
            "prior": prior,
            "scale": scale,
            "truth": truth,
            "ols": ols,
            "starts": starts,
            "repairs": changes,
            "nv": model.nv,
            "data_sha256": hashlib.sha256(
                b"".join(x.tobytes() for x in [Y, tau, Yv, tauv, *states, *statesv])
            ).hexdigest(),
        }
    return fixtures


def ols_sdp(fx):
    """Corrected per-link OLS+SDP baseline, exactly as in the #22 spike."""
    begin = time.perf_counter()
    projected, statuses = [], []
    for x, w in zip(fx["ols"].reshape(-1, 10), (1 / fx["scale"]).reshape(-1, 10)):
        p, report = base.project_p10_lmi(x, weights=w)
        projected.append(p)
        statuses.append(report.status)
    p = np.concatenate(projected)
    return p, {
        "success": all(s == "projected" for s in statuses),
        "runtime_seconds": time.perf_counter() - begin,
    }


# ------------------------------------------------------------------- method
class Objective:
    """The #22 objective in scaled deviations q = (p - prior) / scale.

    f(q) = 0.5 ||Y (prior + scale q) - tau||^2 + 0.5 lam ||q||^2, unchanged.
    """

    def __init__(self, fx):
        self.fx = fx
        self.lam = base.CONFIG["prior_strength"]
        self.A = fx["Y"] * fx["scale"]
        self.e = fx["tau"] - fx["Y"] @ fx["prior"]
        U, S, Vt = np.linalg.svd(self.A, full_matrices=False)
        rank = int(np.sum(S > S[0] * PROTOCOL["rank_cutoff"]))
        self.rank, self.sigma_max = rank, float(S[0])
        self.null = Vt[rank:].T
        # Ridge minimiser: lies in range(A.T), so its null component is zero.
        self.q_star = Vt.T @ (S / (S**2 + self.lam) * (U.T @ self.e))
        self.p_star = fx["prior"] + fx["scale"] * self.q_star
        # Pinning the null component adds a term that vanishes at q_star, so the
        # minimiser is unchanged; it only removes the flat directions.
        self.pin = self.sigma_max
        self.M = np.vstack(
            [self.A, np.sqrt(self.lam) * np.eye(len(S)), self.pin * self.null.T]
        )

    def q(self, p):
        return (p - self.fx["prior"]) / self.fx["scale"]

    def cost(self, p):
        q = self.q(p)
        r = self.A @ q - self.e
        return 0.5 * float(r @ r + self.lam * q @ q)

    def identifiable_rms(self, p):
        """RMS distance to the detected optimum, in training-torque units."""
        d = self.M @ (self.q(p) - self.q_star)
        return float(np.linalg.norm(d) / np.sqrt(len(self.e)))


def min_eig(p):
    return min(v["min_pseudo_eigenvalue"] for v in base.verdict(p))


def fit_interior(obj, start):
    fx = obj.fx
    begin = time.perf_counter()
    sqrt_lam = np.sqrt(obj.lam)
    calls = {"residual": 0, "jacobian": 0}
    stop = PROTOCOL["stop_identifiable_rms_Nm"]

    def residual(z):
        calls["residual"] += 1
        if time.perf_counter() - begin > base.CONFIG["max_seconds_per_fit"]:
            raise TimeoutError("fit exceeded time budget")
        q = obj.q(base.decode(z))
        return np.r_[obj.A @ q - obj.e, sqrt_lam * q, obj.pin * (obj.null.T @ q)]

    def jacobian(z):
        calls["jacobian"] += 1
        Dq = base.derivative(z) / fx["scale"][:, None]
        return np.vstack([obj.A @ Dq, sqrt_lam * Dq, obj.pin * (obj.null.T @ Dq)])

    def callback(intermediate_result):
        if obj.identifiable_rms(base.decode(intermediate_result.x)) <= stop:
            raise StopIteration

    z0 = np.concatenate([base.encode(p) for p in start.reshape(-1, 10)])
    kw = dict(
        jac=jacobian,
        method=PROTOCOL["method"],
        ftol=base.CONFIG["ftol"],
        xtol=base.CONFIG["xtol"],
        gtol=base.CONFIG["gtol"],
        max_nfev=PROTOCOL["max_nfev"],
        callback=callback,
    )
    if PROTOCOL["method"] != "lm":
        kw.update(x_scale="jac", bounds=base.CONFIG["coordinate_bounds"])
    try:
        result = least_squares(residual, z0, **kw)
    except (TimeoutError, FloatingPointError, ValueError) as error:
        return start.copy(), {
            "status": None,
            "message": str(error),
            "calls": calls,
            "runtime_seconds": time.perf_counter() - begin,
        }
    p = base.decode(result.x)
    return p, {
        "status": int(result.status),
        "message": result.message,
        "nfev": int(result.nfev),
        "njev": int(result.njev or 0),
        "calls": calls,
        "identifiable_rms_Nm": obj.identifiable_rms(p),
        "max_abs_z": float(np.abs(result.x).max()),
        "runtime_seconds": time.perf_counter() - begin,
    }


def pseudo_basis():
    """pseudo() is linear in p10; return its ten 4x4 basis matrices."""
    return [base.pseudo(e) for e in np.eye(10)]


def fit_boundary(obj):
    """Convex SDP: the same torque objective with P_j >= margin * I per link."""
    fx = obj.fx
    begin = time.perf_counter()
    n = len(obj.q_star)
    # 0.5||A q - e||^2 + 0.5 lam ||q||^2 = 0.5||R q - c||^2 + constant.
    stacked = np.vstack([obj.A, np.sqrt(obj.lam) * np.eye(n)])
    Q, R = np.linalg.qr(stacked)
    c = Q.T @ np.r_[obj.e, np.zeros(n)]
    constant = 0.5 * float(np.r_[obj.e, np.zeros(n)] @ np.r_[obj.e, np.zeros(n)])
    constant -= 0.5 * float(c @ c)
    E = pseudo_basis()
    problem = pc.Problem()
    q = pc.RealVariable("q", n)
    t = pc.RealVariable("t", 1)
    problem.set_objective("min", t)
    problem.add_constraint(abs(pc.Constant(R) * q - pc.Constant(c)) <= t)
    margin = PROTOCOL["eigenvalue_margin"]
    for j, p0 in enumerate(fx["prior"].reshape(-1, 10)):
        P = pc.Constant(base.pseudo(p0))
        for k in range(10):
            P = P + float(fx["scale"][10 * j + k]) * q[10 * j + k] * pc.Constant(E[k])
        problem.add_constraint(P - margin * np.eye(4) >> 0)
    tol = PROTOCOL["sdp_tolerance"]
    problem.options.solver = PROTOCOL["sdp_solver"]
    problem.options.abs_prim_fsb_tol = tol
    problem.options.abs_dual_fsb_tol = tol
    problem.options.rel_ipm_opt_tol = tol
    problem.options.abs_ipm_opt_tol = tol
    try:
        solution = problem.solve()
        status = solution.claimedStatus
        q_hat = np.asarray(q.value).ravel()
    except pc.SolutionFailure as error:
        return None, {
            "status": "failure",
            "message": str(error),
            "runtime_seconds": time.perf_counter() - begin,
        }
    p = fx["prior"] + fx["scale"] * q_hat
    return p, {
        "status": str(status),
        "primal_status": str(solution.primalStatus),
        "dual_status": str(solution.dualStatus),
        "objective_check": 0.5 * float(np.sum((R @ q_hat - c) ** 2)) + constant,
        "runtime_seconds": time.perf_counter() - begin,
    }


def fit(fx, start):
    """Boundary-aware fit. Returns parameters and a record with `converged`."""
    begin = time.perf_counter()
    obj = Objective(fx)
    star_min_eig = min_eig(obj.p_star)
    interior = star_min_eig >= PROTOCOL["eigenvalue_margin"]
    detection = {
        "branch": "interior" if interior else "boundary",
        "unconstrained_min_pseudo_eigenvalue": star_min_eig,
        "unconstrained_infeasible_links": sum(
            not v["feasible"] for v in base.verdict(obj.p_star)
        ),
        "rank": obj.rank,
    }
    if interior:
        p, report = fit_interior(obj, start)
        converged = (
            report["status"] is not None
            and report["identifiable_rms_Nm"] <= PROTOCOL["stop_identifiable_rms_Nm"]
        )
    else:
        p, report = fit_boundary(obj)
        converged = p is not None and report["status"] == "optimal"
        if p is None:
            p = start.copy()
    return p, {
        **detection,
        **report,
        "converged": bool(converged),
        "cost": obj.cost(p),
        "unconstrained_cost": obj.cost(obj.p_star),
        "runtime_seconds": time.perf_counter() - begin,
    }


# -------------------------------------------------------------------- gates
def record(fx, method, p, extra):
    m = base.metrics(p, fx["Y"], fx["tau"], fx["Yv"], fx["tauv"], fx["nv"])
    return {"method": method, **extra, **m}


def gates(records):
    def find(case, method, start=None):
        for r in records:
            if (r["case"], r["method"], r.get("initialization")) == (
                case,
                method,
                start,
            ):
                return r
        raise KeyError((case, method, start))

    failures = []
    for r in records:
        if r.get("initialization") not in PROTOCOL["gated_starts"]:
            continue
        label = f"{r['case']}/{r['initialization']}"
        if not r["converged"]:
            failures.append(f"{label}: not converged ({r['branch']}, {r['status']})")
        if not all(link["feasible"] for link in r["physical"]):
            failures.append(f"{label}: infeasible")
        if r["runtime_seconds"] > base.CONFIG["go_runtime_seconds"]:
            failures.append(f"{label}: runtime {r['runtime_seconds']:.2f} s")
    sdp = find("noisy", "ols_sdp")
    for start in PROTOCOL["gated_starts"]:
        clean = find("clean", "boundary_aware", start)["heldout_rmse"]
        if clean > base.CONFIG["go_clean_rmse_Nm"]:
            failures.append(f"clean/{start}: held-out {clean:.3e} Nm")
        noisy = find("noisy", "boundary_aware", start)["heldout_rmse"]
        limit = base.CONFIG["go_heldout_vs_sdp_factor"] * sdp["heldout_rmse"]
        if not sdp["success"] or noisy > limit:
            failures.append(f"noisy/{start}: held-out {noisy:.6f} > {limit:.6f} Nm")
    return {"pass": not failures, "failures": failures}


def run_set(name):
    fixtures = build(SEED_SETS[name])
    records, summary = [], {}
    for case, fx in fixtures.items():
        p, extra = ols_sdp(fx)
        records.append(record(fx, "ols_sdp", p, {"case": case, **extra}))
        for start_name, start in fx["starts"].items():
            p, extra = fit(fx, start)
            rec = record(
                fx,
                "boundary_aware",
                p,
                {
                    "case": case,
                    "initialization": start_name,
                    "initial_physical": base.verdict(start),
                    **extra,
                },
            )
            if start_name == "repaired_ols":
                rec["repair_p10_norm_per_link"] = fx["repairs"]
            records.append(rec)
        summary[case] = {
            "data_sha256": fx["data_sha256"],
            "truth": fx["truth"].tolist(),
        }
    return {"fixtures": summary, "records": records, "gates": gates(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sets", nargs="+", choices=list(SEED_SETS), default=list(SEED_SETS)
    )
    args = parser.parse_args()
    sets = {}
    for name in args.sets:
        sets[name] = run_set(name)
        gate = sets[name]["gates"]
        print(f"{name}: {'pass' if gate['pass'] else 'FAIL'}", flush=True)
        for failure in gate["failures"]:
            print(f"  {failure}")
        for r in sets[name]["records"]:
            if r["method"] != "boundary_aware":
                continue
            print(
                f"  {r['case'][:13]:<13} {r['initialization'][:14]:<14} "
                f"{r['branch']:<8} conv {int(r['converged'])} "
                f"nfev {r.get('nfev', '-')!s:>4} "
                f"held-out {r['heldout_rmse']:.3e} t {r['runtime_seconds']:.2f}s"
            )
    confirmed = [name for name in CONFIRMATION if name in sets]
    output = {
        "protocol": PROTOCOL,
        "base_config": base.CONFIG,
        "seed_sets": SEED_SETS,
        "environment": {
            "python": platform.python_version(),
            **{
                key: version(key)
                for key in ["pin", "numpy", "scipy", "picos", "cvxopt"]
            },
        },
        "core_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "scripts_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                _HERE / "log_cholesky_convergence_diagnosis.py",
                _HERE / "log_cholesky_feasibility.py",
            ]
        },
        "go": bool(confirmed)
        and len(confirmed) == len(CONFIRMATION)
        and all(sets[name]["gates"]["pass"] for name in confirmed),
        "sets": sets,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"go: {output['go']}")


if __name__ == "__main__":
    main()
