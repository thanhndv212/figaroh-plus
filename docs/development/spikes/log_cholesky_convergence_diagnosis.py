"""Private issue #30 diagnosis of log-Cholesky TRF non-termination (D5).

Diagnostic only. It imports the frozen issue #22 spike unchanged, rebuilds its
fixtures with the same seeds and verifies them against the recorded results,
then (1) traces per-iteration convergence quantities of the frozen protocol
for the nominal and repaired-OLS starts and (2) runs exploratory sensitivity
probes. Nothing here is a candidate protocol or a confirmation result; every
probe uses the *exploration* fixtures (the original seeds), so a future
confirmation run must use fresh seeds.

Run (figaroh-dev, one BLAS thread):
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python \
        docs/development/spikes/log_cholesky_convergence_diagnosis.py \
        --output <scratch>/diagnosis.json
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

_SPIKE = Path(__file__).with_name("log_cholesky_feasibility.py")
_spec = importlib.util.spec_from_file_location("log_cholesky_feasibility", _SPIKE)
base = importlib.util.module_from_spec(_spec)
sys.modules["log_cholesky_feasibility"] = base
_spec.loader.exec_module(base)

CASES = ["clean", "noisy", "weak_excitation", "near_boundary_truth"]
STARTS = ["nominal", "repaired_ols"]
RESULTS = Path(__file__).with_name("results") / "log-cholesky-pin37.json"


# --------------------------------------------------------------------- data
def build_fixtures(seeds=None):
    """Replicate run() data generation for the two diagnosed starts.

    seeds overrides the frozen seeds (for a future confirmation set); the
    default reproduces the exploration fixtures exactly.
    """
    cfg = dict(base.CONFIG)
    cfg.update(seeds or {})
    model = pin.buildSampleModelManipulator()
    rng = np.random.default_rng(cfg["seed_model"])
    n = model.njoints - 1
    truth_z = np.zeros((n, 10))
    truth_z[:, 0] = rng.uniform(-0.3, 0.5, n)
    truth_z[:, 1:4] = rng.uniform(-1.8, -0.8, (n, 3))
    truth_z[:, 4:] = rng.normal(0, 0.08, (n, 6))
    prior = base.decode((truth_z + rng.normal(0, 0.12, truth_z.shape)).ravel())
    scale = np.tile([1] + [0.2] * 9, n)
    fixtures = {}
    for case in CASES:
        case_z = truth_z.copy()
        if case == "near_boundary_truth":
            case_z[:, 1] -= 4
        truth = base.decode(case_z.ravel())
        for j, p in enumerate(truth.reshape(-1, 10), 1):
            model.inertias[j] = pin.Inertia.FromDynamicParameters(p)
        Y, clean, states = base.trajectory(
            model,
            cfg["seed_train"],
            cfg["samples_train"],
            weak=case == "weak_excitation",
        )
        Yv, tauv, _ = base.trajectory(
            model, cfg["seed_validation"], cfg["samples_validation"]
        )
        tau = clean.copy()
        if case != "clean":
            tau += np.random.default_rng(cfg["seed_noise"]).normal(
                0, cfg["noise_std_Nm"], tau.shape
            )
        ols = prior + np.linalg.lstsq(Y, tau - Y @ prior, rcond=1e-10)[0]
        repaired, _ = base.repair(ols)
        fixtures[case] = {
            "Y": Y,
            "tau": tau,
            "Yv": Yv,
            "tauv": tauv,
            "prior": prior,
            "scale": scale,
            "ols": ols,
            "starts": {"nominal": prior, "repaired_ols": repaired},
            "nv": model.nv,
        }
    return fixtures


def reference_optimum(fx, lam):
    """Global minimiser of the (convex) p-space objective at prior weight lam.

    The log-Cholesky objective is this ridge problem restricted to the PD set,
    so if the ridge minimiser is PD it IS the global optimum of the nonlinear
    problem; otherwise the infimum is on the boundary and not attained.
    """
    sp = np.sqrt(lam) / fx["scale"]
    A = np.vstack([fx["Y"], np.diag(sp)])
    b = np.r_[fx["tau"], sp * fx["prior"]]
    p = np.linalg.lstsq(A, b, rcond=1e-12)[0]
    r = A @ p - b
    verd = base.verdict(p)
    return {
        "p": p,
        "cost": 0.5 * float(r @ r),
        "min_eig": min(v["min_pseudo_eigenvalue"] for v in verd),
        "n_infeasible_links": sum(not v["feasible"] for v in verd),
    }


# ------------------------------------------------------------------ analysis
class Problem:
    def __init__(self, fx, lam, s=1.0):
        self.fx, self.lam, self.s = fx, lam, s
        self.sp = np.sqrt(lam)
        self.Y = fx["Y"]
        self.null_p = None

    def residual(self, z):
        p = base.decode(z)
        fx = self.fx
        return (
            self.s
            * np.r_[self.Y @ p - fx["tau"], self.sp * (p - fx["prior"]) / fx["scale"]]
        )

    def jacobian(self, z):
        D = base.derivative(z)
        fx = self.fx
        return self.s * np.vstack([self.Y @ D, self.sp * D / fx["scale"][:, None]])


def cl_scaled_gradient(g, z, lo, hi):
    v = np.ones_like(g)
    m = g < 0
    v[m] = hi - z[m]
    m = g > 0
    v[m] = z[m] - lo
    return g * v


def analyze(prob, z, bounds, ref=None, prev=None, tols=None):
    """All per-iterate diagnostics at z (called at accepted iterates)."""
    tols = tols or (base.CONFIG["ftol"], base.CONFIG["xtol"], base.CONFIG["gtol"])
    ftol, xtol, gtol = tols
    r, J = prob.residual(z), prob.jacobian(z)
    cost = 0.5 * float(r @ r)
    g = J.T @ r
    D = base.derivative(z)
    A = prob.Y @ D
    Ua, Sa, Vta = np.linalg.svd(A)
    rank = int(np.sum(Sa > Sa[0] * 1e-10))
    N = Vta[rank:].T  # null space of (torque regressor o dp/dz), 60 x k
    free = (z > bounds[0] + 1e-6) & (z < bounds[1] - 1e-6)
    gcl = cl_scaled_gradient(g, z, *bounds)
    sj = np.linalg.svd(J, compute_uv=False)
    colnorm = np.linalg.norm(J, axis=0)
    snorm = np.linalg.svd(J / colnorm, compute_uv=False)
    # Gauss-Newton step and its quadratic-model decrease, total and null-only.
    dz = -np.linalg.lstsq(J, r, rcond=None)[0]
    dz_null = N @ (N.T @ dz)

    def model_decrease(d):
        return float(-g @ d - 0.5 * np.linalg.norm(J @ d) ** 2)

    out = {
        "cost": cost,
        "g_inf": float(np.abs(g).max()),
        "g_2": float(np.linalg.norm(g)),
        "g_cl_inf": float(np.abs(gcl).max()),
        "g_free_inf": float(np.abs(g[free]).max()) if free.any() else 0.0,
        "n_free": int(free.sum()),
        "max_abs_z": float(np.abs(z).max()),
        "yd_rank": rank,
        "yd_sv_gap": [float(Sa[rank - 1]), float(Sa[rank]) if rank < len(Sa) else 0.0],
        "g_null_fraction": float(
            np.linalg.norm(N.T @ g) / max(np.linalg.norm(g), 1e-300)
        ),
        "g_null_inf": float(np.abs(N.T @ g).max()) if N.size else 0.0,
        "g_range_2": float(np.linalg.norm(g - N @ (N.T @ g))),
        "sv_max": float(sj[0]),
        "sv_min": float(sj[-1]),
        "cond_J": float(sj[0] / sj[-1]),
        "cond_J_colscaled": float(snorm[0] / snorm[-1]),
        "n_sv_below_1e-6": int(np.sum(sj < sj[0] * 1e-6)),
        "gn_step_norm": float(np.linalg.norm(dz)),
        "gn_step_null_fraction": float(
            np.linalg.norm(dz_null) / max(np.linalg.norm(dz), 1e-300)
        ),
        "gn_decrease_total": model_decrease(dz),
        "gn_decrease_null_only": model_decrease(dz_null),
        "gn_decrease_rel_cost": model_decrease(dz) / cost,
        "g_over_JnormRnorm": float(np.linalg.norm(g) / (sj[0] * np.linalg.norm(r))),
        "n_pd_links_infeasible": 0,
    }
    # Distance to the global (ridge) optimum, split by null(Y) in p space.
    p = base.decode(z)
    if ref is not None:
        out["cost_excess"] = cost - ref["cost_scaled"]
        dp = p - ref["p"]
        Pn = ref["null_p"]
        out["dp_null_Y"] = float(np.linalg.norm(Pn.T @ dp))
        out["dp_range_Y"] = float(np.linalg.norm(dp - Pn @ (Pn.T @ dp)))
    if prev is not None:
        dF = prev["cost"] - cost
        step = float(np.linalg.norm(z - prev["z"]))
        out["step_norm"] = step
        out["ratio_ftol"] = float(dF / cost / ftol)  # <1 would trigger ftol
        out["ratio_xtol"] = float(
            step / (xtol * (xtol + np.linalg.norm(z)))
        )  # <1 triggers xtol
    out["ratio_gtol"] = out["g_cl_inf"] / gtol  # <1 triggers gtol
    return out, N


def null_basis_p(Y):
    _, S, Vt = np.linalg.svd(Y)
    rank = int(np.sum(S > S[0] * 1e-10))
    return Vt[rank:].T


def solve(
    fx,
    start,
    lam=1e-6,
    s=1.0,
    max_nfev=200,
    x_scale="jac",
    method="trf",
    ftol=None,
    xtol=None,
    gtol=None,
    trace=False,
    bounds=(-12.0, 12.0),
    ref=None,
):
    cfg = base.CONFIG
    ftol = cfg["ftol"] if ftol is None else ftol
    xtol = cfg["xtol"] if xtol is None else xtol
    gtol = cfg["gtol"] if gtol is None else gtol
    prob = Problem(fx, lam, s)
    z0 = np.concatenate([base.encode(p) for p in fx["starts"][start].reshape(-1, 10)])
    if isinstance(x_scale, str) and x_scale == "jac0":
        x_scale = 1.0 / np.linalg.norm(prob.jacobian(z0), axis=0)
    iterates = []
    cnt = {"fev": 0, "jev": 0}

    def fun(z):
        cnt["fev"] += 1
        return prob.residual(z)

    def jac(z):
        cnt["jev"] += 1
        if trace:
            iterates.append(z.copy())
        return prob.jacobian(z)

    kw = dict(
        jac=jac,
        method=method,
        x_scale=x_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
    )
    if method != "lm":
        kw["bounds"] = bounds
    t0 = time.perf_counter()
    res = least_squares(fun, z0, **kw)
    runtime = time.perf_counter() - t0
    p = base.decode(res.x)
    verd = base.verdict(p)
    m = base.metrics(p, fx["Y"], fx["tau"], fx["Yv"], fx["tauv"], fx["nv"])
    summary = {
        "status": int(res.status),
        "nfev": int(res.nfev),
        "njev": int(res.njev or 0),
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "feasible": all(v["feasible"] for v in verd),
        "min_eig": min(v["min_pseudo_eigenvalue"] for v in verd),
        "max_abs_z": float(np.abs(res.x).max()),
        "n_at_bound": int(np.sum(np.abs(res.x) > bounds[1] - 1e-6)),
        "train_rmse": m["train_rmse"],
        "heldout_rmse": m["heldout_rmse"],
        "runtime_s": runtime,
    }
    if ref is not None:
        summary["cost_excess"] = summary["cost"] - ref["cost_scaled"] * 1.0
        summary["rel_cost_excess"] = summary["cost_excess"] / ref["cost_scaled"]
    out = {"summary": summary, "x": res.x}
    if trace:
        out["iterates"] = iterates
        out["prob"] = prob
    return out


def make_ref(fx, lam, s=1.0):
    ref = reference_optimum(fx, lam)
    ref["cost_scaled"] = ref["cost"] * s * s
    ref["null_p"] = null_basis_p(fx["Y"])
    return ref


def trace_run(fx, start, lam=1e-6, max_nfev=200, **kw):
    ref = make_ref(fx, lam)
    out = solve(fx, start, lam=lam, max_nfev=max_nfev, trace=True, ref=ref, **kw)
    prob, iters = out["prob"], out["iterates"]
    rows, prev = [], None
    for z in iters:
        a, N = analyze(prob, z, (-12.0, 12.0), ref=ref, prev=prev)
        a["z"] = z
        rows.append(a)
        prev = a
    final, N = analyze(
        prob, out["x"], (-12.0, 12.0), ref=ref, prev=rows[-1] if rows else None
    )
    # Which tolerance is closest (ratio nearest 1 from above) over the run.
    ratios = {
        k: [r[k] for r in rows if k in r]
        for k in ["ratio_ftol", "ratio_xtol", "ratio_gtol"]
    }
    closest = {k: float(np.min(v)) for k, v in ratios.items() if v}
    closest_final = {
        k: final[k] for k in ["ratio_ftol", "ratio_xtol", "ratio_gtol"] if k in final
    }
    # Rotation of null(Y D) between first and last iterate (curved valley).
    N0 = analyze(prob, iters[0], (-12.0, 12.0))[1]
    ang = np.degrees(
        np.arccos(np.clip(np.linalg.svd(N0.T @ N, compute_uv=False), -1, 1))
    )
    for r in rows:
        r.pop("z")
    keep = sorted(
        set(list(range(0, len(rows), max(1, len(rows) // 12))) + [len(rows) - 1])
    )
    return {
        "summary": out["summary"],
        "ref": {
            "cost": ref["cost"],
            "min_eig": ref["min_eig"],
            "n_infeasible_links": ref["n_infeasible_links"],
        },
        "first": rows[0],
        "final": final,
        "decimated_trace": [
            dict(
                iter=i,
                **{
                    k: rows[i][k]
                    for k in [
                        "cost",
                        "cost_excess",
                        "g_cl_inf",
                        "g_2",
                        "g_null_fraction",
                        "step_norm",
                        "cond_J_colscaled",
                        "dp_null_Y",
                        "dp_range_Y",
                    ]
                    if k in rows[i]
                },
            )
            for i in keep
        ],
        "min_ratio_over_run": closest,
        "final_ratio": closest_final,
        "null_space_rotation_deg_max": float(ang.max()) if ang.size else 0.0,
        "n_iterates": len(rows),
        "rejected_steps": out["summary"]["nfev"] - out["summary"]["njev"],
    }


# -------------------------------------------------------------------- probes
def probe(fixtures, label, **kw):
    """Run nominal+repaired_ols on all four cases with one option set."""
    lam = kw.get("lam", 1e-6)
    s = kw.get("s", 1.0)
    rows = []
    for case in CASES:
        ref = make_ref(fixtures[case], lam, s)
        for start in STARTS:
            r = solve(fixtures[case], start, ref=ref, **kw)["summary"]
            r.update(
                case=case, start=start, ref_feasible=ref["n_infeasible_links"] == 0
            )
            rows.append(r)
    conv = [r["status"] > 0 for r in rows]
    return {
        "label": label,
        "options": {
            k: (v if not isinstance(v, np.ndarray) else "array") for k, v in kw.items()
        },
        "n_converged": int(sum(conv)),
        "n_runs": len(rows),
        "statuses": [r["status"] for r in rows],
        "nfev": [r["nfev"] for r in rows],
        "all_feasible": all(r["feasible"] for r in rows),
        "max_rel_cost_excess": float(max(r["rel_cost_excess"] for r in rows)),
        "clean_heldout_rmse": [r["heldout_rmse"] for r in rows if r["case"] == "clean"],
        "noisy_heldout_rmse": [r["heldout_rmse"] for r in rows if r["case"] == "noisy"],
        "max_runtime_s": max(r["runtime_s"] for r in rows),
        "rows": rows,
    }


def short(p):
    return (
        f"{p['label']:<34} conv {p['n_converged']}/{p['n_runs']} st {''.join(map(str, p['statuses']))} "
        f"nfev {min(p['nfev'])}-{max(p['nfev'])} feas {int(p['all_feasible'])} "
        f"relExcess {p['max_rel_cost_excess']:.1e} cleanRMSE {max(p['clean_heldout_rmse']):.1e} "
        f"t {p['max_runtime_s']:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--long-budget", type=int, default=20000)
    args = ap.parse_args()

    fixtures = build_fixtures()
    recorded = json.loads(RESULTS.read_text())
    out = {"note": "diagnostic only; exploration seeds; not a candidate protocol"}

    # 0. Reproduction of the frozen records (same objective/nfev).
    repro = []
    for rec in recorded["records"]:
        if rec["method"] != "log_cholesky" or rec["initialization"] not in STARTS:
            continue
        mine = solve(fixtures[rec["case"]], rec["initialization"])["summary"]
        repro.append(
            {
                "case": rec["case"],
                "start": rec["initialization"],
                "nfev_rec": rec["nfev"],
                "nfev": mine["nfev"],
                "cost_rec": rec["objective"],
                "cost": mine["cost"],
                "match": rec["nfev"] == mine["nfev"]
                and abs(rec["objective"] - mine["cost"]) <= 1e-9 * rec["objective"],
            }
        )
    out["reproduction"] = repro
    print(
        "reproduces frozen records:",
        all(r["match"] for r in repro),
        f"({len(repro)} runs)",
    )

    # 1. Frozen-protocol traces.
    out["traces"] = {}
    for case in CASES:
        for start in STARTS:
            t = trace_run(fixtures[case], start)
            out["traces"][f"{case}/{start}"] = t
            f, i = t["final"], t["first"]
            print(
                f"{case[:9]:<9} {start[:7]:<7} st{t['summary']['status']} nfev{t['summary']['nfev']:>4} "
                f"cost {t['summary']['cost']:.5f} (ref {t['ref']['cost']:.5f}, refInfeas {t['ref']['n_infeasible_links']}) "
                f"g_cl_inf {f['g_cl_inf']:.1e} g2 {f['g_2']:.1e} gNullFrac {f['g_null_fraction']:.2f} "
                f"condJ {i['cond_J']:.1e}->{f['cond_J']:.1e} condJcs {f['cond_J_colscaled']:.1e} "
                f"minRatio f/x/g {t['min_ratio_over_run'].get('ratio_ftol', 0):.1e}/"
                f"{t['min_ratio_over_run'].get('ratio_xtol', 0):.1e}/{t['min_ratio_over_run'].get('ratio_gtol', 0):.1e} "
                f"nullRot {t['null_space_rotation_deg_max']:.0f}deg rank {f['yd_rank']}"
            )
            print(
                f"{'':<18} GNdec/cost {f['gn_decrease_rel_cost']:.1e} (null-only share "
                f"{f['gn_decrease_null_only'] / max(f['gn_decrease_total'], 1e-300):.2f}) "
                f"|dp| null(Y) {f['dp_null_Y']:.2e} range(Y) {f['dp_range_Y']:.2e} "
                f"relExcess {f['cost_excess'] / t['ref']['cost']:.1e} maxz {f['max_abs_z']:.1f}"
            )

    if args.skip_probes:
        args.output.write_text(json.dumps(out, indent=1, default=float) + "\n")
        return

    # 2. Exploratory probes (nominal + repaired OLS, four cases = 8 fits each).
    probes = []
    configs = [("frozen: lam1e-6 jac B200", {})]
    configs += [
        (f"budget {b}", {"max_nfev": b}) for b in (1000, 2000, args.long_budget)
    ]
    for lam in (1e-8, 1e-4, 1e-2):
        for b in (200, 2000):
            configs.append((f"lam {lam:g} B{b}", {"lam": lam, "max_nfev": b}))
    for xs in (1.0, "jac0"):
        for b in (200, 2000):
            configs.append((f"x_scale {xs} B{b}", {"x_scale": xs, "max_nfev": b}))
    for method in ("dogbox", "lm"):
        for b in (200, 2000):
            configs.append((f"method {method} B{b}", {"method": method, "max_nfev": b}))
    n_tau = fixtures["clean"]["Y"].shape[0]
    for sc, name in ((1 / 0.03, "1/sigma"), (1 / np.sqrt(n_tau), "1/sqrtN")):
        configs.append((f"resid scale {name} B2000", {"s": sc, "max_nfev": 2000}))
    for gt in (1e-8, 1e-6):
        configs.append((f"gtol {gt:g} B2000", {"gtol": gt, "max_nfev": 2000}))
    configs.append(
        ("ftol=xtol 1e-8 B2000", {"ftol": 1e-8, "xtol": 1e-8, "max_nfev": 2000})
    )
    configs.append(
        ("lam1e-4 gtol1e-8 B2000", {"lam": 1e-4, "gtol": 1e-8, "max_nfev": 2000})
    )
    configs.append(
        (
            "lam1e-4 ftol=xtol1e-8 g1e-8 B2000",
            {"lam": 1e-4, "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "max_nfev": 2000},
        )
    )
    for label, kw in configs:
        pr = probe(fixtures, label, **kw)
        probes.append(pr)
        print(short(pr), flush=True)
    out["probes"] = probes
    args.output.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
