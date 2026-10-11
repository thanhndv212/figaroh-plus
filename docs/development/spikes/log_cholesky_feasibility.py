"""Private issue #22 experiment; not an installable solver or supported API.

Run with one BLAS thread in figaroh-dev. Configuration below is frozen before
measurement; generated trajectories have analytic derivatives. No robot data
or preprocessing is involved. Output includes the full reproducibility contract.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from scipy.optimize import least_squares

from figaroh.identification.physical_consistency import project_p10_lmi

CONFIG = {
    "seed_model": 2201,
    "seed_train": 2202,
    "seed_validation": 2203,
    "seed_noise": 2204,
    "seed_starts": [2205, 2206],
    "samples_train": 120,
    "samples_validation": 160,
    "noise_std_Nm": 0.03,
    "prior_strength": 1e-6,
    "ftol": 1e-10,
    "xtol": 1e-10,
    "gtol": 1e-10,
    "max_nfev": 200,
    "max_seconds_per_fit": 20.0,
    "repair_eigenvalue_floor": 1e-6,
    "coordinate_bounds": [-12.0, 12.0],
    "physical_tolerance": 1e-8,
    "go_heldout_vs_sdp_factor": 1.05,
    "go_clean_rmse_Nm": 1e-5,
    "go_runtime_seconds": 20.0,
}
KEYS = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]


def pseudo(p):
    """Independent oracle: rotational inertia about origin -> second moment."""
    inertia = np.array([[p[4], p[5], p[7]], [p[5], p[6], p[8]], [p[7], p[8], p[9]]])
    out = np.empty((4, 4))
    out[:3, :3] = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    out[:3, 3] = out[3, :3] = p[1:4]
    out[3, 3] = p[0]
    return out


def dynamic(P):
    inertia = np.trace(P[:3, :3]) * np.eye(3) - P[:3, :3]
    return np.array(
        [
            P[3, 3],
            *P[:3, 3],
            inertia[0, 0],
            inertia[0, 1],
            inertia[1, 1],
            inertia[0, 2],
            inertia[1, 2],
            inertia[2, 2],
        ]
    )


def encode(p):
    # Reverse Cholesky gives P = U U.T with U upper triangular.
    U = np.linalg.cholesky(pseudo(p)[::-1, ::-1])[::-1, ::-1]
    alpha = np.log(U[3, 3])
    U = U / U[3, 3]
    return np.array(
        [
            alpha,
            *np.log(np.diag(U)[:3]),
            U[0, 1],
            U[1, 2],
            U[0, 2],
            U[0, 3],
            U[1, 3],
            U[2, 3],
        ]
    )


def decode(z):
    return np.concatenate(
        [pin.LogCholeskyParameters(x).toDynamicParameters() for x in z.reshape(-1, 10)]
    )


def derivative(z):
    return block_diag(
        *[pin.LogCholeskyParameters(x).calculateJacobian() for x in z.reshape(-1, 10)]
    )


def repair(p):
    blocks, changes = [], []
    for x in p.reshape(-1, 10):
        eig, V = np.linalg.eigh(pseudo(x))
        fixed = dynamic((V * np.maximum(eig, CONFIG["repair_eigenvalue_floor"])) @ V.T)
        blocks.append(fixed)
        changes.append(float(np.linalg.norm(fixed - x)))
    return np.concatenate(blocks), changes


def verdict(p):
    output = []
    for x in p.reshape(-1, 10):
        P = pseudo(x)
        mass = float(x[0])
        minimum = float(np.linalg.eigvalsh(P).min())
        # Independent centre-of-mass second moment / principal triangle check.
        if mass > 0:
            C = P[:3, :3] - np.outer(x[1:4], x[1:4]) / mass
            moments = np.linalg.eigvalsh(np.trace(C) * np.eye(3) - C)
            margin = float(moments[0] + moments[1] - moments[2])
        else:
            margin = None
        output.append(
            {
                "mass": mass,
                "min_pseudo_eigenvalue": minimum,
                "triangle_margin": margin,
                "feasible": mass > 0
                and minimum >= -CONFIG["physical_tolerance"]
                and margin is not None
                and margin >= -CONFIG["physical_tolerance"],
            }
        )
    return output


def trajectory(model, seed, samples, weak=False):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, samples)
    amplitude = rng.uniform(0.15, 0.45, (model.nv, 3))
    frequency = rng.uniform(0.4, 2.3, (model.nv, 3))
    phase = rng.uniform(-np.pi, np.pi, (model.nv, 3))
    argument = t[:, None, None] * frequency + phase
    q = (amplitude * np.sin(argument)).sum(axis=2)
    v = (amplitude * frequency * np.cos(argument)).sum(axis=2)
    a = (-amplitude * frequency**2 * np.sin(argument)).sum(axis=2)
    if weak:
        q[:, 1:] = v[:, 1:] = a[:, 1:] = 0
    data = model.createData()
    Y, tau = [], []
    for qi, vi, ai in zip(q, v, a):
        Y.append(pin.computeJointTorqueRegressor(model, data, qi, vi, ai).copy())
        tau.append(pin.rnea(model, data, qi, vi, ai).copy())
    # Explicit joint-major stacking, matching FIGAROH's regression contract.
    Y = np.stack(Y).transpose(1, 0, 2).reshape(model.nv * samples, -1)
    tau = np.asarray(tau).T.reshape(-1)
    return Y, tau, (q, v, a)


def metrics(p, Y, tau, Yv, tauv, nv):
    train = (Y @ p - tau).reshape(nv, -1)
    validation = (Yv @ p - tauv).reshape(nv, -1)
    return {
        "train_rmse": float(np.sqrt(np.mean(train**2))),
        "heldout_rmse": float(np.sqrt(np.mean(validation**2))),
        "train_per_joint_rmse": np.sqrt(np.mean(train**2, axis=1)).tolist(),
        "heldout_per_joint_rmse": np.sqrt(np.mean(validation**2, axis=1)).tolist(),
        "physical": verdict(p),
        "parameters": p.tolist(),
    }


def fit(Y, tau, prior, scale, start):
    begin = time.perf_counter()
    z0 = np.concatenate([encode(p) for p in start.reshape(-1, 10)])
    sqrt_prior = np.sqrt(CONFIG["prior_strength"])
    calls = {"residual": 0, "jacobian": 0}

    def residual(z):
        calls["residual"] += 1
        if time.perf_counter() - begin > CONFIG["max_seconds_per_fit"]:
            raise TimeoutError("fit exceeded time budget")
        p = decode(z)
        if not np.isfinite(p).all():
            raise FloatingPointError("nonfinite dynamic parameters")
        return np.r_[Y @ p - tau, sqrt_prior * (p - prior) / scale]

    def jacobian(z):
        calls["jacobian"] += 1
        D = derivative(z)
        return np.vstack([Y @ D, sqrt_prior * D / scale[:, None]])

    try:
        result = least_squares(
            residual,
            z0,
            jac=jacobian,
            method="trf",
            x_scale="jac",
            bounds=CONFIG["coordinate_bounds"],
            ftol=CONFIG["ftol"],
            xtol=CONFIG["xtol"],
            gtol=CONFIG["gtol"],
            max_nfev=CONFIG["max_nfev"],
        )
        p = decode(result.x)
        return p, {
            "success": bool(result.success),
            "status": int(result.status),
            "message": result.message,
            "objective": float(result.cost),
            "nfev": result.nfev,
            "njev": result.njev,
            "optimality": float(result.optimality),
            "calls": calls,
            "runtime_seconds": time.perf_counter() - begin,
        }
    except (TimeoutError, FloatingPointError, ValueError) as error:
        return start.copy(), {
            "success": False,
            "message": str(error),
            "calls": calls,
            "runtime_seconds": time.perf_counter() - begin,
        }


def run():
    model = pin.buildSampleModelManipulator()
    rng = np.random.default_rng(CONFIG["seed_model"])
    truth_z = np.zeros((model.njoints - 1, 10))
    truth_z[:, 0] = rng.uniform(-0.3, 0.5, len(truth_z))
    truth_z[:, 1:4] = rng.uniform(-1.8, -0.8, (len(truth_z), 3))
    truth_z[:, 4:] = rng.normal(0, 0.08, (len(truth_z), 6))
    prior = decode((truth_z + rng.normal(0, 0.12, truth_z.shape)).ravel())
    scale = np.tile([1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], len(truth_z))
    # Verify inverse and Jacobian independently before any fit.
    for z in truth_z:
        p = pin.LogCholeskyParameters(z).toDynamicParameters()
        np.testing.assert_allclose(encode(p), z, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            pseudo(p),
            pin.LogCholeskyParameters(z).toPseudoInertia().toMatrix(),
            rtol=1e-10,
            atol=1e-12,
        )
    z = truth_z.ravel()
    step = 1e-6
    Dfd = np.column_stack(
        [
            (decode(z + step * d) - decode(z - step * d)) / (2 * step)
            for d in np.eye(len(z))
        ]
    )
    np.testing.assert_allclose(derivative(z), Dfd, rtol=1e-5, atol=1e-7)
    records, fixtures = [], {}
    for case in ["clean", "noisy", "weak_excitation", "near_boundary_truth"]:
        case_z = truth_z.copy()
        if case == "near_boundary_truth":
            case_z[:, 1] -= 4
        truth = decode(case_z.ravel())
        for j, p in enumerate(truth.reshape(-1, 10), 1):
            model.inertias[j] = pin.Inertia.FromDynamicParameters(p)
        Y, clean, states = trajectory(
            model,
            CONFIG["seed_train"],
            CONFIG["samples_train"],
            weak=case == "weak_excitation",
        )
        Yv, tauv, statesv = trajectory(
            model, CONFIG["seed_validation"], CONFIG["samples_validation"]
        )
        np.testing.assert_allclose(Y @ truth, clean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Yv @ truth, tauv, rtol=1e-10, atol=1e-12)
        tau = clean.copy()
        if case != "clean":
            tau += np.random.default_rng(CONFIG["seed_noise"]).normal(
                0, CONFIG["noise_std_Nm"], tau.shape
            )
        singular = np.linalg.svd(Y, compute_uv=False)
        rank = int(np.sum(singular > singular[0] * 1e-10))
        begin = time.perf_counter()
        # OLS predictions, with null-space component anchored to the same nominal prior.
        ols = prior + np.linalg.lstsq(Y, tau - Y @ prior, rcond=1e-10)[0]
        records.append(
            {
                "case": case,
                "method": "ols",
                "runtime_seconds": time.perf_counter() - begin,
                "success": True,
                **metrics(ols, Y, tau, Yv, tauv, model.nv),
            }
        )
        begin = time.perf_counter()
        projected, reports = [], []
        for x, w in zip(ols.reshape(-1, 10), (1 / scale).reshape(-1, 10)):
            p, report = project_p10_lmi(x, weights=w)
            projected.append(p)
            reports.append({"status": report.status, "objective": report.objective})
        records.append(
            {
                "case": case,
                "method": "ols_sdp",
                "runtime_seconds": time.perf_counter() - begin,
                "success": all(r["status"] == "projected" for r in reports),
                "projection": reports,
                **metrics(np.concatenate(projected), Y, tau, Yv, tauv, model.nv),
            }
        )
        repaired, changes = repair(ols)
        boundary_z = np.concatenate([encode(x) for x in prior.reshape(-1, 10)]).reshape(
            -1, 10
        )
        boundary_z[:, 1:4] -= 3
        starts = [
            ("nominal", prior, []),
            ("repaired_ols", repaired, changes),
            ("near_boundary", decode(boundary_z.ravel()), []),
        ]
        for seed in CONFIG["seed_starts"]:
            perturbed = truth_z + np.random.default_rng(seed).normal(
                0, 0.3, truth_z.shape
            )
            starts.append((f"perturbed_{seed}", decode(perturbed.ravel()), []))
        for name, start, repairs in starts:
            p, report = fit(Y, tau, prior, scale, start)
            records.append(
                {
                    "case": case,
                    "method": "log_cholesky",
                    "initialization": name,
                    "repair_p10_norm_per_link": repairs,
                    "initial_physical": verdict(start),
                    **report,
                    **metrics(p, Y, tau, Yv, tauv, model.nv),
                }
            )
        fixtures[case] = {
            "truth": truth.tolist(),
            "rank": rank,
            "columns": Y.shape[1],
            "observed_condition": float(singular[0] / singular[rank - 1]),
            "data_sha256": hashlib.sha256(
                b"".join(x.tobytes() for x in [Y, tau, Yv, tauv, *states, *statesv])
            ).hexdigest(),
        }
    return {
        "config": CONFIG,
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
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": {
            "builder": "buildSampleModelManipulator",
            "joint_names": list(model.names)[1:],
            "placements": [
                np.asarray(x.homogeneous).tolist() for x in model.jointPlacements
            ],
            "gravity": model.gravity.vector.tolist(),
            "prior": prior.tolist(),
            "p10_order": KEYS,
            "row_order": "joint-major",
            "scale": scale.tolist(),
        },
        "oracle_jacobian_max_abs_error": float(np.max(np.abs(derivative(z) - Dfd))),
        "fixtures": fixtures,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
