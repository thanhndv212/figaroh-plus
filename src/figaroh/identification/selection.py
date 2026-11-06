# Copyright [2021-2026] Thanh Nguyen
# Copyright [2022-2024] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which estimate an identification run reports (#61).

``tasks.identification.select_stage`` picks the estimate that validation,
``verify()``, the reports, the run archive and the URDF export use:

* ``fit`` (default): the base-parameter least-squares fit, unchanged.
* ``reconstruction``: the full parameter vector reconstructed from the fit,
  accepted only when it is physically consistent and was produced by the
  requested method.
* ``physical_fit``: a physically consistent effort fit chosen by
  ``physical_fit.method``: ``direct`` (default), the direct LMI effort fit
  (:func:`figaroh.identification.physical_fit.solve_direct_effort_fit`), or
  ``log_cholesky``, the experimental boundary-aware log-Cholesky fit
  (:func:`figaroh.identification.log_cholesky_fit.solve_log_cholesky_fit`,
  #25). Both minimise the same objective on the same training rows; the
  result names the method and, for ``log_cholesky``, the branch.

A requested stage that does not meet its acceptance conditions is
*rejected*: nothing else is substituted, the selected stage is ``none`` and
the verdict fails. The fit numbers stay under their usual result keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STAGE = "fit"
BASE_REL_TOL = 1e-6


@dataclass
class SelectedEstimate:
    """The estimate chosen by ``select_stage`` and the verdict on it.

    Attributes:
        stage: Stage whose numbers are reported: ``fit``, ``reconstruction``
            or ``physical_fit`` when accepted, ``none`` when rejected.
        requested: The stage that was asked for.
        status: ``accepted`` or ``rejected``.
        reason: Why, in one line (names the solver(s) and the failed test).
        space: ``standard`` (values are the full standard parameter
            vector).
        names: Parameter names aligned with ``values``.
        values: The candidate vector; kept for diagnosis when rejected.
        phi_base_equivalent: ``M @ theta`` for a standard candidate
            (diagnostic only, never used for prediction).
        base_residual_rel: Relative distance of ``phi_base_equivalent`` from
            the fitted base parameters.
        feasibility: Per link ``{mass, min_eig, ok}`` for standard candidates.
        effective_method: Method that produced a reconstruction, or the
            ``physical_fit.method`` (``direct``/``log_cholesky``) of a
            physical fit.
        solvers: Solver names tried for a physical fit.
        psd_eig_tol: Pseudo-inertia eigenvalue tolerance the links were
            judged feasible with; export uses the same one.
    """

    stage: str
    requested: str
    status: str
    reason: str
    space: str
    names: List[str] = field(default_factory=list)
    values: Optional[np.ndarray] = None
    phi_base_equivalent: Optional[np.ndarray] = None
    base_residual_rel: Optional[float] = None
    feasibility: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    effective_method: Optional[str] = None
    solvers: List[str] = field(default_factory=list)
    psd_eig_tol: float = -1e-10
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def predict(self, W_full: np.ndarray) -> np.ndarray:
        """Effort predicted by this estimate: the full regressor times the
        standard parameter vector (never ``phi_base_equivalent``)."""
        if self.values is None:
            raise ValueError("no estimate values to predict with")
        return W_full @ self.values

    def as_dict(self) -> Dict[str, Any]:
        def arr(x):
            return None if x is None else [float(v) for v in np.asarray(x).ravel()]

        d = {
            "stage": self.stage,
            "requested": self.requested,
            "status": self.status,
            "reason": self.reason,
            "space": self.space,
            "names": list(self.names),
            "values": arr(self.values),
            "phi_base_equivalent": arr(self.phi_base_equivalent),
            "base_residual_rel": self.base_residual_rel,
            "feasibility": {
                k: {
                    "mass": float(v["mass"]),
                    "min_eig": float(v["min_eig"]),
                    "ok": bool(v["ok"]),
                }
                for k, v in self.feasibility.items()
            },
            "effective_method": self.effective_method,
            "solvers": list(self.solvers),
        }
        d.update(self.extra)
        return d

    def parameter_dict(self) -> Dict[str, float]:
        """``{name: value}`` of the candidate (empty when no values)."""
        if self.values is None:
            return {}
        return {n: float(v) for n, v in zip(self.names, self.values)}

    def link_p10(self, joint_names: Sequence[str]) -> Dict[str, np.ndarray]:
        """Per-link 10-vectors of a standard-space estimate."""
        if self.space != "standard":
            raise ValueError("link inertias need a standard-space estimate")
        from figaroh.identification.physical_consistency import (
            p10_by_joint_from_param_dict,
        )

        return p10_by_joint_from_param_dict(
            parameter_dict=self.parameter_dict(), joint_names=joint_names
        )


def requested_stage(identif) -> str:
    """The configured ``select_stage`` (``fit`` when absent)."""
    cfg = getattr(identif, "identif_config", None) or {}
    return str(cfg.get("select_stage") or DEFAULT_STAGE)


def physical_fit_config(identif) -> Dict[str, Any]:
    """The ``physical_fit`` settings with every key present.

    The config parser already merges the defaults; a programmatic config
    may hold a partial mapping (or none), which is completed here. Raises
    ``ValueError`` for an unknown key or a setting that
    :func:`~figaroh.identification.config.check_physical_fit_config`
    rejects, so a bad programmatic config fails as clearly as a parsed one.
    """
    from figaroh.identification.config import (
        PHYSICAL_FIT_DEFAULTS,
        check_physical_fit_config,
    )

    raw = (getattr(identif, "identif_config", None) or {}).get("physical_fit")
    cfg = dict(PHYSICAL_FIT_DEFAULTS)
    if isinstance(raw, dict):
        unknown = set(raw) - set(PHYSICAL_FIT_DEFAULTS)
        if unknown:
            raise ValueError(
                f"unknown physical_fit keys {sorted(unknown)}; "
                f"allowed: {sorted(PHYSICAL_FIT_DEFAULTS)}"
            )
        cfg.update(raw)
    return check_physical_fit_config(cfg)


def physical_fit_wanted(identif) -> bool:
    return requested_stage(identif) == "physical_fit" or bool(
        physical_fit_config(identif)["enabled"]
    )


def _nominal_vector(identif) -> np.ndarray:
    return np.array(list(identif.standard_parameter.values()), dtype=float)


def _equivalent_base(identif, theta_std: np.ndarray):
    """``M @ theta`` over the active parameters and its distance to phi_base."""
    names = list(identif.standard_parameter.keys())
    M = getattr(identif, "_M_matrix", None)
    params_r = getattr(identif, "_params_r_for_recon", None)
    if M is None or params_r is None:
        return None, None
    pos = {n: i for i, n in enumerate(names)}
    theta_r = np.array([theta_std[pos[p]] for p in params_r], dtype=float)
    equiv = np.asarray(M, dtype=float) @ theta_r
    phi = np.asarray(identif.phi_base, dtype=float).reshape(-1)
    rel = float(np.linalg.norm(equiv - phi) / max(np.linalg.norm(phi), 1e-300))
    return equiv, rel


def _link_feasibility(identif, theta_std, mass_min, psd_eig_tol):
    """Feasibility of the links that carry an identified parameter.

    Links whose values are only the nominal CAD ones (no parameter in
    ``params_r``) were not estimated and are not judged. A mass on its lower
    bound passes within the solver tolerance
    (:func:`~figaroh.identification.physical_consistency.mass_bound_tolerance`).
    """
    from figaroh.identification.physical_consistency import (
        check_p10_feasibility,
        mass_bound_tolerance,
        p10_by_joint_from_param_dict,
    )
    from figaroh.identification.reconstruction import _INERTIAL_KEYS

    names = list(identif.standard_parameter.keys())
    joints = list(identif.model.names[1:])
    params_r = getattr(identif, "_params_r_for_recon", None)
    if params_r is not None:
        identified = set(params_r)
        joints = [
            j for j in joints if any(f"{k}_{j}" in identified for k in _INERTIAL_KEYS)
        ]
    p10 = p10_by_joint_from_param_dict(
        parameter_dict=dict(zip(names, map(float, theta_std))),
        joint_names=joints,
    )
    out = {}
    for joint, v in p10.items():
        rep = check_p10_feasibility(
            v,
            mass_min=mass_min - mass_bound_tolerance(mass_min),
            psd_eig_tol=psd_eig_tol,
        )
        out[joint] = {
            "mass": rep.mass,
            "min_eig": rep.min_eig,
            "ok": rep.status == "feasible",
        }
    return out


# ---------------------------------------------------------------------------
# physical_fit adapter
# ---------------------------------------------------------------------------


def log_cholesky_summary(record) -> Dict[str, Any]:
    """JSON-ready branch, detection, start and solver metadata of a
    log-Cholesky :class:`~figaroh.identification.physical_fit.SolveRecord`.

    Reports the numerical rank and pinned null-space dimension separately
    from per-link feasibility, the links that sent the fit to the boundary,
    start repairs and, for the interior branch, the stop reason, evaluation
    counts and identifiable residual distance.
    """
    notes = getattr(record, "notes", None) or {}
    det = notes.get("detection") or {}
    start = notes.get("start") or {}
    out: Dict[str, Any] = {
        "branch": notes.get("branch"),
        "branch_forced": bool(notes.get("branch_forced", False)),
        "rank": det.get("rank"),
        "n_params": det.get("n_params"),
        "null_space_dim": det.get("null_space_dim"),
        "rank_rtol": det.get("rank_rtol"),
        "eigenvalue_margin": det.get("eigenvalue_margin"),
        "unconstrained_min_eig": det.get("unconstrained_min_eig"),
        "active_links": {
            j: list(v.get("active", []))
            for j, v in (det.get("per_link") or {}).items()
            if v.get("active")
        },
        "start_used": bool(start.get("used", False)),
        "start_source": start.get("source"),
        "start_repairs": dict(start.get("repairs") or {}),
        "interior": None,
    }
    interior = notes.get("interior")
    if interior:
        keys = (
            "stop_reason",
            "nfev",
            "njev",
            "identifiable_rms",
            "scipy_status",
            "scipy_message",
            "max_abs_log_scale",
            "runtime_s",
        )
        out["interior"] = {k: interior[k] for k in keys if k in interior}
    import json

    return json.loads(json.dumps(out, default=float))


def _run_log_cholesky(problem, cfg, solvers):
    """One log-Cholesky fit; returns ``(record, steps)``."""
    from figaroh.identification.config import log_cholesky_options
    from figaroh.identification.log_cholesky_fit import solve_log_cholesky_fit

    record = solve_log_cholesky_fit(problem, options=log_cholesky_options(cfg))
    branch = record.notes.get("branch")
    status = record.solver_status
    if record.exception:
        status = f"{status} ({record.exception})"
    if branch == "boundary":
        step = f"log_cholesky boundary branch ({solvers[0]} SDP): {status}"
    else:
        step = f"log_cholesky interior branch (scipy): {status}"
    return record, [step]


def run_physical_fit(identif) -> Dict[str, Any]:
    """Solve the configured physical fit on the data the fit used.

    Returns an outcome dict: ``record`` (a
    :class:`~figaroh.identification.physical_fit.SolveRecord` or ``None``),
    ``method``, ``solvers`` (names tried), ``reason``, ``accepted`` and,
    for ``method: log_cholesky``, a ``log_cholesky`` summary
    (:func:`log_cholesky_summary`). Solver failures do not raise; an invalid
    ``physical_fit`` config does. The training rows (``_solve_W``,
    ``_solve_tau``) are scattered into all standard columns (zeros where
    columns were eliminated); held-out data never enters. Friction and
    offsets are free parameters and the prior (also the log-Cholesky start)
    is the nominal standard parameter vector.
    """
    cfg = physical_fit_config(identif)
    method = cfg["method"]
    solvers = [str(cfg["solver"])]
    outcome: Dict[str, Any] = {
        "record": None,
        "method": method,
        "solvers": solvers,
        "reason": "",
        "accepted": False,
        "config": cfg,
    }
    try:
        from figaroh.identification import physical_fit as pf

        names = list(identif.standard_parameter.keys())
        n = len(names)
        W = np.asarray(identif._solve_W, dtype=float)
        tau = np.asarray(identif._solve_tau, dtype=float).reshape(-1)
        eliminated = set(int(i) for i in (identif._idx_eliminated or []))
        keep = [i for i in range(n) if i not in eliminated]
        if W.shape[1] != len(keep):
            raise ValueError(
                f"processed regressor has {W.shape[1]} columns, expected {len(keep)}"
            )
        Y = np.zeros((W.shape[0], n))
        Y[:, keep] = W
        problem = pf.build_problem(
            Y,
            tau,
            names,
            list(identif.model.names[1:]),
            prior=_nominal_vector(identif),
            policy=pf.PhysicalPolicy(
                mass_min=float(cfg["mass_min"]),
                prior_weight=float(cfg["prior_weight"]),
            ),
            row_weight=getattr(identif, "_wls_row_weight", None),
        )
        if method == "log_cholesky":
            record, steps = _run_log_cholesky(problem, cfg, solvers)
            outcome["log_cholesky"] = log_cholesky_summary(record)
            if record.notes.get("branch") != "boundary":
                solvers[:] = ["scipy"]
        else:
            opts = {}
            if cfg["max_seconds"] is not None:
                opts["max_seconds"] = float(cfg["max_seconds"])
            record = pf.solve_direct_effort_fit(problem, solver=solvers[0], **opts)
            steps = [f"{solvers[0]}: {record.solver_status}"]
            if record.solver_status != "optimal" and cfg["second_solver"]:
                second = str(cfg["second_solver"])
                solvers.append(second)
                record = pf.solve_direct_effort_fit(problem, solver=second, **opts)
                steps.append(f"{second}: {record.solver_status}")
        outcome["record"] = record
        outcome["psd_eig_tol"] = problem.policy.feas_tol
        outcome["accepted"] = bool(
            record.solver_status == "optimal" and record.accepted
        )
        verdict = (
            "all links feasible"
            if outcome["accepted"]
            else (
                "a link is not physically feasible"
                if record.solver_status == "optimal"
                else "no optimal solution"
            )
        )
        outcome["reason"] = "; ".join(steps) + f"; {verdict}"
    except Exception as exc:  # solver stack missing, shape problems, ...
        logger.warning("physical fit failed: %s", exc)
        outcome["reason"] = f"{type(exc).__name__}: {exc}"
    return outcome


def physical_fit_result(out: Dict[str, Any]) -> Dict[str, Any]:
    """The ``result["physical_fit"]`` entry for a :func:`run_physical_fit`
    outcome: verdict, method, solver status, elapsed time, objective,
    per-link feasibility and the log-Cholesky summary when present."""
    record = out.get("record")
    info = {
        "status": "accepted" if out["accepted"] else "rejected",
        "method": out.get("method", "direct"),
        "reason": out["reason"],
        "solvers": list(out["solvers"]),
        "solver_status": None if record is None else record.solver_status,
        "runtime_s": None if record is None else float(record.runtime_s),
        "objective_value": None if record is None else record.objective_value,
        "feasibility": {} if record is None else record.as_dict()["feasibility"],
    }
    if "log_cholesky" in out:
        info["log_cholesky"] = out["log_cholesky"]
    return info


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _select_reconstruction(identif) -> SelectedEstimate:
    recon = getattr(identif, "_recon_result", None)
    rejected = dict(
        stage="none",
        requested="reconstruction",
        status="rejected",
        space="standard",
        names=list(identif.standard_parameter.keys()),
    )
    if recon is None:
        return SelectedEstimate(reason="reconstruction did not run", **rejected)
    rcfg = identif.identif_config.get("reconstruction") or {}
    requested = str(rcfg.get("method", "nullspace")).lower().strip()
    mass_min = float(rcfg.get("mass_min", 1e-6))
    psd_tol = float(rcfg.get("psd_eig_tol", -1e-10))

    theta = _nominal_vector(identif)
    names = rejected["names"]
    pos = {n: i for i, n in enumerate(names)}
    for name, val in recon.as_dict().items():
        theta[pos[name]] = val
    equiv, _ = _equivalent_base(identif, theta)
    resid_rel = float(
        np.linalg.norm(recon.residual) / max(np.linalg.norm(identif.phi_base), 1e-300)
    )
    feas = _link_feasibility(identif, theta, mass_min, psd_tol)

    problems = []
    swapped = False
    if recon.status != "ok":
        problems.append(f"reconstruction status {recon.status}")
    eff = recon.effective_method
    if requested != "auto" and eff is not None and eff != requested:
        swapped = True
        problems.append(f"requested {requested} but {eff} was used")
    if resid_rel > BASE_REL_TOL:
        problems.append(f"base residual {resid_rel:.2e} > {BASE_REL_TOL:.0e}")
    bad = [j for j, v in feas.items() if not v["ok"]]
    if bad:
        problems.append("infeasible links: " + ", ".join(bad))

    common = dict(
        values=theta,
        phi_base_equivalent=equiv,
        base_residual_rel=resid_rel,
        feasibility=feas,
        effective_method=eff,
        psd_eig_tol=psd_tol,
        extra={"swapped_method": swapped},
    )
    if problems:
        return SelectedEstimate(reason="; ".join(problems), **{**rejected, **common})
    return SelectedEstimate(
        stage="reconstruction",
        requested="reconstruction",
        status="accepted",
        reason=f"{eff} reconstruction, all links feasible",
        space="standard",
        names=names,
        **common,
    )


def _select_physical_fit(identif) -> SelectedEstimate:
    out = getattr(identif, "_physical_fit", None)
    names = list(identif.standard_parameter.keys())
    rejected = dict(
        stage="none",
        requested="physical_fit",
        status="rejected",
        space="standard",
        names=names,
    )
    if out is None:
        return SelectedEstimate(reason="physical fit did not run", **rejected)
    record = out.get("record")
    solvers = list(out.get("solvers", []))
    method = out.get("method", "direct")
    extra: Dict[str, Any] = {"physical_fit_method": method}
    if "log_cholesky" in out:
        extra["log_cholesky_branch"] = out["log_cholesky"].get("branch")
    if record is None or record.theta is None:
        return SelectedEstimate(
            reason=out["reason"],
            solvers=solvers,
            effective_method=method,
            extra=extra,
            **rejected,
        )
    theta = np.asarray(record.theta, dtype=float)
    equiv, rel = _equivalent_base(identif, theta)
    feas = {
        j: {"mass": v["mass"], "min_eig": v["min_eig"], "ok": v["ok"]}
        for j, v in record.feasibility.items()
    }
    common = dict(
        values=theta,
        phi_base_equivalent=equiv,
        base_residual_rel=rel,
        feasibility=feas,
        solvers=solvers,
        effective_method=method,
        psd_eig_tol=float(out.get("psd_eig_tol", -1e-10)),
        extra=extra,
    )
    if out["accepted"]:
        return SelectedEstimate(
            stage="physical_fit",
            requested="physical_fit",
            status="accepted",
            reason=out["reason"],
            **{**common, "names": names, "space": "standard"},
        )
    return SelectedEstimate(reason=out["reason"], **{**rejected, **common})


def stage_label(selected: Dict[str, Any]) -> str:
    """Display name of a selected-estimate dict (``SelectedEstimate.as_dict``).

    A log-Cholesky physical fit names its method and branch, e.g.
    ``physical_fit (log_cholesky, boundary branch)``; the default ``direct``
    method keeps the plain ``physical_fit`` label. A rejection reads
    ``none (requested ...: rejected)``.
    """
    if selected.get("status") != "accepted":
        return f"none (requested {selected.get('requested')}: rejected)"
    stage = str(selected.get("stage"))
    if stage != "physical_fit" or selected.get("physical_fit_method") in (
        None,
        "direct",
    ):
        return stage
    branch = selected.get("log_cholesky_branch")
    method = selected["physical_fit_method"]
    detail = f"{method}, {branch} branch" if branch else str(method)
    return f"{stage} ({detail})"


def select_estimate(identif) -> SelectedEstimate:
    """Build the :class:`SelectedEstimate` for the configured stage."""
    stage = requested_stage(identif)
    if stage == DEFAULT_STAGE:
        raise ValueError("select_stage='fit' reports the base fit; nothing to select")
    if stage == "reconstruction":
        return _select_reconstruction(identif)
    if stage == "physical_fit":
        return _select_physical_fit(identif)
    raise ValueError(f"select_stage={stage!r} is not selectable")
