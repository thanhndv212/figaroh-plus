# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

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
* ``physical_fit``: the direct LMI effort fit
  (:func:`figaroh.identification.physical_fit.solve_direct_effort_fit`).

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
        space: ``base`` (values are base parameters) or ``standard`` (values
            are the full standard parameter vector).
        names: Parameter names aligned with ``values``.
        values: The candidate vector; kept for diagnosis when rejected.
        phi_base_equivalent: ``M @ theta`` for a standard candidate
            (diagnostic only, never used for prediction).
        base_residual_rel: Relative distance of ``phi_base_equivalent`` from
            the fitted base parameters.
        feasibility: Per link ``{mass, min_eig, ok}`` for standard candidates.
        effective_method: Method that produced a reconstruction.
        solvers: Solver names tried for a physical fit.
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
    base_indices: Optional[Sequence[int]] = field(default=None, repr=False)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def predict(self, W_full: np.ndarray, W_reduced: np.ndarray) -> np.ndarray:
        """Effort predicted by this estimate.

        A base-space estimate uses the reduced regressor's base columns; a
        standard-space estimate uses the full regressor. The standard
        candidates are never evaluated through ``phi_base_equivalent``.
        """
        if self.values is None:
            raise ValueError("no estimate values to predict with")
        if self.space == "base":
            return W_reduced[:, list(self.base_indices)] @ self.values
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
    from figaroh.identification.config import PHYSICAL_FIT_DEFAULTS

    cfg = dict(PHYSICAL_FIT_DEFAULTS)
    raw = (getattr(identif, "identif_config", None) or {}).get("physical_fit")
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg


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
    from figaroh.identification.physical_consistency import (
        check_p10_feasibility,
        p10_by_joint_from_param_dict,
    )

    names = list(identif.standard_parameter.keys())
    p10 = p10_by_joint_from_param_dict(
        parameter_dict=dict(zip(names, map(float, theta_std))),
        joint_names=list(identif.model.names[1:]),
    )
    out = {}
    for joint, v in p10.items():
        rep = check_p10_feasibility(v, mass_min=mass_min, psd_eig_tol=psd_eig_tol)
        out[joint] = {
            "mass": rep.mass,
            "min_eig": rep.min_eig,
            "ok": rep.status == "feasible",
        }
    return out


# ---------------------------------------------------------------------------
# physical_fit adapter
# ---------------------------------------------------------------------------


def run_physical_fit(identif) -> Dict[str, Any]:
    """Solve the direct LMI effort fit on the data the fit used.

    Returns an outcome dict (never raises): ``record`` (a
    :class:`~figaroh.identification.physical_fit.SolveRecord` or ``None``),
    ``solvers`` (names tried), ``reason`` and ``accepted``. The processed
    regressor is scattered into all standard columns (zeros where columns
    were eliminated); friction and offsets are free parameters and the
    prior is the nominal standard parameter vector.
    """
    cfg = physical_fit_config(identif)
    solvers = [str(cfg["solver"])]
    outcome: Dict[str, Any] = {
        "record": None,
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
        outcome["accepted"] = bool(record.solver_status == "optimal" and record.accepted)
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


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _select_fit(identif) -> SelectedEstimate:
    return SelectedEstimate(
        stage="fit",
        requested="fit",
        status="accepted",
        reason="base-parameter least-squares fit",
        space="base",
        names=list(identif.params_base),
        values=np.asarray(identif.phi_base, dtype=float),
        base_indices=list(identif._base_indices),
    )


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
        return SelectedEstimate(
            reason="reconstruction did not run", **rejected
        )
    rcfg = (identif.identif_config.get("reconstruction") or {})
    requested = str(rcfg.get("method", "nullspace")).lower().strip()
    mass_min = float(rcfg.get("mass_min", 1e-6))
    psd_tol = float(rcfg.get("psd_eig_tol", -1e-10))

    theta = _nominal_vector(identif)
    names = rejected["names"]
    pos = {n: i for i, n in enumerate(names)}
    for name, val in recon.as_dict().items():
        theta[pos[name]] = val
    equiv, rel = _equivalent_base(identif, theta)
    resid_rel = float(
        np.linalg.norm(recon.residual)
        / max(np.linalg.norm(identif.phi_base), 1e-300)
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
        extra={"swapped_method": swapped},
    )
    if problems:
        return SelectedEstimate(
            reason="; ".join(problems), **{**rejected, **common}
        )
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
    if record is None or record.theta is None:
        return SelectedEstimate(reason=out["reason"], solvers=solvers, **rejected)
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


def select_estimate(identif) -> SelectedEstimate:
    """Build the :class:`SelectedEstimate` for the configured stage."""
    stage = requested_stage(identif)
    if stage == "fit":
        return _select_fit(identif)
    if stage == "reconstruction":
        return _select_reconstruction(identif)
    if stage == "physical_fit":
        return _select_physical_fit(identif)
    raise ValueError(f"select_stage={stage!r} is not selectable")
