r"""QR decomposition utilities for robot parameter identification.

This module provides QR-based helpers to:

- identify a set of base parameters (a full-rank subset) from a rank-deficient
    regressor
- express those base parameters as linear combinations of the original/full
    parameters

In particular, both QR workflows compute a matrix $M$ such that:

$$
\phi_{base} = M\,\theta_{full}
$$

where columns of $M$ correspond to the *remaining* parameter vector ordered as
``params_r``.

Note:
    In FIGAROH, ``params_r`` typically refers to the list of
    *standard parameter
    names remaining after column elimination* (e.g., after removing all-zero
    regressor columns). It is not necessarily the complete set of standard
    parameters (which is often available as a dictionary like ``params_std``).

    Use :meth:`QRDecomposer.expand_mapping_matrix_to_full` if you need an "M"
    defined over a full, canonical parameter ordering.
The resulting row ordering matches the returned base-parameter expressions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg

# Constants
TOL_QR = 1e-8
TOL_BETA = 1e-6


@dataclass
class QRResult:
    """Structured output of a QR base-parameter decomposition.

    All numerical fields are stored at full float64 precision.  Rounding for
    display is done only at the expression-building stage.

    Attributes:
        rank: Identified numerical rank of the regressor.
        base_indices: Column indices into ``params_r`` that form the
            independent (base) set.
        pivot_order: Full pivot permutation used by the pivoting path
            (``P`` from ``scipy.linalg.qr``); ``None`` for the double path.
        W_b: Base regressor matrix, shape ``(m, rank)``.
        beta: Dependency coefficient matrix, shape ``(rank, n - rank)``.
            Full precision — not rounded.
        M: Base mapping matrix satisfying ``phi_base = M @ theta_r``,
            shape ``(rank, n)``.
        base_param_expressions: Human-readable expression strings, length
            ``rank``.
        phi_b: Identified base-parameter values, shape ``(rank,)``; ``None``
            when computed without a ``tau`` vector.
        phi_b_nom: Nominal base-parameter values from ``params_std`` prior,
            shape ``(rank,)``; ``None`` when ``params_std`` was not provided.
        method: ``"pivoting"`` or ``"double"``.
        diag_R: Absolute diagonal of ``R`` at the factorisation point,
            shape ``(min(m, n),)``.  Useful for stability diagnostics.
        cond_R1: Condition number of the ``R[:rank, :rank]`` block.  Measures
            how well-conditioned the base part of the factorisation is.
    """

    rank: int
    base_indices: List[int]
    pivot_order: Optional[List[int]]
    W_b: np.ndarray
    beta: np.ndarray
    M: np.ndarray
    base_param_expressions: List[str]
    phi_b: Optional[np.ndarray]
    phi_b_nom: Optional[np.ndarray]
    method: str
    diag_R: np.ndarray
    cond_R1: float


class QRDecomposer:
    """Enhanced QR decomposition handler for robot parameter identification."""

    def __init__(
        self,
        tolerance: float = TOL_QR,
        beta_tolerance: float = TOL_BETA,
        relative_tolerance: Optional[float] = None,
    ):
        self.tolerance = tolerance
        self.beta_tolerance = beta_tolerance
        self.relative_tolerance = relative_tolerance

        # Last computed base-mapping matrix and its labels.
        # M maps the remaining parameter vector (ordered as params_r) to base.
        self.M: Optional[np.ndarray] = None
        self.M_params_r: Optional[List[str]] = None
        self.M_base_params_expr: Optional[List[str]] = None
        self.M_method: Optional[str] = None
        # Diagnostics from the last decomposition (available via get_diagnostics)
        self._last_diag_R: Optional[np.ndarray] = None
        self._last_cond_R1: Optional[float] = None

    def get_M(self) -> Optional[np.ndarray]:
        """Return the last computed base mapping matrix ``M`` (or None)."""
        return self.M

    def get_M_labels(self) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Return (base_param_expr, params_r) labels for the last stored M."""
        return self.M_base_params_expr, self.M_params_r

    @staticmethod
    def _build_M_from_partition(
        *,
        beta: np.ndarray,
        base_indices: List[int],
        regroup_indices: List[int],
        n_params: int,
    ) -> np.ndarray:
        """Build M for a base/dependent column partition.

        Produces an M such that ``phi_base = M @ theta_r`` where ``theta_r``
        is ordered as ``params_r``.
        """
        rank = len(base_indices)
        M = np.zeros((rank, n_params), dtype=float)

        for row_idx, param_idx in enumerate(base_indices):
            M[row_idx, param_idx] = 1.0

        if beta.size:
            for dep_col, param_idx in enumerate(regroup_indices):
                M[:, param_idx] = beta[:, dep_col]

        return M

    @staticmethod
    def _build_M_from_pivoting(
        *,
        beta: np.ndarray,
        P: np.ndarray,
        rank: int,
        n_params: int,
    ) -> np.ndarray:
        """Build M from pivoted QR outputs.

        In pivoted ordering, ``phi_base = [I, beta] @ theta_sorted``.
        This method maps that back to the original ``params_r`` ordering.
        """
        M_sorted = np.zeros((rank, n_params), dtype=float)
        M_sorted[:, :rank] = np.eye(rank)
        if beta.size:
            M_sorted[:, rank:] = beta

        M = np.zeros_like(M_sorted)
        for sorted_idx, original_idx in enumerate(P.tolist()):
            M[:, original_idx] = M_sorted[:, sorted_idx]
        return M

    def decompose_with_pivoting(
        self, tau: np.ndarray, W_e: np.ndarray, params_r: List[str]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform QR decomposition with column pivoting.

        This identifies an effective rank, computes base parameters, and
        returns a base regressor ``W_b`` along with a dict mapping
        base-parameter expressions to values.

        Args:
            tau: Torque/effort vector, shape (m,).
            W_e: Full regressor matrix, shape (m, n).
            params_r: Remaining parameter names (columns of W_e), length n.

        Returns:
            (W_b, base_parameters) where:
              - W_b has shape (m, r) with r = rank(W_e)
              - base_parameters maps expression strings (length r) to values.
        """
        Q, R, P = linalg.qr(W_e, pivoting=True)

        # Reorder parameters according to pivoting
        params_sorted = [params_r[P[i]] for i in range(P.shape[0])]

        # Find effective rank
        rank = self._find_rank(R)

        # Extract base components
        R1, Q1, R2 = self._extract_base_components(R, Q, rank)

        # Compute base parameters
        beta = np.around(np.linalg.solve(R1, R2), 6)
        phi_b = np.round(np.linalg.solve(R1, Q1.T @ tau), 6)
        W_b = Q1 @ R1

        # Build parameter expressions
        base_params = self._build_parameter_expressions(
            params_sorted[:rank], params_sorted[rank:], beta
        )

        # Store mapping matrix M such that: phi_base = M @ theta_r
        n_params = len(params_r)
        M = self._build_M_from_pivoting(
            beta=beta,
            P=P,
            rank=rank,
            n_params=n_params,
        )

        self.M = M
        self.M_params_r = list(params_r)
        self.M_base_params_expr = list(base_params)
        self.M_method = "pivoting"

        return W_b, dict(zip(base_params, phi_b))

    def double_decomposition(
        self,
        tau: np.ndarray,
        W_e: np.ndarray,
        params_r: List[str],
        params_std: Optional[Dict[str, float]] = None,
    ) -> Union[
        Tuple[np.ndarray, Dict, List, np.ndarray],
        Tuple[np.ndarray, Dict, List, np.ndarray, np.ndarray],
    ]:
        """Perform the "double QR" workflow for base-parameter identification.

        This follows a two-stage procedure:
            1) Identify an independent/base subset via QR on ``W_e``.
            2) Regroup columns and run a second QR to compute linear
               dependencies.

        The output base parameters are returned as expression strings built
        from the original parameter names.

        Args:
            tau: Torque/effort vector, shape (m,).
            W_e: Full regressor matrix, shape (m, n).
            params_r: Remaining parameter names (columns of W_e), length n.
            params_std: Optional mapping from full parameter name -> value. If
                provided, also returns the standard-parameter values of
                the base expressions.

                Returns:
                        If params_std is None:
                            (W_b, base_parameters, params_base_expr, phi_b)
                        else:
                            (W_b, base_parameters, params_base_expr, phi_b,
                             phi_b_nom)

            Where:
              - W_b has shape (m, r)
              - params_base_expr is a list[str] of length r
              - phi_b is a vector of length r
                            - phi_b_nom is length r if params_std is provided
        """

        # First QR to identify base parameters
        base_indices, regroup_indices = self._identify_base_parameters(
            W_e, params_r
        )

        # Regroup and second QR
        (
            W_base,
            W_regroup,
            params_base,
            params_regroup,
        ) = self._regroup_parameters(
            W_e,
            params_r,
            base_indices,
            regroup_indices,
        )

        # Second QR decomposition
        W_regrouped = np.c_[W_base, W_regroup]
        Q_r, R_r = np.linalg.qr(W_regrouped)

        rank = len(base_indices)
        R1, Q1, R2 = self._extract_base_components(R_r, Q_r, rank)

        # Compute parameters
        beta = np.around(np.linalg.solve(R1, R2), 6)
        phi_b = np.round(np.linalg.solve(R1, Q1.T @ tau), 6)
        W_b = Q1 @ R1

        # Verify consistency
        assert np.allclose(W_base, W_b), "Base regressor calculation error"

        # Build expressions and compute standard parameters if provided
        params_base_expr = self._build_parameter_expressions(
            params_base, params_regroup, beta
        )
        base_parameters = dict(zip(params_base_expr, phi_b))

        # Store mapping matrix M such that: phi_base = M @ theta_r
        n_params = len(params_r)
        M = self._build_M_from_partition(
            beta=beta,
            base_indices=base_indices,
            regroup_indices=regroup_indices,
            n_params=n_params,
        )

        self.M = M
        self.M_params_r = list(params_r)
        self.M_base_params_expr = list(params_base_expr)
        self.M_method = "double"

        if params_std is not None:
            phi_b_nom = self.compute_nominal_base_parameters(
                params_base, params_regroup, beta, params_std
            )

            return W_b, base_parameters, params_base_expr, phi_b, phi_b_nom

        return W_b, base_parameters, params_base_expr, phi_b

    def _find_rank(self, R: np.ndarray) -> int:
        """Find effective numerical rank from an upper-triangular R.

        A diagonal entry with absolute value <= ``self.tolerance`` is treated
        as zero.  Returns 0 for an all-zero (or empty) matrix.

        When ``self.relative_tolerance`` is set, the threshold is
        ``relative_tolerance * |R[0, 0]|`` (largest pivot), which is more
        robust across different regressor scalings.  The absolute
        ``self.tolerance`` is used as a floor so that a near-zero leading
        diagonal still yields rank 0.
        """
        diag_R = np.abs(np.diag(R))
        if self.relative_tolerance is not None and diag_R.size > 0:
            thr = max(self.relative_tolerance * diag_R[0], self.tolerance)
        else:
            thr = self.tolerance
        return int(np.sum(diag_R > thr))

    def _extract_base_components(
        self, R: np.ndarray, Q: np.ndarray, rank: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract (R1, Q1, R2) blocks used for base/dependency computations.

        Given a QR of the form W = Q R (with Q orthonormal), and a chosen
        numerical rank r:
          - R1 = R[:r, :r]
          - Q1 = Q[:, :r]
          - R2 = R[:r, r:]
        """
        R1 = R[:rank, :rank]
        Q1 = Q[:, :rank]
        R2 = (
            R[:rank, rank:]
            if rank < R.shape[1]
            else np.array([]).reshape(rank, 0)
        )
        return R1, Q1, R2

    def _identify_base_parameters(
        self, W_e: np.ndarray, params_r: List[str]
    ) -> Tuple[List[int], List[int]]:
        """Identify base and dependent (regrouped) parameter indices.

        This uses a non-pivoted QR and a diagonal threshold on R to select
        columns. The returned indices are positions in ``params_r`` / columns
        of ``W_e``.
        """
        Q, R = np.linalg.qr(W_e)
        diag_R = np.abs(np.diag(R))

        base_indices = [
            i for i, val in enumerate(diag_R) if val > self.tolerance
        ]
        regroup_indices = [
            i for i, val in enumerate(diag_R) if val <= self.tolerance
        ]

        return base_indices, regroup_indices

    def _regroup_parameters(
        self,
        W_e: np.ndarray,
        params_r: List[str],
        base_indices: List[int],
        regroup_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Split W_e into base and dependent column blocks.

        Returns:
            (W_base, W_regroup, params_base, params_regroup)
        """
        W_base = W_e[:, base_indices]
        W_regroup = (
            W_e[:, regroup_indices]
            if regroup_indices
            else np.array([]).reshape(W_e.shape[0], 0)
        )

        params_base = [params_r[i] for i in base_indices]
        params_regroup = [params_r[i] for i in regroup_indices]

        return W_base, W_regroup, params_base, params_regroup

    def _build_parameter_expressions(
        self,
        base_params: List[str],
        regroup_params: List[str],
        beta: np.ndarray,
    ) -> List[str]:
        r"""Build human-readable base parameter expressions.

        Each returned expression corresponds to:

        $$
        \phi_{base,i} = \theta_{base,i} + \sum_j \beta_{i,j}\,\theta_{dep,j}
        $$

        where ``base_params`` are the independent columns and
        ``regroup_params`` are the dependent columns in the chosen ordering.
        """
        expressions = base_params.copy()

        for i, base_param in enumerate(expressions):
            for j, regroup_param in enumerate(regroup_params):
                if j < beta.shape[1] and abs(beta[i, j]) > self.beta_tolerance:
                    sign = " - " if beta[i, j] < 0 else " + "
                    coefficient = abs(beta[i, j])
                    expressions[i] += f"{sign}{coefficient}*{regroup_param}"

        return expressions

    def compute_nominal_base_parameters(
        self,
        base_params: List[str],
        regroup_params: List[str],
        beta: np.ndarray,
        params_std: Dict[str, float],
    ) -> np.ndarray:
        """Compute numerical values of base parameters from priors.

        This computes the numerical value of each base expression using:

            phi_std[i] = params_std[base_param]
                         + sum_j beta[i, j] * params_std[dep_param]
        """
        phi_std = [params_std[param] for param in base_params]

        for i in range(len(phi_std)):
            for j, regroup_param in enumerate(regroup_params):
                if j < beta.shape[1]:
                    phi_std[i] += beta[i, j] * params_std[regroup_param]

        return np.around(phi_std, 5)

    def get_base_mapping_matrix_pivoting(
        self, W_e: np.ndarray, params_r: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        r"""Return the base-parameter mapping matrix M for pivoting-QR.

        The matrix ``M`` satisfies ``phi_base = M @ theta_r`` where
        ``theta_r`` is ordered as ``params_r`` (the remaining parameters).

        Row order matches the returned base-parameter expression strings.

        Args:
            W_e: Full regressor matrix, shape (m, n).
            params_r: Remaining parameter names (columns of W_e), length n.

        Returns:
            (M, base_params_expr)
              - M has shape (r, n) where r is the identified rank
              - base_params_expr is a list[str] of length r
        """
        _, R, P = linalg.qr(W_e, pivoting=True)
        rank = self._find_rank(R)

        # Dependency coefficients in the pivoted ordering
        R1 = R[:rank, :rank]
        R2 = (
            R[:rank, rank:]
            if rank < R.shape[1]
            else np.array([]).reshape(rank, 0)
        )
        beta = np.around(np.linalg.solve(R1, R2), 6)
        n_params = len(params_r)
        M = self._build_M_from_pivoting(
            beta=beta,
            P=P,
            rank=rank,
            n_params=n_params,
        )

        params_sorted = [params_r[P[i]] for i in range(P.shape[0])]
        base_params_expr = self._build_parameter_expressions(
            params_sorted[:rank], params_sorted[rank:], beta
        )
        return M, base_params_expr

    def get_base_mapping_matrix_double(
        self, W_e: np.ndarray, params_r: List[str]
    ) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
        """Return the base-parameter mapping matrix M for the double-QR
        workflow.

        The returned ``M`` satisfies ``phi_base = M @ theta_r`` with columns
        ordered as ``params_r`` (the remaining parameters).

        Args:
            W_e: Full regressor matrix, shape (m, n).
            params_r: Remaining parameter names (columns of W_e), length n.

        Returns:
            (M, base_params_expr, base_indices, regroup_indices)
              - M has shape (r, n)
              - base_params_expr is a list[str] of length r
              - base_indices and regroup_indices index into params_r
        """
        base_indices, regroup_indices = self._identify_base_parameters(
            W_e, params_r
        )
        (
            W_base,
            W_regroup,
            params_base,
            params_regroup,
        ) = self._regroup_parameters(
            W_e,
            params_r,
            base_indices,
            regroup_indices,
        )

        W_regrouped = np.c_[W_base, W_regroup]
        _, R_r = np.linalg.qr(W_regrouped)
        rank = len(base_indices)

        R1 = R_r[:rank, :rank]
        R2 = (
            R_r[:rank, rank:]
            if rank < R_r.shape[1]
            else np.array([]).reshape(rank, 0)
        )
        beta = np.around(np.linalg.solve(R1, R2), 6)

        n_params = len(params_r)
        M = self._build_M_from_partition(
            beta=beta,
            base_indices=base_indices,
            regroup_indices=regroup_indices,
            n_params=n_params,
        )

        base_params_expr = self._build_parameter_expressions(
            params_base, params_regroup, beta
        )
        return M, base_params_expr, base_indices, regroup_indices

    def get_base_mapping_matrix(
        self, W_e: np.ndarray, params_r: List[str], method: str = "double"
    ) -> Tuple[np.ndarray, List[str]]:
        """Convenience wrapper to retrieve the base mapping matrix.

        Args:
            W_e: Full regressor matrix, shape (m, n).
            params_r: Remaining parameter names (columns of W_e), length n.
            method: "double" (default) or "pivoting".

        Returns:
            (M, base_params_expr)

        Raises:
            ValueError: if method is not recognized.
        """
        method_norm = method.strip().lower()
        if method_norm in {"double", "double_qr", "double-qr"}:
            M, base_params_expr, _, _ = self.get_base_mapping_matrix_double(
                W_e, params_r
            )
            return M, base_params_expr
        if method_norm in {"pivoting", "pivot", "qr_pivoting", "qr-pivoting"}:
            return self.get_base_mapping_matrix_pivoting(W_e, params_r)
        raise ValueError(
            f"Unknown method={method!r}. Expected 'double' or 'pivoting'."
        )

    @staticmethod
    def expand_mapping_matrix_to_full(
        M: np.ndarray,
        params_r: List[str],
        full_param_names: List[str],
    ) -> np.ndarray:
        """Expand an M defined on ``params_r`` into a full parameter ordering.

        This is useful when the QR routines operate on a reduced set of
        parameters (after removing all-zero columns), but you want a mapping
        matrix defined over the complete standard-parameter list.

        Args:
            M: Base mapping matrix for remaining parameters, shape (r, n_r).
            params_r: Remaining parameter names (columns of ``M``), length n_r.
            full_param_names: Full/canonical parameter ordering to expand into.

        Returns:
            M_full of shape (r, n_full) where columns not present in
            ``params_r`` are set to 0.

        Raises:
            ValueError: if M has incompatible shape or if a name in params_r is
                missing from full_param_names.
        """
        if M.ndim != 2:
            raise ValueError("M must be a 2D array")
        if M.shape[1] != len(params_r):
            raise ValueError(
                "M has incompatible column count: "
                f"M.shape[1]={M.shape[1]} but len(params_r)={len(params_r)}"
            )

        full_index = {name: idx for idx, name in enumerate(full_param_names)}
        missing = [name for name in params_r if name not in full_index]
        if missing:
            raise ValueError(
                "params_r contains names not present in full_param_names: "
                f"{', '.join(missing)}"
            )

        M_full = np.zeros((M.shape[0], len(full_param_names)), dtype=M.dtype)
        for j_r, name in enumerate(params_r):
            M_full[:, full_index[name]] = M[:, j_r]
        return M_full


# Backward compatibility functions
def QR_pivoting(
    tau: np.ndarray,
    W_e: np.ndarray,
    params_r: list,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy QR pivoting function for backward compatibility."""
    decomposer = QRDecomposer(tolerance=tol_qr)
    return decomposer.decompose_with_pivoting(tau, W_e, params_r)


def double_QR(
    tau: np.ndarray,
    W_e: np.ndarray,
    params_r: list,
    params_std: dict = None,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy double QR function for backward compatibility."""
    decomposer = QRDecomposer(tolerance=tol_qr)
    return decomposer.double_decomposition(tau, W_e, params_r, params_std)


def get_baseParams(
    W_e: np.ndarray,
    params_r: list,
    params_std: dict = None,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy function for getting base parameters."""
    decomposer = QRDecomposer(tolerance=tol_qr)
    dummy_tau = np.zeros(W_e.shape[0])  # Not used in this function
    result = decomposer.double_decomposition(
        dummy_tau, W_e, params_r, params_std
    )
    base_indices, _ = decomposer._identify_base_parameters(W_e, params_r)
    return result[0], result[2], base_indices


def get_baseIndex(
    W_e: np.ndarray,
    params_r: list,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy function for getting base indices."""
    decomposer = QRDecomposer(tolerance=tol_qr)
    base_indices, _ = decomposer._identify_base_parameters(W_e, params_r)
    return tuple(base_indices)


def build_baseRegressor(W_e: np.ndarray, idx_base: tuple) -> np.ndarray:
    """Legacy function for building base regressor."""
    return W_e[:, list(idx_base)]


def cond_num(W_b: np.ndarray, norm_type: str = None) -> float:
    """Calculate condition number with various norms."""
    if norm_type == "fro":
        return np.linalg.cond(W_b, "fro")
    elif norm_type == "max_over_min_sigma":
        return np.linalg.cond(W_b, 2) / np.linalg.cond(W_b, -2)
    else:
        return np.linalg.cond(W_b)
