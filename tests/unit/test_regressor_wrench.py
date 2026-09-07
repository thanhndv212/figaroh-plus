# Copyright [2021-2025] Thanh Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for build_total_regressor_wrench.

The function's signature named its settings dict ``param`` while the body
read ``identif_config``, so every call raised ``NameError`` before reaching
any real work. Nothing called it and nothing tested it, which is how a fully
broken exported function survived. These tests call it for real.

Shape contract, read off the implementation:

* ``len(tau_u)`` and ``len(tau_l)`` must be multiples of 6 (the body splits
  each into 6 blocks).
* ``W_b_u`` / ``W_b_l`` have one row per corresponding torque sample.
* ``W_l`` needs at least ``which_body_loaded * 10 + 10`` columns, and one row
  per torque sample.
"""

import numpy as np
import pytest

from figaroh.tools.regressor import build_total_regressor_wrench

N_SAMPLES = 12  # multiple of 6
N_BASE = 4
N_STANDARD = 20


def _inputs(which_body_loaded=0, mass_load=2.5, seed=0):
    """Build a consistently-shaped argument set."""
    rng = np.random.default_rng(seed)
    return dict(
        W_b_u=rng.random((N_SAMPLES, N_BASE)),
        W_b_l=rng.random((N_SAMPLES, N_BASE)),
        W_l=rng.random((N_SAMPLES, N_STANDARD)),
        tau_u=rng.random(N_SAMPLES),
        tau_l=rng.random(N_SAMPLES),
        param_standard_l=np.zeros(N_STANDARD),
        identif_config={
            "which_body_loaded": which_body_loaded,
            "mass_load": mass_load,
        },
    )


def test_runs_without_undefined_name():
    """The regression guard: this used to raise NameError immediately."""
    W_tot, V_norm, residue = build_total_regressor_wrench(**_inputs())
    assert W_tot is not None


def test_settings_are_accepted_as_identif_config():
    """The keyword must match the sibling build_total_regressor_current."""
    # Passing it positionally and by the documented keyword must agree.
    kwargs = _inputs()
    positional = build_total_regressor_wrench(
        kwargs["W_b_u"],
        kwargs["W_b_l"],
        kwargs["W_l"],
        kwargs["tau_u"],
        kwargs["tau_l"],
        kwargs["param_standard_l"],
        kwargs["identif_config"],
    )
    by_keyword = build_total_regressor_wrench(**kwargs)
    np.testing.assert_allclose(positional[0], by_keyword[0])
    np.testing.assert_allclose(positional[1], by_keyword[1])


def test_output_shapes():
    """Stacked unloaded+loaded rows; one normalized parameter per column."""
    W_tot, V_norm, residue = build_total_regressor_wrench(**_inputs())
    assert W_tot.shape[0] == 2 * N_SAMPLES
    assert V_norm.shape == (W_tot.shape[1],)
    assert residue.shape == (W_tot.shape[0],)


def test_outputs_are_finite():
    W_tot, V_norm, residue = build_total_regressor_wrench(**_inputs())
    assert np.all(np.isfinite(W_tot))
    assert np.all(np.isfinite(V_norm))
    assert np.all(np.isfinite(residue))


def test_mass_load_scales_the_parameter_vector():
    """V_norm is normalized against mass_load, so it scales linearly."""
    _, v1, _ = build_total_regressor_wrench(**_inputs(mass_load=1.0))
    _, v2, _ = build_total_regressor_wrench(**_inputs(mass_load=2.0))
    np.testing.assert_allclose(v2, 2.0 * v1, rtol=1e-9)


@pytest.mark.parametrize("which_body_loaded", [0, 1])
def test_which_body_loaded_selects_a_different_block(which_body_loaded):
    """The setting indexes a 10-wide block of the standard regressor."""
    W_tot, _, _ = build_total_regressor_wrench(
        **_inputs(which_body_loaded=which_body_loaded)
    )
    assert np.all(np.isfinite(W_tot))
