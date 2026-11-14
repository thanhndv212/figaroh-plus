"""solve_with_custom_solver follows the same path as solve (#253, Q1.2).

It used to repeat part of solve by hand: the sample mask was not applied,
no ``fit`` record was written, the base-column selection was not stored (so
validation metrics were skipped) and records of an earlier solve survived.
"""

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from test_data_contract_wiring import N, _Ident, _trajectory  # noqa: E402

MASKED = 50


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


def _ident(model, masked=True, **config):
    mask = None
    if masked:
        mask = np.ones(N, bool)
        mask[100 : 100 + MASKED] = False
    ident = _Ident(model)
    ident.trajectory_to_return = _trajectory(model, mask=mask)
    ident.initialize()
    ident.identif_config.update(config)
    return ident


def _stages(ident):
    return {s.stage: s for s in ident.stages}


@pytest.mark.parametrize("decimate", [False, True])
def test_masked_samples_are_left_out(model, decimate):
    reference = _ident(model)
    reference.solve(decimate=decimate, decimation_factor=4, plotting=False)
    custom = _ident(model)
    custom.solve_with_custom_solver(decimate=decimate, decimation_factor=4)
    assert len(custom.tau_noised) == len(reference.tau_noised)
    if not decimate:
        assert len(custom.tau_noised) == model.nv * (N - MASKED)
    np.testing.assert_allclose(custom.tau_noised, reference.tau_noised)
    np.testing.assert_allclose(
        custom.dynamic_regressor_base, reference.dynamic_regressor_base
    )


def test_lstsq_gives_the_base_parameters_of_solve(model):
    joints = [model.names[j] for j in (1, 2, 4)]
    reference = _ident(model, torque_fit_joints=joints)
    reference.solve(decimate=False, plotting=False)
    custom = _ident(model, torque_fit_joints=joints)
    phi = custom.solve_with_custom_solver(decimate=False)
    assert custom.params_base == reference.params_base
    assert list(custom.result["base parameters"]) == reference.params_base
    # solve rounds its estimate to six decimals
    np.testing.assert_allclose(phi, reference.phi_base, atol=1e-6)
    np.testing.assert_allclose(
        list(custom.result["base parameters"].values()), phi, rtol=0, atol=0
    )
    np.testing.assert_allclose(custom.result["torque estimated"], custom.tau_identif)


def test_fit_and_validation_are_recorded(model):
    ident = _ident(model)
    ident.solve_with_custom_solver(method="ridge", alpha=1e-6, decimate=False)
    stages = _stages(ident)
    assert list(stages) == ["data", "fit", "validation"]
    fit = stages["fit"]
    assert fit.status == "ok" and fit.reason == "custom solver: ridge"
    assert fit.metrics["rows"]["value"] == model.nv * (N - MASKED)
    assert fit.metrics["base_parameters"]["value"] == len(ident.phi_base)
    assert [d["stage"] for d in ident.result["stages"]] == list(stages)
    # the base-column selection is stored, so the metrics are computed
    assert ident._base_indices is not None
    assert "validation_metrics" in ident.result
    assert ident.result["solver_method"] == "ridge"
    assert ident.result["solver_info"]["alpha"] == 1e-6
    assert ident._run_started_at <= ident._run_finished_at


def test_nothing_of_an_earlier_solve_survives(model):
    ident = _ident(model)
    ident.solve(decimate=True, decimation_factor=4, plotting=False)
    decimated_rows = len(ident.tau_noised)
    ident._M_matrix = "stale"
    ident._base_indices = "stale"
    ident.solve_with_custom_solver(decimate=False)
    fit = _stages(ident)["fit"]
    assert fit.reason == "custom solver: lstsq"
    assert fit.metrics["rows"]["value"] == len(ident.tau_noised) != decimated_rows
    assert ident._decimate_used is False
    assert not isinstance(ident._M_matrix, str)
    assert not isinstance(ident._base_indices, str)


def test_a_failing_custom_solve_records_the_failure(model):
    ident = _ident(model)
    ident.solve(decimate=False, plotting=False)
    assert _stages(ident)["fit"].status == "ok"
    with pytest.raises(ValueError):
        # three bounds for many more base parameters
        ident.solve_with_custom_solver(
            method="constrained", bounds=[(0, 1)] * 3, decimate=False
        )
    fit = _stages(ident)["fit"]
    assert fit.status == "failed" and "ValueError" in fit.reason


def test_a_refused_option_leaves_the_earlier_run_untouched(model):
    ident = _ident(model)
    ident.solve(decimate=False, plotting=False)
    before = (ident._run_started_at, ident._solve_W, _stages(ident)["fit"])
    with pytest.raises(ValueError, match="bounds"):
        ident.solve_with_custom_solver(method="ridge", alpha=0.1, bounds=(0, None))
    assert (ident._run_started_at, ident._solve_W, _stages(ident)["fit"]) == before


def test_relative_qr_tolerance_applies(model, monkeypatch):
    from figaroh.identification import identification_tools

    seen = []
    real = identification_tools.identification_qr_decomposer

    def spy(identif_config, **kwargs):
        seen.append(identif_config.get("qr_relative_tolerance"))
        return real(identif_config, **kwargs)

    monkeypatch.setattr(identification_tools, "identification_qr_decomposer", spy)
    ident = _ident(model, qr_relative_tolerance=1e-4)
    ident.solve_with_custom_solver(decimate=False)
    assert seen == [1e-4]
