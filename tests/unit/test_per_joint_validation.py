"""Per-joint held-out identification error (#103).

Pooled validation numbers mix units and are dominated by the largest
torques. On synthetic data where one joint's effort is pure noise, the
per-joint metrics flag that joint and the others stay predictive.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from test_data_contract_wiring import _Ident, _trajectory  # noqa: E402


class _WithValidation(_Ident):
    validation_to_return = None

    def load_trajectory_data(self, data_source=None):
        if data_source is not None:
            return self.validation_to_return
        return self.trajectory_to_return


def _noisy_last_joint(traj, seed):
    effort = traj.effort.copy()
    effort[:, -1] = np.random.default_rng(seed).normal(0.0, 2.0, len(effort))
    return replace(traj, effort=effort)


@pytest.fixture(scope="module")
def ident():
    import pinocchio as pin

    model = pin.buildSampleModelManipulator()
    ident = _WithValidation(model)
    ident.trajectory_to_return = _noisy_last_joint(_trajectory(model), seed=1)
    ident.validation_to_return = _noisy_last_joint(_trajectory(model), seed=2)
    ident.identif_config["validation_data_file"] = "held-out"
    ident.initialize()
    ident.solve(decimate=False, plotting=False)
    return ident


def test_unpredictable_joint_is_flagged(ident):
    val = ident.result["validation_metrics"]
    assert val["validation_source"] == "validation_data"
    names = val["joint_names"]
    assert val["unpredictable_joints"] == [names[-1]]
    noisy = val["per_joint"][names[-1]]
    assert not noisy["predictive"] and noisy["nrmse"] >= 1.0
    for name in names[:-1]:
        m = val["per_joint"][name]
        assert m["predictive"] and m["r2"] > 0.9
        assert m["unit"] == "N·m"  # revolute joints


def test_verdict_uses_the_per_joint_view(ident):
    verdict = ident.verify(scope="execution")
    val = ident.result["validation_metrics"]
    assert verdict.metrics["validation_correlation"] == pytest.approx(
        val["correlation_normalised"]
    )
    assert verdict.metrics["validation_unpredictable_joints"] == 1.0
    # the pooled value is kept for existing readers
    assert "correlation" in val


def test_reports_flag_the_joint(ident, tmp_path, capsys):
    name = ident.result["validation_metrics"]["joint_names"][-1]
    html = Path(ident.export_html_report(output_path=str(tmp_path / "r.html")))
    text = html.read_text()
    assert "not better than a constant" in text
    assert f"Held-out effort of {name}" in text
    ident.print_quality_report()
    out = capsys.readouterr().out
    assert "<- not better than a constant" in out
