"""BaseIdentification.export_urdf and the export stage (#61)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

pytest.importorskip("picos")
sys.path.insert(0, os.path.dirname(__file__))

from test_data_contract_wiring import _trajectory  # noqa: E402
from test_estimate_selection import _break_solver, _run  # noqa: E402

PENDULUM = str(Path(__file__).resolve().parent.parent / "fixtures" / "pendulum.urdf")


@pytest.fixture(scope="module")
def model():
    return pin.buildModelFromUrdf(PENDULUM)


@pytest.fixture(scope="module")
def traj(model):
    return _trajectory(model)


def _stage(ident, name):
    return {s.stage: s for s in ident.stages}[name]


def test_export_reloads_as_the_selected_estimate(model, traj, tmp_path):
    ident = _run(model, traj, select_stage="physical_fit")
    assert ident.selected.accepted
    out = ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))

    reloaded = pin.buildModelFromUrdf(out)
    data = reloaded.createData()
    rng = np.random.default_rng(1)
    from figaroh.tools.regressor import build_regressor_basic

    q = rng.uniform(-1, 1, (20, model.nq))
    dq = rng.uniform(-1, 1, (20, model.nv))
    ddq = rng.uniform(-1, 1, (20, model.nv))
    W = build_regressor_basic(ident.robot, q, dq, ddq, ident.identif_config)
    want = W @ ident.selected.values  # joint-major, 20 samples per joint
    got = np.stack(
        [pin.rnea(reloaded, data, q[i], dq[i], ddq[i]) for i in range(20)]
    ).T.reshape(-1)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)

    st = _stage(ident, "export")
    assert st.status == "ok" and st.artifacts == [out]
    assert ident.verify(scope="execution").stages["export"] == "pass"


def test_export_of_the_fit_is_refused_and_recorded(model, traj, tmp_path):
    ident = _run(model, traj)
    with pytest.raises(ValueError, match="export failed"):
        ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))
    assert _stage(ident, "export").status == "failed"
    assert not (tmp_path / "o.urdf").exists()


def test_export_of_a_rejected_estimate_is_refused(model, traj, tmp_path, monkeypatch):
    _break_solver(monkeypatch)
    ident = _run(model, traj, select_stage="physical_fit")
    with pytest.raises(ValueError, match="rejected"):
        ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))
    assert _stage(ident, "export").status == "failed"
    assert not (tmp_path / "o.urdf").exists()
