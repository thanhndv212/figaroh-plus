"""SelectedEstimate data class (#61)."""

import json

import numpy as np

from figaroh.identification.selection import SelectedEstimate


def test_predict_uses_the_full_regressor_and_standard_values():
    W = np.arange(12.0).reshape(4, 3)
    std = SelectedEstimate(
        "physical_fit",
        "physical_fit",
        "accepted",
        "ok",
        "standard",
        names=["a", "b", "c"],
        values=np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(std.predict(W), W @ std.values)


def test_as_dict_is_json_and_rejected_reports_none():
    est = SelectedEstimate(
        "none",
        "reconstruction",
        "rejected",
        "infeasible links: j1",
        "standard",
        names=["m_j1"],
        values=np.array([-1.0]),
        feasibility={"j1": {"mass": -1.0, "min_eig": -1.0, "ok": False}},
    )
    d = json.loads(json.dumps(est.as_dict()))
    assert not est.accepted
    assert d["stage"] == "none" and d["requested"] == "reconstruction"
    assert d["feasibility"]["j1"]["ok"] is False
