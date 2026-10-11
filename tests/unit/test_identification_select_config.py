"""Parsing of tasks.identification.select_stage / physical_fit (#61)."""

import pytest

from figaroh.identification.config import (
    PHYSICAL_FIT_DEFAULTS,
    _extract_selection_config,
)


def _parse(src):
    out = {}
    _extract_selection_config(out, src)
    return out


def test_absent_keys_write_nothing():
    assert _parse({}) == {}
    assert _parse({"select_stage": None}) == {}


@pytest.mark.parametrize("stage", ["fit", "reconstruction", "physical_fit"])
def test_selectable_stages(stage):
    assert _parse({"select_stage": stage}) == {"select_stage": stage}


def test_stage_case_insensitive():
    assert _parse({"select_stage": " Physical_Fit "})["select_stage"] == "physical_fit"


@pytest.mark.parametrize("stage", ["projected", "bogus", ""])
def test_unselectable_stage_raises(stage):
    with pytest.raises(ValueError, match="select_stage"):
        _parse({"select_stage": stage})


def test_projected_message_names_reason():
    with pytest.raises(ValueError, match="nominal model"):
        _parse({"select_stage": "projected"})


def test_physical_fit_defaults_merge():
    out = _parse({"physical_fit": {"enabled": True, "second_solver": "qics"}})
    pf = out["physical_fit"]
    assert pf["enabled"] is True and pf["second_solver"] == "qics"
    assert pf["solver"] == "cvxopt" and pf["mass_min"] == 1e-6
    assert set(pf) == set(PHYSICAL_FIT_DEFAULTS)


def test_physical_fit_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown physical_fit"):
        _parse({"physical_fit": {"nonsense": 1}})
