"""regularization_coefficient: default 0, deprecated when set (#120)."""

import warnings

import pytest

from figaroh.calibration.config import (
    _extract_calibration_params,
    regularization_coefficient,
)


def _parse(parameters):
    cfg = {}
    _extract_calibration_params(cfg, None, parameters)
    return cfg


def test_default_is_zero_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = _parse({"calibration_level": "joint_offset"})
    assert cfg["coeff_regularize"] == 0.0


@pytest.mark.parametrize("value", [0, 0.0, None])
def test_zero_or_unset_does_not_warn(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert regularization_coefficient(value) == value


def test_nonzero_is_kept_and_points_to_map():
    with pytest.warns(DeprecationWarning, match="estimation.method: map"):
        cfg = _parse({"regularization_coefficient": 0.01})
    assert cfg["coeff_regularize"] == 0.01
