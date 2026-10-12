"""regularization_coefficient: default 0; non-zero removed in 1.0 (#120, #185)."""

import warnings

import pytest

from figaroh.calibration.config import (
    _extract_calibration_params,
    regularization_coefficient,
)
from figaroh.utils.error_handling import ConfigurationError


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
def test_zero_or_unset_is_accepted(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert regularization_coefficient(value) == value


def test_nonzero_is_an_error_pointing_to_map():
    with pytest.raises(ConfigurationError, match="estimation.method: map"):
        _parse({"regularization_coefficient": 0.01})
