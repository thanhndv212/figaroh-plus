"""Behavioral regressions for evidence-based, scoped acceptance."""

from types import SimpleNamespace
import numpy as np
import pytest
from figaroh.tools._report_common import evaluate_thresholds
from figaroh.identification.base_identification import BaseIdentification


def identifier(independent=False, prediction=(1.0, 2.0)):
    result = {
        "condition number": 100000.0,
        "rmse norm (N/m)": 0.1,
        "base parameters names": ["p"],
        "base parameters values": np.array([1.0]),
        "torque estimated": np.array(prediction),
        "torque processed": np.array([1.0, 2.0]),
        "validation_metrics": {
            "validation_source": (
                "validation_data" if independent else "identification_data_fallback"
            ),
            "joint_names": ["joint"],
            "n_val_samples": 2,
            "tau_measured_per_joint": {"joint": [1.0, 2.0]},
            "tau_identified_per_joint": {"joint": list(prediction)},
            "improvement_pct": 0.0,
            "correlation": 1.0,
        },
    }
    return SimpleNamespace(
        result=result,
        std_relative=[],
        robot_name="test",
        identif_config={},
        _config_file_path=None,
    )


def test_good_nominal_and_large_raw_condition_do_not_reject_execution():
    v = BaseIdentification.verify(identifier(), scope="execution")
    assert v.passed and v.scope == "execution"
    assert v.stages["prediction"] == "not_evaluated"
    assert not any(
        c.name
        in ("condition_number", "validation_improvement_pct", "validation_correlation")
        for c in v.checks
    )


def test_empty_policy_has_no_acceptance_evidence():
    v = evaluate_thresholds({}, {})
    assert not v.passed and v.status == "not_evaluated"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_required_metric_is_failure(value):
    v = evaluate_thresholds(
        {"error": value}, {"error": {"threshold": 1.0, "comparison": "max"}}
    )
    assert not v.passed and v.status == "fail"


def test_missing_required_and_optional_checks_are_explicit():
    v = evaluate_thresholds(
        {"error": 0.1},
        {
            "error": {"threshold": 1.0, "comparison": "max"},
            "extra": {"threshold": 1.0, "comparison": "max", "required": False},
        },
    )
    assert v.passed and v.checks[1].status == "not_evaluated"
    v = evaluate_thresholds({}, {"error": {"threshold": 1.0, "comparison": "max"}})
    assert not v.passed and v.status == "not_evaluated"


@pytest.mark.parametrize("independent", [False, True])
def test_prediction_without_error_profile_cannot_pass(independent):
    v = BaseIdentification.verify(identifier(independent), scope="prediction")
    assert not v.passed and v.status == "not_evaluated"


def test_training_fallback_cannot_pass_independent_prediction_policy():
    v = BaseIdentification.verify(
        identifier(),
        scope="prediction",
        thresholds={"validation_rmse:joint": {"threshold": 0.5, "comparison": "max"}},
    )
    assert not v.passed and v.status == "not_evaluated"


def test_high_correlation_biased_prediction_fails_error_limit():
    v = BaseIdentification.verify(
        identifier(True, (11.0, 12.0)),
        scope="prediction",
        thresholds={"validation_rmse:joint": {"threshold": 0.5, "comparison": "max"}},
    )
    assert not v.passed and v.status == "fail"


def test_independent_prediction_passes_justified_explicit_error_limit():
    v = BaseIdentification.verify(
        identifier(True),
        scope="prediction",
        thresholds={"validation_rmse:joint": {"threshold": 0.5, "comparison": "max"}},
    )
    assert v.passed and v.stages["prediction"] == "pass"
    assert v.stages["physical"] == v.stages["export"] == "not_evaluated"


@pytest.mark.parametrize("prediction", [(float("nan"), 2.0), (1.0,)])
def test_nonfinite_or_misaligned_fit_cannot_pass_execution(prediction):
    v = BaseIdentification.verify(identifier(prediction=prediction), scope="execution")
    assert not v.passed and v.status == "fail"


def test_explicitly_requested_but_failed_validation_is_execution_failure():
    obj = identifier()
    obj.identif_config["validation_data_file"] = "missing-validation.csv"
    v = BaseIdentification.verify(obj, scope="execution")
    assert not v.passed and v.status == "fail"


def test_invalid_evidence_exports_strict_json_with_failure(tmp_path):
    import json

    obj = identifier()
    obj.result["rmse norm (N/m)"] = float("nan")
    obj.verify = lambda **kw: BaseIdentification.verify(obj, **kw)
    destination = tmp_path / "verdict.json"
    BaseIdentification.export_verification_report(obj, output_path=str(destination))
    data = json.loads(
        destination.read_text(), parse_constant=lambda value: pytest.fail(value)
    )
    assert data["status"] == "fail" and data["metrics"]["rmse"] is None


def test_prediction_profile_can_enforce_peak_and_bias_limits():
    obj = identifier(True, (1.0, 2.2))
    v = BaseIdentification.verify(
        obj,
        scope="prediction",
        thresholds={
            "validation_rmse:joint": {"threshold": 0.5, "comparison": "max"},
            "validation_peak_error:joint": {"threshold": 0.1, "comparison": "max"},
            "validation_abs_bias:joint": {"threshold": 0.15, "comparison": "max"},
        },
    )
    assert v.status == "fail"
    assert (
        next(c for c in v.checks if c.name == "validation_peak_error:joint").status
        == "fail"
    )


def test_calibration_failed_solver_is_not_numerical_execution_success():
    from figaroh.calibration.base_calibration import BaseCalibration

    obj = SimpleNamespace(
        evaluation_metrics={"rmse": 0.001, "optimization_success": False},
        calib_config={"calibration_index": 3},
        results_data={"calibrated parameters values": [1.0], "residuals": [0.001]},
        _config_file_path=None,
        robot_name="test",
    )
    v = BaseCalibration.verify(obj, scope="execution")
    assert v.status == "fail" and v.stages["solver"] == "fail"
    # No check may report a failed solve as passing evidence.
    assert all(c.status != "pass" for c in v.checks if "solver" in c.name)


def test_calibration_training_fallback_is_labelled_training_evidence():
    from figaroh.calibration.base_calibration import BaseCalibration

    obj = SimpleNamespace(
        evaluation_metrics={"rmse": 0.001, "optimization_success": True},
        calib_config={"calibration_index": 3},
        _val_available=False,
        results_data={
            "calibrated parameters values": [1.0],
            "residuals": [0.001],
            "validation_metrics": {"pos_rmse_calibrated_mm": 0.5},
        },
        _config_file_path=None,
        robot_name="test",
    )
    v = BaseCalibration.verify(
        obj,
        scope="prediction",
        thresholds={"position_rmse_mm": {"threshold": 1.0, "comparison": "max"}},
    )
    assert "training_position_rmse_mm" in v.metrics
    assert "position_rmse_mm" not in v.metrics
    assert v.stages["prediction"] == "not_evaluated" and not v.passed


def test_unscoped_library_verification_does_not_silently_accept_a_model():
    v = BaseIdentification.verify(identifier())
    assert v.scope == "prediction" and v.status == "not_evaluated" and not v.passed


def test_unmeasured_orientation_is_not_zero_error_acceptance():
    from figaroh.calibration.base_calibration import BaseCalibration

    obj = SimpleNamespace(
        evaluation_metrics={"rmse": 0.001, "optimization_success": True},
        calib_config={"calibration_index": 3},
        _val_available=True,
        results_data={
            "calibrated parameters values": [1.0],
            "residuals": [0.001],
            "validation_metrics": {
                "pos_rmse_calibrated_mm": 0.5,
                "orient_rmse_calibrated_deg": 0.0,
            },
        },
        _config_file_path=None,
        robot_name="test",
    )
    v = BaseCalibration.verify(
        obj,
        scope="prediction",
        thresholds={
            "position_rmse_mm": {"threshold": 1.0, "comparison": "max"},
            "orientation_rmse_deg": {"threshold": 0.1, "comparison": "max"},
        },
    )
    assert v.status == "not_evaluated" and not v.passed
