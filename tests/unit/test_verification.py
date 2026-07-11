# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the machine-readable verification verdict
(figaroh.tools._report_common.evaluate_thresholds /
BaseCalibration.verify / BaseIdentification.verify).

Uses minimal fake calibrator/identifier objects exposing only the
attributes ``verify()`` reads, following the same pattern as
test_report.py / test_identification_report.py, rather than a full
Base* fixture. ``verify()``/``export_verification_report()`` are called
as unbound methods (``BaseCalibration.verify(fake, ...)``) against these
fakes so the real numerical solve is never exercised here.
"""

import json

import numpy as np
import pytest

from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.identification.base_identification import BaseIdentification
from figaroh.tools._report_common import (
    build_provenance_metadata,
    evaluate_thresholds,
)
from figaroh.utils.results_manager import ResultsManager


class TestEvaluateThresholds:
    def test_all_pass(self):
        metrics = {"condition_number": 50.0, "rmse": 0.01}
        thresholds = {
            "condition_number": {"threshold": 1000.0, "comparison": "max"},
            "rmse": {"threshold": 1.0, "comparison": "max"},
        }
        verdict = evaluate_thresholds(metrics, thresholds)
        assert verdict.passed is True
        assert len(verdict.checks) == 2
        assert all(c.passed for c in verdict.checks)

    def test_max_comparison_fails_above_threshold(self):
        metrics = {"condition_number": 5000.0}
        thresholds = {
            "condition_number": {"threshold": 1000.0, "comparison": "max"}
        }
        verdict = evaluate_thresholds(metrics, thresholds)
        assert verdict.passed is False
        assert verdict.checks[0].passed is False

    def test_min_comparison_passes_above_threshold(self):
        metrics = {"correlation": 0.95}
        thresholds = {"correlation": {"threshold": 0.9, "comparison": "min"}}
        verdict = evaluate_thresholds(metrics, thresholds)
        assert verdict.passed is True

    def test_min_comparison_fails_below_threshold(self):
        metrics = {"correlation": 0.5}
        thresholds = {"correlation": {"threshold": 0.9, "comparison": "min"}}
        verdict = evaluate_thresholds(metrics, thresholds)
        assert verdict.passed is False

    def test_missing_metric_is_skipped_not_failed(self):
        thresholds = {
            "condition_number": {"threshold": 1000.0, "comparison": "max"}
        }
        verdict = evaluate_thresholds({}, thresholds)
        assert verdict.checks == []
        assert verdict.passed is True

    def test_nan_metric_is_skipped_not_failed(self):
        metrics = {"rmse": float("nan")}
        thresholds = {"rmse": {"threshold": 1.0, "comparison": "max"}}
        verdict = evaluate_thresholds(metrics, thresholds)
        assert verdict.checks == []
        assert verdict.passed is True

    def test_unknown_comparison_raises(self):
        metrics = {"x": 1.0}
        thresholds = {"x": {"threshold": 1.0, "comparison": "bogus"}}
        with pytest.raises(ValueError):
            evaluate_thresholds(metrics, thresholds)

    def test_metrics_field_preserves_all_inputs(self):
        metrics = {"a": 1.0, "b": float("nan")}
        verdict = evaluate_thresholds(metrics, {})
        assert verdict.metrics == metrics
        assert verdict.checks == []


class TestProvenanceMetadata:
    def test_keys_present(self):
        meta = build_provenance_metadata(None, "ur10")
        assert set(meta.keys()) == {
            "git_commit", "config_sha256", "timestamp", "robot_name",
        }
        assert meta["robot_name"] == "ur10"

    def test_missing_config_file_is_unknown_not_raising(self):
        meta = build_provenance_metadata(
            "/nonexistent/path/config.yaml", "ur10"
        )
        assert meta["config_sha256"] == "unknown"

    def test_real_config_file_is_hashed(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("a: 1")
        meta = build_provenance_metadata(str(cfg), "ur10")
        assert meta["config_sha256"] != "unknown"
        assert len(meta["config_sha256"]) == 64  # sha256 hex digest length


class FakeCalibrator:
    """Stand-in for BaseCalibration exposing only what verify() reads."""

    def __init__(
        self,
        evaluation_metrics,
        calib_config,
        results_data=None,
        robot_name="fake_robot",
        config_file_path=None,
        results_manager=None,
    ):
        self.evaluation_metrics = evaluation_metrics
        self.calib_config = calib_config
        self.results_data = results_data or {}
        self.robot_name = robot_name
        self.model = None
        self._config_file_path = config_file_path
        self.results_manager = results_manager

    # export_verification_report() calls self.verify() internally, so the
    # fake needs it bound as a real method, not just callable unbound.
    verify = BaseCalibration.verify
    export_verification_report = BaseCalibration.export_verification_report


def _calib_eval(**overrides):
    e = {
        "condition_number": 42.0,
        "rmse": 0.001,
        "outlier_percentage": 1.0,
        "optimization_success": True,
        "n_iterations": 5,
        "cost": 0.0001,
        "n_outliers": 1,
        "param_stddev_percentage": [5.0],
        "correlated_pairs": [],
    }
    e.update(overrides)
    return e


class TestBaseCalibrationVerify:
    def test_passes_with_good_metrics(self):
        calib = FakeCalibrator(
            _calib_eval(), {"NbSample": 100, "param_name": ["p1"]}
        )
        verdict = BaseCalibration.verify(calib)
        assert verdict.passed is True
        assert verdict.metadata["robot_name"] == "fake_robot"

    def test_fails_on_ill_conditioned(self):
        calib = FakeCalibrator(
            _calib_eval(condition_number=5000.0),
            {"NbSample": 100, "param_name": ["p1"]},
        )
        verdict = BaseCalibration.verify(calib)
        assert verdict.passed is False
        failed_names = {c.name for c in verdict.checks if not c.passed}
        assert "condition_number" in failed_names

    def test_validation_metrics_feed_into_checks(self):
        calib = FakeCalibrator(
            _calib_eval(),
            {"NbSample": 100, "param_name": ["p1"]},
            results_data={
                "validation_metrics": {
                    "pos_rmse_calibrated_mm": 5.0,
                    "orient_rmse_calibrated_deg": 0.2,
                }
            },
        )
        verdict = BaseCalibration.verify(calib)
        assert "position_rmse_mm" in verdict.metrics
        # 5mm exceeds the 2mm default threshold
        assert verdict.passed is False

    def test_no_validation_data_skips_those_checks(self):
        calib = FakeCalibrator(
            _calib_eval(), {"NbSample": 100, "param_name": ["p1"]}
        )
        verdict = BaseCalibration.verify(calib)
        names = {c.name for c in verdict.checks}
        assert "position_rmse_mm" not in names
        assert "orientation_rmse_deg" not in names

    def test_custom_thresholds_override_defaults(self):
        calib = FakeCalibrator(
            _calib_eval(condition_number=5000.0),
            {"NbSample": 100, "param_name": ["p1"]},
        )
        verdict = BaseCalibration.verify(
            calib,
            thresholds={
                "condition_number": {
                    "threshold": 10000.0, "comparison": "max",
                }
            },
        )
        assert verdict.passed is True

    def test_raises_before_solve(self):
        calib = FakeCalibrator.__new__(FakeCalibrator)  # no attrs set
        with pytest.raises(AttributeError):
            BaseCalibration.verify(calib)

    def test_insights_reuse_html_report_text(self):
        calib = FakeCalibrator(
            _calib_eval(
                condition_number=5000.0, condition_label="ill-conditioned"
            ),
            {"NbSample": 100, "param_name": ["p1"]},
        )
        verdict = BaseCalibration.verify(calib)
        assert any("ill-conditioned" in text for text in verdict.insights)

    def test_compat_populated_without_validation_data(self):
        calib = FakeCalibrator(
            _calib_eval(),
            {"NbSample": 100, "param_name": ["p1"], "calibration_index": 6},
        )
        verdict = BaseCalibration.verify(calib)
        assert verdict.compat["dof_names"] == [
            "X (mm)", "Y (mm)", "Z (mm)",
            "rx (deg)", "ry (deg)", "rz (deg)",
        ]
        assert verdict.compat["sample_count"] == 100
        assert verdict.compat["config_sha256"] == "unknown"

    def test_series_empty_without_validation_data(self):
        calib = FakeCalibrator(
            _calib_eval(), {"NbSample": 100, "param_name": ["p1"]}
        )
        verdict = BaseCalibration.verify(calib)
        assert verdict.series == {}

    def test_series_populated_with_validation_data(self):
        calib = FakeCalibrator(
            _calib_eval(),
            {"NbSample": 100, "param_name": ["p1"], "calibration_index": 2},
            results_data={
                "validation_metrics": {
                    "pos_rmse_calibrated_mm": 1.0,
                    "orient_rmse_calibrated_deg": 0.05,
                    "n_val_samples": 3,
                    "dof_names": ["X (mm)", "Y (mm)"],
                    "error_nominal_per_dof": {
                        "X (mm)": [1.0, 2.0, 3.0],
                        "Y (mm)": [4.0, 5.0, 6.0],
                    },
                    "error_fitted_per_dof": {
                        "X (mm)": [0.1, 0.2, 0.3],
                        "Y (mm)": [0.4, 0.5, 0.6],
                    },
                }
            },
        )
        verdict = BaseCalibration.verify(calib)
        assert verdict.series["time"] == [0, 1, 2]
        assert verdict.series["dof_names"] == ["X (mm)", "Y (mm)"]
        assert verdict.series["nominal"]["X (mm)"] == [1.0, 2.0, 3.0]
        assert verdict.series["fitted"]["Y (mm)"] == [0.4, 0.5, 0.6]
        assert verdict.series["measured"]["X (mm)"] == [0.0, 0.0, 0.0]


class TestBaseCalibrationExportVerificationReport:
    def test_writes_valid_json(self, tmp_path):
        calib = FakeCalibrator(
            _calib_eval(), {"NbSample": 100, "param_name": ["p1"]}
        )
        out = tmp_path / "verdict.json"
        path = BaseCalibration.export_verification_report(
            calib, output_path=str(out)
        )
        assert path == str(out)
        data = json.loads(out.read_text())
        assert "passed" in data and "checks" in data and "metadata" in data
        assert "series" in data and "compat" in data

    def test_numpy_metrics_are_json_safe(self, tmp_path):
        """condition_number/rmse in real code are numpy scalars, not
        python floats — export must not crash on them (needs a real
        ResultsManager, which is what actually does the conversion)."""
        calib = FakeCalibrator(
            _calib_eval(condition_number=np.float64(42.0)),
            {"NbSample": 100, "param_name": ["p1"]},
            results_manager=ResultsManager("calibration", "fake_robot"),
        )
        out = tmp_path / "verdict.json"
        BaseCalibration.export_verification_report(
            calib, output_path=str(out)
        )
        data = json.loads(out.read_text())
        assert data["metrics"]["condition_number"] == 42.0

    def test_default_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calib = FakeCalibrator(
            _calib_eval(), {"NbSample": 100, "param_name": ["p1"]}
        )
        path = BaseCalibration.export_verification_report(calib)
        assert path == "results/calibration_verification.json"
        assert (tmp_path / path).exists()


class FakeIdentifier:
    """Stand-in for BaseIdentification exposing only what verify() reads."""

    def __init__(
        self,
        result,
        std_relative=None,
        robot_name="fake_robot",
        config_file_path=None,
        results_manager=None,
        identif_config=None,
        decimate_used=False,
    ):
        self.result = result
        self.std_relative = std_relative
        self.robot_name = robot_name
        self._config_file_path = config_file_path
        self.results_manager = results_manager
        self.identif_config = identif_config or {}
        self._decimate_used = decimate_used

    # export_verification_report() calls self.verify() internally, so the
    # fake needs it bound as a real method, not just callable unbound.
    verify = BaseIdentification.verify
    export_verification_report = BaseIdentification.export_verification_report


def _identif_result(**overrides):
    r = {
        "base parameters names": ["p1"],
        "condition number": 42.0,
        "rmse norm (N/m)": 0.1,
    }
    r.update(overrides)
    return r


class TestBaseIdentificationVerify:
    def test_passes_with_good_metrics(self):
        ident = FakeIdentifier(_identif_result())
        verdict = BaseIdentification.verify(ident)
        assert verdict.passed is True
        assert verdict.metadata["robot_name"] == "fake_robot"

    def test_fails_on_ill_conditioned(self):
        ident = FakeIdentifier(
            _identif_result(**{"condition number": 5000.0})
        )
        verdict = BaseIdentification.verify(ident)
        assert verdict.passed is False
        failed_names = {c.name for c in verdict.checks if not c.passed}
        assert "condition_number" in failed_names

    def test_validation_metrics_feed_into_checks(self):
        ident = FakeIdentifier(
            _identif_result(
                validation_metrics={
                    "correlation": 0.5, "improvement_pct": 10.0,
                }
            )
        )
        verdict = BaseIdentification.verify(ident)
        assert "validation_correlation" in verdict.metrics
        assert "validation_improvement_pct" in verdict.metrics
        assert verdict.passed is False

    def test_no_validation_data_skips_those_checks(self):
        ident = FakeIdentifier(_identif_result())
        verdict = BaseIdentification.verify(ident)
        names = {c.name for c in verdict.checks}
        assert "validation_correlation" not in names
        assert "validation_improvement_pct" not in names

    def test_custom_thresholds_override_defaults(self):
        ident = FakeIdentifier(
            _identif_result(**{"condition number": 5000.0})
        )
        verdict = BaseIdentification.verify(
            ident,
            thresholds={
                "condition_number": {
                    "threshold": 10000.0, "comparison": "max",
                }
            },
        )
        assert verdict.passed is True

    def test_raises_before_solve(self):
        ident = FakeIdentifier(result=None)
        with pytest.raises(AttributeError):
            BaseIdentification.verify(ident)

    def test_compat_populated_without_validation_data(self):
        ident = FakeIdentifier(
            _identif_result(**{"num samples": 500}),
            identif_config={"active_joints": ["j1", "j2"]},
            decimate_used=True,
        )
        verdict = BaseIdentification.verify(ident)
        assert verdict.compat == {
            "active_joints": ["j1", "j2"],
            "decimate": True,
            "sample_count": 500,
            "config_sha256": "unknown",
        }

    def test_series_empty_without_validation_data(self):
        ident = FakeIdentifier(_identif_result())
        verdict = BaseIdentification.verify(ident)
        assert verdict.series == {}

    def test_series_populated_with_validation_data(self):
        ident = FakeIdentifier(
            _identif_result(
                validation_metrics={
                    "correlation": 0.99,
                    "improvement_pct": 90.0,
                    "n_val_samples": 2,
                    "joint_names": ["j1", "j2"],
                    "tau_nominal_per_joint": {
                        "j1": [1.0, 2.0], "j2": [3.0, 4.0],
                    },
                    "tau_identified_per_joint": {
                        "j1": [1.1, 2.1], "j2": [3.1, 4.1],
                    },
                    "tau_measured_per_joint": {
                        "j1": [1.2, 2.2], "j2": [3.2, 4.2],
                    },
                }
            )
        )
        verdict = BaseIdentification.verify(ident)
        assert verdict.series["time"] == [0, 1]
        assert verdict.series["joint_names"] == ["j1", "j2"]
        assert verdict.series["nominal"]["j1"] == [1.0, 2.0]
        assert verdict.series["fitted"]["j2"] == [3.1, 4.1]
        assert verdict.series["measured"]["j1"] == [1.2, 2.2]


class TestBaseIdentificationExportVerificationReport:
    def test_writes_valid_json(self, tmp_path):
        ident = FakeIdentifier(_identif_result())
        out = tmp_path / "verdict.json"
        path = BaseIdentification.export_verification_report(
            ident, output_path=str(out)
        )
        assert path == str(out)
        data = json.loads(out.read_text())
        assert "passed" in data and "checks" in data and "metadata" in data
        assert "series" in data and "compat" in data

    def test_numpy_metrics_are_json_safe(self, tmp_path):
        ident = FakeIdentifier(
            _identif_result(**{"condition number": np.float64(42.0)}),
            results_manager=ResultsManager("identification", "fake_robot"),
        )
        out = tmp_path / "verdict.json"
        BaseIdentification.export_verification_report(
            ident, output_path=str(out)
        )
        data = json.loads(out.read_text())
        assert data["metrics"]["condition_number"] == 42.0
