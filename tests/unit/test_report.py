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
Unit tests for the HTML calibration diagnostic report (figaroh.tools.report).

Uses a minimal fake calibrator exposing only the attributes
``generate_calibration_report`` reads (``evaluation_metrics``,
``calib_config``, ``results_data``) so tests do not depend on a full
BaseCalibration fixture.
"""

import pytest

from figaroh.tools.report import (
    _build_insights,
    _uncertainty_tier,
    generate_calibration_report,
)


class FakeCalibrator:
    """Stand-in for BaseCalibration exposing only what the report reads."""

    def __init__(
        self, evaluation_metrics, calib_config, results_data=None,
        redistributed=None,
    ):
        self.evaluation_metrics = evaluation_metrics
        self.calib_config = calib_config
        self.results_data = results_data or {}
        self._redistributed = redistributed

    def redistribute_parameters(self):
        """Mirrors BaseCalibration.redistribute_parameters: raises when
        unavailable (the default, matching a calibrator that never
        populated the base-mapping), returns a fixed dict otherwise."""
        if self._redistributed is None:
            raise AttributeError("redistribute_parameters not available")
        return self._redistributed


def _base_eval(**overrides):
    eval_ = {
        "optimization_success": True,
        "n_iterations": 12,
        "cost": 0.001234,
        "n_outliers": 2,
        "outlier_percentage": 2.0,
        "condition_number": 42.0,
        "condition_label": "well-conditioned",
        "per_dof_stats": {
            "dof_names": ["X (mm)", "Y (mm)", "Z (mm)"],
            "mean": [0.1, -0.2, 0.05],
            "std": [0.3, 0.4, 0.2],
            "rmse": [0.32, 0.45, 0.21],
            "max_abs": [1.1, 1.4, 0.9],
            "r_squared": [0.98, 0.97, 0.99],
            "overall": {
                "pos_rmse_mm": 0.5,
                "orient_rmse_deg": 0.02,
                "pos_max_mm": 1.4,
                "orient_max_deg": 0.05,
            },
        },
        "param_stdev": [0.001, 0.002, 0.05],
        "param_stddev_percentage": [5.0, 15.0, 45.0],
        "correlated_pairs": [],
    }
    eval_.update(overrides)
    return eval_


def _base_config():
    return {
        "NbSample": 100,
        "param_name": ["base_x", "base_y", "base_z"],
    }


class TestUncertaintyTier:
    def test_good_below_caution_threshold(self):
        assert _uncertainty_tier(5.0) == "good"

    def test_fair_between_thresholds(self):
        assert _uncertainty_tier(15.0) == "fair"

    def test_poor_above_warn_threshold(self):
        assert _uncertainty_tier(45.0) == "poor"

    def test_unknown_for_nan(self):
        assert _uncertainty_tier(float("nan")) == "unknown"


class TestBuildInsights:
    def test_healthy_fit_reports_no_issues(self):
        eval_ = _base_eval(param_stddev_percentage=[5.0, 8.0, 9.0])
        insights = _build_insights(
            eval_, 100, _base_config()["param_name"],
            validation={"pos_improvement_pct": 90.0},
        )
        assert any("No issues detected" in i["text"] for i in insights)

    def test_flags_ill_conditioned(self):
        eval_ = _base_eval(
            condition_number=5000.0, condition_label="ill-conditioned"
        )
        insights = _build_insights(
            eval_, 100, _base_config()["param_name"], validation=None
        )
        assert any("ill-conditioned" in i["text"] for i in insights)
        assert any(i["level"] == "warn" for i in insights)

    def test_flags_poorly_identified_parameters(self):
        eval_ = _base_eval(param_stddev_percentage=[5.0, 15.0, 45.0])
        insights = _build_insights(
            eval_, 100, _base_config()["param_name"], validation=None
        )
        assert any("base_z" in i["text"] for i in insights)

    def test_flags_missing_validation_data(self):
        eval_ = _base_eval()
        insights = _build_insights(
            eval_, 100, _base_config()["param_name"], validation=None
        )
        assert any("No held-out validation" in i["text"] for i in insights)

    def test_flags_failed_convergence(self):
        eval_ = _base_eval(optimization_success=False)
        insights = _build_insights(
            eval_, 100, _base_config()["param_name"], validation=None
        )
        assert any("did not report success" in i["text"] for i in insights)


class TestGenerateCalibrationReport:
    def test_produces_self_contained_html(self):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        doc = generate_calibration_report(calibrator)

        assert doc.startswith("<!doctype html>")
        assert "<style>" in doc
        # Self-contained: no external network requests.
        assert "http://" not in doc
        assert "https://" not in doc
        assert "FakeCalibrator" in doc

    def test_handles_missing_validation_data(self):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        doc = generate_calibration_report(calibrator)
        assert "No separate validation data provided" in doc

    def test_renders_validation_section_when_present(self):
        results_data = {
            "validation_metrics": {
                "n_val_samples": 50,
                "pos_rmse_nominal_mm": 12.3,
                "pos_rmse_calibrated_mm": 0.7,
                "pos_improvement_pct": 94.3,
                "orient_rmse_nominal_deg": 2.1,
                "orient_rmse_calibrated_deg": 0.02,
                "orient_improvement_pct": 99.0,
                "pos_max_nominal_mm": 20.0,
                "pos_max_calibrated_mm": 2.0,
                "orient_max_nominal_deg": 5.0,
                "orient_max_calibrated_deg": 0.1,
            }
        }
        calibrator = FakeCalibrator(
            _base_eval(), _base_config(), results_data
        )
        doc = generate_calibration_report(calibrator)
        assert "94.3" in doc
        assert "n=50" in doc

    def test_handles_no_correlated_pairs(self):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        doc = generate_calibration_report(calibrator)
        assert "No parameter pairs exceed" in doc

    def test_renders_correlated_pairs(self):
        eval_ = _base_eval(
            correlated_pairs=[
                {"param_i": "base_x", "param_j": "pEE_x", "correlation": 0.93}
            ]
        )
        calibrator = FakeCalibrator(eval_, _base_config())
        doc = generate_calibration_report(calibrator)
        assert "base_x" in doc and "pEE_x" in doc
        assert "0.930" in doc

    def test_escapes_param_names(self):
        eval_ = _base_eval(param_stddev_percentage=[5.0, 15.0, 45.0])
        config = _base_config()
        config["param_name"] = ["<script>", "base_y", "base_z"]
        calibrator = FakeCalibrator(eval_, config)
        doc = generate_calibration_report(calibrator)
        assert "<script>alert" not in doc
        assert "&lt;script&gt;" in doc

    def test_parameter_uncertainty_table_shows_values(self):
        eval_ = _base_eval(param_values=[0.012345, -0.02, 0.5])
        calibrator = FakeCalibrator(eval_, _base_config())
        doc = generate_calibration_report(calibrator)
        assert "<th>Value</th>" in doc
        assert "0.012345" in doc

    def test_parameter_uncertainty_table_handles_missing_values(self):
        """param_values omitted entirely (e.g. an older cached
        evaluation_metrics dict) -- must render '—', not crash."""
        eval_ = _base_eval()
        assert "param_values" not in eval_
        calibrator = FakeCalibrator(eval_, _base_config())
        doc = generate_calibration_report(calibrator)
        assert "<th>Value</th>" in doc
        assert "—" in doc

    def test_writes_to_output_path(self, tmp_path):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        out_file = tmp_path / "report.html"
        doc = generate_calibration_report(
            calibrator, output_path=str(out_file)
        )
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == doc

    def test_custom_title(self):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        doc = generate_calibration_report(calibrator, title="UR10 Report")
        assert "<h1>UR10 Report</h1>" in doc

    def test_handles_nan_condition_number(self):
        eval_ = _base_eval(
            condition_number=float("nan"),
            condition_label="unavailable (no Jacobian)",
        )
        calibrator = FakeCalibrator(eval_, _base_config())
        doc = generate_calibration_report(calibrator)
        assert "unavailable (no Jacobian)" in doc

    def test_empty_per_dof_stats_does_not_crash(self):
        eval_ = _base_eval(per_dof_stats={})
        calibrator = FakeCalibrator(eval_, _base_config())
        doc = generate_calibration_report(calibrator)
        assert "No per-DOF residual data available" in doc

    def test_before_after_series_unavailable_without_validation(self):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        doc = generate_calibration_report(calibrator)
        assert "before/after series unavailable" in doc.lower()
        assert "initSeriesPanel" not in doc.split("<script>")[0]

    def test_renders_before_after_series_when_present(self):
        results_data = {
            "validation_metrics": {
                "n_val_samples": 3,
                "pos_rmse_nominal_mm": 12.3,
                "pos_rmse_calibrated_mm": 0.7,
                "pos_improvement_pct": 94.3,
                "orient_rmse_nominal_deg": 2.1,
                "orient_rmse_calibrated_deg": 0.02,
                "orient_improvement_pct": 99.0,
                "pos_max_nominal_mm": 20.0,
                "pos_max_calibrated_mm": 2.0,
                "orient_max_nominal_deg": 5.0,
                "orient_max_calibrated_deg": 0.1,
                "dof_names": ["X (mm)", "Y (mm)"],
                "error_nominal_per_dof": {
                    "X (mm)": [1.0, 2.0, 3.0], "Y (mm)": [4.0, 5.0, 6.0],
                },
                "error_fitted_per_dof": {
                    "X (mm)": [0.1, 0.2, 0.3], "Y (mm)": [0.4, 0.5, 0.6],
                },
            }
        }
        calibrator = FakeCalibrator(
            _base_eval(), _base_config(), results_data
        )
        doc = generate_calibration_report(calibrator)
        assert "initSeriesPanel(" in doc
        assert "function initSeriesPanel" in doc
        assert "series-panel-select" in doc
        assert '"names": ["X (mm)", "Y (mm)"]' in doc
        assert "\"measured\": {\"X (mm)\": [0.0, 0.0, 0.0]" in doc
        # Regression: the function definition must appear (in <head>)
        # before the invocation further down the page — scripts execute
        # in document order, so a call before its definition would throw
        # ReferenceError in a real browser.
        assert doc.index("function initSeriesPanel") < doc.index(
            'initSeriesPanel("series-panel"'
        )

    def test_self_contained_html_check_survives_series_panel(self):
        """The series panel's SVG-namespace string must not reintroduce
        a literal external-request-looking substring."""
        results_data = {
            "validation_metrics": {
                "n_val_samples": 1,
                "pos_rmse_nominal_mm": 1.0,
                "pos_rmse_calibrated_mm": 1.0,
                "pos_improvement_pct": 1.0,
                "orient_rmse_nominal_deg": 1.0,
                "orient_rmse_calibrated_deg": 1.0,
                "orient_improvement_pct": 1.0,
                "pos_max_nominal_mm": 1.0,
                "pos_max_calibrated_mm": 1.0,
                "orient_max_nominal_deg": 1.0,
                "orient_max_calibrated_deg": 1.0,
                "dof_names": ["X (mm)"],
                "error_nominal_per_dof": {"X (mm)": [1.0]},
                "error_fitted_per_dof": {"X (mm)": [0.1]},
            }
        }
        calibrator = FakeCalibrator(
            _base_eval(), _base_config(), results_data
        )
        doc = generate_calibration_report(calibrator)
        assert "http://" not in doc
        assert "https://" not in doc


class TestRedistributedSection:
    """The 'Redistributed standard parameters' section, driven by
    calibrator.redistribute_parameters()."""

    def test_not_available_message_when_missing(self):
        calibrator = FakeCalibrator(_base_eval(), _base_config())
        doc = generate_calibration_report(calibrator)
        assert "Redistribution not available for this run" in doc

    def test_renders_values_when_available(self):
        redistributed = {
            "d_px_arm_1_joint": {"value": 0.0123, "std_dev": 0.0005},
            "d_py_arm_1_joint": {"value": 0.0123, "std_dev": 0.0005},
        }
        calibrator = FakeCalibrator(
            _base_eval(), _base_config(), redistributed=redistributed
        )
        doc = generate_calibration_report(calibrator)
        assert "Redistributed standard parameters" in doc
        assert "d_px_arm_1_joint" in doc
        assert "d_py_arm_1_joint" in doc
        assert "0.0123" in doc
        assert "Redistribution not available" not in doc

    def test_handles_zero_value_without_crashing(self):
        redistributed = {"d_pz_arm_2_joint": {"value": 0.0, "std_dev": 0.0002}}
        calibrator = FakeCalibrator(
            _base_eval(), _base_config(), redistributed=redistributed
        )
        doc = generate_calibration_report(calibrator)
        assert "d_pz_arm_2_joint" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
