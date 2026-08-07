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
Unit tests for the HTML identification diagnostic report
(figaroh.tools.identification_report).

Uses a minimal fake identifier exposing only the attributes
``generate_identification_report`` reads (``result``, ``std_relative``,
``phi_base``, ``correlation``, optionally ``_compute_per_joint_stats``)
so tests do not depend on a full BaseIdentification fixture.
"""

import numpy as np
import pytest

from figaroh.tools.identification_report import (
    _build_insights,
    _condition_label,
    generate_identification_report,
)


class FakeIdentifier:
    """Stand-in for BaseIdentification exposing only what the report reads."""

    def __init__(
        self,
        result,
        std_relative=None,
        phi_base=None,
        correlation=float("nan"),
        per_joint_stats=None,
    ):
        self.result = result
        self.std_relative = std_relative
        self.phi_base = phi_base
        self.correlation = correlation
        self._per_joint_stats = per_joint_stats

    def _compute_per_joint_stats(self):
        return self._per_joint_stats


def _base_result(**overrides):
    result = {
        "base parameters names": ["p1", "p2", "p3"],
        "condition number": 42.0,
        "rmse norm (N/m)": 1.234,
        "num samples": 100,
    }
    result.update(overrides)
    return result


class TestConditionLabel:
    def test_well_conditioned(self):
        assert _condition_label(42.0) == "well-conditioned"

    def test_moderately_conditioned(self):
        assert _condition_label(500.0) == "moderately conditioned"

    def test_ill_conditioned(self):
        assert _condition_label(5000.0) == "ill-conditioned"

    def test_unavailable_for_nan(self):
        assert _condition_label(float("nan")) == "unavailable"


class TestBuildInsights:
    def test_healthy_fit_reports_no_issues(self):
        result = _base_result()
        insights = _build_insights(
            result, [5.0, 8.0, 9.0], result["base parameters names"],
            validation={"correlation": 0.99, "improvement_pct": 90.0},
        )
        assert any("No issues detected" in i["text"] for i in insights)

    def test_flags_ill_conditioned(self):
        result = _base_result(**{"condition number": 5000.0})
        insights = _build_insights(
            result, None, result["base parameters names"], validation=None
        )
        assert any("ill-conditioned" in i["text"] for i in insights)
        assert any(i["level"] == "warn" for i in insights)

    def test_flags_poorly_identified_parameters(self):
        result = _base_result()
        insights = _build_insights(
            result, [5.0, 15.0, 45.0], result["base parameters names"],
            validation=None,
        )
        assert any("p3" in i["text"] for i in insights)

    def test_flags_missing_validation_data(self):
        result = _base_result()
        insights = _build_insights(
            result, None, result["base parameters names"], validation=None
        )
        assert any("No held-out validation" in i["text"] for i in insights)

    def test_flags_low_correlation_validation(self):
        result = _base_result()
        insights = _build_insights(
            result, None, result["base parameters names"],
            validation={"correlation": 0.5, "improvement_pct": 90.0},
        )
        assert any("correlation is only" in i["text"] for i in insights)

    def test_flags_weak_validation_improvement(self):
        result = _base_result()
        insights = _build_insights(
            result, None, result["base parameters names"],
            validation={"correlation": 0.99, "improvement_pct": 10.0},
        )
        assert any("improved by only" in i["text"] for i in insights)

    def test_flags_physical_consistency_error(self):
        result = _base_result(
            **{"physical consistency": {"status": "error"}}
        )
        insights = _build_insights(
            result, None, result["base parameters names"], validation=None
        )
        assert any("Physical-consistency" in i["text"] for i in insights)

    def test_ignores_feasible_physical_consistency(self):
        result = _base_result(
            **{"physical consistency": {"status": "already_feasible"}}
        )
        insights = _build_insights(
            result, None, result["base parameters names"], validation=None
        )
        assert not any("Physical-consistency" in i["text"] for i in insights)


class TestGenerateIdentificationReport:
    def test_produces_self_contained_html(self):
        identifier = FakeIdentifier(_base_result())
        doc = generate_identification_report(identifier)

        assert doc.startswith("<!doctype html>")
        assert "<style>" in doc
        assert "http://" not in doc
        assert "https://" not in doc
        assert "FakeIdentifier" in doc

    def test_raises_without_result(self):
        identifier = FakeIdentifier(None)
        identifier.result = None
        with pytest.raises(AttributeError):
            generate_identification_report(identifier)

    def test_handles_missing_validation_data(self):
        identifier = FakeIdentifier(_base_result())
        doc = generate_identification_report(identifier)
        assert "No separate validation data provided" in doc

    def test_renders_validation_section_when_present(self):
        result = _base_result(
            validation_metrics={
                "n_val_samples": 50,
                "rmse_nominal": 12.3,
                "rmse_identified": 0.7,
                "improvement_pct": 94.3,
                "max_nominal": 20.0,
                "max_identified": 2.0,
                "correlation": 0.98,
            }
        )
        identifier = FakeIdentifier(result)
        doc = generate_identification_report(identifier)
        assert "94.3" in doc
        assert "n=50" in doc

    def test_handles_missing_per_joint_stats(self):
        identifier = FakeIdentifier(_base_result())
        doc = generate_identification_report(identifier)
        assert "Per-joint residuals unavailable" in doc

    def test_before_after_series_unavailable_without_validation(self):
        identifier = FakeIdentifier(_base_result())
        doc = generate_identification_report(identifier)
        assert "before/after series unavailable" in doc.lower()
        assert "initSeriesPanel" not in doc.split("<script>")[0]

    def test_renders_before_after_series_when_present(self):
        result = _base_result(
            validation_metrics={
                "n_val_samples": 2,
                "rmse_nominal": 12.3,
                "rmse_identified": 0.7,
                "improvement_pct": 94.3,
                "max_nominal": 20.0,
                "max_identified": 2.0,
                "correlation": 0.98,
                "joint_names": ["joint_1", "joint_2"],
                "tau_nominal_per_joint": {
                    "joint_1": [1.0, 2.0], "joint_2": [3.0, 4.0],
                },
                "tau_identified_per_joint": {
                    "joint_1": [1.1, 2.1], "joint_2": [3.1, 4.1],
                },
                "tau_measured_per_joint": {
                    "joint_1": [1.2, 2.2], "joint_2": [3.2, 4.2],
                },
            }
        )
        identifier = FakeIdentifier(result)
        doc = generate_identification_report(identifier)
        assert "initSeriesPanel(" in doc
        assert "function initSeriesPanel" in doc
        assert "series-panel-select" in doc
        assert '"names": ["joint_1", "joint_2"]' in doc
        assert '"joint_1": [1.2, 2.2]' in doc
        # Regression: function definition (in <head>) must precede the
        # invocation further down the page — see test_report.py's
        # analogous test for why.
        assert doc.index("function initSeriesPanel") < doc.index(
            'initSeriesPanel("series-panel"'
        )

    def test_renders_per_joint_stats_when_present(self):
        identifier = FakeIdentifier(
            _base_result(),
            per_joint_stats={
                "joint_names": ["joint_1", "joint_2"],
                "mean": [0.1, -0.2],
                "std": [0.3, 0.4],
                "rmse": [0.32, 0.45],
                "max_abs": [1.1, 1.4],
            },
        )
        doc = generate_identification_report(identifier)
        assert "joint_1" in doc and "joint_2" in doc

    def test_escapes_param_names(self):
        result = _base_result(
            **{"base parameters names": ["<script>", "p2", "p3"]}
        )
        identifier = FakeIdentifier(
            result,
            std_relative=np.array([5.0, 15.0, 45.0]),
            phi_base=np.array([1.0, 2.0, 3.0]),
        )
        doc = generate_identification_report(identifier)
        assert "<script>alert" not in doc
        assert "&lt;script&gt;" in doc

    def test_handles_numpy_std_relative_without_crashing(self):
        # std_relative/phi_base are numpy arrays in real usage — must not
        # hit "truth value of an array is ambiguous" errors.
        identifier = FakeIdentifier(
            _base_result(),
            std_relative=np.array([5.0, 15.0, 45.0]),
            phi_base=np.array([1.0, 2.0, 3.0]),
        )
        doc = generate_identification_report(identifier)
        assert "p3" in doc

    def test_base_parameter_table_shows_values(self):
        identifier = FakeIdentifier(
            _base_result(),
            std_relative=np.array([5.0, 15.0, 45.0]),
            phi_base=np.array([1.234, 2.0, 3.0]),
        )
        doc = generate_identification_report(identifier)
        assert "<th>Value</th>" in doc
        assert "1.234" in doc

    def test_base_parameter_table_handles_missing_phi_base(self):
        identifier = FakeIdentifier(
            _base_result(), std_relative=np.array([5.0, 15.0, 45.0])
        )
        doc = generate_identification_report(identifier)
        assert "<th>Value</th>" in doc
        assert "—" in doc

    def test_writes_to_output_path(self, tmp_path):
        identifier = FakeIdentifier(_base_result())
        out_file = tmp_path / "report.html"
        doc = generate_identification_report(
            identifier, output_path=str(out_file)
        )
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == doc

    def test_custom_title(self):
        identifier = FakeIdentifier(_base_result())
        doc = generate_identification_report(identifier, title="UR10 Report")
        assert "<h1>UR10 Report</h1>" in doc

    def test_handles_nan_condition_number(self):
        result = _base_result(**{"condition number": float("nan")})
        identifier = FakeIdentifier(result)
        doc = generate_identification_report(identifier)
        assert "Condition number" in doc
        assert "<div class=\"stat-value\">unavailable</div>" in doc

    def test_renders_consistency_section_when_present(self):
        result = _base_result(
            **{
                "physical consistency": {"status": "projected"},
                "reconstruction": {"status": "success"},
            }
        )
        identifier = FakeIdentifier(result)
        doc = generate_identification_report(identifier)
        assert "projected" in doc
        assert "success" in doc

    def test_handles_missing_consistency_sections(self):
        identifier = FakeIdentifier(_base_result())
        doc = generate_identification_report(identifier)
        assert "were not enabled for this" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
