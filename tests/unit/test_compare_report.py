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
Unit tests for the static two-run compare page
(figaroh.tools.compare_report, Feature 6 Phase C / Step 5).

Unlike the calibration/identification report tests, this page has no
run object to feed it: it is a static shell that loads two
export_verification_report() JSON files client-side. So these tests
check the *shell itself* — self-containment, required DOM hooks the
driver script depends on (getElementById calls), definition-before-use
ordering, output_path writing — rather than any data-populated markup.

The compatibility-check / diff-table / series-overlay *logic* embedded
in the page's JavaScript is exercised separately with a hand-rolled
Node DOM stub (mirroring how Step 4's chart script was verified) as
part of manual verification; see the roadmap's Step 5 entry.
"""

import re

import pytest

from figaroh.tools.compare_report import generate_compare_page


class TestGenerateComparePage:
    def test_produces_self_contained_html(self):
        doc = generate_compare_page()
        assert doc.startswith("<!doctype html>")
        assert "</html>" in doc
        assert "http://" not in doc
        assert "https://" not in doc

    def test_default_title(self):
        doc = generate_compare_page()
        assert "Two-Run Comparison" in doc

    def test_custom_title_is_escaped_into_title_and_h1(self):
        doc = generate_compare_page(title="My Compare Page")
        assert "<title>My Compare Page</title>" in doc
        assert "<h1>My Compare Page</h1>" in doc

    def test_writes_to_output_path(self, tmp_path):
        out = tmp_path / "compare.html"
        doc = generate_compare_page(output_path=str(out))
        assert out.read_text(encoding="utf-8") == doc

    def test_no_output_path_does_not_write_anything(self, tmp_path):
        generate_compare_page()
        assert list(tmp_path.iterdir()) == []

    def test_contains_two_file_inputs_and_drop_zones(self):
        doc = generate_compare_page()
        for element_id in (
            "zone-a", "zone-b", "file-a", "file-b",
            "filename-a", "filename-b",
        ):
            assert f'id="{element_id}"' in doc

    def test_contains_compat_and_results_dom_hooks(self):
        """Every id the driver script's getElementById calls reference
        must exist in the emitted markup, or the real page throws at
        runtime the moment a file is loaded."""
        doc = generate_compare_page()
        # Only literal-string getElementById("...") calls — dynamic ones
        # like getElementById("zone-" + slot) are checked separately
        # (test_dynamic_id_prefixes_have_matching_static_elements).
        get_by_id_calls = set(
            re.findall(r'getElementById\("([^"]+)"\)', doc)
        )
        assert get_by_id_calls, "expected at least one getElementById call"
        for element_id in get_by_id_calls:
            assert f'id="{element_id}"' in doc, (
                f"driver script references #{element_id} via "
                "getElementById, but no element with that id is "
                "rendered into the page"
            )

    def test_dynamic_id_prefixes_have_matching_static_elements(self):
        """getElementById("zone-" + slot) etc. are built from a static
        prefix + "a"/"b" at runtime — assert the concrete ids they
        resolve to (zone-a/zone-b/file-a/file-b/filename-a/filename-b)
        actually exist, since the regex above can't see through string
        concatenation."""
        doc = generate_compare_page()
        prefixes = set(
            re.findall(r'getElementById\("([a-z-]+-)"\s*\+\s*slot\)', doc)
        )
        assert prefixes, "expected at least one dynamic-id getElementById"
        for prefix in prefixes:
            for slot in ("a", "b"):
                assert f'id="{prefix}{slot}"' in doc

    def test_force_compare_checkbox_present(self):
        doc = generate_compare_page()
        assert 'id="force-compare"' in doc
        assert 'id="force-compare-row"' in doc

    def test_series_overlay_controls_present(self):
        doc = generate_compare_page()
        for element_id in (
            "compare-toggle-a", "compare-toggle-b",
            "compare-select", "compare-reset", "compare-svg",
            "compare-legend", "compare-tooltip",
        ):
            assert f'id="{element_id}"' in doc

    def test_initComparePanel_defined_before_any_use(self):
        """Same class of bug Step 4 caught: a function referenced before
        its <script> definition throws ReferenceError in a real
        browser. initComparePanel is defined in the head chart script
        and only ever invoked from inside the body driver script's
        evaluate()/renderSeriesPanel() — never at parse time — but this
        regression guard keeps that true if the page is ever
        restructured."""
        doc = generate_compare_page()
        definition_pos = doc.index("function initComparePanel")
        first_call_pos = doc.index("initComparePanel(", definition_pos + 1)
        assert definition_pos < first_call_pos

    def test_domain_and_compat_field_names_referenced_match_schema(self):
        """Guards the driver script's compatibility check against silent
        drift from the VerificationVerdict.compat schema populated by
        BaseCalibration.verify()/BaseIdentification.verify() (Step 3):
        dof_names/active_joints/decimate/sample_count."""
        doc = generate_compare_page()
        for field in (
            "dof_names", "active_joints", "decimate", "sample_count",
        ):
            assert field in doc

    def test_metrics_and_checks_fields_referenced(self):
        """Guards the diff-table logic against drift from the
        VerificationVerdict schema's `metrics`/`checks` fields."""
        doc = generate_compare_page()
        assert "a.metrics" in doc or ".metrics" in doc
        assert ".checks" in doc

    def test_svg_namespace_not_a_literal_external_url(self):
        """The SVG XML namespace URI must be built via concatenation
        (matching _report_common.py's _SERIES_CHART_SCRIPT convention),
        not a literal 'http://...' string, or it trips the
        self-containment check above via a false positive."""
        doc = generate_compare_page()
        assert '"http:" + "//www.w3.org/2000/svg"' in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
