"""Stage-aware verdicts and reports (#63).

Separate data / solver / fit / validation / physical / export verdicts, a
fallback validation that is never held-out evidence, the stage whose
parameters are reported, the data splits, the paired revisions, and the
existing verdict fields kept for current consumers.
"""

import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from figaroh.tools._report_common import VerificationVerdict  # noqa: E402
from figaroh.tools.provenance import collect_run_provenance  # noqa: E402
from figaroh.tools.stages import apply_to_verdict, record_stage  # noqa: E402
from test_data_contract_wiring import _solve, _trajectory  # noqa: E402


@pytest.fixture(scope="module")
def model():
    import pinocchio as pin

    return pin.buildSampleModelManipulator()


@pytest.fixture(scope="module")
def ident(model):
    return _solve(model, _trajectory(model))


def test_verdict_separates_stages(ident):
    verdict = ident.verify(scope="prediction")
    stages = verdict.stages
    # existing scoped entries are kept for current consumers
    for key in ("numerical_execution", "prediction", "solver", "data_provenance"):
        assert key in stages
    assert stages["data"] == stages["data_provenance"] == "pass"
    assert stages["fit"] == "pass"
    assert stages["validation"] == "fallback"
    # a training fallback never makes prediction acceptance pass
    assert stages["prediction"] != "pass"
    assert stages["physical"] == stages["export"] == "not_evaluated"
    assert verdict.selected_stage == "fit"
    assert [r["stage"] for r in verdict.stage_records] == ["data", "fit", "validation"]


def test_splits_name_the_data(ident):
    splits = ident.verify(scope="execution").splits
    assert splits["validation_source"] == "training_fallback"
    assert splits["validation"] is None
    assert splits["training"]["samples"] == 400
    assert splits["training"]["masked_samples"] == 0


def test_failed_fit_selects_nothing():
    obj = types.SimpleNamespace(stages=[])
    record_stage(obj, "data", "ok")
    record_stage(obj, "fit", "failed", "LinAlgError: singular")
    verdict = VerificationVerdict(passed=False, checks=[], metrics={})
    apply_to_verdict(verdict, obj)
    assert verdict.stages["fit"] == "fail"
    assert verdict.stages["validation"] == "not_evaluated"
    assert verdict.selected_stage == "none"
    assert verdict.splits["training"] == "legacy input (see provenance data)"


def test_provenance_names_both_revisions(ident):
    software = collect_run_provenance(ident, "identification")["software"]
    assert "git_commit" in software  # working directory, kept
    core = software["figaroh_revision"]
    head = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if head.returncode == 0 and core["commit"] != "unknown":
        assert core["commit"] == head.stdout.strip()
        assert isinstance(core["dirty"], bool)


def test_exported_verdict_keeps_existing_fields(ident, tmp_path):
    path = ident.export_verification_report(
        output_path=str(tmp_path / "v.json"), scope="execution"
    )
    data = json.loads(Path(path).read_text())
    for key in ("passed", "checks", "metrics", "status", "scope", "compat"):
        assert key in data
    assert isinstance(data["stages"], dict) and isinstance(data["stage_records"], list)
    assert data["selected_stage"] == "fit"
    assert data["splits"]["validation_source"] == "training_fallback"


def test_reports_show_stages(ident, tmp_path, capsys):
    html = Path(ident.export_html_report(output_path=str(tmp_path / "r.html")))
    text = html.read_text()
    assert "Stages and data" in text
    assert "training data (fallback, not held-out evidence)" in text
    ident.print_quality_report()
    out = capsys.readouterr().out
    assert (
        "Stages:" in out and "validation fallback (training data, not held-out)" in out
    )
    assert np.isfinite(ident.rms_error)


def test_archive_records_stages(ident, tmp_path):
    from figaroh.tools.run_archive import archive_run

    run_dir = tmp_path / "runs" / "asset" / "identification" / "run1"
    run_dir.mkdir(parents=True)
    ident.export_verification_report(
        output_path=str(run_dir / "verdict.json"), scope="execution"
    )
    archive_run(ident, run_dir)
    stages = json.loads((run_dir / "stages.json").read_text())
    assert [s["stage"] for s in stages["stages"]] == ["data", "fit", "validation"]
    entry = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert entry["stages"]["validation"] == "fallback"
    assert entry["selected_stage"] == "fit"
