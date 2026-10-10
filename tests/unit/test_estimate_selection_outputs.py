"""Reports, terminal summary and run archive for select_stage (#61)."""

import csv
import os
import sys
from pathlib import Path

import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from test_data_contract_wiring import _trajectory  # noqa: E402
from test_estimate_selection import _break_solver, _run  # noqa: E402

from figaroh.tools.run_archive import archive_run  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


@pytest.fixture(scope="module")
def traj(model):
    return _trajectory(model)


def _archive(ident, tmp_path):
    run_dir = tmp_path / "runs" / "asset" / "identification" / "run1"
    run_dir.mkdir(parents=True)
    ident.export_verification_report(
        output_path=str(run_dir / "verdict.json"), scope="execution"
    )
    archive_run(ident, run_dir)
    return run_dir


def _rows(path):
    with open(path) as f:
        return list(csv.reader(f))[1:]


def test_archive_writes_selected_standard_parameters(model, traj, tmp_path):
    pytest.importorskip("picos")
    ident = _run(model, traj, select_stage="physical_fit")
    run_dir = _archive(ident, tmp_path)
    rows = _rows(run_dir / "parameters.csv")
    assert [r[0] for r in rows] == list(ident.standard_parameter)
    fit = _rows(run_dir / "fit_parameters.csv")
    assert [r[0] for r in fit] == ident.result["base parameters names"]


def test_archive_of_a_rejected_estimate_has_no_parameters(
    model, traj, tmp_path, monkeypatch
):
    pytest.importorskip("picos")
    _break_solver(monkeypatch)
    ident = _run(model, traj, select_stage="physical_fit")
    run_dir = _archive(ident, tmp_path)
    assert not (run_dir / "parameters.csv").exists()
    assert (run_dir / "fit_parameters.csv").exists()


def test_default_archive_is_unchanged(model, traj, tmp_path):
    ident = _run(model, traj)
    run_dir = _archive(ident, tmp_path)
    assert (run_dir / "parameters.csv").exists()
    assert not (run_dir / "fit_parameters.csv").exists()


def test_terminal_and_html_report_name_the_selection(model, traj, tmp_path, capsys):
    pytest.importorskip("picos")
    ident = _run(model, traj, select_stage="physical_fit")
    ident.print_quality_report()
    assert "Reported estimate: physical_fit" in capsys.readouterr().out
    text = Path(
        ident.export_html_report(output_path=str(tmp_path / "r.html"))
    ).read_text()
    assert "Reported estimate" in text and "physical_fit" in text
    assert "Min pseudo-inertia eigenvalue" in text
    for joint in ident.identif_config["active_joints"]:
        assert joint in text


def test_html_report_of_a_rejection_says_none(model, traj, tmp_path, monkeypatch):
    pytest.importorskip("picos")
    _break_solver(monkeypatch)
    ident = _run(model, traj, select_stage="physical_fit")
    text = Path(
        ident.export_html_report(output_path=str(tmp_path / "r.html"))
    ).read_text()
    assert "none (requested physical_fit: rejected)" in text


def test_projected_block_is_labelled_as_nominal_projection(model, traj, tmp_path):
    pytest.importorskip("picos")
    ident = _run(model, traj, physical_consistency={"enabled": True})
    text = Path(
        ident.export_html_report(output_path=str(tmp_path / "r.html"))
    ).read_text()
    assert "projects the reconstructed fit" in text
