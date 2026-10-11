"""The direct LMI effort fit is a public module; the old name is a shim."""

import importlib

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("picos")

NAMES = [
    "FixedExtras",
    "PhysicalPolicy",
    "ComparatorProblem",
    "SolveRecord",
    "build_problem",
    "solve_exact_reconstruction",
    "solve_direct_effort_fit",
    "solve_per_link_projection",
    "diagnose_exact",
    "_solve_core",
    "_feasibility",
    "_make_record",
]


def test_public_module_documented():
    pf = importlib.import_module("figaroh.identification.physical_fit")
    assert pf.__doc__ and "solve_direct_effort_fit" in pf.__doc__
    for n in ("build_problem", "solve_direct_effort_fit", "SolveRecord"):
        assert getattr(pf, n).__doc__


def test_shim_reexports_same_objects():
    pf = importlib.import_module("figaroh.identification.physical_fit")
    sh = importlib.import_module("figaroh.identification._physical_comparator")
    for n in NAMES:
        assert getattr(sh, n) is getattr(pf, n)
