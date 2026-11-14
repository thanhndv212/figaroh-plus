"""Validation raises real exceptions, not ``assert`` (#262, plan item Q2.2).

``assert`` disappears under ``python -O`` and reports ``AssertionError``,
which callers cannot tell apart from a failed internal invariant.
"""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import figaroh
from figaroh.utils import error_handling
from figaroh.utils.error_handling import (
    CalibrationError,
    FigarohError,
    IdentificationError,
)
from figaroh.utils.results_manager import ResultsManager

SRC = Path(figaroh.__file__).resolve().parent


def test_library_has_no_assert_statements():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        offenders += [
            f"{path.relative_to(SRC)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        ]
    assert offenders == []


def test_library_does_not_raise_assertion_error():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, ast.Name) and target.id == "AssertionError":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []


def test_base_exception_is_figaroh_error():
    assert issubclass(CalibrationError, FigarohError)
    assert issubclass(IdentificationError, FigarohError)


@pytest.mark.parametrize("module", [error_handling, figaroh.utils])
def test_old_base_name_is_a_deprecated_alias(module):
    with pytest.warns(DeprecationWarning, match="FigarohError") as caught:
        old = module.FigarohExampleError
    assert old is FigarohError
    # The warning names the caller, not the library's own lookup hook.
    assert caught[0].filename == __file__
    # Code that catches the old name still catches the library's errors.
    with pytest.raises(old):
        raise CalibrationError("x")


def test_importing_the_package_does_not_warn():
    # A fresh interpreter: reloading the module here would rebind the
    # exception classes other tests already imported.
    code = (
        "import sys, warnings; sys.path.insert(0, sys.argv[1]); "
        "warnings.filterwarnings('error', message='.*FigarohExampleError.*'); "
        "import figaroh.utils, figaroh.utils.error_handling"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(SRC.parent)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_results_manager_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="Unsupported task type"):
        ResultsManager("no-such-task", "robot")


def test_results_manager_rejects_mismatched_results():
    with pytest.raises(ValueError, match="must match task type"):
        ResultsManager("calibration", "robot", {"task type": "identification"})


def test_unknown_joint_name_is_a_value_error():
    pin = pytest.importorskip("pinocchio")
    from figaroh.calibration.data_loader import get_idxq_from_jname

    model = pin.buildSampleModelManipulator()
    with pytest.raises(ValueError, match="does not exist"):
        get_idxq_from_jname(model, "no_such_joint")


def test_wrong_variable_count_is_a_value_error():
    pin = pytest.importorskip("pinocchio")
    from figaroh.calibration.calibration_tools import calc_updated_fkm

    model = pin.buildSampleModelManipulator()
    calib_config = {"calib_model": "full_params", "param_name": ["a", "b", "c"]}
    with pytest.raises(ValueError, match="Length of variables"):
        calc_updated_fkm(
            model,
            model.createData(),
            np.zeros(2),
            np.zeros((1, model.nq)),
            calib_config,
        )
