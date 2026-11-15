"""Library output goes through figaroh.* loggers; printing by default is
deprecated (#267, plan item Q2.3).
"""

import ast
import logging
import warnings
from pathlib import Path

import pytest

import figaroh
import figaroh.calibration.base_calibration as bc
from figaroh.tools._deprecation import resolve_changing_defaults

import test_calibration_export_refusal as _export_refusal

# the synthetic TIAGo calibration fixture, shared with the export tests
make_calib = _export_refusal.make_calib

SRC = Path(figaroh.__file__).parent

# an explicit helper for example scripts, called by the application
_ALLOWED_LOGGER_NAMES = {"figaroh_examples"}


def test_library_loggers_live_under_figaroh():
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getLogger"
            ):
                continue
            if not node.args:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno} (root)")
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                name = arg.value
                if name in _ALLOWED_LOGGER_NAMES:
                    continue
                if name != "figaroh" and not name.startswith("figaroh."):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {name!r}")
    assert offenders == []


@pytest.fixture
def calib_logger():
    logger = logging.getLogger(bc.__name__)
    handlers, level = list(logger.handlers), logger.level
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_console_logging_keeps_the_application_handlers(make_calib, calib_logger):
    app = logging.NullHandler()
    calib_logger.addHandler(app)
    calib = make_calib()
    calib._setup_logging()
    calib._setup_logging()

    assert app in calib_logger.handlers
    console = [
        h for h in calib_logger.handlers if h.get_name() == bc._CONSOLE_HANDLER_NAME
    ]
    assert len(console) == 1


def _future_warnings(record):
    return [w for w in record if issubclass(w.category, FutureWarning)]


def test_calibration_default_solve_warns_once_and_still_reports(
    make_calib, calib_logger, capsys
):
    calib = make_calib()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        calib.solve(plotting=False)
    future = _future_warnings(record)
    assert len(future) == 1
    message = str(future[0].message)
    assert "enable_logging=" in message and "report=" in message
    assert future[0].filename == __file__
    assert "CALIBRATION QUALITY REPORT" in capsys.readouterr().out.upper()


def test_calibration_explicit_flags_do_not_warn(make_calib, calib_logger, capsys):
    calib = make_calib()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        calib.solve(plotting=False, enable_logging=False, report=False)
    assert capsys.readouterr().out == ""
    assert not [
        h for h in calib_logger.handlers if h.get_name() == bc._CONSOLE_HANDLER_NAME
    ]


try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None


def _identification():
    from test_data_contract_wiring import _Ident, _trajectory

    model = pin.buildSampleModelManipulator()
    ident = _Ident(model)
    ident.trajectory_to_return = _trajectory(model)
    ident.initialize()
    return ident


@pytest.mark.skipif(pin is None, reason="Pinocchio not available")
def test_identification_default_solve_warns_once(monkeypatch, capsys):
    ident = _identification()
    plotted = []
    monkeypatch.setattr(ident, "plot_results", lambda: plotted.append(True))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ident.solve(decimate=False)
    future = _future_warnings(record)
    assert len(future) == 1
    assert "plotting=" in str(future[0].message)
    assert "report=" in str(future[0].message)
    assert future[0].filename == __file__
    assert plotted == [True]
    assert capsys.readouterr().out != ""


@pytest.mark.skipif(pin is None, reason="Pinocchio not available")
def test_identification_explicit_flags_are_quiet(monkeypatch, capsys):
    ident = _identification()
    monkeypatch.setattr(ident, "plot_results", lambda: pytest.fail("plotted"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ident.solve(decimate=False, plotting=False, report=False)
    assert capsys.readouterr().out == ""


def test_resolve_changing_defaults():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert resolve_changing_defaults("f", {"a": False, "b": True}) == {
            "a": False,
            "b": True,
        }
    with pytest.warns(FutureWarning, match=r"f\(\) called without a=: .*2\.0"):
        assert resolve_changing_defaults("f", {"a": None, "b": False}) == {
            "a": True,
            "b": False,
        }
