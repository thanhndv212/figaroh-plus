"""Behavioral checks shared by the Pinocchio 3.7 and 4.1 CI profiles."""

import warnings
from unittest.mock import Mock

import numpy as np
import pinocchio as pin
import pytest

from figaroh.calibration.calibration_tools import get_rel_kinreg
from figaroh.tools.robotvisualization import RobotVisualizer


def test_relative_kinematic_regressor_uses_current_frame_api():
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        regressor = get_rel_kinreg(
            model, data, "universe", model.frames[-1].name, pin.neutral(model)
        )
    assert regressor.shape == (6, 6 * (model.njoints - 1))
    assert np.isfinite(regressor).all()
    assert np.linalg.matrix_rank(regressor) == 6


def test_com_visualization_uses_current_frame_api_without_gui():
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    visualizer = RobotVisualizer(model, data, Mock())
    frames = [model.getFrameId("shoulder1_body"), model.getFrameId("shoulder2_body")]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        visualizer.display_com(pin.neutral(model), frames)
        visualizer.display_bounding_boxes(
            pin.neutral(model), np.zeros(6), np.ones(6), frames
        )
    visualizer.viz.viewer.gui.addSphere.assert_called_once()
    assert visualizer.viz.viewer.gui.addBox.call_count == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_cholesky_dynamic_jacobian_matches_finite_difference(seed):
    coordinates = np.random.default_rng(seed).uniform(-0.5, 0.5, 10)
    parameters = pin.LogCholeskyParameters(coordinates)
    pseudo = parameters.toPseudoInertia()
    dynamic = parameters.toDynamicParameters()
    canonical = pin.PseudoInertia.FromDynamicParameters(dynamic)
    np.testing.assert_allclose(pseudo.toMatrix(), canonical.toMatrix(), atol=1e-12)
    assert np.linalg.eigvalsh(pseudo.toMatrix()).min() > 0
    assert dynamic[0] > 0

    step = 1e-6
    finite_difference = np.column_stack(
        [
            (
                pin.LogCholeskyParameters(
                    coordinates + step * direction
                ).toDynamicParameters()
                - pin.LogCholeskyParameters(
                    coordinates - step * direction
                ).toDynamicParameters()
            )
            / (2 * step)
            for direction in np.eye(10)
        ]
    )
    np.testing.assert_allclose(
        parameters.calculateJacobian(), finite_difference, rtol=1e-5, atol=1e-7
    )
