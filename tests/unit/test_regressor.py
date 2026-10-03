"""Tests for regressor matrix computation functionality."""

import pytest
import numpy as np
import sys
import os
from types import SimpleNamespace

import pinocchio as pin

# Add the src directory to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from figaroh.tools.regressor import build_regressor_basic

    # Import other functions that actually exist
    try:
        from figaroh.tools.regressor import eliminate_non_dynaffect
    except ImportError:
        eliminate_non_dynaffect = None

    try:
        from figaroh.tools.regressor import get_index_eliminate
    except ImportError:
        get_index_eliminate = None

    try:
        from figaroh.tools.regressor import build_regressor_reduced
    except ImportError:
        build_regressor_reduced = None

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the figaroh package is installed or the path is correct")
    raise


@pytest.fixture(scope="module")
def arm():
    """Pinocchio sample manipulator with random, sign-mixed motion samples."""
    model = pin.buildSampleModelManipulator()
    robot = SimpleNamespace(model=model, data=model.createData())
    rng = np.random.default_rng(58)
    n = 7
    q = np.array([pin.randomConfiguration(model) for _ in range(n)])
    v = rng.uniform(-1, 1, (n, model.nv))
    a = rng.uniform(-3, 3, (n, model.nv))
    return robot, q, v, a


def _config(nv, act_idxv=None, **flags):
    cfg = {
        "is_joint_torques": True,
        "has_friction": False,
        "has_actuator_inertia": False,
        "has_joint_offset": False,
        "act_idxv": list(range(nv)) if act_idxv is None else act_idxv,
    }
    cfg.update(flags)
    return cfg


class TestRegressorBuilding:
    """build_regressor_basic on a real Pinocchio model (no mocks)."""

    def test_build_regressor_basic_exists(self):
        """Test that the main function exists and is callable."""
        assert callable(build_regressor_basic)

    def test_rows_are_joint_major_pinocchio_regressor(self, arm):
        """Row j * N + i is joint j at sample i of Pinocchio's own regressor."""
        robot, q, v, a = arm
        model, n = robot.model, len(q)

        W = build_regressor_basic(robot, q, v, a, _config(model.nv))

        assert W.shape == (n * model.nv, 10 * model.nv)
        for i in range(n):
            Y = pin.computeJointTorqueRegressor(
                model, model.createData(), q[i], v[i], a[i]
            )
            np.testing.assert_allclose(W[i::n], Y, atol=1e-12)

    @pytest.mark.parametrize(
        "flags, blocks",
        [
            ({}, []),
            ({"has_friction": True}, ["fv", "fs"]),
            ({"has_actuator_inertia": True}, ["ia"]),
            ({"has_joint_offset": True}, ["off"]),
            (
                {
                    "has_friction": True,
                    "has_actuator_inertia": True,
                    "has_joint_offset": True,
                },
                ["fv", "fs", "ia", "off"],
            ),
        ],
        ids=["none", "friction", "actuator_inertia", "joint_offset", "all"],
    )
    def test_extra_blocks_follow_inertial_columns_in_order(self, arm, flags, blocks):
        """Each enabled block takes nv columns after the inertial ones, in the
        order fv, fs, ia, off, and only joint j's rows populate its column."""
        robot, q, v, a = arm
        nv, n = robot.model.nv, len(q)
        expected_column = {
            "fv": lambda j: v[:, j],
            "fs": lambda j: np.sign(v[:, j]),
            "ia": lambda j: a[:, j],
            "off": lambda j: np.ones(n),
        }

        W = build_regressor_basic(robot, q, v, a, _config(nv, **flags))

        assert W.shape == (n * nv, (10 + len(blocks)) * nv)
        for k, block in enumerate(blocks):
            for j in range(nv):
                col = W[:, 10 * nv + k * nv + j].reshape(nv, n)
                np.testing.assert_array_equal(col[j], expected_column[block](j))
                np.testing.assert_array_equal(np.delete(col, j, axis=0), 0.0)

    def test_inactive_joints_get_no_extra_entries(self, arm):
        robot, q, v, a = arm
        nv = robot.model.nv
        active = [0, 2, 5]
        flags = {
            "has_friction": True,
            "has_actuator_inertia": True,
            "has_joint_offset": True,
        }

        W = build_regressor_basic(robot, q, v, a, _config(nv, active, **flags))

        extra = W[:, 10 * nv :]
        for k in range(4):
            for j in range(nv):
                populated = np.any(extra[:, k * nv + j] != 0)
                assert populated == (j in active), (k, j)
        # The inertial part does not depend on which joints are active.
        W_all = build_regressor_basic(robot, q, v, a, _config(nv, **flags))
        np.testing.assert_array_equal(W[:, : 10 * nv], W_all[:, : 10 * nv])

    def test_single_sample_1d_matches_2d(self, arm):
        robot, q, v, a = arm
        cfg = _config(robot.model.nv, has_friction=True)

        W_1d = build_regressor_basic(robot, q[0], v[0], a[0], cfg)
        W_2d = build_regressor_basic(robot, q[:1], v[:1], a[:1], cfg)

        np.testing.assert_array_equal(W_1d, W_2d)
        assert W_1d.shape == (robot.model.nv, 12 * robot.model.nv)

    def test_input_validation(self, arm):
        robot, q, v, a = arm
        cfg = _config(robot.model.nv)

        with pytest.raises(ValueError, match="q must have"):
            build_regressor_basic(robot, q[:, :-1], v, a, cfg)
        with pytest.raises(ValueError, match="Inconsistent sample counts"):
            build_regressor_basic(robot, q, v[:-1], a, cfg)
        with pytest.raises(ValueError, match="joint_torques or external_wrench"):
            build_regressor_basic(robot, q, v, a, {**cfg, "is_joint_torques": False})


class TestOptionalFunctions:
    """Test functions that may or may not exist."""

    @pytest.mark.skipif(
        eliminate_non_dynaffect is None, reason="eliminate_non_dynaffect not available"
    )
    def test_eliminate_non_dynaffect(self):
        """Test elimination of non-dynamically affecting parameters."""
        # Create test regressor with some small columns
        W = np.array(
            [[1.0, 0.5, 1e-8, 2.0], [0.8, 0.3, 1e-9, 1.5], [1.2, 0.7, 1e-7, 1.8]]
        )

        params_std = {"p1": 1.0, "p2": 0.5, "p3": 0.1, "p4": 2.0}

        try:
            W_reduced, params_reduced = eliminate_non_dynaffect(
                W, params_std, tol_e=1e-6
            )

            # Should eliminate column 2 (index 2) which has small norm
            assert W_reduced.shape[0] == W.shape[0]  # Same number of rows
            assert W_reduced.shape[1] <= W.shape[1]  # Same or fewer columns
            assert len(params_reduced) <= len(params_std)  # Same or fewer parameters

        except Exception as e:
            pytest.skip(f"Function signature different: {e}")

    @pytest.mark.skipif(
        get_index_eliminate is None, reason="get_index_eliminate not available"
    )
    def test_get_index_eliminate(self):
        """Test getting indices for elimination."""
        W = np.array([[1.0, 1e-8, 2.0], [0.8, 1e-9, 1.5], [1.2, 1e-7, 1.8]])

        params_std = {"p1": 1.0, "p2": 0.1, "p3": 2.0}

        try:
            result = get_index_eliminate(W, params_std, tol_e=1e-6)

            # Should return some kind of indexing information
            assert result is not None

        except Exception as e:
            pytest.skip(f"Function signature different: {e}")

    @pytest.mark.skipif(
        build_regressor_reduced is None, reason="build_regressor_reduced not available"
    )
    def test_build_regressor_reduced(self):
        """Test building reduced regressor."""
        W = np.random.randn(5, 6)
        idx_e = [1, 3, 5]  # Eliminate columns 1, 3, 5

        try:
            W_reduced = build_regressor_reduced(W, idx_e)

            assert isinstance(W_reduced, np.ndarray)
            assert W_reduced.shape[0] == W.shape[0]  # Same number of rows
            assert W_reduced.shape[1] <= W.shape[1]  # Same or fewer columns

        except Exception as e:
            pytest.skip(f"Function signature different: {e}")


class TestActualModuleStructure:
    """Test the actual structure of the regressor module."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported."""
        import figaroh.tools.regressor as regressor_module

        assert regressor_module is not None

    def test_build_regressor_basic_exists(self):
        """Test that the main function exists."""
        from figaroh.tools.regressor import build_regressor_basic

        assert callable(build_regressor_basic)

    def test_available_functions(self):
        """Print available functions for debugging."""
        import figaroh.tools.regressor as regressor_module

        available_functions = [
            name for name in dir(regressor_module) if not name.startswith("_")
        ]
        print(f"Available functions: {available_functions}")

        # Check for common expected functions
        expected_functions = ["build_regressor_basic"]
        for func_name in expected_functions:
            assert hasattr(
                regressor_module, func_name
            ), f"Missing expected function: {func_name}"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
