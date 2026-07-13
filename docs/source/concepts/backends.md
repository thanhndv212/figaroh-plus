# Backends

FIGAROH's algorithms (regressor construction, calibration, identification)
are written once against an abstract `DynamicsBackend` interface, not
against a specific simulator. Swapping the simulator underneath a workflow
is a config change, not a rewrite.

## Available backends

| Backend | Formats | Status | Best for |
|---|---|---|---|
| **Pinocchio** | URDF | Always available (core dependency) | Research & identification, CPU |
| **MuJoCo** | URDF, MJCF | Optional (`pip install mujoco`) | Contact dynamics, optimal control, sparse ops |
| **Genesis** | URDF, MJCF, USD | Optional (`pip install genesis-world`) | GPU, large-scale parallel/multi-robot |
| **Isaac Sim** | USD, URDF | Optional, requires NVIDIA Omniverse | Sim-to-real, RL, synthetic data |

```python
from figaroh.backends import get_backend, list_backends

print(list_backends())
# {'pinocchio': True, 'mujoco': True, 'genesis': False, 'isaacsim': False}

backend = get_backend("mujoco", model_path="robot.urdf")
```

## The `DynamicsBackend` interface

Every backend implements the same set of rigid-body dynamics primitives, so
identification/calibration code never branches on which simulator is
active:

- `compute_mass_matrix(q)` — `M(q)` in the equation of motion
- `compute_coriolis_matrix(q, v)` — `C(q, v̇)`
- `compute_gravity_vector(q)` — `g(q)`
- `compute_forward_kinematics(q)` — per-frame position/orientation
- `compute_jacobian(q, frame)` — stacked `[linear; angular]` geometric Jacobian
- `compute_regressor(q, v, a)` — the linear-in-parameters regressor `W` such
  that `τ = W(q, v̇, a) · θ`; this is the core computation identification
  builds on
- `compute_inverse_dynamics(q, v, a)` / `compute_forward_dynamics(q, v, τ)` —
  RNEA / ABA
- `nq`, `nv`, `model_format` properties

A handful of methods are optional (default to `NotImplementedError`) because
not every backend can implement them efficiently: `compute_dynamics_derivatives`,
`get_joint_names`, `get_frame_names`, `get_inertias`, `compute_difference`/
`compute_integrate` (Lie-group operations for free-flyer bases),
`random_configuration`, and `get_model_object()` — an escape hatch that
returns the underlying `pin.Model` or `mj.MjModel` for advanced features not
yet abstracted (e.g. `PseudoInertia`, collision models).

See [`figaroh.backends.base.DynamicsBackend`](../api/backends.md) for the
full signatures.

## Choosing a backend

- **Research & identification** → Pinocchio or MuJoCo
- **Contact-rich tasks** → MuJoCo or Genesis
- **Large-scale / parallel** → Genesis (GPU)
- **Sim-to-real** → Isaac Sim

## Next steps

- [Integration API](integration.md) — the high-level, backend-selecting
  entry point (`RobotIdentificationSystem`) built on top of this interface.
- [API Reference → Backends](../api/backends.md) — full class docs.
