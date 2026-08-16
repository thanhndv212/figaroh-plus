# URDF Exporter — Plan & Specification

**Status:** Implemented, with deviations from this spec (function-based API,
no `URDFExporter` class; metrology frame params surfaced but not
auto-applied; inertia-tensor handler unimplemented; camera-YAML and
multi-format export deferred). Section 5's parameter-name registry is still
the accurate behavioral spec — kept as reference, not superseded.
**Scope:** `figaroh/src/figaroh/tools/urdf_exporter.py` — applying
identified/calibrated parameter overlays onto a nominal URDF. Originally
written 2026-06-27 as a pre-implementation design doc.

---

## Implementation status (verified 2026-08-16, against current `main`)

The header originally read "Review document — NO implementation yet." That's
stale — `export_urdf()` shipped, but not exactly as specced in §2/§4 below.

| Planned item (§2, §4, §7) | Status | Note |
|---|---|---|
| `export_urdf()` function | ✅ Done | `urdf_exporter.py:545` |
| `URDFExporter` class (§2 "Internal Architecture") | ❌ Not built | Deviation — implemented as module-level functions + a handler-dispatch table instead of a class |
| Joint placement, additive (`d_px_*` etc.) | ✅ Done | `_apply_joint_placement` |
| Joint offset/calibration, additive (`offsetRX_*` etc., `<calibration rising>`) | ✅ Done | `_apply_joint_offset` |
| Mass, absolute (`m_*`) | ✅ Done | `_apply_mass` |
| Viscous/static friction, absolute (`fv_*`, `fs_*`) | ✅ Done | `_apply_viscous_friction`, `_apply_static_friction` |
| Armature, absolute (`Ia_*`) | ✅ Done | `_apply_armature` |
| Elasticity, additive (`k_*`) | ✅ Done | `_apply_elasticity` |
| Inertia tensor, absolute (`Ixx_*` etc.) | 🟡 Stub only | Category is registered and parsed, but the handler is a no-op (`logger.debug("inertia handler not implemented", ...)`) — `Ixx_*`/`Ixy_*`/etc. params are silently accepted and do nothing |
| Base placement (`base_px` etc.) | 🟡 Deviation | Parsed and recognized, but **deliberately not auto-applied** to the URDF — surfaced via `frame_settings_doc()` for the caller to configure separately, unlike §2's original "ADD to base frame placement" plan |
| EE marker (`pEEx`/`phiEEx` etc.) | 🟡 Deviation | Same as base placement — parsed but not auto-applied; §2's original "create new link for marker" behavior was not implemented |
| Unknown-param `ValueError` | ✅ Done | Matches §2 spec |
| Test suite (§3) | ✅ Present | `tests/unit/test_urdf_exporter.py` (18 test functions) + `tests/fixtures/pendulum.urdf`, matching the planned Phase-1 numerical-validation design |
| Visual/viser tests (§3 Phase 2) | ✅ Done | `TestURDFExporterVisual` class present, gated on `FIGAROH_TEST_VIZ=1`/`--viz`, matching the planned design |
| Camera-YAML export | ❌ Not started | No `camera*yaml*export` module anywhere in `src/figaroh` |
| Multi-format export — MJCF/SDF/USD (§4 Phase 4) | ❌ Deferred, as planned | No `export_mjcf`/`export_sdf`/`export_usd` in core; `examples/shared/exporters.py` does not exist on `main` (only ever existed on the `h1v2` branch) |
| Integration into example scripts (§4 Phase 3) | ✅ Done, broader than planned | `export_urdf`/`urdf_exporter` used in `talos/calibration_upperbody.py`, `tiago/calibration.py`, and `ur10/calibration.py` — §4 only planned UR10 `update_model.py` + TIAGo eye-hand, but eye-hand calibration itself was never ported (see the TIAGo port-review doc), so the actual integration path differs from what was planned |

**Bottom line:** the core additive/absolute joint- and dynamics-parameter
pipeline is solid and tested, but three of the plan's design decisions
changed during implementation (no class, metrology frames deliberately
excluded from auto-apply, inertia tensor left as a stub) and two scoped-out
items (camera YAML, multi-format) remain undone as planned.

---

## 1. Executive Summary

**Problem:** After running identification or calibration in figaroh, the identified parameters (inertial deltas, joint offsets, geometric placement deltas, base frame) exist only as a flat dict or YAML. There is no way to materialize them into a modified URDF that downstream tools (simulation, model-based control, visualization) can consume directly.

**Solution:** A `urdf_exporter.export_urdf()` function that reads a nominal URDF and applies a `params: dict[str, float]` overlay, writing a new URDF. Parameter names encode the semantics — `d_px_jointX` is additive (geometric offset), `m_link1` is absolute (inertial override).

**Scope:** URDF export only (Phase 1). MJCF/SDF/USD export deferred.

**Validation:** Numerical FK comparison between original and exported model + viser overlay visualization with trajectory/static-pose error measurement.

---

## 2. API Design

### Location

```python
# figaroh/src/figaroh/tools/urdf_exporter.py
```

### Function Signature

```python
def export_urdf(
    nominal_urdf_path: str,
    params: dict[str, float],
    *,
    output_path: Optional[Union[str, Path]] = None,
    package_dirs: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Apply identified/calibrated parameters to a nominal URDF and write the result.

    Args:
        nominal_urdf_path: Path to the nominal (reference) URDF file.
        params: Dictionary of {parameter_name: value} pairs as produced by
            figaroh identification or calibration routines.
        output_path: Path for the modified URDF. If None (default), writes to
            <stem>_modified.urdf beside the nominal URDF.
        package_dirs: Mesh package directories (passed through if URDF
            references meshes by package:// paths).
        verbose: If True, log which params were applied additively vs. absolutely.

    Returns:
        str: Absolute path to the modified URDF file.

    Raises:
        FileNotFoundError: If nominal_urdf_path does not exist.
        ValueError: If an unknown parameter name is encountered.
    """
```

### Add-vs-Replace Rules (Parameter-Name Driven)

The decision is encoded entirely in the parameter name prefix — no boolean flag needed:

| Prefix / Pattern | Rule | Examples |
|------------------|------|----------|
| `d_px_`, `d_py_`, `d_pz_`, `d_phix_`, `d_phiy_`, `d_phiz_` | **ADD** to `origin xyz` or `origin rpy` of the matching joint/placement | `d_px_joint2` → joint2.origin +0.05m in x |
| `offsetPX_`, `offsetPY_`, `offsetPZ_`, `offsetRX_`, `offsetRY_`, `offsetRZ_` | **ADD** to the matching joint's calibration/offset | `offsetRX_joint1` → joint1.calibration +0.1 rad |
| `base_px`, `base_py`, `base_pz`, `base_phix`, `base_phiy`, `base_phiz` | **ADD** to the base frame placement (first joint) | `base_px` → root joint origin +0.02m in x |
| `m_` (inertial mass) | **REPLACE** `<mass value="..."/>` | `m_link1` → link1 inertial mass becomes this value |
| `mx_`, `my_`, `mz_` (first moments) | **REPLACE** `<inertial><mass .../><origin .../><com .../></inertial>` | `mx_link1` → link1 first moment of mass around x |
| `Ixx_`, `Ixy_`, etc. (inertia tensor) | **REPLACE** the corresponding `<inertia>` element | `Ixx_link1` → replace link1 inertia Ixx |
| `fv_` (viscous friction) | **REPLACE** `<dynamics damping="..."/>` | `fv_joint1` → joint1 damping replaced |
| `fs_` (static friction / Coulomb) | **REPLACE** `<dynamics friction="..."/>` | `fs_joint1` → joint1 friction replaced |
| `Ia_` (armature inertia) | **REPLACE** `<dynamics armature="..."/>` | `Ia_joint1` → joint1 armature replaced |
| `off_` (joint offset, legacy) | **REPLACE** `<calibration rising="..."/>` | `off_joint1` → joint1 calibration replaced |
| `k_` (elasticity, suspension) | **ADD** to a `<dynamics elasticity="..."/>` element | `k_PX_joint1` → joint1 linear stiffness |
| `pEEx`, `phiEEx` (end-effector marker) | **ADD** as a new fixed joint + link after the tool frame | `pEEx_1` → create new link for marker 1 |

> **Note:** Unknown parameter names raise `ValueError` with a helpful message listing known categories.

### Internal Architecture

```python
class URDFExporter:
    """Applies parameter overlays to a URDF model."""

    # ── Registry: parameter prefix → rule type + handler ──
    ADDITIVE_PREFIXES: ClassVar[dict[str, str]] = {
        "d_px": "joint_placement", "d_py": "joint_placement",
        "d_pz": "joint_placement", "d_phix": "joint_placement",
        "d_phiy": "joint_placement", "d_phiz": "joint_placement",
        "offsetPX": "joint_calibration", "offsetPY": "joint_calibration",
        "offsetPZ": "joint_calibration", "offsetRX": "joint_calibration",
        "offsetRY": "joint_calibration", "offsetRZ": "joint_calibration",
        "base_px": "base_placement", "base_py": "base_placement",
        "base_pz": "base_placement", "base_phix": "base_placement",
        "base_phiy": "base_phiz": "base_placement",  # [sic: phiy/phiz]
        "k_": "elasticity",
        "pEEx": "ee_marker", "pEEy": "ee_marker", "pEEz": "ee_marker",
        "phiEEx": "ee_marker", "phiEEy": "ee_marker", "phiEEz": "ee_marker",
    }

    ABSOLUTE_PREFIXES: ClassVar[dict[str, str]] = {
        "m_": "mass",
        "mx_": "first_moment", "my_": "first_moment", "mz_": "first_moment",
        "Ixx_": "inertia", "Ixy_": "inertia", "Ixz_": "inertia",
        "Iyy_": "inertia", "Iyz_": "inertia", "Izz_": "inertia",
        "fv_": "viscous_friction",
        "fs_": "static_friction",
        "Ia_": "armature",
        "off_": "joint_offset",
    }

    def __init__(self, nominal_urdf_path: str):
        self._path = nominal_urdf_path
        self._doc: xml.etree.ElementTree.Element = ...  # parsed URDF

    def apply(self, params: dict[str, float]) -> None:
        """Apply parameter overlay in-place."""

    def write(self, output_path: str) -> str:
        """Write modified URDF to file."""

    # ── Handlers (one per rule type) ──
    def _apply_joint_placement_additive(...):
    def _apply_joint_calibration_additive(...):
    def _apply_base_placement_additive(...):
    def _apply_mass_absolute(...):
    def _apply_inertia_absolute(...):
    # ...
```

### Integration with Calibration Output

The calibration `full_params` dict (6 DOF joint placement deltas) uses `d_px_{joint_name}` etc. — this matches the additive `joint_placement` category and maps directly to `update_joint_placement()` convention in `calibration_tools.py:693`.

### Integration with Identification Output

The identification `params` dict uses `m_{link}`, `fv_{joint}`, etc. — these match the absolute `REPLACE` categories and overwrite the URDF dynamics/inertial elements directly.

---

## 3. Test Design

### Location

```
figaroh/tests/unit/test_urdf_exporter.py
figaroh/tests/fixtures/pendulum.urdf         # inline, self-contained
```

### Test Fixture: Inline Pendulum URDF

A 2-link planar pendulum (revolute-revolute) with:
- No mesh files (pure primitives)
- Deterministic FK
- Known joint placement, mass, friction parameters

```xml
<?xml version="1.0" ?>
<robot name="pendulum">
  <link name="world"/>
  <joint name="joint1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14"/>
    <dynamics damping="0.05"/>
  </joint>
  <link name="link2">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

### Phase 1 — Numerical Validation (always runs, CI-safe)

```python
class TestURDFExporterNumerical:
    """CI-safe FK comparison between original and exported model."""

    def setup_method(self):
        self.nominal = FIXTURES / "pendulum.urdf"
        self.params = {
            # Additive: geometric offset
            "d_px_joint2": 0.05,           # joint2 origin +0.05m in x
            "d_phiz_joint2": 0.1,          # joint2 origin +0.1 rad in z
            "offsetRX_joint1": 0.25,       # joint1 calibration +0.25 rad
            # Absolute: inertial/dynamics override
            "m_link1": 2.5,                # link1 mass becomes 2.5
            "fv_joint1": 0.2,              # joint1 damping becomes 0.2
            "base_pz": 0.1,                # base origin +0.1m in z
        }
        self.modified = export_urdf(self.nominal, self.params)

    # ── Trajectory tracking ──

    def test_trajectory_position_rmse(self):
        """100 random configs → FK → RMSE position error matches applied deltas."""
        errors_6d = compute_trajectory_errors(self.nominal, self.modified, n_samples=100)
        # joint2 origin shifted by d_px=0.05, d_phiz=0.1 → expected positional delta
        assert errors_6d.rmse_position < 0.06      # close to d_px magnitude
        assert errors_6d.rmse_orientation < 0.11    # close to d_phiz magnitude

    def test_trajectory_max_error(self):
        """Max single-point error does not exceed parameter bounds."""
        errors_6d = compute_trajectory_errors(self.nominal, self.modified, n_samples=100)
        assert errors_6d.max_position < 0.07
        assert errors_6d.max_orientation < 0.12

    def test_zero_params_identity(self):
        """Empty params → exported URDF produces identical FK."""
        zero = export_urdf(self.nominal, {})
        errors_6d = compute_trajectory_errors(self.nominal, zero, n_samples=50)
        assert errors_6d.rmse_position < 1e-10
        assert errors_6d.rmse_orientation < 1e-10

    # ── Static configurations ──

    def test_static_singular_config(self):
        """Home config: joint2 placement delta = 0.05m in world x."""
        # When both joints at 0, joint2 origin shift is purely along original x axis
        q_home = np.array([0.0, 0.0])
        delta = compute_pose_delta(self.nominal, self.modified, q_home)
        assert abs(delta.translation[0] - 0.05) < 1e-6   # d_px applied

    def test_static_rotated_config(self):
        """Config with joint1=90°: joint2 x-shift projected through rotation."""
        q_rot = np.array([np.pi / 2, 0.0])
        delta = compute_pose_delta(self.nominal, self.modified, q_rot)
        # With joint1 at 90°, d_px is now along world y
        assert abs(delta.translation[1] - 0.05) < 1e-6

    # ── Parameter name routing ──

    def test_additive_params_change_placement(self):
        """d_px params produce different placement XML, not mass XML."""
        orig_placements = extract_joint_origins(self.nominal)
        mod_placements = extract_joint_origins(self.modified)
        assert mod_placements["joint2"] != orig_placements["joint2"]

    def test_absolute_params_change_mass(self):
        """m_ params produce different mass XML, not placement XML."""
        orig_masses = extract_link_masses(self.nominal)
        mod_masses = extract_link_masses(self.modified)
        assert mod_masses["link1"] == 2.5              # exactly replaced
        assert mod_masses["link2"] == orig_masses["link2"]  # untouched

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError, match="Unknown parameter.*foobar"):
            export_urdf(self.nominal, {"foobar_joint1": 1.0})
```

### Phase 2 — Visual Validation (interactive, `--viz` flag or `$FIGAROH_TEST_VIZ=1`)

```python
# pytest -xvs --viz tests/unit/test_urdf_exporter.py

@pytest.mark.skipif("not os.environ.get('FIGAROH_TEST_VIZ')")
class TestURDFExporterVisual:
    """Interactive viser-based overlay visualization. Not for CI."""

    def test_overlay_both_models(self, viser_server):
        """Original (blue, 60% opacity) + Modified (red, 60%) overlaid."""
        urdf_orig = yourdfpy.URDF.load(self.nominal)
        urdf_mod  = yourdfpy.URDF.load(self.modified)

        ViserUrdf(
            viser_server, urdf_or_path=urdf_orig,
            root_node_name="/robot/original",
            color=(0.2, 0.4, 0.9, 0.6),
        )
        ViserUrdf(
            viser_server, urdf_or_path=urdf_mod,
            root_node_name="/robot/modified",
            color=(0.9, 0.2, 0.2, 0.6),
        )
        # → User inspects visually: joint2 origin visibly shifted, masses identical

    def test_trajectory_animation(self, viser_server):
        """Animate through configs, draw end-effector path trace."""
        configs = sample_random_configs(self.nominal, n=100)
        orig_path, mod_path = [], []

        for q in configs:
            update_viser_urdf(viser_server, "/robot/original", q)
            update_viser_urdf(viser_server, "/robot/modified", q)
            orig_path.append(compute_ee_pose(self.nominal, q))
            mod_path.append(compute_ee_pose(self.modified, q))
            time.sleep(0.05)

        # Draw paths as line sets
        draw_path(viser_server, "/paths/original", orig_path, color=(0.2, 0.4, 0.9))
        draw_path(viser_server, "/paths/modified", mod_path, color=(0.9, 0.2, 0.2))
        # → User sees path divergence caused by d_px_joint2

    def test_static_config_grid(self, viser_server):
        """5×4 grid of configs, each cell showing both models."""
        configs = sample_singular_configs(n=20)
        for i, q in enumerate(configs):
            row, col = divmod(i, 4)
            x, z = col * 2.0, row * 2.0  # grid layout
            # Place both models at this grid position
            place_model(viser_server, "/grid/orig", q, x, z, color=BLUE)
            place_model(viser_server, "/grid/mod", q, x, z, color=RED)
            # Annotate error
            delta = compute_pose_delta(self.nominal, self.modified, q)
            add_label(viser_server, f"/grid/label_{i}",
                      f"{norm(delta.translation)*1000:.1f}mm\n{norm(delta.rotation)*180/np.pi:.1f}°")
```

### Helper: `compute_trajectory_errors`

```python
@dataclass
class TrajectoryErrors6D:
    rmse_position: float
    rmse_orientation: float
    max_position: float
    max_orientation: float
    per_sample: list

def compute_trajectory_errors(
    nominal_urdf: str, modified_urdf: str,
    n_samples: int = 100, seed: int = 42
) -> TrajectoryErrors6D:
    """Load both URDFs, sample configs, compute FK pose deltas, return metrics."""
    model_a = pin.buildModelFromURDF(nominal_urdf)
    model_b = pin.buildModelFromURDF(modified_urdf)
    data_a, data_b = model_a.createData(), model_b.createData()
    rng = np.random.default_rng(seed)

    frame_id_a = model_a.getFrameId(model_a.frames[-1].name)
    frame_id_b = model_b.getFrameId(model_b.frames[-1].name)
    # end-effector is the last frame in both

    samples = []
    for _ in range(n_samples):
        q = rng.uniform(-np.pi, np.pi, size=model_a.nq)
        pin.forwardKinematics(model_a, data_a, q)
        pin.updateFramePlacements(model_a, data_a)
        pin.forwardKinematics(model_b, data_b, q)
        pin.updateFramePlacements(model_b, data_b)
        M_a = data_a.oMf[frame_id_a]
        M_b = data_b.oMf[frame_id_b]

        # 6D error: se3 twist log(M_a⁻¹ * M_b)
        delta_SE3 = M_a.inverse() * M_b
        delta_twist = pin.log(delta_SE3)  # [v, ω] ∈ ℝ⁶
        samples.append(delta_twist)

    samples = np.array(samples)  # (N, 6)
    return TrajectoryErrors6D(
        rmse_position=float(np.sqrt(np.mean(samples[:, :3]**2))),
        rmse_orientation=float(np.sqrt(np.mean(samples[:, 3:]**2))),
        max_position=float(np.max(np.linalg.norm(samples[:, :3], axis=1))),
        max_orientation=float(np.max(np.linalg.norm(samples[:, 3:], axis=1))),
        per_sample=samples.tolist(),
    )
```

### Helper: `compute_pose_delta`

```python
def compute_pose_delta(
    nominal_urdf: str, modified_urdf: str, q: np.ndarray
) -> pin.SE3:
    """Compute SE3 delta between end-effector poses at a single config."""
    model_a = pin.buildModelFromURDF(nominal_urdf)
    model_b = pin.buildModelFromURDF(modified_urdf)
    data_a, data_b = model_a.createData(), model_b.createData()
    pin.forwardKinematics(model_a, data_a, q)
    pin.updateFramePlacements(model_a, data_a)
    pin.forwardKinematics(model_b, data_b, q)
    pin.updateFramePlacements(model_b, data_b)
    frame_id = model_a.getFrameId(model_a.frames[-1].name)
    return data_a.oMf[frame_id].inverse() * data_b.oMf[frame_id]
```

### viser Test Fixture

```python
# conftest.py
@pytest.fixture(scope="session")
def viser_server():
    """Shared viser server for visual tests."""
    import viser
    server = viser.ViserServer(port=8080, verbose=False)
    yield server
    # Fixture cleanup: scene stays open (user closes manually),
    # or shutdown on test end
    server.stop()
```

---

## 4. Implementation Plan

### Phase 1: Tests First (TDD)

| Step | File | Content | Est. lines |
|------|------|---------|------------|
| 1.1 | `tests/fixtures/pendulum.urdf` | Inline 2-link pendulum URDF | 40 |
| 1.2 | `tests/unit/test_urdf_exporter.py` | Numerical tests (CI-safe) | 180 |
| 1.3 | `tests/unit/test_urdf_exporter.py` | Visual tests (viser, --viz flag) | 120 |

**Exit criteria:** Tests run (with `pytest -xvs tests/unit/test_urdf_exporter.py`), numerical tests pass (visual tests skip).

### Phase 2: Core API

| Step | File | Content | Est. lines |
|------|------|---------|------------|
| 2.1 | `src/figaroh/tools/urdf_exporter.py` | `URDFExporter` class + `export_urdf()` function | 250 |
| 2.2 | `src/figaroh/tools/__init__.py` | Re-export `export_urdf` | 1 |

**Key implementation details:**
- Parse nominal URDF with `xml.etree.ElementTree`
- Walk `<joint>` elements for placement/calibration/dynamics
- Walk `<link><inertial>` elements for mass/inertia
- Apply additive params by parsing the existing numeric string, adding the param value, formatting back
- Apply absolute params by replacing the numeric string entirely
- Preserve comments, whitespace, and mesh references (`package://` paths)

**Parameter-name → element mapping:**

| Param prefix | URDF element | Attribute | Action |
|-------------|--------------|-----------|--------|
| `d_px_` | `<origin xyz="x y z">` | x | `x += value` |
| `d_py_` | `<origin xyz="x y z">` | y | `y += value` |
| `d_pz_` | `<origin xyz="x y z">` | z | `z += value` |
| `d_phix_` | `<origin rpy="r p y">` | r | `r += value` |
| `d_phiy_` | `<origin rpy="r p y">` | p | `p += value` |
| `d_phiz_` | `<origin rpy="r p y">` | y | `y += value` |
| `offsetRX_` | `<calibration rising="..."/>` | rising | `rising += value` |
| `m_` | `<mass value="..."/>` | value | `value = value` (replace) |
| `fv_` | `<dynamics damping="..."/>` | damping | `damping = value` (replace) |
| `base_px` | root joint `<origin xyz="x y z">` | x | `x += value` |

### Phase 3: Integration & Examples

| Step | File | Content | Est. lines |
|------|------|---------|------------|
| 3.1 | `figaroh-examples/examples/ur10/update_model.py` | Use `export_urdf` for UR10 calibration output | 10 |
| 3.2 | `figaroh-examples/examples/tiago/eye_hand_calibration.py` | Use `export_urdf` for eye-hand output | 5 |

### Phase 4 (deferred): Multi-format

| Format | Priority | Depends on |
|--------|----------|------------|
| MJCF | Low | MuJoCo backend stabilization |
| SDF | Low | User demand |
| USD | Lowest | Production workflow need |

---

## 5. Parameter-Name Registry (Reference)

For implementation matching in `urdf_exporter.py`, the canonical naming is defined in `figaroh/src/figaroh/calibration/parameter.py`:

```python
FULL_PARAMTPL = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"]
JOINT_OFFSETTPL = ["offsetPX", "offsetPY", "offsetPZ", "offsetRX", "offsetRY", "offsetRZ"]
BASE_TPL = ["base_px", "base_py", "base_pz", "base_phix", "base_phiy", "base_phiz"]
ELAS_TPL = ["k_PX", "k_PY", "k_PZ", "k_RX", "k_RY", "k_RZ"]
EE_TPL = ["pEEx", "pEEy", "pEEz", "phiEEx", "phiEEy", "phiEEz"]
```

All additive params are in these templates. Anything not in these templates and not matching `m_`, `mx_`, `my_`, `mz_`, `Ixx_`, `Ixy_`, `Ixz_`, `Iyy_`, `Iyz_`, `Izz_`, `fv_`, `fs_`, `Ia_`, `off_` is an unknown parameter → `ValueError`.

---

## 6. Open Questions

1. **Calibration output contains `full_params` (6-DOF per joint) and `base_frame` (6-DOF).** Should `export_urdf()` accept the calibration result object directly, or always take the flat `params: dict[str, float]`? **Current plan:** flat dict only — one input format, adapter exists at call site.

2. **Joint calibration XML element:** The URDF spec has `calibration` with `rising`/`falling` attributes. Figaroh's `offsetRX_` convention maps to `rising` — but is this the correct element for joint angle offsets? **Confirmed:** Yes, `calibration rising="value"` is the standard URDF representation for joint offsets.

3. **Tool frame / EE marker in URDF:** The eye-hand calibration produces a tool tip pose. How should this be represented — a new fixed joint + blank link appended to the chain? **Current plan:** Append `<joint name="tool_frame" type="fixed">` + `<link name="tool_tip"/>`. This preserves downstream FK.

4. **Mesh handling for overlay:** When visualizing with viser, should the test handle robots with mesh files (not just inline pendulum)? **Deferred** — Phase 1 tests use inline pendulum; Phase 3 may add a real-robot integration test.

---

## 7. Key Design Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Input format | `dict[str, float]` | Universal — accepts both calibration and identification output |
| 2 | Add-vs-replace detection | Parameter name prefix | No boolean flag; human-readable; matches figaroh naming conventions |
| 3 | XML parser | `xml.etree.ElementTree` | Stdlib, no deps. No need for lxml for basic URDF mutation |
| 4 | Output path default | `nominal_stem + "_modified.urdf"` | Non-destructive, predictable, parallel-safe |
| 5 | Primary visualizer | viser (`ViserUrdf`) | Interactive overlay of two models with distinct colors; supports trajectory animation |
| 6 | Fallback visualizer | meshcat (`pinocchio.visualize.MeshcatVisualizer`) | No viser server dependency; works headless in CI |
| 7 | Test model | Inline 2-link pendulum | No mesh files; deterministic FK; fast; easy to hand-verify deltas |
| 8 | Verification metric | se3 twist `log(M₁⁻¹ · M₂)` | Single 6D error vector; translation + rotation in one metric |
