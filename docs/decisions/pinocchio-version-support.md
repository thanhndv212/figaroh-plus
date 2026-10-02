# Decision: Pinocchio version support

- Status: Accepted
- Date: 2026-10-02
- Issue: [#28](https://github.com/thanhndv212/figaroh-plus/issues/28);
  supports Feature 3 [#20](https://github.com/thanhndv212/figaroh-plus/issues/20)
  separately from the log-Cholesky go/no-go.
- Supersedes / superseded by: None

## Context

FIGAROH's unbounded `pin` dependency can resolve to a new major version while
local development still uses Pinocchio 3.7.0. Upstream 4.x removes deprecated
interfaces and updates the native dependency stack. The
[3-to-4 migration guide](https://github.com/stack-of-tasks/pinocchio/blob/v4.1.0/doc/_porting/porting-3-to-4.md)
replaces frame `parent` with `parentJoint` and documents the Coal transition.
FIGAROH uses the affected frame field in calibration and COM visualization.

Pinocchio 3.7 already provides `LogCholeskyParameters`, including dynamic
parameter conversion and its analytic Jacobian. Feature 3 can use this public
API across the selected versions without requiring a new major version.

## Decision

- Declare `pin>=3.7,<5`. The lower bound matches the retained baseline;
  the upper bound excludes an unreviewed future major version.
- Pin and test 3.7.0 and 4.1.0 on Python 3.12. Versions between those pins
  are resolver-eligible, not individually validated configurations.
- Select compatible ndcurves versions with each Pinocchio profile:
  2.0.0.1 for Pinocchio 3.7.0, 2.3.0 for Pinocchio 4.1.0. Do not upgrade
  Pinocchio alone inside an old environment and retain an incompatible
  ndcurves/native dependency set.
- Apply profile constraints before environment creation and subsequent
  pip installs. Require native imports, exact version assertions and
  `python -m pip check`; archive the constraints and installed versions.
- Keep native wheel constraints in `ci/pinocchio-3.7.0.txt` and
  `ci/pinocchio-4.1.0.txt`. In addition to pin/ndcurves, constrain Assimp,
  urdfdom and tinyxml2 to the compatible tested stack. For example, urdfdom
  6.0.0 loads tinyxml2 11, while the 3.7 baseline uses urdfdom 4.0.1 and
  tinyxml2 10. Package metadata alone did not catch the native load failure
  when the old tinyxml2 remained installed during the local upgrade.
- Use `parentJoint`, supported by both profiles. Keep semantic behavior
  unchanged in this compatibility change.
- Add a dedicated 4.1 core profile. Retain existing check names and test
  MuJoCo 3.9 with Pinocchio 3.7, MuJoCo 3.14 with Pinocchio 4.1.
- Keep production dependency ranges distinct from reproducible CI pins.
  Refresh CI pins deliberately after validating the replacement versions.

## Alternatives and consequences

Requiring Pinocchio 4.x immediately would exclude working 3.7 installations
without benefiting the log-Cholesky API. Leaving `pin` unbounded offers no
guard against future major changes. Pinning one exact version in package
metadata would prevent compatible dependency resolution for downstream users.

The chosen matrix adds one full-suite job and needs coordinated updates when
Pinocchio or ndcurves change. It covers representative backend combinations,
not the Cartesian product of all supported versions, platforms and Python
versions. Package Python metadata is unchanged; Python 3.12 is the tested
baseline and this decision adds no claim of testing older Python versions.

## Validation and implementation

The compatibility suite runs real frame/COM operations without a GUI and
fails on deprecated frame access. It verifies log-Cholesky conversion and
checks the analytic dynamic-parameter Jacobian against central differences.
The full suite retains physical-consistency, reconstruction and backend tests.

Local validation uses the original `figaroh-dev` for 3.7 and a separate clone
named `figaroh-dev` for 4.1. The clone keeps the original environment intact.
Full-suite counts, representative example metrics, native dependency versions
and remaining coverage limitations belong in the validation audit. Hosted
checks require publishing the change; an edited workflow is not proof of CI
success.

See the [2026-10-02 validation audit](../development/pinocchio-compatibility-audit-2026-10-02.md)
for local results and coverage limits.
