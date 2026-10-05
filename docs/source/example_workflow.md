# Plan, fit and validate a robot example

**Draft guideline for review — 2026-10-02.** This is a decision process for
using FIGAROH with a new robot or dataset. It complements the runnable
[tutorials](tutorials/index.md); it does not introduce a new pipeline API.
Record each decision in the examples repository's
[experiment brief](https://github.com/thanhndv212/figaroh-examples/blob/main/docs/experiment-brief-template.md).
Reference workflows demonstrate the process; a new example must still audit
its own sensors, model, assumptions and validation evidence.

## 1. Define the outcome and inventory available evidence

Start by answering these questions before selecting a solver:

| Question | Record | Consequence |
| --- | --- | --- |
| What should improve? | Effort prediction, geometric accuracy, or an exported model for a declared application | Determines residuals and acceptance metrics |
| Which model is available? | URDF/source revision, geometry packages, joint names/order, active chain, fixed/free base, payload and nominal parameters | Defines coordinates, frames and model assumptions |
| What measurements exist? | Raw files, timestamps, units, sensors, effort source, measured pose/contact components and missing channels | Determines which estimation problem is supported |
| What can be collected? | Accessible joints/poses, instrumentation, controller, logging rate, operating limits and recording time | Determines feasible excitation and validation design |
| What independent evidence exists? | Separate recording, unused trajectory/postures, synthetic truth or none | Limits the validation claim |
| What must remain fixed? | Known payload, unobserved joints, sensor transforms, geometry, friction or other nuisance parameters | Reduces ambiguity and defines the estimation scope |

Classify every input as measured, commanded, numerically derived, inferred or
simulated. Encoder position and commanded position are different signals.
Current-derived effort needs a documented conversion, gear ratio, sign and
joint-side convention; it is not automatically a calibrated torque measurement.
Keep unknown quantities unknown rather than filling them with undocumented
constants. Missing dynamic measurements may support a reduced gravity/friction
study, but do not establish full inertial identification.

**Output:** a task statement, input inventory, missing-data list and intended
validation level. If critical data are absent, plan acquisition or narrow the
model before fitting.

## 2. Choose the model and parameter scope

For **dynamic identification**, declare the rigid-body inertial blocks and
any friction, actuator inertia, coupling or external-force terms. State which
are fitted, fixed or excluded. Motion/effort data identify parameter
combinations; an apparently precise full inertial vector may simply be one
choice in an unobservable null space. Distinguish base parameters from a
physical full model and from nuisance parameters.

For **geometric calibration**, declare the chain, measured frame, sensor/world
frame, joint offsets/link corrections and any base/tool/marker transforms.
State the gauge: which frame or transform anchors otherwise interchangeable
corrections. A position-only marker supports position residuals; plane contact
constrains the observed contact component. Neither supplies full 6D pose truth.
Use measured joint configurations where available and document substitutes.

For calibration, which identifiable corrections to estimate, and whether with
priors, is a separate choice from the calibration level: see
[Calibration: choosing what to estimate](tutorials/calibration_estimation_guide.md).

Start with the smallest model that explains the measured phenomenon. Add a
parameter block only with a physical reason, observability evidence and a
validation test. Record payload, temperature/contact regime and other operating
conditions that could invalidate a single-model assumption.

**Output:** parameter/block table, observation equation, units, frame conventions,
fixed quantities, gauge and unsupported effects.

## 3. Select methods with explicit objectives

| Method | Appropriate question | Required interpretation |
| --- | --- | --- |
| Base-parameter OLS | Which observable dynamic combinations explain effort? | Prediction baseline; does not guarantee physical individual inertias |
| WLS or regularized estimation | Is there a justified noise model or prior? | Estimate weights/scales from training evidence; disclose prior and its influence |
| Exact SDP/LMI reconstruction | Can a physical full model preserve the chosen base vector? | Check solver status, physical constraints and base residual; a returned fallback is not success |
| Per-link physical projection | Can an estimated link vector be made physically consistent? | Recompute effort/base fit afterward; projection may alter prediction |
| Direct physical constrained fit | Which physical model minimizes the declared effort objective? | Separate experiment from exact reconstruction; confirm the selected implementation is supported |
| Log-Cholesky fit | Can positive pseudo-inertia coordinates produce an adequate physical fit? | Feasibility and optimizer convergence are separate; currently a research path pending convergence review |
| Geometric least-squares calibration | Which identifiable corrections reduce the measured pose/contact residual? | Declare residual components, scaling, gauge, priors and redistribution semantics; choose the [estimation method](concepts/calibration_estimation.md) (`structural`, `excitation`, `map`, `map_cv`, `cv_subset`) and record it |

This table is a selection guide, not a promise that every method is exposed by
every robot entry point. Check the [identification](api/identification.md),
[calibration](api/calibration.md) and [tools](api/tools.md) interfaces and the
robot adapter. Link a missing capability to a scoped issue instead of silently
substituting a different objective.

Compare methods with the same training/validation samples, conventions,
residual scaling and extra-parameter policy. Freeze extras to isolate inertial
methods, or compare joint estimation as a separately declared protocol. Record
initialization, priors, bounds, tolerances, solver/version, evaluation budget
and timeout. Select hyperparameters on training/development data; reserve the
final validation set for assessment.

**Output:** baseline, candidate methods, objective/constraint definitions and a
comparison protocol frozen before inspecting final validation results.

## 4. Design acquisition and validation together

If data already exist, first assess what they excite and where coverage is
missing. Do not assume a long recording supplies identifiable parameters.
For a new experiment, define both fitting and validation recordings up front.

For dynamics, vary configurations and motion to expose the selected gravity,
inertial and friction effects. Inspect the relevant regressor's rank and
singular spectrum under declared column scaling. Include direction/speed
coverage when estimating friction. For calibration, vary chain configurations
and marker observations to expose the chosen corrections; inspect the
measurement Jacobian and gauge dependencies. Repeated similar samples cannot
replace missing directions of information.

Use [optimal experiment design](tutorials/optimal_design.md) where applicable,
with the same parameter scope and constraints as the intended task. Compare
against a feasible baseline; matrix conditioning alone does not establish
better real measurements or validation accuracy. Check generated motion/poses
against the actual robot's position, velocity, acceleration, effort, workspace,
collision and acquisition constraints before execution. An offline optimizer
result is a candidate experiment, not evidence of executed motion.

Prefer an independent validation recording with different unused excitation or
postures in the intended operating range. A temporal block from one recording
is useful but weaker evidence. For filtered time series, separate blocks by a
guard interval sufficient for the documented filter/derivative support.
Random neighboring-sample splits can leak strongly correlated information.

**Output:** acquisition plan, excitation rationale, coverage diagnostics,
training/development/final-validation assignments and predeclared thresholds.

## 5. Collect and preserve measurements

Preserve immutable raw recordings and nominal models. Record acquisition date,
robot/controller state, payload, sensor configuration, calibration/conversion,
clock sources, actual sample intervals and missing/saturated/dropped samples.
Document whether streams share a clock and how offset/drift is established.
Store enough metadata to reconstruct joint order, units and all frame transforms.

Run a small acquisition check before the full experiment: inspect channel
signs, ranges, timestamps, synchronization and observation availability. Keep
recorded executed motion separate from the planned trajectory. Synthetic runs
need known model parameters, independently checked states/derivatives and saved
noise seeds; using the same code to generate and fit data alone can conceal a
shared convention bug.

**Output:** raw-data manifest, acquisition notes and sanity-check figures.

## 6. Process data through an auditable adapter

Keep robot-specific import/conversion in `examples/<robot>/utils/`; reusable
numerical processing belongs in the core library. Preserve a mapping from raw
samples to accepted, trimmed and decimated samples.

1. Check timestamps, missing values, joint order, dimensions, units and frames.
2. Align streams with an explicit policy; record interpolation and exclusions.
3. Establish the actual rate from timestamps. Set filter cutoffs in the correct
   units and document assumptions if timestamps are unavailable.
4. For dynamics, check velocity/acceleration provenance for **every active
   coordinate**. Handle `nq != nv` deliberately; do not blindly differentiate
   configuration columns as if all were scalar joint angles.
5. Document filter type/order, phase behavior, derivative method and edge trim.
   Apply consistent index changes to all corresponding channels.
6. Process split partitions independently so smoothing, learned noise weights,
   outlier rules or normalization do not borrow final-validation information.
7. Inspect raw/processed overlays and derived signals before constructing the
   regressor or calibration residual.

For geometric observations, check pose representation, rotation conventions
and marker validity; do not average or subtract rotation coordinates without
considering their representation. Label offline noncausal filtering if relevant
to the eventual deployment.

**Output:** processing recipe, accepted indices, split provenance and processed
signal diagnostics. Current rate/derivative limitations in the delivery plan
must be checked for the chosen adapter before trusting a fit.

## 7. Fit and read the result in layers

Fit the nominal/baseline and candidates under the frozen protocol. Preserve
raw estimates, reconstructed/projected estimates and failed candidates as
separate stages. Read the result in this order:

| Layer | Inspect | A useful conclusion |
| --- | --- | --- |
| Input correctness | Rates, signs, frames, synchronized channels and coverage | Whether the fitted problem matches the observations |
| Numerical termination | Success flag, residual/objective, budget, bounds and fallback | Whether the intended solve completed |
| Training fit | Per-joint/component RMSE, bias, residual traces and nominal comparison | What the model explains in the fitted data |
| Parameter interpretation | Rank, observable combinations, physical checks, gauge and prior sensitivity (calibration: `estimation_report`, posterior vs prior; see the [guide](tutorials/calibration_estimation_guide.md#5-read-the-diagnostics)) | Which parameter claims are justified |
| Held-out prediction | Same metrics on unused data, coverage and leakage checks | Whether the improvement transfers |
| Export parity | Reloaded-model effort or FK vs selected fitted stage | Whether the delivered model represents the fitted result |

Plot residuals against time and relevant configuration/velocity/acceleration,
as applicable. Persistent bias, phase lag or structured errors warrant checking
synchronization, conversion or missing effects before adding model complexity.
Correlation alone can conceal amplitude/bias error. Report each effort in its
proper unit (N or Nm), and translation/rotation separately unless justified
whitening defines a dimensionless residual. Do not combine mixed units into an
unlabeled aggregate error. Define relative errors and zero-signal behavior.

Good effort prediction does not prove each link parameter is correct. Synthetic
truth comparisons should use identifiable coordinates where appropriate;
real-data parameter analysis needs physical plausibility and sensitivity,
not an invented ground-truth vector. A positive pseudo-inertia candidate can
still have an unsuccessful optimizer termination or poor validation fit.

**Output:** methodology, procedure, model fitting results, parameter analysis,
held-out validation and limitations, with every candidate's status retained.

## 8. Validate, export and make the example reusable

Evaluate final held-out data with the selected model and frozen processing.
Label evidence as training-only, temporal holdout, separate simulation or
independent real recording. If no genuine held-out data exist, state that
validation is unavailable; a training fallback does not fill this gap.

Use [reporting and verification](reporting_and_verification.md), but inspect
which checks were computed, skipped and passed. A passing verdict with absent
validation checks does not establish generalization. Choose task thresholds
from the application and measurement limits before measuring; preserve
existing regression gates unless a separate reviewed change justifies them.

Export only an explicitly selected, accepted result stage to a new output.
Reload the model and compare FK for calibration or inverse dynamics for
identification on declared states. Check inertial frames/CoM/tensors and
physical consistency where relevant. Existing inertial-export limitations
require explicit checks; a written URDF alone is insufficient evidence.

Archive paired core/examples revisions, dependency versions, input/model/config
hashes, commands, processing/splits, solver settings/status, metrics, reports,
selected stage and export/reload evidence. Specify which fields the existing
archive captures and attach the remaining evidence explicitly. Hardware control
or rollout claims require their own executed experiment.

A reusable example includes a README explaining purpose and limitations, data
and model provenance, unified config, a small supported adapter, reproducible
commands, an evidence report and regression tests for its essential behavior.
See the examples repository's
[new-example guide](https://github.com/thanhndv212/figaroh-examples/blob/main/docs/new-example-guide.md)
for the folder layout and review checklist.
