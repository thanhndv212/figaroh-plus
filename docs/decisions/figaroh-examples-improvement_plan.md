# figaroh-examples Improvement Plan

**Status:** Phases 1–6 closed out (35/35 items verified). Phase 7 mostly open —
only 7.3 (TIAGo `update_model.py`) is done; 7.1, 7.2, 7.4 remain.
**Scope:** Cross-example audit of UR10, TIAGo, TALOS, Staubli TX40 + shared
infrastructure, generated Fri Jun 19 2026.

---

## Implementation status (verified 2026-08-16, against current `main`)

The header above previously read "Phases 1-6 complete (39/39 items)" — that
total was wrong on its own terms (Phases 1–6 sum to 35 items; 39 is the
all-7-phases total) and Phase 7's four `[x]` marks below don't match the
checked-in code.

| Item | Header claimed | Verified |
|---|---|---|
| Phases 1–6 (35 items: §1–§6 below) | ✅ complete | ✅ Confirmed — argparse/CLI, `if __name__` guards, INFO logging, CI workflow, end-to-end smoke tests, URDF symlinks, "colission" typo fix, `extends` template inheritance all present in the checked-in examples |
| 7.1 Add missing scripts to TALOS | ✅ `[x]` | ❌ **False** — `examples/talos/` has only `calibration_upperbody.py` + `update_model.py`; no `identification.py`, `optimal_config.py`, or `optimal_trajectory.py` |
| 7.2 Add missing scripts to Staubli TX40 | ✅ `[x]` | ❌ **False** — `examples/staubli_tx40/` has only `identification.py`; no `calibration.py`, `optimal_config.py`, `optimal_trajectory.py`, or `update_model.py` |
| 7.3 Add `update_model.py` to TIAGo | ✅ `[x]` | ✅ Confirmed — `examples/tiago/update_model.py` exists |
| 7.4 Rename TALOS `calibration_upperbody.py` → `calibration.py` | ✅ `[x]` | ❌ **False** — file is still named `calibration_upperbody.py` |

**Net: Phase 7 is 1/4 done (7.3 only), not 4/4.** Total across all 7 phases:
**36/39**, not 39/39. Phase 7's checkboxes below are corrected to match; all
other content (Phases 1–6, the audit tables) is left as originally written
since it's independently verified accurate.

---

## Audit Summary

### Script Coverage Matrix

| Script | UR10 | TIAGo | TALOS | Staubli TX40 |
|--------|:----:|:-----:|:-----:|:------------:|
| `calibration.py` | yes | yes | renamed `calibration_upperbody.py` | missing |
| `identification.py` | yes | yes | missing | yes |
| `optimal_config.py` | yes | yes | missing | missing |
| `optimal_trajectory.py` | yes | yes | missing | missing |
| `update_model.py` | yes | missing | yes (broken) | missing |

- **TALOS**: 2/5 scripts
- **Staubli TX40**: 1/5 scripts (most incomplete)
- **TIAGo**: 4/5 (missing update_model.py)
- **UR10**: 5/5 (but identification.py main() commented out)

### Findings Count by Priority

| Priority | Category | Count |
|----------|----------|-------|
| P0 | Broken / must fix | 7 |
| P1 | Copy-paste errors & typos | 3 |
| P2 | Non-standard practices (cross-cutting) | 7 |
| P3 | Config & model management | 6 |
| P4 | Dead code & placeholders | 6 |
| P5 | Infrastructure gaps | 6 |
| P6 | Complete incomplete examples | 4 |
| | **Total** | **39** |

---

## Phase 1: Stop the Bleeding (P0 — Broken Things)

Fix things that are actively broken or produce wrong results. No dependencies between these — can be parallelized.

### 1.1 Fix 10 failing tests
- **Scope:** `tests/`
- **Files:** `test_robot_configs.py`, `test_backward_compatibility.py`, mock configs
- **Problem:** `signal_processing` section missing in mock configs; `KeyError` on `tls_params`/`dq_lim_def` in legacy format tests
- **Action:** Add missing sections to mock configs; fix legacy test expectations
- **Effort:** M
- **Dependencies:** None
- **Status:** [x] completed

### 1.2 Fix TALOS update_model.py path bug
- **Scope:** TALOS
- **Files:** `examples/talos/update_model.py` (lines 102, 114)
- **Problem:** `dirname(dirname(abspath(__file__)))` goes one level too high — writes to `examples/data/` instead of `examples/talos/data/`
- **Action:** Remove one `dirname()` call so output goes to `talos/data/`
- **Effort:** S
- **Dependencies:** None
- **Status:** [x] completed

### 1.3 Connect TALOS update_model.py to calibration
- **Scope:** TALOS
- **Files:** `examples/talos/calibration_upperbody.py`, `examples/talos/update_model.py`
- **Problem:** `calibration_upperbody.py` never saves/returns `res`; `update_model.py` requires `res` but can't get it — the two scripts are completely disconnected
- **Action:** Have calibration save `res` to file (e.g. `data/calibration_results.npz`); update_model reads it
- **Effort:** M
- **Dependencies:** 1.2 (fix path first)
- **Status:** [x] completed

### 1.4 Fix UR10 identification.py — uncomment main()
- **Scope:** UR10
- **Files:** `examples/ur10/identification.py` (lines 35-104)
- **Problem:** `main()` is commented out; entire logic runs at module level on import
- **Action:** Uncomment `main()`, add `if __name__ == "__main__": main()`, remove module-level execution
- **Effort:** S
- **Dependencies:** None
- **Status:** [x] completed

### 1.5 Fix TIAGo broken data file paths
- **Scope:** TIAGo
- **Files:** `config/tiago_config_hey5.yaml`, `config/tiago_config_palgripper.yaml`, `config/tiago_config_mocap.yaml`, `config/tiago_config_mocap_vicon.yaml`
- **Problem:** 5+ configs reference non-existent CSV files; `tiago_config_hey5.yaml` has `data_file: data/` (directory, not file)
- **Action:** Either remove configs with missing data, add the data files, or fix the paths to point to existing data
- **Effort:** M
- **Dependencies:** None
- **Status:** [x] completed

### 1.6 Fix Staubli dead import
- **Scope:** Staubli TX40
- **Files:** `examples/staubli_tx40/utils/staubli_tx40_tools.py` (lines 36-47)
- **Problem:** `from ...shared.config_manager import ConfigManager` — `shared/` module does not exist; silently falls back to trivial YAML loader
- **Action:** Remove dead import and fallback; use direct YAML loading
- **Effort:** S
- **Dependencies:** None
- **Status:** [x] completed

### 1.7 Fix package_dirs path bugs
- **Scope:** Staubli TX40, TALOS
- **Files:** `examples/staubli_tx40/identification.py:45`, `examples/talos/calibration_upperbody.py:60`
- **Problem:** Both use `package_dirs="models"` (1 level up) but scripts run from `examples/<robot>/` which needs `../../models`
- **Action:** Standardize to `"../../models"` (relative from `examples/<robot>/`)
- **Effort:** S
- **Dependencies:** None
- **Status:** [x] completed

---

## Phase 2: Fix Copy-Paste Errors & Typos (P1)

Quick wins, mechanical fixes. No dependencies. Can run in parallel with Phase 1.

### 2.1 Fix "TX40" in TIAGo identification.py
- **Scope:** TIAGo
- **Files:** `examples/tiago/identification.py:106`
- **Problem:** Print statement says "TX40 DYNAMIC PARAMETER IDENTIFICATION RESULTS" — copy-paste from Staubli
- **Action:** Change to "TIAGo"
- **Effort:** S
- **Status:** [x] completed

### 2.2 Fix "TIAGo" in Staubli tools
- **Scope:** Staubli TX40
- **Files:** `examples/staubli_tx40/utils/staubli_tx40_tools.py:66`
- **Problem:** Print statement says "TiagoIdentification initialized for TIAGo robot" — copy-paste from TIAGo
- **Action:** Change to "TX40"
- **Effort:** S
- **Status:** [x] completed

### 2.3 Fix "colission" typo everywhere
- **Scope:** shared, TIAGo
- **Files:** `examples/create_example.sh`, `examples/tiago/utils/simplified_colission_model.py` (rename + all references), `examples/tiago/utils/__init__.py`
- **Problem:** "colission" should be "collision" — propagates from create_example.sh to all generated examples
- **Action:** Rename file, update all imports and references, fix create_example.sh
- **Effort:** S
- **Status:** [x] completed

---

## Phase 3: Standardize Practices (P2 — Cross-Cutting)

Apply consistently across all 4 examples. Best done per-example to keep changes coherent.

### 3.1 Add if __name__ == "__main__" guards
- **Scope:** ALL examples
- **Files:** UR10 (3 scripts: calibration, identification, optimal_trajectory), TIAGo (3 scripts: calibration, identification, optimal_trajectory), Staubli (1 script: identification)
- **Problem:** Code runs at module level on import — only UR10's optimal_config.py has the guard
- **Action:** Wrap entry-point logic in `main()`, add `if __name__ == "__main__": main()`
- **Effort:** S
- **Dependencies:** 1.4 (UR10 identification main() fix)
- **Status:** [x] completed

### 3.2 Fix logging levels
- **Scope:** ALL examples
- **Files:** Every script's `logging.basicConfig()`
- **Problem:** All scripts set `level=logging.CRITICAL` — suppresses all debug/info output, opposite of what you'd want
- **Action:** Change to `logging.INFO` (or `logging.WARNING` with `--verbose` flag for INFO)
- **Effort:** S
- **Status:** [x] completed

### 3.3 Add argparse/CLI to all entry-point scripts
- **Scope:** ALL examples
- **Files:** 9 scripts total (all calibration.py, identification.py, optimal_config.py, optimal_trajectory.py)
- **Problem:** No CLI interface — users must edit source to change anything. Only TIAGo's optimal_config.py has argparse.
- **Action:** Add at minimum `--config`, `--urdf`, `--verbose` flags to each script
- **Effort:** M
- **Status:** [x] completed

### 3.4 Add file existence validation + error handling
- **Scope:** ALL examples
- **Files:** 9 scripts + utils
- **Problem:** No file existence checks, no try/except at entry points — users get cryptic Pinocchio/Python errors on missing files
- **Action:** Add validation before loading (URDF, config, data files); wrap in try/except with user-friendly error messages
- **Effort:** M
- **Status:** [x] completed

### 3.5 Move hardcoded joint params to YAML config
- **Scope:** UR10, TIAGo, Staubli
- **Files:** `identification.py`, `optimal_trajectory.py` in each
- **Problem:** 40+ lines of joint config (active_joints, reduction_ratios, joint IDs, indices) hardcoded in scripts instead of read from YAML
- **Action:** Move all joint parameters into the unified config YAML; scripts read from config
- **Effort:** L
- **Dependencies:** 3.3 (argparse for --config)
- **Status:** [x] completed

### 3.6 Eliminate DRY violations
- **Scope:** UR10, TIAGo
- **Files:** 4 scripts (identification.py + optimal_trajectory.py in each)
- **Problem:** `active_joints` list duplicated across scripts; same joint config repeated
- **Action:** Extract to config (done in 3.5) or shared utils; single source of truth
- **Effort:** M
- **Dependencies:** 3.5
- **Status:** [x] completed

### 3.7 Add type hints to all public functions/methods
- **Scope:** ALL examples
- **Files:** ~15 Python files across all examples
- **Problem:** Zero type annotations anywhere
- **Action:** Add type hints to all public functions and method signatures
- **Effort:** M
- **Status:** [x] completed

---

## Phase 4: Config & Model Cleanup (P3)

### 4.1 Consolidate TIAGo config files
- **Scope:** TIAGo
- **Files:** `examples/tiago/config/`
- **Problem:** 10 YAML files with inconsistent formats; mix of legacy flat format + unified `tasks:` format; some reference missing data
- **Action:** Remove or archive broken/unused configs; keep only working configs; document which to use
- **Effort:** M
- **Dependencies:** 1.5 (fix broken paths first)
- **Status:** [x] completed

### 4.2 Enable extends in TIAGo unified config
- **Scope:** TIAGo
- **Files:** `examples/tiago/config/tiago_unified_config.yaml:6`
- **Problem:** `extends` is commented out — template inheritance disabled, won't catch schema drift
- **Action:** Uncomment `extends: "templates/humanoid_robot.yaml"` and verify it works
- **Effort:** S
- **Dependencies:** 4.1
- **Status:** [x] completed

### 4.3 Remove duplicate base_robot_config.yaml
- **Scope:** TALOS, Staubli
- **Files:** `examples/talos/config/base_robot_config.yaml`, `examples/staubli_tx40/config/base_robot_config.yaml`
- **Problem:** Copied into config dirs instead of referenced via template inheritance — template updates won't propagate
- **Action:** Remove local copies; rely on `extends` chain to templates
- **Effort:** S
- **Status:** [x] completed

### 4.4 Standardize package_dirs across all examples
- **Scope:** ALL examples
- **Files:** All entry-point scripts
- **Problem:** UR10 uses `"../../models"`, TIAGo uses `"tiago_description"`, TALOS/Staubli use `"models"` — no standard
- **Action:** Pick one pattern (likely `"../../models"`) and apply everywhere
- **Effort:** S
- **Dependencies:** 1.7 (Phase 1 path fix)
- **Status:** [x] completed

### 4.5 Replace URDF copies with symlinks to models/
- **Scope:** ALL examples
- **Files:** `urdf/` dirs in all examples
- **Problem:** URDFs are full copies (TIAGo has 7 copies, 72-129KB each); version drift risk; models/ also has copies
- **Action:** Replace with symlinks to `models/` where possible, or at minimum document the duplication
- **Effort:** M
- **Status:** [x] completed

### 4.6 Fix config references to non-existent files
- **Scope:** TALOS, TIAGo
- **Files:** YAML configs
- **Problem:** `sample_configurations_file` paths point to missing `data/optimal_configs/` directories
- **Action:** Remove references or create the directories/files
- **Effort:** S
- **Status:** [x] completed

---

## Phase 5: Remove Dead Code (P4)

No dependencies. Can run in parallel with earlier phases.

### 5.1 Remove validate_robot_config()
- **Scope:** TIAGo
- **Files:** `examples/tiago/utils/tiago_tools.py:42`
- **Problem:** Always returns True — placeholder code providing false confidence
- **Action:** Remove function and all calls
- **Effort:** S
- **Status:** [x] completed

### 5.2 Remove _save_tx40_parameters()
- **Scope:** Staubli TX40
- **Files:** `examples/staubli_tx40/utils/staubli_tx40_tools.py:491-526`
- **Problem:** Method defined but never called — dead code
- **Action:** Remove method
- **Effort:** S
- **Status:** [x] completed

### 5.3 Remove build_tiago_normal()
- **Scope:** TIAGo
- **Files:** `examples/tiago/utils/simplified_colission_model.py:183-197`
- **Problem:** Function never called — dead code
- **Action:** Remove function
- **Effort:** S
- **Status:** [x] completed

### 5.4 Fix or remove broken main() in collision model
- **Scope:** TIAGo
- **Files:** `examples/tiago/utils/simplified_colission_model.py:203`
- **Problem:** `from tiago_tools import load_robot` — wrong import path; `../urdf/tiago_48_hey5.urdf` path wrong
- **Action:** Fix import to `from examples.tiago.utils.tiago_tools import ...` or remove main() if not needed
- **Effort:** S
- **Dependencies:** 2.3 (filename rename)
- **Status:** [x] completed

### 5.5 Remove dead data_type parameter
- **Scope:** TALOS
- **Files:** `examples/talos/calibration_upperbody.py:44`
- **Problem:** `data_type` parameter accepted but never used
- **Action:** Remove from function signature
- **Effort:** S
- **Status:** [x] completed

### 5.6 Remove commented-out CSV writer block
- **Scope:** TALOS
- **Files:** `examples/talos/update_model.py:125-139`
- **Problem:** Commented-out code block for saving estimation results to CSV — never used
- **Action:** Remove commented block
- **Effort:** S
- **Status:** [x] completed

---

## Phase 6: Infrastructure (P5)

### 6.1 Add GitHub Actions CI
- **Scope:** shared
- **Files:** `.github/workflows/` (new)
- **Problem:** No automated testing on push/PR — tests only run manually
- **Action:** Add workflow that runs pytest on push/PR, activates `figaroh-dev` conda env
- **Effort:** M
- **Dependencies:** 1.1 (tests must pass first)
- **Status:** [x] completed

### 6.2 Fix run_tests.py
- **Scope:** shared
- **Files:** `tests/run_tests.py`
- **Problem:** Only runs 3 of 8 test files; parses stdout instead of using pytest API
- **Action:** Include all test files; use pytest API or just document `pytest` as the runner
- **Effort:** S
- **Status:** [x] completed

### 6.3 Fix web-interface config discovery
- **Scope:** shared
- **Files:** `web-interface/core/example_loader.py:48`
- **Problem:** Looks for `config.yaml`, `calibration.yaml` etc. but examples use `*_unified_config.yaml` — no configs discovered
- **Action:** Update discovery to look for `*_unified_config.yaml` patterns
- **Effort:** S
- **Status:** [x] completed

### 6.4 Expand templates README
- **Scope:** shared
- **Files:** `examples/templates/README.md`
- **Problem:** Only 14 lines, no examples of `extends`, `inherits_from`, `variants`
- **Action:** Add usage examples and explanations for template inheritance features
- **Effort:** S
- **Status:** [x] completed

### 6.5 Fix create_example.sh bugs
- **Scope:** shared
- **Files:** `examples/create_example.sh`
- **Problem:** Title-case conversion wrong (produces `Ur10` for `UR10`); replace order semantically wrong; temp file cleanup risk on interrupt
- **Action:** Fix awk title-case logic; fix replace order; add trap for cleanup
- **Effort:** M
- **Dependencies:** 2.3 (colission typo fix)
- **Status:** [x] completed

### 6.6 Add end-to-end smoke tests
- **Scope:** shared
- **Files:** `tests/` (new)
- **Problem:** All tests use mocks — none actually run example scripts
- **Action:** Add smoke tests that verify each example script imports and initializes without crashing
- **Effort:** M
- **Dependencies:** Phases 1-3 (scripts must be fixed first)
- **Status:** [x] completed

---

## Phase 7: Complete Incomplete Examples (Future)

New feature work, not cleanup. Should come after Phases 1-6 so new scripts follow standardized patterns.

### 7.1 Add missing scripts to TALOS
- **Scope:** TALOS
- **Scripts:** `identification.py`, `optimal_config.py`, `optimal_trajectory.py`
- **Effort:** L
- **Status:** [ ] open — none of the three scripts exist on `main` as of 2026-08-16

### 7.2 Add missing scripts to Staubli TX40
- **Scope:** Staubli TX40
- **Scripts:** `calibration.py`, `optimal_config.py`, `optimal_trajectory.py`, `update_model.py`
- **Effort:** L
- **Status:** [ ] open — none of the four scripts exist on `main` as of 2026-08-16

### 7.3 Add update_model.py to TIAGo
- **Scope:** TIAGo
- **Scripts:** `update_model.py`
- **Effort:** M
- **Status:** [x] completed

### 7.4 Rename TALOS calibration_upperbody.py
- **Scope:** TALOS
- **Problem:** Named `calibration_upperbody.py` instead of `calibration.py` — inconsistent with convention
- **Action:** Rename to `calibration.py` (or document why it differs)
- **Effort:** S
- **Status:** [ ] open — still named `calibration_upperbody.py` on `main` as of 2026-08-16

---

## Execution Strategy

```
Phase 1 (P0 broken)     ──┐
Phase 2 (P1 typos)      ──┤── parallel, no deps
Phase 5 (P4 dead code)  ──┘
         │
         ▼
Phase 3 (P2 standardize) ── per-example lanes (4 parallel)
         │
         ▼
Phase 4 (P3 config cleanup) ── depends on 1.7 (path fix)
         │
         ▼
Phase 6 (P5 infra) ── 6.1 depends on 1.1 (tests pass)
         │
         ▼
Phase 7 (complete examples) ── future, after patterns established
```

### Effort Legend
- **S** = small (<30 min)
- **M** = medium (30-90 min)
- **L** = large (2+ hours)

### Totals
- **39 work items** across 7 phases
- **Phase 1-2 + 5**: 16 items, all parallelizable, mostly S effort → fast wins
- **Phase 3**: 7 items, the core standardization work → most impactful
- **Phase 4**: 6 items, config/model cleanup → medium complexity
- **Phase 6**: 6 items, infrastructure → enables automation
- **Phase 7**: 4 items, new feature work → future scope — **1/4 done (7.3
  only); 7.1, 7.2, 7.4 still open, see Implementation status above**

---

## Detailed Audit Findings (Reference)

### P0 — Broken / Must Fix

| # | Issue | Scope | Location |
|---|-------|-------|----------|
| 1 | 10 failing tests — `signal_processing` missing in mock configs, `tls_params`/`dq_lim_def` KeyErrors | shared | `tests/` |
| 2 | `update_model.py` path bug (TALOS) — writes to `examples/data/` instead of `examples/talos/data/` | TALOS | `update_model.py:102` |
| 3 | `update_model.py` disconnected (TALOS) — calibration never saves `res`; update_model can't get results | TALOS | `calibration_upperbody.py` + `update_model.py` |
| 4 | `identification.py` main() commented out (UR10) — entire logic runs at module level on import | UR10 | `identification.py:35-104` |
| 5 | Broken data file paths (TIAGo) — 5+ configs reference non-existent CSV files | TIAGo | `config/tiago_config_hey5.yaml` et al. |
| 6 | Dead import (Staubli) — `from ...shared.config_manager import ConfigManager` silently falls back | Staubli | `staubli_tx40_tools.py:36-47` |
| 7 | `package_dirs` path bugs — Staubli & TALOS use `"models"` but need `../../models` | Staubli, TALOS | `identification.py:45`, `calibration_upperbody.py:60` |

### P1 — Copy-Paste Errors & Typos

| # | Issue | Scope | Location |
|---|-------|-------|----------|
| 8 | "TX40" printed in TIAGo identification — copy-paste from Staubli | TIAGo | `identification.py:106` |
| 9 | "TIAGo" printed in Staubli tools — copy-paste from TIAGo | Staubli | `staubli_tx40_tools.py:66` |
| 10 | "colission" typo — propagates from create_example.sh to all generated examples + TIAGo | shared, TIAGo | `create_example.sh`, `examples/tiago/utils/` |

### P2 — Non-Standard Practices (Cross-Cutting)

| # | Issue | Affected | Details |
|---|-------|----------|---------|
| 11 | Logging set to CRITICAL — suppresses all debug/info output | ALL | Every script: `logging.basicConfig(level=logging.CRITICAL)` |
| 12 | Missing `if __name__` guards — code runs on import | UR10 (3), TIAGo (3), Staubli (1) | Only UR10's optimal_config.py has it |
| 13 | Hardcoded params bypassing YAML — 40+ lines of joint config in code | UR10, TIAGo, Staubli | `identification.py` and `optimal_trajectory.py` |
| 14 | No argparse/CLI — users must edit source | ALL | Only TIAGo's optimal_config.py has argparse |
| 15 | No error handling/validation — no file checks, no try/except | ALL | Cryptic errors on missing files |
| 16 | No type hints anywhere | ALL | Zero type annotations |
| 17 | DRY violations — `active_joints` duplicated across scripts | UR10, TIAGo | identification.py + optimal_trajectory.py |

### P3 — Config & Model Management

| # | Issue | Scope | Details |
|---|-------|-------|---------|
| 18 | Config file proliferation — TIAGo has 10 YAML files, inconsistent formats | TIAGo | Mix of legacy + unified; `extends` commented out |
| 19 | Duplicate `base_robot_config.yaml` — copied into TALOS & Staubli | TALOS, Staubli | Template updates won't propagate |
| 20 | URDFs are copies, not symlinks — version drift risk | ALL | TIAGo has 7 copies (72-129KB each) |
| 21 | `package_dirs` inconsistency — 3 different patterns | ALL | No standard path resolution |
| 22 | Config references non-existent files — `data/optimal_configs/` missing | TALOS, TIAGo | YAML configs point to missing dirs |

### P4 — Dead Code & Placeholders

| # | Issue | Scope | Location |
|---|-------|-------|----------|
| 23 | Dead functions — `validate_robot_config()` returns True; `_save_tx40_parameters()` never called; `build_tiago_normal()` never called | TIAGo, Staubli | `tiago_tools.py:42`, `staubli_tx40_tools.py:491`, `simplified_colission_model.py:183` |
| 24 | Dead parameter — `data_type` in TALOS main() never used | TALOS | `calibration_upperbody.py:44` |
| 25 | Broken `main()` in collision model — wrong import path | TIAGo | `simplified_colission_model.py:203` |
| 26 | Commented-out CSV writer in update_model | TALOS | `update_model.py:125-139` |

### P5 — Infrastructure Gaps

| # | Issue | Scope | Details |
|---|-------|-------|---------|
| 27 | No CI workflows — no GitHub Actions | shared | `.github/` has only a Skill markdown |
| 28 | `run_tests.py` incomplete — only 3 of 8 test files | shared | `tests/run_tests.py` |
| 29 | Web-interface config discovery mismatch — looks for `config.yaml` not `*_unified_config.yaml` | shared | `web-interface/core/example_loader.py:48` |
| 30 | Templates README too brief — no `extends`/`variants` examples | shared | `examples/templates/README.md` |
| 31 | No end-to-end tests — all tests use mocks | shared | `tests/` |
| 32 | `create_example.sh` bugs — title-case, replace order, temp file cleanup | shared | `examples/create_example.sh` |
