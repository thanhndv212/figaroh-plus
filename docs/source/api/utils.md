# Utilities

## Configuration Parser

::: figaroh.utils.config_parser
    options:
      show_root_heading: false

## Cubic Spline

::: figaroh.utils.cubic_spline
    options:
      show_root_heading: false

## Results Manager

Shared fallback-plotting helper and the joint-major torque-plotting fix used
by both calibration and identification (`plot_with_fallback`,
`plot_calibration_results`, `plot_identification_results`).

::: figaroh.utils.results_manager
    options:
      show_root_heading: false

## Error Handling

::: figaroh.utils.error_handling
    options:
      show_root_heading: false

## Config Migration

Converts a legacy flat (`calibration:`/`identification:`) config to the
unified `extends:`/`tasks:` format, including an optional round-trip
self-check against a URDF. See "Migrating from legacy to unified format"
in [Config Parameters](../concepts/config_parameters.md).

::: figaroh.utils.config_migration
    options:
      show_root_heading: false
