# Identification

The identification module provides tools for dynamic parameter
identification of robots, including the
[reporting & verification suite](../reporting_and_verification.md)
(`print_quality_report()`, `export_html_report()`, `verify()`,
`export_verification_report()`) attached directly to `BaseIdentification`.

## base_identification

::: figaroh.identification.base_identification
    options:
      show_root_heading: false

## identification_tools

::: figaroh.identification.identification_tools
    options:
      show_root_heading: false

## config

::: figaroh.identification.config
    options:
      show_root_heading: false

## parameter

::: figaroh.identification.parameter
    options:
      show_root_heading: false

## physical_fit

Direct LMI effort fit over all standard parameters (public since #61;
`figaroh.identification._physical_comparator` remains as an alias).

::: figaroh.identification.physical_fit
    options:
      show_root_heading: false

## selection

::: figaroh.identification.selection
    options:
      show_root_heading: false

## Physical inertial conventions

Physical-consistency utilities use Pinocchio dynamic-parameter order:
`[m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`. The first moments are
`h = m*c`; the six tensor entries describe rotational inertia `I_O` about
the link-frame origin. They are distinct from inertia `I_C` about the centre
of mass: `I_O = I_C + m * ((c.T*c)*eye(3) - c*c.T)`.

The physical constraint is positive semidefiniteness of the pseudo-inertia
`P = [[Sigma, h], [h.T, m]]`, with
`Sigma = 0.5*trace(I_O)*eye(3) - I_O`. Projection uses this second-moment
block as its decision variable and converts back using
`I_O = trace(Sigma)*eye(3) - Sigma`. Its objective and weights apply to the
ten dynamic parameters. SDP reconstruction imposes the same pseudo-inertia
constraint while preserving its base-parameter equalities.

The existing CAD `com_bounds` option bounds first moments `h`, in kg*m,
rather than CoM coordinates in metres. Physical feasibility does not establish
identifiability or complete inertial URDF export.
