# Welcome to FIGAROH's documentation!

**FIGAROH** (Free dynamics Identification and Geometrical cAlibration of
RObot and Human) is a comprehensive Python toolbox for robot calibration
and identification.

## Key Features

- Dynamic parameter identification for rigid multi-body systems
- Geometric calibration for serial and tree-structure robots
- **Reporting & verification (V&V) suite** — terminal and self-contained
  HTML diagnostic reports, a machine-readable pass/fail verdict
  (`verify()`), and a static two-run compare page. See
  [Reporting & Verification](reporting_and_verification.md).
- **Advanced linear solver with 10 methods** (lstsq, QR, SVD, Ridge, Lasso,
  Elastic Net, Tikhonov, constrained, robust, weighted)
- **Regularization and constraint optimization** (L1/L2 regularization, box
  constraints, linear equality/inequality)
- [Unified configuration system with template inheritance](concepts/configuration.md)
- Advanced regressor computation with object-oriented design
- Support for URDF modeling convention
- [Pluggable dynamics backends](concepts/backends.md) (Pinocchio, MuJoCo,
  Genesis, Isaac Sim)
- Optional physical-consistency projection and base→full parameter
  reconstruction for identification
- [Extensive examples and tutorials](tutorials/index.md) covering UR10,
  TIAGo, TALOS, and Staubli TX40

## Quick Links

- [PyPI Package](https://pypi.org/project/figaroh/)
- [Examples Repository](https://github.com/thanhndv212/figaroh-examples)
- [GitHub Repository](https://github.com/thanhndv212/figaroh-plus)

## Where to start

| I want to... | Go to |
|---|---|
| Install FIGAROH and run my first calibration/identification | [Getting Started](getting_started.md) |
| Understand the theory behind calibration/identification/optimal design | [Tutorials](tutorials/index.md) |
| Understand how the library is structured, or how backends/config work | [Concepts](concepts/architecture.md) |
| Generate quality reports and CI-gateable pass/fail verdicts | [Reporting & Verification](reporting_and_verification.md) |
| See a complete, runnable example for my robot (or a similar one) | [Examples Gallery](examples/index.md) |
| Look up a specific class or function | [API Reference](api/index.md) |
| Check what's planned, what changed, or troubleshoot an issue | [Further Reading](further_reading/faq.md) |
