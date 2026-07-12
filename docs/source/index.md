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
  [Reporting & Verification](guides/reporting_and_verification.md).
- **Advanced linear solver with 10 methods** (lstsq, QR, SVD, Ridge, Lasso,
  Elastic Net, Tikhonov, constrained, robust, weighted)
- **Regularization and constraint optimization** (L1/L2 regularization, box
  constraints, linear equality/inequality)
- Unified configuration system with template inheritance
- Advanced regressor computation with object-oriented design
- Support for URDF modeling convention
- Optional physical-consistency projection and base→full parameter
  reconstruction for identification
- Extensive examples and tutorials

## Quick Links

- [PyPI Package](https://pypi.org/project/figaroh/)
- [Examples Repository](https://github.com/thanhndv212/figaroh-examples)
- [GitHub Repository](https://github.com/thanhndv212/figaroh-plus)

## Where to start

| I want to... | Go to |
|---|---|
| Install FIGAROH and run my first calibration/identification | [Getting Started](getting_started.md) |
| Generate quality reports and CI-gateable pass/fail verdicts | [Reporting & Verification](guides/reporting_and_verification.md) |
| Understand how the library is structured | [Architecture](architecture.md) |
| Look up a specific class or function | Core Modules / Tools & Utilities in the nav |
