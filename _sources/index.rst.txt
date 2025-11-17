Welcome to FIGAROH's documentation!
===================================

FIGAROH (Free dynamics Identification and Geometrical cAlibration of RObot and Human) 
is a comprehensive Python toolbox for robot calibration and identification.

**Key Features:**
- Dynamic parameter identification for rigid multi-body systems
- Geometric calibration for serial and tree-structure robots  
- **Advanced linear solver with 10 methods** (lstsq, QR, SVD, Ridge, Lasso, Elastic Net, Tikhonov, constrained, robust, weighted)
- **Regularization and constraint optimization** (L1/L2 regularization, box constraints, linear equality/inequality)
- Unified configuration system with template inheritance
- Advanced regressor computation with object-oriented design
- Support for URDF modeling convention
- Extensive examples and tutorials

**What's New in v0.3.0:**
- Advanced linear solver for robot parameter identification
- Module reorganization for better maintainability
- Enhanced BaseIdentification with flexible solver methods
- Comprehensive test coverage with 18 unit tests

**Quick Links:**
- `PyPI Package <https://pypi.org/project/figaroh/>`_
- `Examples Repository <https://github.com/thanhndv212/figaroh-examples>`_
- `GitHub Repository <https://github.com/thanhndv212/figaroh-plus>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: Core Modules:

   modules/calibration
   modules/identification 

.. toctree::
   :maxdepth: 2
   :caption: Tools and Utilities:
   
   modules/tools
   modules/utils
   modules/measurements
   modules/visualisation

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features:
   
   modules/optimal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
