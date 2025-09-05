Getting Started
===============

Installation
-----------

Install FIGAROH from PyPI:

.. code-block:: bash

    pip install figaroh

For development with all dependencies:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate figaroh-dev
    pip install -e .

Configuration System
-------------------

FIGAROH uses a flexible YAML-based configuration system that supports both modern unified format and legacy format.

Unified Configuration Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The modern unified format provides better organization and template inheritance:

.. code-block:: yaml

    # modern_config.yaml
    inherit_from: "templates/base_robot.yaml"

    robot:
      name: "tiago"
      urdf_path: "urdf/tiago.urdf"

    calibration:
      method: "full_params"
      sensor_type: "camera"
      
      markers:
        - ref_joint: "wrist_3_joint"  
          position: [0.1, 0.0, 0.05]
          measure: [true, true, true, true, true, true]

    identification:
      mechanics:
        friction_coefficients:
          viscous: [0.01, 0.02, 0.015]
          static: [0.001, 0.002, 0.0015]
        actuator_inertias: [0.1, 0.15, 0.12]
        
      signal_processing:
        sampling_frequency: 5000.0
        cutoff_frequency: 100.0

Legacy Format Support
^^^^^^^^^^^^^^^^^^^^

Existing configurations continue to work without modification:

.. code-block:: yaml

    # legacy_config.yaml  
    calibration:
      calib_level: full_params
      markers:
        - ref_joint: wrist_3_joint
          measure: [True, True, True, True, True, True]

    identification:
      robot_params:
        - fv: [0.01, 0.02, 0.015] 
          fs: [0.001, 0.002, 0.0015]
      processing_params:
        - ts: 0.0002
          cut_off_frequency_butterworth: 100.0

Quick Start Examples
------------------

Basic Calibration
^^^^^^^^^^^^^^^^

.. code-block:: python

    from figaroh.calibration import BaseCalibration
    from figaroh.tools.robot import load_robot

    # Load robot and run calibration
    robot = load_robot("path/to/robot.urdf")
    calibration = BaseCalibration(robot, "config/calibration_config.yaml")
    calibration.load_data("data/calibration_data.csv")
    results = calibration.run_calibration()

Basic Identification  
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from figaroh.identification import BaseIdentification
    from figaroh.tools.robot import load_robot

    # Load robot and run identification
    robot = load_robot("path/to/robot.urdf")
    identification = BaseIdentification(robot, "config/identification_config.yaml")
    identification.load_data("data/identification_data.csv")
    params = identification.run_identification()

Advanced Regressor Building
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from figaroh.tools.regressor import RegressorBuilder, RegressorConfig
    
    # Configure regressor
    config = RegressorConfig(
        has_friction=True,
        has_actuator_inertia=True,
        is_joint_torques=True
    )
    
    # Build regressor matrix
    builder = RegressorBuilder(robot, config)
    W = builder.build_basic_regressor(q, dq, ddq)

Configuration Management
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from figaroh.utils.config_parser import UnifiedConfigParser
    
    # Parse any configuration format
    parser = UnifiedConfigParser("config/robot_config.yaml")
    config = parser.parse()
    
    # Create task-specific configuration
    calib_config = parser.create_task_config(robot, config, "calibration")
    identif_config = parser.create_task_config(robot, config, "identification")

Next Steps
---------

- Explore the `Examples Repository <https://github.com/thanhndv212/figaroh-examples>`_ for complete workflows
- Check out the API documentation for detailed module information
- Review the configuration templates for your specific robot type
