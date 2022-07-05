# FIGAROH

(Free dynamics Identification and Geometrical cAlibration of RObot and Human)

FIGAROH  is a python toolbox aiming at automatzing dynamics  identification of rigid multi-body systems. The considered systems can be serial (industrial manipulator) or tree-structures (human, humanoid robots). As described in the following figure it provides:
- Continuous optimal exciting postures and trajectories that can be played onto the robot
- Identification pipepline (data pre-processing) that will elaborate measured data
- Calculation of physically consistent inertial parameters (data processing) that can be used in a URDF file


## Tools

generation_EM_main.py: generate continuous optimized excitations motions file
identification_pipeline_main.py: identify inertial parameters (mass, COM and  inertia matrix) starting from an URDF description and from torque and/or external wrench measurements
visualization_reporting.py: show 3D robot model of URDF, trajectory motion, list of base and standard inertial parameters
parameters_setting.py: the main configuration file allowing to choose between different cost functions, trajectory formulation (Fourier series, B_splines, etc), model (static or dynamic), filtering parameters, etc

## Features

- find optimized excitation trajectories with non-linear  optimization
- data preprocessing
- derive velocity and acceleration values from position readings

- Optimal combination of data blocks to yield a better condition number [Venture, 2009] TO BE DONE

- implemented estimation methods:

    - Ordinary Least Squares, OLS

    - Moore-Penrose pseudo inverse

    - Weighted Least Squares [Zak, 1994]

    - Total least square [Gautier, 2003]

    - estimation of parameter error using previously known CAD values based on a constrained QP [Jovic, 2016]

    - essential standard parameters [Pham, Gautier, 2013], estimating only those that are most certain for the measurement data and leaving the others unchanged
