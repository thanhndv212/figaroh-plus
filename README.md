# FIGAROH

(Free dynamics Identification and Geometrical cAlibration of RObot and Human)

FIGAROH is a python toolbox aiming at automatzing dynamics identification and geometric calibration of rigid multi-body systems based on the popular modeling convention URDF. The considered systems can be serial (industrial manipulator) or tree-structures (human, humanoid robots). 

## Features

As described in the following figure it provides:
+ Dynamic Identification:
    - Generation of continuous optimal exciting trajectories that can be played onto the robot.
    - Guide line on data filtering/pre-processing.
    - Identification pipeline with a selection of dynamic parameter estimation algorithms.
    - Calculation of physically consistent standard inertial parameters that can be updated in a URDF file.
+ Geometric Calibration:
    - Generation of optimal calibration postures based on combinatorial optimization.
    - Calibration pipeline with customized kinematic chains and different selection of external sensoring methods (eye-hand camera, motion capture) or non-external methods (planar constraints).
    - Calculatation of kinematic parameters that can be updated in URDF model.

## How to use (TODO: input link and sample code)
+ Step 1: Define a config file with template.
+ Step 2: Generate sampled exciting postures and trajectories for experimentation.
+ Step 3: Collect and prepare data in the correct format.
+ Step 3: Create a script implementing identification/calibration algorithms with templates.
+ Step 4: Analysize results and update model with identified parameters. 

## Examples (TODO: input link)
### 1/ Human model

### 2/ Industrial manipulator Staubli TX40

### 3/ Industrial manipulator Universal UR10 
### 4/ Mobile base manipulator TIAGo

## Tools (TODO: explanation for main methods)
+ Calibration tools
+ Identification tools
+ Measurements
+ Meshcat viewer
+ Common tools
## Prequisites 
+ cyipopt
+ numdifftools
+ quadprog
+ numpy
+ scipy
+ pinocchio
+ ndcurves
+ hppfcl
## Reference
+ Pinocchio (source)
+ Ipopt (source)