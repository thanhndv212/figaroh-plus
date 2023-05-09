# Adaptation to PAL calibration pipeline
## 1/ Current state of PAL eye-hand calibration
a. Methodology
- Selected kinematic chain: camera-hand.
- Calibration parameters: joint angle offsets and camera location w.r.t the head frame.
- Measurability: Head camera-chessboard/Aruco+OpenCV
- Regression: minimize the errors between projected points by camera and predicted points by forward kinematics (using rodigues expression)

b. Problems
- Uncomplete model: the ```torso_lift_joint``` the camera-hand kinematic chain consists of 10 series joints:  ``` head_1_joint -> head_2_joint -> torso_lift_joint -> arm_1_joint -> arm_2_joint -> arm_3_joint -> arm_4_joint -> arm_5_joint -> arm_6_joint -> arm_7_joint ``` 
- Only joint angle offsets considered as the source of kinematic modeling errors.
- Too many iterations with bad convergence. (500 random configs/ 4-5 hrs of work on average per robot) 
- No existing cross-validation process of the identified parameters

### 2/ What Figaroh offers to improve the current framework
- Complete model for any desired kinematic chain.
- Offering selection of joint angle offset/full-set kinematic parameters (URDF compatible) calibration (further level of non-geometric parameters such as joint elasticity (TIAGo 2, CANOPIES), base suspension effect are in development)
- Optimizer to select a minimal set of experiment configurations (reduced by factor of 10, at least with current framework)
- Cross-validation process with more accurate measurement(laser tracker/mocap): discuss and TBD 

### 3/ Organization of Figaroh to adapt in PAL code framework

The toolbox Figaroh to be delivered to PAL will contain the following elements:
    
- The installable core package of Figaroh
- 3 ROS nodes and launch files:

    1/ Experiment design: generate minimal set of exciting calibration configurations

        -> Input: joint configurations (.yaml, .csv) 

        -> Output: joint configurations (.yaml, .csv)
    
    2/ Experiementation and data collection: excecuting motions visting designed configurations and collect synchronized data including joint encoder readings and camera frame at designed configuration
        
        -> Input: joint configurations (.yaml)

        -> Output: end-effector position and/or orientation; measured joint configurations by encoders (.csv)
    
    3/ Parameters identification node: input experiment data, run least-square algo to identify selected parameters (joint offsets/full-set kinematic parameters) + camera location

        -> Input: end-effector position and/or orientation; measured joint configurations by encoders (.csv)

        -> Output: parametes file (.xacro)

- Cross-validation experiments: required more accurate measurement system or perform a certain iterating application