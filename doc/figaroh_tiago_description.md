# FIGAROH

(Free dynamics Identification and Geometrical cAlibration of RObot and Human)

FIGAROH is a python toolbox aiming at providing efficient and highly flexible frameworks for dynamics identification and geometric calibration of rigid multi-body systems based on the popular modeling convention URDF. The considered systems can be serial (industrial manipulator) or tree-structures (human, humanoid robots). 

## Installing dependencies

First, set up the robotpkg repositories in your source list as explained here: http://robotpkg.openrobots.org/debian.html

Then, execute:
```bash
sudo apt-get install robotpkg-py36-pinocchio robotpkg-py36-hpp-fcl robotpkg-py36-ndcurves
pip3 install --user cyipopt numdifftools quadprog numpy scipy
```

## Features
The toolbox provides:
+ Dynamic Identification:
    - Dynamic model including effects of frictions, actuator inertia and joint torque offset. 
    - Generation of continuous optimal exciting trajectories that can be played onto the robot.
    - Guide line on data filtering/pre-processing.
    - Identification pipeline with a selection of dynamic parameter estimation algorithms.
    - Calculation of physically consistent standard inertial parameters that can be updated in a URDF file.
+ Geometric Calibration:
    - Calibration model with full-set kinematic parameters/joint offsets.
    - Generation of optimal calibration postures based on combinatorial optimization.
    - Calibration pipeline with customized kinematic chains and different selection of external sensoring methods (eye-hand camera, motion capture) or non-external methods (planar constraints).
    - Calculatation of kinematic parameters that can be updated in URDF model.
## Core package FIGAROH

+ calibration
    - calibration_tools.py

        - get_param_from_yaml(robot, calib_data): Read a .yaml config, then contain parsed parameters in a dictionary.

        - get_joint_offset(joint_names): This function give a dictionary of joint offset parameters.
            - Input:  joint_names: a list of joint names (from model.names)
            - Output: joint_off: a dictionary of joint offsets.

        - get_geo_offset(joint_names): This function give a dictionary of variations (offset) of kinematics parameters.
            - Input:  joint_names: a list of joint names (from model.names)
            - Output: geo_params: a dictionary of variations of kinematics parameters.


        - add_pee_name(param): Add additional kinematic parameters.

        - add_eemarker_frame(frame_name, p, rpy, model, data): Add a frame at the end_effector.
        - load_data(path_to_file, model, param, del_list=[]):
        Read a csv file into dataframe by pandas, then transform to the form
        of full joint configuration and markers' position/location.
        NOTE: indices matter! Pay attention.
            - Input:  path_to_file: str, path to csv
                param: Param, a class contain neccesary constant info.
            - Output: np.ndarray, joint configs
                1D np.ndarray, markers' position/location 
            - Csv headers: 
                i-th marker position: xi, yi, zi
                i-th marker orientation: phixi, phiyi, phizi (not used atm)
                active joint angles: 
                    example: tiago: torso, arm1, arm2, arm3, arm4, arm5, arm6, arm7
        - rank_in_configuration(model, joint_name): Get index of a given joint in joint configuration vector
        - cartesian_to_SE3(X): Convert (6,) cartesian coordinates to SE3
            Input: 1D (6,) numpy array
            Output: SE3 placement
        - get_rel_transform(model, data, start_frame, end_frame): Calculate relative transformation between any two frames
        in the same kinematic structure in pinocchio
            Output: SE3 placement
        Note: assume framesForwardKinematics and updateFramePlacements already
        updated
        - get_sup_joints(model, start_frame, end_frame):
        Find supporting joints between two frames
            Output: a list of supporting joints' Id 
        - get_rel_kinreg(model, data, start_frame, end_frame, q):
        Calculate kinematic regressor between start_frame and end_frame
            Output: 6x6n matrix 
        - get_rel_jac(model, data, start_frame, end_frame, q):
        Calculate frameJacobian between start_frame and end_frame
            Output: 6xn matrix 
        - get_LMvariables(param, mode=0, seed=0):
        Create a initial zero/range-bounded random search varibale for Leverberg-Marquardt algo.
        - update_forward_kinematics(model, data, var, q, param):
        Update jointplacements with offset parameters, recalculate forward kinematics
        to find end-effector's position and orientation.
        - update_joint_placement(model, joint_idx, xyz_rpy):
        Update joint placement given a vector of 6 offsets.
        - init_var(param, mode=0, base_model=True):
        Creates variable vector, mode = 0: initial guess, mode = 1: predefined values(randomized)
        - calculate_kinematics_model(q_i, model, data, param):
        Calculate jacobian matrix and kinematic regressor given ONE configuration.
        Details of calculation at regressor.hxx and se3-tpl.hpp
        - calculate_identifiable_kinematics_model(q, model, data, param):
        Calculate jacobian matrix and kinematic regressor and aggreating into one matrix, given a set of configurations or random configurations if not given.
        - calculate_base_kinematics_regressor(q, model, data, param):
        Calculate base regressor and base parameters for a calibration model from given configuration data.
+ identification
    - identification_tools.py
        - get_param_from_yaml(robot,identif_data): This function allows to create a dictionnary of the settings set in a yaml file.
            - Input:  robot: (Robot Tpl) a robot (extracted from an URDF for instance)
            identif_data: (dict) a dictionnary containing the parameters settings for identification (set in a config yaml file)  
            - Output: param: (dict) a dictionnary of parameters settings
        - set_missing_params_setting(robot, params_settings): Helper function to set customized parameters.
        - base_param_from_standard(phi_standard,params_base): This function allows to calculate numerically the base parameters with the values of the standard ones.
            - Input:  phi_standard: (tuple) a dictionnary containing the values of the standard parameters of the model (usually from get_standard_parameters)
            params_base: (list) a list containing the analytical relations between standard parameters to give the base parameters  
            - Output: phi_base: (list) a list containing the numeric values of the base parameters
        - relative_stdev(W_b, phi_b, tau):
        Calculates relative standard deviation of estimated parameters using the residual errro[Pressé & Gautier 1991]
        - weigthed_least_squares(robot,phi_b,W_b,tau_meas,tau_est,param):
        This function computes the weigthed least square solution of the identification problem see [Gautier, 1997] for details.
        - calculate_first_second_order_differentiation(model,q,param,dt=None):This function calculates the derivatives (velocities and accelerations here) by central difference for given angular configurations accounting that the robot has a freeflyer or not (which is indicated in the params_settings).
            - Input:  model: (Model Tpl) the robot model
            q: (array) the angular configurations whose derivatives need to be calculated
            param: (dict) a dictionnary containing the settings 
            dt:  (list) a list containing the different timesteps between the samples (set to None by default, which means that the timestep is constant and to be found in param['ts'])  
            - Output: q: (array) angular configurations (whose size match the samples removed by central differences)
            dq: (array) angular velocities
            ddq: (array) angular accelerations
        - low_pass_filter_data(data,param,nbutter):
        This function filters and elaborates data used in the identification process. It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes, France)

+ measurements
    - measurement.py
        - class Measurement:
            - __init__(self,name,joint,frame,type,value):
                name (str) : the name of the measurement
                joint (str) : the joint in which the measurement is expressed
                frame (str) : the closest frame from the measurement
                type (str) : type of measurement, choice between SE3, wrench or current
                value (6D array) : value of the measurement. Suppose that value is given wrt the joint placement
                      (what if current ?)
            - add_SE3_measurement(self,model): Adds the SE3 measurement to a given model
+ meshcat_viewer_wrapper
    - colors.py
        - rgb2int(r,g,b):
            Convert 3 integers (chars) 0<= r,g,b < 256 into one single integer = 256**2*r+256*g+b, as expected by Meshcat.
        - material( color, transparent=False):
        - colormap = {
                'red':red,
                'blue': blue,
                'green': green,
                'yellow': yellow,
                'magenta': magenta,
                'cyan': cyan,
                'black': black,
                'white': white,
                'grey': grey
                }
    - visualizer.py
        - class   MeshcatVisualizer(PMV):
            - __init__(self, robot=None, model=None, collision_model=None, visual_model=None, url=None):
            - addSphere(self, name, radius, color):
            - addCylinder(self, name, length, radius, color=None):
            - addCylinder(self, name, length, radius, color=None):
            - addBox(self, name, dims, color):
            - applyConfiguration(self, name, placement):
            - delete(self, name):
+ tools 
    - qrdecomoposition.py
        - QR_pivoting(tau, W_e, params_r): This function calculates QR decompostion with pivoting, finds rank of regressor,
    and calculates base parameters
            - Input:  W_e: regressor matrix (normally after eliminating zero columns)
                    params_r: a list of parameters corresponding to W_e
            - Output: W_b: base regressor
                    base_parametes: a dictionary of base parameters
        - double_QR(tau, W_e, params_r, params_std=None): This function calculates QR decompostion 2 times, first to find symbolic 
    expressions of base parameters, second to find their values after re-organizing 
    regressor matrix.
            - Input:  W_e: regressor matrix (normally after eliminating zero columns)
                    params_r: a list of parameters corresponding to W_e
            - Output: W_b: base regressor
                    base_parametes: a dictionary of base parameters
        - get_baseParams(W_e, params_r, params_std=None):
        Returns symbolic expressions of base parameters and base regressor matrix and idenx of the base regressor matrix. """
        - get_baseIndex(W_e, params_r): This function finds the linearly independent parameters.
            - Input:  W_e: regressor matrix
                    params_r: a dictionary of parameters
            - Output: idx_base: a tuple of indices of only independent parameters.
        - build_baseRegressor(W_e, idx_base): Create base regressor matrix corresponding to base parameters.
        - cond_num(W_b, norm_type=None):
        Calculates different types of condition number of a matrix.

    - randomdata.py
        - generate_waypoints(N, robot, mlow, mhigh): This function generates N random values for joints' position,velocity, acceleration.
            - Input:  N: number of samples
                    nq: length of q, nv : length of v
                    mlow and mhigh: the bound for random function
            - Output: q, v, a: joint's position, velocity, acceleration
        - generate_waypoints_fext(N, robot, mlow, mhigh): This function generates N random values for joints' position,velocity, acceleration.
            - Input:  N: number of samples
                    nq: length of q, nv : length of v
                    mlow and mhigh: the bound for random function
            - Output: q, v, a: joint's position, velocity, acceleration
        - get_torque_rand(N, robot, q, v, a, param): Get joint torques values from random configurations.
    - regressor.py
        - build_regressor_basic(robot, q, v, a, param, tau=None): This function builds the basic regressor of the 10(+4) parameters 'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m'+ ('ia','fs','fv','off') using pinocchio library depending on param.
            - Input:  robot: (robot) a robot extracted from an urdf (for instance)
                q: (ndarray) a configuration position vector (size robot.model.nq)
                v: (ndarray) a configuration velocity vector (size robot.model.nv)
                a: (ndarray) a configutation acceleration vectore (size robot.model.na)
                param: (dict) a dictionnary setting the options, i.e., here add two
                parameters, 'ia' if the flag 'has_actuator_inertia' is true,'fs' and 'fv' if the flag 'has friction' is true, 'off' is the flag "has_joint_offset' is true
                tau : (ndarray) of stacked torque measurements (Fx,Fy,Fz), None if the torque offsets are not identified 
            - Output: W_mod: (ndarray) basic regressor for 10(+4) parameters
        - add_actuator_inertia(W, robot, q, v, a, param):
        - add_friction(W,robot, q, v, a, param):
        - add_joint_offset(W, robot, q, v, a, param):
        - eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
        - get_index_eliminate(W, params_std, tol_e=1e-6):
        - build_regressor_reduced(W, idx_e):
    - robot.py
        - Robot(RobotWrapper):
            - __init__(
            self,
            robot_urdf,
            package_dirs,
            isFext=False,
            ):
            - get_standard_parameters(self, param): This function prints out the standard inertial parameters defined in urdf model.
            Output: params_std: a dictionary of parameter names and their values
            - display_q0(self):If you want to visualize the robot in this example, you can choose which visualizer to employ by specifying an option from the command line: GepettoVisualizer: -g MeshcatVisualizer: -m
    - robotcollisions.py
        - CollisionWrapper:
            - __init__(self, robot, geom_model=None, geom_data=None, viz=None)
            - add_collisions(self):
            - remove_collisions(self, srdf_model_path):
            - computeCollisions(self, q, geom_data=None):
            - getCollisionList(self): Return a list of triplets [ index,collision,result ] where index is the index of the collision pair, colision is gmodel.collisionPairs[index] and result is gdata.collisionResults[index].
            - getCollisionDistances(self, collisions=None):
            - getDistances(self):
            - getAllpairs(self):
            - check_collision(self, q):
            - displayContact(self, ipatch, contact):
                Display a small red disk at the position of the contact, perpendicular to the
                contact normal.

                @param ipatchf: use patch named "world/contact_%d" % contactRef.
                @param contact: the contact object, taken from Pinocchio (HPP-FCL) e.g.
                geomModel.collisionResults[0].getContact(0).
            - displayCollisions(self, collisions=None):
                Display in the viewer the collision list get from getCollisionList().
    - robotipopt.py
    - robotvisualization.py
+ utils
    - square_fitting.py
    - transformation.py
## Calibration and identification procedure on TIAGo
Overall, a calibration/identification project folder for TIAGo is presented as follow: 
```
\tiago
    \config
        tiago_config.yaml
    \data
        tiago_effort.csv    #identification data
        tiago_position.csv  #identification data
        tiago_velocity.csv  #identification data
        tiago_nov_30_64.csv #calibration data
    optimal_config.py
    optimal_trajectory.py
    calibration.py
    identification.py
    update_model.py
    cubic_spline.py
    simplified_colission_model.py
```

A step-by-step procedure is presented as follow.
+ Step 1: Define a config file with sample template.\
    A .yaml file containing information of the considered system and characteristics of the calibration/identification problem has a structure as follow. Also, depending on the measuring system selected, the ```markers``` should be defined accordingly.
    ```
    calibration:
        calib_level: full_params
        non_geom: False
        base_frame: universe
        tool_frame: wrist_ft_tool_link
        markers:
            - ref_joint: arm_7_joint
            measure: [True, True, True, False, False, False]
            - ref_joint: arm_7_joint
            measure: [True, True, True, False, False, False]
            - ref_joint: arm_7_joint
            measure: [True, True, True, False, False, False]
            - ref_joint: arm_7_joint
            measure: [True, True, True, False, False, False]  
        free_flyer: False
        nb_sample: 35
    ```

    ```
    identification:
        robot_params:
            - q_lim_def: 1.57
            fv : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            fs : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Ia : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            offset : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Iam6 : 0
            fvm6 : 0
            fsm6 : 0
            N : [32.0, 32.0, 45.0, -48.0, 45.0, 32.0]
            ratio_essential : 30.0
            dq_lim_: [0.05009095, 0.05009095, 0.07504916, 0.0715585,  0.05585054, 0.12217305]
            ddq_lim_: 20.0
            tau_lim_: 4.0
        problem_params:
            - is_external_wrench : False
            is_joint_torques : True
            force_torque : None
            external_wrench_offsets : False
            has_friction : True
            has_joint_offset : True
            has_actuator_inertia : True
            is_static_regressor : True
            is_inertia_regressor : True
            has_coupled_wrist: True
            embedded_forces : False
        processing_params:
            - cut_off_frequency_butterworth: 100.0
            ts : 0.0002
        tls_params:
            - mass_load : 0.0
            which_body_loaded : 0.0  
            sync_joint_motion : False
    ```
+ Step 2: Generate sampled exciting postures and trajectories for experimentation.
    - For geomeotric calibration: Firstly, considering the infinite possibilities of combination of postures can be generated, a finite pool of feasible sampled postures in working space for the considered system needs to be provided thanks to simulator. Then, the pool can be input for a script ```optimal_config.py``` with a combinatorial optimization algorithm which will calculate and propose an optimal set of calibration postures chosen from the pool with much less number of postures while maximizing the excitation.
    - For dynamic identification: A nonlinear optimization problem needs to formulated and solved thanks to Ipopt solver in a script namde ```optimal_trajectory.py```. Cost function can be chosen amongst different criteria such as condition number. Joint constraints, self-collision constraints should be obligatory, and other dedicated constraints can be included in constraint functions. Then, the Ipopt solver will iterate and find the best cubic spline that sastifies all constraints and optimize the defined cost function which aims to maximize the excitation for dynamics of the considered system.
+ Step 3: Collect and prepare data in the correct format.\
    To standardize the handling of data, we propose a sample format for collected data in csv format. These datasets should be stored in a ```data``` folder for such considered system.
+ Step 3: Create a script implementing identification/calibration algorithms with templates.
    Dedicated template scripts ```calibration.py``` and ```identification.py``` are provided. Users needs to fill in essential parts to adapt to their systems. At the end, calibration/identification results will be displayed with visualization and statistical analysis. Then, it is up to users to justify the quality of calibration/identification based on their needs.
+ Step 4: Update model with identified parameters.\
    Once the results are accepted, users can update calibrated/identified parameters to their urdf model by scripts ```update_model.py``` or simply save to a ```xacro``` file for later usage.


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
