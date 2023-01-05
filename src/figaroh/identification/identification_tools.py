import pinocchio as pin
import numpy as np
from scipy import signal
from ..tools.regressor import build_regressor_reduced, get_index_eliminate
from ..tools.robot import Robot
import quadprog
import operator
import pprint

def get_params_settings():
    params_settings = {
       #### parameters related to the identification model 
        'is_external_wrench':True,
        'is_joint_torques':False,
        'force_torque':['All'], # forces and torques you desire under the format : Fx,Fy,Fz,Mx,My,Mz (NB : answer 'All' gives the regressor for Fx,Fy,Fz,Mx,My,Mz
        'external_wrench_offsets':False, # add to the pipeline the identification of the offset for Mx, My, Mz
        'has_friction':False,
        'has_joint_offset':False,
        'has_actuator_inertia':False,
        'has_coupled_wrist':False,
        'K_1':2, # default values of drive gains in case they are used
        'K_2':2, # default values of drive gains in case they are used
        'is_static_regressor':True, # Use this flag to work only with the static regressor (segment's masses and COM)
        'is_inertia_regressor':True, # Use this flag to work only with the inertial regressor (segment's inertia)
        'embedded_forces':True, #True = Forces acquired separetely from the mocap, False = Forces acquired with the mocap

        #### defaut values to be set in case the URDF is incomplete
        'q_lim_def': 1.57,# default value for joint limits in case the URDF does not have the info
        'dq_lim_def':5, # in rad.s-1
        'ddq_lim_def':20, # in rad.s-2
        'tau_lim_def':4, # in N.m
        #### parameters related to all exciting trajectories
        'ts':1/100,# Sampling frequency of the trajectories to be recorded
         
        #### parameters related to the data elaboration (filtering, etc...) 
        'cut_off_frequency_butterworth': 10,
        
        #### parameters related to the trapezoidal trajectory generation
        'nb_repet_trap': 2, # number of repetition of the trapezoidal motions
        'trapez_vel_steps':10,# Velocity step in % of the max velocity

        #### parameters related to the OEM generation
        'tf':1,# duration of one OEM
        'nb_iter_OEM':1, # number of OEM to be optimized
        'nb_harmonics': 4,# number of harmonics of the fourier serie
        'freq': 1, # frequency of the fourier serie coefficient
        'q_safety': 0.08, # value in radian (=5deg) to remove from the actual joint limits
        'eps_gradient':1e-5,# numerical gradient step
        'is_fourier_series':True,# use Fourier series for trajectory generation
        'time_static':2, # in seconds the time for each static pose
        
        #### parameters related to the Totl least square identification
        'mass_lsoad':3.0,
        'which_body_loaded':19, #Use this flag to specify the index of the body on which you have set the additional mass (for the loaded part), base link counts as 0
        'sync_joint_motion':False, # move all joint simultaneously when using quintic polynomial interpolation
        
        #### parameters related to  animation/saving data
        'ANIMATE':True, # plot flag for gepetto-viewer
        'SAVE_FILE':True
    }
    params_settings["nb_samples"] = int(params_settings["tf"] / params_settings["ts"])
    return params_settings

def set_missing_params_setting(robot, params_settings):

    diff_limit = np.setdiff1d(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    # upper and lower joint limits are the same so use defaut values for all joints
    if not diff_limit.any:
        print("No joint limits. Set default values")
        for ii in range(robot.model.nq):
            robot.model.lowerPositionLimit[ii] = -params_settings["q_lim_def"]
            robot.model.upperPositionLimit[ii] = params_settings["q_lim_def"]

    if np.sum(robot.model.velocityLimit) == 0:
        print("No velocity limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.velocityLimit[ii] = params_settings["dq_lim_def"]

    # maybe we need to check an other field somethnibg weird here
    if np.sum(robot.model.velocityLimit) == 0:
        print("No joint torque limit. Set default value")
        for ii in range(robot.model.nq):
            robot.model.effortLimit[ii] = -params_settings["tau_lim_def"]

    accelerationLimit = np.zeros(robot.model.nq)
    for ii in range(robot.model.nq):
        # accelerationLimit to be consistent with PIN naming
        accelerationLimit[ii] = params_settings["ddq_lim_def"]
    params_settings["accelerationLimit"] = accelerationLimit
    # print(model.accelerationLimit)

    if params_settings["has_friction"]:

        for ii in range(robot.model.nv):
            if ii == 0:
                # default values of the joint viscous friction in case they are used
                fv = [(ii + 1) / 10]
                # default value of the joint static friction in case they are used
                fs = [(ii + 1) / 10]
            else:
                fv.append((ii + 1) / 10)
                fs.append((ii + 1) / 10)

        params_settings["fv"] = fv
        params_settings["fs"] = fs
    
    if params_settings["external_wrench_offsets"]: # set for a fp of dim (1.8mx0.9m) at its center 
        params_settings["OFFX"]=900
        params_settings["OFFY"]=450
        params_settings["OFFZ"]=0

    return params_settings


def get_param_from_yaml(robot,identif_data):
    """This function allows to create a dictionnary of the settings set in a yaml file.
    Input:  robot: (Robot Tpl) a robot (extracted from an URDF for instance)
            identif_data: (dict) a dictionnary containing the parameters settings for identification (set in a config yaml file)  
    Output: param: (dict) a dictionnary of parameters settings
    """
    # robot_name: anchor as a reference point for executing
    robot_name = robot.model.name

    robots_params = identif_data['robot_params'][0]
    problem_params = identif_data['problem_params'][0]
    process_params = identif_data['processing_params'][0]
    tls_params = identif_data['tls_params'][0]

    param = {
        'robot_name': robot_name,
        'nb_samples': int(1/(process_params['ts'])),
        'q_lim_def': robots_params['q_lim_def'],
        'is_external_wrench': problem_params['is_external_wrench'],
        'is_joint_torques': problem_params['is_joint_torques'],
        'force_torque': problem_params['force_torque'],
        'external_wrench_offsets': problem_params['external_wrench_offsets'],
        'has_friction': problem_params['has_friction'],
        'cut_off_frequency_butterworth': process_params['cut_off_frequency_butterworth'],
        'ts': process_params['ts'],
        'mass_load': tls_params['mass_load'],
        'which_body_loaded': tls_params['which_body_loaded'],
    }
    pprint.pprint(param)
    return param


def base_param_from_standard(phi_standard,params_base):
    """This function allows to calculate numerically the base parameters with the values of the standard ones.
    Input:  phi_standard: (tuple) a dictionnary containing the values of the standard parameters of the model (usually from get_standard_parameters)
            params_base: (list) a list containing the analytical relations between standard parameters to give the base parameters  
    Output: phi_base: (list) a list containing the numeric values of the base parameters
    """
    phi_base=[]
    ops = { "+": operator.add, "-": operator.sub }
    for ii in range(len(params_base)):
        param_base_i=params_base[ii].split(' ')
        values=[]
        list_ops=[]
        for jj in range(len(param_base_i)):
            param_base_j=param_base_i[jj].split('*')
            if len(param_base_j)==2:
                value=float(param_base_j[0])*phi_standard[param_base_j[1]]
                values.append(value)
            elif param_base_j[0]!='+' and param_base_j[0]!='-':
                value=phi_standard[param_base_j[0]]
                values.append(value)
            else:
                list_ops.append(ops[param_base_j[0]])
        value_phi_base=values[0]
        for kk in range(len(list_ops)):
            value_phi_base=list_ops[kk](value_phi_base,values[kk+1])
        phi_base.append(value_phi_base)
    return phi_base

def remove_zero_crossing_velocity(robot,W,tau):
    # eliminate qd crossing zero
    for i in range(len(W)):
        idx_qd_cross_zero = []
        for j in range(W[i].shape[0]):
            if abs(W[i][j, i * 14 + 11]) < robot.qd_lim[i]:  # check columns of fv_i
                idx_qd_cross_zero.append(j)
        # if i == 4 or i == 5:  # joint 5 and 6
        #     for k in range(W_list[i].shape[0]):
        #         if abs(W_list[i][k, 4 * 14 + 11] + W_list[i][k, 5 * 14 + 11]) < qd_lim[4] + qd_lim[5]:  # check sum cols of fv_5 + fv_6
        #             idx_qd_cross_zero.append(k)
        # indices with vels around zero
        idx_eliminate = list(set(idx_qd_cross_zero))
        W[i] = np.delete(W[i], idx_eliminate, axis=0)
        W[i] = np.delete(tau[i], idx_eliminate, axis=0)
        print(W[i].shape, tau[i].shape)


def relative_stdev(W_b, phi_b, tau):
    """ Calculates relative standard deviation of estimated parameters using the residual errro[Pressé & Gautier 1991]"""
    # stdev of residual error ro
    sig_ro_sqr = np.linalg.norm((tau - np.dot(W_b, phi_b))) ** 2 / (
        W_b.shape[0] - phi_b.shape[0]
    )

    # covariance matrix of estimated parameters
    C_x = sig_ro_sqr * np.linalg.inv(np.dot(W_b.T, W_b))

    # relative stdev of estimated parameters
    std_x_sqr = np.diag(C_x)
    std_xr = np.zeros(std_x_sqr.shape[0])
    for i in range(std_x_sqr.shape[0]):
        std_xr[i] = np.round(100 * np.sqrt(std_x_sqr[i]) / np.abs(phi_b[i]), 2)

    return std_xr

def weigthed_least_squares(robot,phi_b,W_b,tau_meas,tau_est,param):
    '''This function computes the weigthed least square solution of the identification problem see [Gautier, 1997] for details
            inputs:
            - robot: pinocchio robot structure
            - W_b: base regressor matrix
            - tau: measured joint torques
            '''
    sigma=np.zeros(robot.model.nq) # Needs to be modified for taking into account the GRFM
    zero_identity_matrix=np.identity(len(tau_meas))
    P=np.zeros((len(tau_meas),len(tau_meas)))
    nb_samples=int (param['idx_tau_stop'][0])
    start_idx=int(0)
    for ii in range(robot.model.nq):
        
        sigma[ii]=np.linalg.norm( tau_meas[int(start_idx):int (param['idx_tau_stop'][ii]) ]-tau_est[int(start_idx):int (param['idx_tau_stop'][ii]) ] )/(len(tau_meas[int(start_idx):int (param['idx_tau_stop'][ii]) ])-len(phi_b))
       
        start_idx=param['idx_tau_stop'][ii]
        
    
        for jj in range(nb_samples):
           
            P[jj+ii*(nb_samples),jj+ii*(nb_samples)]=1/sigma[ii]
        
        phi_b=np.matmul( np.linalg.pinv(np.matmul(P,W_b)) ,np.matmul(P,tau_meas) )

         # sig_ro_joint[ii] = np.linalg.norm( tau[ii] - np.dot(W_b[a: (a + tau[ii]), :], phi_b) ) ** 2 / (tau[ii])
        #diag_SIGMA[a: (a + tau[ii])] = np.full(    tau[ii], sig_ro_joint[ii])
        #a += tau[ii]
    #SIGMA = np.diag(diag_SIGMA)
    
    
    # Covariance matrix
    # C_X = np.linalg.inv(np.matmul(np.matmul(W_b.T, np.linalg.inv(SIGMA)), W_b))  # (W^T*SIGMA^-1*W)^-1
    # WLS solution
   # phi_b = np.matmul(np.matmul(np.matmul(C_X, W_b.T), np.linalg.inv(SIGMA)), tau)  # (W^T*SIGMA^-1*W)^-1*W^T*SIGMA^-1*TAU
    phi_b = np.around(phi_b, 6)
    
    return phi_b


def calculate_first_second_order_differentiation(model,q,param,dt=None):
    """This function calculates the derivatives (velocities and accelerations here) by central difference for given angular configurations accounting that the robot has a freeflyer or not (which is indicated in the params_settings).
    Input:  model: (Model Tpl) the robot model
            q: (array) the angular configurations whose derivatives need to be calculated
            param: (dict) a dictionnary containing the settings 
            dt:  (list) a list containing the different timesteps between the samples (set to None by default, which means that the timestep is constant and to be found in param['ts'])  
    Output: q: (array) angular configurations (whose size match the samples removed by central differences)
            dq: (array) angular velocities
            ddq: (array) angular accelerations
    """

    if param['is_joint_torques']:
        dq = np.zeros([q.shape[0]-1, q.shape[1]])
        ddq = np.zeros([q.shape[0]-1, q.shape[1]])

    if param['is_external_wrench']:
        dq = np.zeros([q.shape[0]-1, q.shape[1]-1])
        ddq = np.zeros([q.shape[0]-1, q.shape[1]-1])
    
    if dt is None:
        dt = param['ts']
        for ii in range(q.shape[0]-1):
            dq[ii,:] = pin.difference(model,q[ii,:],q[ii+1,:])/dt

        for jj in range(model.nq-1):
            ddq[:,jj] = np.gradient(dq[:,jj], edge_order=1)/dt
    else :
        for ii in range(q.shape[0]-1):
            dq[ii,:] = pin.difference(model,q[ii,:],q[ii+1,:])/dt[ii]

        for jj in range(model.nq-1):
            ddq[:,jj] = np.gradient(dq[:,jj], edge_order=1) / dt


    q = np.delete(q, len(q)-1, 0)
    q = np.delete(q, len(q)-1, 0)

    dq = np.delete(dq, len(dq)-1, 0)
    ddq = np.delete(ddq, len(ddq)-1,0)

    return q, dq, ddq

def low_pass_filter_data(data,param):
    '''This function filters and elaborates data used in the identification process. 
    It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes, France)'''

    # median order 3 => butterworth zerophase filtering
    nbutter = 5
    
    b, a = signal.butter(nbutter, param['ts']*param['cut_off_frequency_butterworth'] / 2, "low")
   
    #data = signal.medfilt(data, 3)
    data= signal.filtfilt(
            b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1) )
    
    
    # suppress end segments of samples due to the border effect
    nbord = 5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    data = np.delete(data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0)
     
    return data

# inertial parameters of link2 from urdf model

def buildAugmentedRegressor(W_b_u, W_l, W_b_l, tau_u, tau_l, param):
    '''Inputs:  W_b_u  base regressor for unloaded case 
                W_b_l:  base Regressor for loaded case 
                W_l: Full  regressor for loaded case
                I_u: measured current in uloaded case in A
                I_l: measured current in loaded case in A
        Ouputs: W_tot: total regressor matrix
                V_norm= Normalised solution vector'''
                
    # augmented regressor matrix
    
    tau=np.concatenate((tau_u, tau_l), axis=0)


    W=np.concatenate((W_b_u, W_b_l), axis=0)
    
    W_upayload=np.concatenate((np.zeros((len(W_l),2)),W_l[:,[-9, -7] ]), axis=0)

    W=np.concatenate((W,W_upayload), axis=1) 
     
    W_kpayload=np.concatenate((np.zeros((len(W_l),1)),W_l[:,-10].reshape(len(W_l),1)), axis=0)
    W=np.concatenate((W,W_kpayload), axis=1) 
    
 
    Phi_b=np.matmul(np.linalg.pinv(W),tau)
    
    # Phi_b_ref=np.copy(Phi_b)
    

    return W, Phi_b

def build_total_regressor(W_b_u, W_b_l,W_l, I_u, I_l,param_standard_l, param):
    '''Inputs:  W_b_u  base regressor for unloaded case 
                W_b_l:  base Regressor for loaded case 
                W_l: Full  regressor for loaded case
                I_u: measured current in uloaded case in A
                I_l: measured current in loaded case in A
        Ouputs: W_tot: total regressor matrix
                V_norm= Normalised solution vector
                residue'''
             
    # build the total regressor matrix for TLS
    # we have to add a minus in front of the regressors for tTLS
    W_tot=np.concatenate((-W_b_u, -W_b_l), axis=0)
  
    nb_j=int(len(I_u)/param['nb_samples'])
   
    # nv (or 6) columns for the current
    V_a=np.concatenate( (I_u[0:param['nb_samples']].reshape(param['nb_samples'],1), np.zeros(((nb_j-1)*param['nb_samples'],1))), axis=0) 
    V_b=np.concatenate( (I_l[0:param['nb_samples']].reshape(param['nb_samples'],1), np.zeros(((nb_j-1)*param['nb_samples'],1))), axis=0) 

    for ii in range(1,nb_j):
        V_a_ii=np.concatenate((np.concatenate((np.zeros((param['nb_samples']*(ii),1)),I_u[param['nb_samples']*(ii):(ii+1)*param['nb_samples']].reshape(param['nb_samples'],1)), axis=0),np.zeros((param['nb_samples']*(5-(ii)),1))), axis=0)
        V_b_ii=np.concatenate((np.concatenate((np.zeros((param['nb_samples']*(ii),1)),I_l[param['nb_samples']*(ii):(ii+1)*param['nb_samples']].reshape(param['nb_samples'],1)), axis=0),np.zeros((param['nb_samples']*(5-(ii)),1))), axis=0)
        V_a=np.concatenate((V_a, V_a_ii), axis=1) 
        V_b=np.concatenate((V_b, V_b_ii), axis=1) 
    
    W_current=np.concatenate((V_a, V_b), axis=0)
     
    
    W_tot=np.concatenate((W_tot,W_current), axis=1)

    
    # selection and reduction of the regressor for the unknown parameters for the mass

    if param['has_friction']: #adds fv and fs
        W_l_temp=np.zeros((len(W_l),12))
        for k in [0,1,2,3,4,5,6,7,8,10,11]:
            W_l_temp[:, k]=W_l[:,(param['which_body_loaded'])*12 + k] # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz fs fv
        idx_e_temp,params_r_temp= get_index_eliminate(W_l_temp,param_standard_l, 1e-6)
        W_e_l=build_regressor_reduced(W_l_temp,idx_e_temp)
        W_upayload=np.concatenate((np.zeros((len(W_l),W_e_l.shape[1])),-W_e_l), axis=0)
        W_tot=np.concatenate((W_tot,W_upayload), axis=1) 
        W_kpayload=np.concatenate((np.zeros((len(W_l),1)),-W_l[:,(param['which_body_loaded'])*12+9].reshape(len(W_l),1)), axis=0)# the mass
        W_tot=np.concatenate((W_tot,W_kpayload), axis=1) 

    elif param['has_actuator_inertia']: #adds ia fv fs off 
        W_l_temp=np.zeros((len(W_l),14))
        for k in [0,1,2,3,4,5,6,7,8,10,11,12,13]:
            W_l_temp[:, k]=W_l[:,(param['which_body_loaded'])*14 + k] # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz ia fv fs off
        idx_e_temp,params_r_temp= get_index_eliminate(W_l_temp,param_standard_l, 1e-6)
        W_e_l=build_regressor_reduced(W_l_temp,idx_e_temp)
        W_upayload=np.concatenate((np.zeros((len(W_l),W_e_l.shape[1])),-W_e_l), axis=0)
        W_tot=np.concatenate((W_tot,W_upayload), axis=1) 
        W_kpayload=np.concatenate((np.zeros((len(W_l),1)),-W_l[:,(param['which_body_loaded'])*14+9].reshape(len(W_l),1)), axis=0)# the mass
        W_tot=np.concatenate((W_tot,W_kpayload), axis=1)

    else:
        W_l_temp=np.zeros((len(W_l),9))
        for k in range(9):
            W_l_temp[:, k]=W_l[:,(param['which_body_loaded'])*10 + k] # adds columns belonging to Ixx Ixy Iyy Iyz Izz mx my mz
        idx_e_temp,params_r_temp= get_index_eliminate(W_l_temp,param_standard_l, 1e-6)
        W_e_l=build_regressor_reduced(W_l_temp,idx_e_temp)
        W_upayload=np.concatenate((np.zeros((len(W_l),W_e_l.shape[1])),-W_e_l), axis=0)
        W_tot=np.concatenate((W_tot,W_upayload), axis=1) 
        W_kpayload=np.concatenate((np.zeros((len(W_l),1)),-W_l[:,(param['which_body_loaded'])*10+9].reshape(len(W_l),1)), axis=0)# the mass
        W_tot=np.concatenate((W_tot,W_kpayload), axis=1) 

    print(W_tot.shape)
    print(np.linalg.matrix_rank(W_tot))
    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    ind_min= np.argmin(S)
    
    V = np.transpose(Vh).conj()
    
    # for validation purpose
    # W_tot_est=W_tot#-S[-1]*np.matmul(U[:,-1].reshape(len(W_tot),1),np.transpose(V[:,-1].reshape(len(Vh),1)))
  
    V_norm=param['mass_load']*np.divide(V[:,-1],V[-1,-1])
    
    residue=np.matmul(W_tot,V_norm)
    
    return W_tot, V_norm, residue


# Building regressor


def iden_model(model, data, q, dq, ddq, param):
    """This function calculates joint torques and generates the joint torque regressor.
            Note: a parameter Friction as to be set to include in dynamic model
            Input: 	model, data: model and data structure of robot from Pinocchio
                    q, v, a: joint's position, velocity, acceleration
                    N : number of samples
                    nq: length of q
            Output: tau: vector of joint torque
                    W : joint torque regressor"""
    nb_samples=len(q)
    tau = np.empty(model.nq*nb_samples)
    W = np.empty([nb_samples*model.nq, 10*model.nq])

    for i in range(nb_samples):
        tau_temp = pin.rnea(model, data, q[i, :], dq[i, :], ddq[i, :])
        W_temp = pin.computeJointTorqueRegressor(
            model, data, q[i, :], dq[i, :], ddq[i, :])
        for j in range(model.nq):
            tau[j*nb_samples + i] = tau_temp[j]
            W[j*nb_samples + i, :] = W_temp[j, :]

    if param['Friction']:
        W = np.c_[W, np.zeros([nb_samples*model.nq, 2*model.nq])]
        for i in range(nb_samples):
            for j in range(model.nq):
                tau[j*nb_samples + i] = tau[j*nb_samples + i] + dq[i, j]*param['fv'] + np.sign(dq[i, j])*param['fc']
                W[j*nb_samples + i, 10*model.nq+2*j] = dq[i, j]
                W[j*nb_samples + i, 10*model.nq+2*j + 1] = np.sign(dq[i, j])

    return tau, W

# SIP QP OPTIMISATION

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T) + np.eye(P.shape[0])*(1e-5)   # make sure P is symmetric, pos,def
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]  

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def calculate_standard_parameters(robot,W,tau,COM_max,COM_min,params_settings):
    """This function retrieves the 10 standard parameters (m, 3D COM, 6D Inertias) for each body in the body tree thanks to a QP optimisation (cf Jovic 2016). 
    Input:  robot : (Robot Tpl) a robot extracted from an urdf for instance
            W : ((Nsamples*njoints,10*nbodies) array) the full dynamic regressor (calculated thanks to build regressor basic for instance)   
            tau : ((Nbsamples*njoints,) array) the joint torque array      
            COM_max : (list) sup boundaries for COM in the form (x,y,z) for each body 
            COM_min : (list) sup boundaries for COM in the form (x,y,z) for each body
            params_settings : (dict) a dictionnary indicating the settings (extracted from get_parameters_from_yaml)
    Output: phi_standard: (list) a list containing the numeric values of the standard parameters
            phi_ref : (list) a list containing the numeric values of the standard parameters as they are set in the urdf
    """

    alpha=0.8
    phi_ref=[]
    id_inertias=[]
    id_virtual=[]

    for jj in range(len(robot.model.inertias.tolist())):
        if robot.model.inertias.tolist()[jj].mass !=0:
            id_inertias.append(jj-1)
        else:
            id_virtual.append(jj-1)

    nreal=len(id_inertias)
    nvirtual=len(id_virtual)
    nbodies=nreal+nvirtual

    params_standard_u = robot.get_standard_parameters_v2(params_settings)

    params_name = (
                    "Ixx",
                    "Ixy",
                    "Ixz",
                    "Iyy",
                    "Iyz",
                    "Izz",
                    "mx",
                    "my",
                    "mz",
                    "m",
                )

    for k in range(nbodies):
        for j in params_name:
            phi_ref_temp=params_standard_u[j+str(k)]
            phi_ref.append(phi_ref_temp)
   

    phi_ref=np.array(phi_ref)

    P=np.matmul(W.T,W) + alpha*np.eye(10*(nbodies))
    r=-(np.matmul(tau.T,W)+alpha*phi_ref.T)

    # Setting constraints
    epsilon=0.001
    v=sample_spherical(2000) # vectors over the unit sphere

    G=np.zeros(((7+len(v[0]))*(nreal),10*(nbodies)))
    h=np.zeros((((7+len(v[0]))*(nreal),1)))
    # A=np.zeros((10*(nvirtual),10*(nbodies)))
    # b=np.zeros((10*nvirtual,1))

    for ii in range(len(id_inertias)):
        for k in range(len(v[0])): # inertia matrix def pos for enough (ie. 2000 here) vectors on unit sphere
            G[ii*(len(v[0])+7)+k][id_inertias[ii]*10+0]=-v[0][k]**2
            G[ii*(len(v[0])+7)+k][id_inertias[ii]*10+1]=-2*v[0][k]*v[1][k]
            G[ii*(len(v[0])+7)+k][id_inertias[ii]*10+2]=-2*v[0][k]*v[2][k]
            G[ii*(len(v[0])+7)+k][id_inertias[ii]*10+3]=-v[1][k]**2
            G[ii*(len(v[0])+7)+k][id_inertias[ii]*10+4]=-2*v[1][k]*v[2][k]
            G[ii*(len(v[0])+7)+k][id_inertias[ii]*10+5]=-v[2][k]**2
            h[ii*(len(v[0])+7)+k]=epsilon
        G[len(v[0])+ii*(len(v[0])+7)][id_inertias[ii]*10+6]=1 # mx<mx+
        G[len(v[0])+ii*(len(v[0])+7)][id_inertias[ii]*10+9]=-COM_max[3*ii] # mx<mx+
        G[len(v[0])+ii*(len(v[0])+7)+1][id_inertias[ii]*10+6]=-1 # mx>mx-
        G[len(v[0])+ii*(len(v[0])+7)+1][id_inertias[ii]*10+9]=COM_min[3*ii] # mx>mx-
        G[len(v[0])+ii*(len(v[0])+7)+2][id_inertias[ii]*10+7]=1 # my<my+
        G[len(v[0])+ii*(len(v[0])+7)+2][id_inertias[ii]*10+9]=-COM_max[3*ii+1] # my<my+
        G[len(v[0])+ii*(len(v[0])+7)+3][id_inertias[ii]*10+7]=-1 # my>my-
        G[len(v[0])+ii*(len(v[0])+7)+3][id_inertias[ii]*10+9]=COM_min[3*ii+1] # my>my-
        G[len(v[0])+ii*(len(v[0])+7)+4][id_inertias[ii]*10+8]=1 # mz<mz+
        G[len(v[0])+ii*(len(v[0])+7)+4][id_inertias[ii]*10+9]=-COM_max[3*ii+2] # mz<mz+
        G[len(v[0])+ii*(len(v[0])+7)+5][id_inertias[ii]*10+8]=-1 # mz>mz-
        G[len(v[0])+ii*(len(v[0])+7)+5][id_inertias[ii]*10+9]=COM_min[3*ii+2] # mz>mz-
        G[len(v[0])+ii*(len(v[0])+7)+6][id_inertias[ii]*10+9]=-1 # m>0


    # SOLVING
    phi_standard=quadprog_solve_qp(P,r,G,h.reshape(((7+len(v[0]))*(nreal),))) # ,A,b.reshape((10*(nvirtual),)))

    return phi_standard,phi_ref


 
