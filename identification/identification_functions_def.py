import pinocchio as pin
import numpy as np
from scipy import signal
from tools.regressor import build_regressor_reduced, get_index_eliminate
import operator

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


def calculate_first_second_order_differentiation(model,data,param,dt=None):
    # calculate vel and acc by central difference
    

    if param['is_joint_torques']:
        ddata = np.zeros([data.shape[0]-1, data.shape[1]])
        dddata = np.zeros([data.shape[0]-1, data.shape[1]])

    if param['is_external_wrench']:
        ddata = np.zeros([data.shape[0]-1, data.shape[1]-1])
        dddata = np.zeros([data.shape[0]-1, data.shape[1]-1])
    
    if dt is None:
        dt = param['ts']
        for ii in range(data.shape[0]-1):
            ddata[ii,:] = pin.difference(model,data[ii,:],data[ii+1,:])/dt

        for jj in range(model.nq-1):
            dddata[:,jj] = np.gradient(ddata[:,jj], edge_order=1) / dt
    else :
        for ii in range(data.shape[0]-1):
            ddata[ii,:] = pin.difference(model,data[ii,:],data[ii+1,:])/dt[ii]

        for jj in range(model.nq-1):
            dddata[:,jj] = np.gradient(ddata[:,jj], edge_order=1) / dt


    data = np.delete(data, len(data)-1, 0)
    data = np.delete(data, len(data)-1, 0)

    ddata = np.delete(ddata, len(ddata)-1, 0)
    dddata = np.delete(dddata, len(dddata)-1,0)

    return data, ddata, dddata

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



 
