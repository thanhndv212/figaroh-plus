import os

import matplotlib.pyplot as plt
import numpy as np

# import pinocchio as pin

from tools.robot import load_model
from parameters_settings import get_params_settings

# from parameters_settings import set_missing_params_setting

# from OEM_generation import generate_optimal_excitation_motion
from tools.regressor import (
    get_standard_parameters,
    build_regressor_basic,
    get_index_eliminate,
    build_regressor_reduced,
    set_missing_urdf_parameters,
)
from tools.qrdecomposition import get_baseParams
from tools.identification_functions_def import (
    low_pass_filter_data,
    # relative_stdev,
    # calculate_first_second_order_differentiation,
    # weigthed_least_squares,
)


def main():

    # "staubli_tx40_description/urdf",
    # "tx40_mdh_modified.urdf",

    params_settings = get_params_settings()

    robot = load_model(
        "staubli_tx40_description", "tx40_mdh_modified.urdf", params_settings
    )
    print(robot.model.name)
    model = robot.model
    # data = robot.data
    nq = model.nq

    if params_settings["SAVE_FILE"]:
        path_results = "results/identification/" + model.name
        if not os.path.exists(path_results):
            os.makedirs(path_results)

    if params_settings["ANIMATE"] == 1:
        robot.initViewer()
        robot.loadViewerModel()

    params_settings = set_missing_urdf_parameters(robot, params_settings)

    # manual input of addtiional parameters provided by Staubli for the TX40

    # params_settings['fv'] = (8.05e0, 5.53e0, 1.97e0, 1.11e0, 1.86e0, 6.5e-1)
    # params_settings['fs'] = (7.14e0, 8.26e0, 6.34e0, 2.48e0, 3.03e0, 2.82e-1)
    # params_settings['Ia'] = (3.62e-1, 3.62e-1, 9.88e-2, 3.13e-2, 4.68e-2, 1.05e-2)
    # params_settings['off'] = (3.92e-1, 1.37e0, 3.26e-1, -1.02e-1, -2.88e-2, 1.27e-1)
    # self.Iam6 = 9.64e-3
    # self.fvm6 = 6.16e-1
    # self.fsm6 = 1.95e0
    # self.N1 = 32
    # self.N2 = 32
    # self.N3 = 45
    # self.N4 = -48
    # self.N5 = 45
    # self.N6 = 32
    # self.qd_lim = 0.01 * \
    #     np.array([287, 287, 430, 410, 320, 700]) * np.pi / 180
    # self.ratio_essential = 30

    # generate a list containing the full set of standard parameters
    # params_standard = robot.get_standard_parameters()
    params_standard = get_standard_parameters(robot, params_settings)

    # 1. First we build the structural base identification model, i.e. the one that can
    # be observed, using random samples

    q = np.random.uniform(low=-6, high=6, size=(10 * params_settings["nb_samples"], nq))
    dq = np.random.uniform(
        low=-6, high=6, size=(10 * params_settings["nb_samples"], nq)
    )
    ddq = np.random.uniform(
        low=-30, high=30, size=(10 * params_settings["nb_samples"], nq)
    )

    W = build_regressor_basic(robot, q, dq, ddq, params_settings)

    # remove zero cols and build a zero columns free regressor matrix
    idx_e, params_r = get_index_eliminate(W, params_standard, 1e-6)
    W_e = build_regressor_reduced(W, idx_e)

    # Calulate the base regressor matrix, the base regroupings equations params_base and
    # get the idx_base, ie. the index of base parameters
    # in the initial regressor matrix W
    _, params_base, idx_base = get_baseParams(W_e, params_r, params_standard)

    print("The structural base parameters are: ")
    for ii in range(len(params_base)):
        print(params_base[ii])

    print(model.name)

    # 2. Load experimenatal data
    data_load = np.loadtxt(
        "experimental_data/" + model.name + "/pos_read_data.csv",
        delimiter=",",
        skiprows=1,
    )
    print(data_load.shape)

    print(nq)

    print(data_load.shape)
    q = np.empty((len(data_load), nq))
    q_filt = np.empty((len(data_load) - 40, nq))
    dq = np.empty((len(data_load), nq))
    ddq = np.empty((len(data_load), nq))
    # time_traj = data_load[:, 0]

    for ii in range(nq):

        q[:, ii] = np.rad2deg(data_load[:, ii])
        # dq[:, ii] = data_load[:, ii + nq + 1]
        # ddq[:, ii] = data_load[:, ii + 2 * nq + 1]
        q_filt[:, ii] = low_pass_filter_data(q[:, ii], params_settings)

    # plt.plot(q)
    # plt.show()

    plt.plot(q, "r")
    plt.plot(q_filt, "--k")
    plt.ylabel("Joint traj [rad]")
    plt.show()

    # calculate vel and acc using central difference
    dt = 1 / params_settings["ts"]
    dq = np.zeros([q.shape[0], q.shape[1]])
    ddq = np.zeros([q.shape[0], q.shape[1]])
    t = np.linspace(0, dq.shape[0], num=dq.shape[0]) / params_settings["ts"]
    for i in range(data_load.shape[1]):
        dq[:, i] = np.gradient(q[:, i], edge_order=1) / dt
        ddq[:, i] = np.gradient(dq[:, i], edge_order=1) / dt
        plt.plot(t, q[:, i])

    '''
    nb_samples = len(q)
    # update the regressor matrix using new joint trajectory
    W = build_regressor_basic(robot, q, dq, ddq, params_settings)
    # select only the columns of the regressor corresponding to the structural base
    # parameters
    W_base = W[:, idx_base]
    print("When using trap traj cond num is", int(np.linalg.cond(W_base)))

    for ii in range(nq):
        if ii == 0:
            idx_fs = [params_base.index("fs" + str(ii + 1))]
            idx_fv = [params_base.index("fv" + str(ii + 1))]
        else:
            idx_fs.append(params_base.index("fs" + str(ii + 1)))
            idx_fv.append(params_base.index("fv" + str(ii + 1)))
        # print('The index of', 'fs'+str(ii+1)  ,'is:', index)

    # simulation of the measured joint torques including friction

    tau_meas = np.empty(model.nq * nb_samples)

    for ii in range(nb_samples):
        tau_temp = pin.rnea(model, data, q[ii, :], dq[ii, :], ddq[ii, :])
        for j in range(model.nq):
            tau_meas[j * nb_samples + ii] = tau_temp[j]
            tau_meas[j * nb_samples + ii] = (
                tau_meas[j * nb_samples + ii]
                + dq[ii, j] * params_settings["fv"][j]
                + np.sign(dq[ii, j]) * params_settings["fs"][j]
            )
    # Least-square identification process
    phi_base = np.matmul(np.linalg.pinv(W_base), tau_meas)

    print("The values of joint viscous friction are:")
    fv = phi_base[idx_fv]
    print(fv, "N.m/rad")

    print("The values of joint viscous friction are:")
    fs = phi_base[idx_fs]
    print(fs, "N.m")

    std_xr = relative_stdev(W_base, phi_base, tau_meas)

    print(std_xr)

    # 2. Identify the rest of the inertial parameters using static postures and dynamic
    # motions

    data_load = np.loadtxt(
        "results/trajectory/"
        + model.name
        + "/optimal_exciting_postures_trajectory.txt",
        delimiter=",",
    )

    q = np.empty((len(data_load), nq))
    dq = np.empty((len(data_load), nq))
    ddq = np.empty((len(data_load), nq))

    for ii in range(nq):

        q[:, ii] = data_load[:, ii + 1]
        dq[:, ii] = data_load[:, ii + nq + 1]
        ddq[:, ii] = data_load[:, ii + 2 * nq + 1]
        if ii == 0:
            q_filt_postures = low_pass_filter_data(q[:, ii], params_settings)
        else:
            q_filt_postures = np.column_stack(
                (q_filt_postures, low_pass_filter_data(q[:, ii], params_settings))
            )

    dq_filt_postures, ddq_filt_postures = calculate_first_second_order_differentiation(
        q_filt_postures, params_settings
    )

    data_load = np.loadtxt(
        "results/trajectory/" + model.name + "/optimal_exciting_motions_trajectory.txt",
        delimiter=",",
    )

    q = np.empty((len(data_load), nq))
    dq = np.empty((len(data_load), nq))
    ddq = np.empty((len(data_load), nq))

    for ii in range(nq):

        q[:, ii] = data_load[:, ii + 1]
        dq[:, ii] = data_load[:, ii + nq + 1]
        ddq[:, ii] = data_load[:, ii + 2 * nq + 1]

        if ii == 0:
            q_filt_OEM = low_pass_filter_data(q[:, ii], params_settings)
        else:
            q_filt_OEM = np.column_stack(
                (q_filt_OEM, low_pass_filter_data(q[:, ii], params_settings))
            )

    dq_filt_OEM, ddq_filt_OEM = calculate_first_second_order_differentiation(
        q_filt_OEM, params_settings
    )

    q_filt = np.concatenate((q_filt_postures, q_filt_OEM))
    dq_filt = np.concatenate((dq_filt_postures, q_filt_OEM))
    ddq_filt = np.concatenate((ddq_filt_postures, q_filt_OEM))

    """q_filt=np.random.uniform(low=-6, high=6, size=(2,nq))
    dq_filt=np.random.uniform(low=-6, high=6, size=(2,nq))
    ddq_filt=np.random.uniform(low=-30, high=30, size=(2,nq))"""

    nb_samples = len(q_filt)
    # update the regressor matrix using new joint trajectory
    W = build_regressor_basic(robot, q_filt, dq_filt, ddq_filt, params_settings)
    # select only the columns of the regressor corresponding to the structural base
    # parameters
    W_base = W[:, idx_base]
    print("When using all trajectories the cond num is", int(np.linalg.cond(W_base)))

    # simulation of the measured joint torques
    tau_meas = np.empty(model.nq * nb_samples)

    for ii in range(nb_samples):
        tau_temp = pin.rnea(model, data, q_filt[ii, :], dq_filt[ii, :], ddq_filt[ii, :])
        for j in range(model.nq):
            tau_meas[j * nb_samples + ii] = tau_temp[j]
            # tau_meas[j * nb_samples + ii] = (
            # tau_meas[j * nb_samples + ii]
            # + dq[ii, j] * param["fv"][j]
            # + np.sign(dq[ii, j]) * param["fs"][j]
            # )

    # plt.plot(q[:,0],'r')
    # plt.plot(dq_filt ,'--k')
    # plt.ylabel("Joint traj [rad]")
    # plt.show()

    # store the index corresponding of the number of samples for each joint
    params_settings["idx_tau_stop"] = np.zeros(model.nq)
    params_settings["idx_tau_stop"][0] = int(len(tau_meas) / 2)
    params_settings["idx_tau_stop"][1] = int(len(tau_meas))

    print(params_settings["idx_tau_stop"])
    print(len(tau_meas))
    # Least-square identification process
    phi_base = np.matmul(np.linalg.pinv(W_base), tau_meas)
    # phi_base=phi_base+np.random.uniform(low=-1, high=1, size=(len(phi_base)))# Add a
    # bit of noise
    tau_est = np.matmul(W_base, phi_base)
    # phi_base_weigthed=weigthed_least_squares(robot,phi_base,
    # W_base,tau_meas,tau_est,param)

    # print(phi_base_weigthed)

    print(phi_base)

    plt.plot(tau_meas, "k")
    plt.plot(tau_est, "r")
    # plt.xlabel("Joint vel [rad.s-2]")
    # plt.ylabel("Joint torque [N.m]")
    plt.show()

    """fig, ax = plt.subplots(3)
    ax[0].plot(time_traj,q)
    #ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Joint pos [rad]")
    ax[1].plot(time_traj,dq)
    #ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Joint vel [rad.s-1]")
    ax[2].plot(time_traj,ddq)
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Joint acc [rad.s-2]")"""
    # plt.show()
    # x = data[:, 0]
    # y = data[:, 1]

    # Generate model and simualated data
    # Build a regressor matrix using trapezoidal data
    # tau_trap, W = iden_model(model, data, q, dq,ddq, param)

    # 2. identification of the friction and segment/motor inertia

    # W_b=W[:,idx_b]

    # W_b_u, numrank_W_u, params_rsorted_u,phi_b_u = double_QR(tau_u, W_e_u, params_r_u)

    #

    # print(param_r)
    # W_b_u, phi_b_u, numrank_W_u, params_rsorted_u = QR_pivoting(W, tau_trap,
    # params_r_u)

    #

    # print(  q.shape )

    # ig, ax = plt.subplots(5)
    # ax[0].plot(q)
    # ax[1].plot(dq)
    # ax[2].plot(ddq)

    # ax[3].plot(tau_trap[0:int ((tau_trap.shape[0])/2)])
    # ax[4].plot( tau_trap[int ((tau_trap.shape[0])/2):-1] )

    # plt.show()

    # fig = plt.subplots( )
    # plt.plot(q)
    # plt.plot(dq)
    # plt.plot(ddq)
    # plt.show()#(block=False)
    # plt.pause(0.001) # Pause for interval seconds.
    # input("hit[enter] to close figure .")
    # plt.close('all') # all open plots are correctly closed after each run

    # param['Nb_repet_trap']*1/param['Ts']
    # q=np.zeros((param['NbSample_interpolate'],model.nq))
    # dq=np.zeros((param['NbSample_interpolate'],model.nq))
    # ddq=np.zeros((param['NbSample_interpolate'],model.nq))

    # q, dq, ddq=generateQuinticPolyTraj(Jc0,Jcf,3,[], model, param)

    # Jc0=np.array([0,0,0])# initial joint configuration pos, vel, acc
    # Jcf=np.array([1,0,0])

    # fig = plt.subplots( )
    # plt.plot(q)
    # plt.plot(dq)
    # plt.plot(ddq)
    # plt.show()

    # generateQuinticPolyTraj(Jc0,Jcf, model, param)

    # coeff=0.01*np.ones((2*param['NbH']+1,model.nq))+np.random.uniform(low=-0.1,
    # high=1, size=((2*param['NbH']+1,model.nq)))
    # q_opt, dq_opt, ddq_opt= generateFourierTraj(coeff, model, param)

    # fig = plt.subplots( )

    # plt.plot(q_opt,'r')
    # plt.show()

    # Uncomment if you would like to generate non-optimal joint trajectories using
    # fourier series or random joint configurations
    """coeff=0.01*np.ones((2*param['NbH']+1,model.nq))
    #+ np.random.uniform(low=-0.1, high=1, size=((2*param['NbH']+1,model.nq)))
    coeff[0,0]=3.14
    coeff[0,1]=3.14
    q, dq, ddq=generateFourierTraj(coeff, model, param)"""

    # generate NbSample random joint configurations
    # q, dq, ddq = generateWaypoints(model, param)

    """fig = plt.subplots( )
    plt.plot(q)
    plt.plot(dq)
    plt.plot(ddq)
    plt.show()"""

    # generate_optimal_excitation_motion(data, model, param)

    if params_settings["ANIMATE"] == 1:
        robot.display(q)
        robot.viewer.gui.refresh()

    """#Display the placement of each joint of the kinematic tree
    for name, oMi in zip(model.names, data.oMi):
        print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))


    # create a list of standar inertial parameters and parse urdf to retrieve the
    # numerical values
    params_sip = standardParameters(model, param)
    print("Standard inertial parameters: ", params_sip)


    # Generate model and simualated data
    # Nominal unloaded _u case
    tau_u, W_u = iden_model(model, data, q, dq,ddq, param)

    # create current from torques
    I_u=tau_u[0:NbSample]*param['K_1']
    I_u=np.append(I_u,tau_u[NbSample:]*param['K_2'] )
    I_u=I_u+ np.random.normal(0,0.1, size=(model.nq*NbSample))

    W_e_u, params_r_u= eliminateNonAffecting(W_u, params_sip, 1e-6)

    #W_b_u, phi_b_u, numrank_W_u, params_rsorted_u = QR_pivoting(W_e_u, tau_u,
    params_r_u)

    W_b_u, numrank_W_u, params_rsorted_u,phi_b_u = double_QR(tau_u, W_e_u, params_r_u)




    #tau_est=np.matmul(W_b_u,phi_b_u)"""

    # Add additinnal load at the end-effector
    # the modifictain of com is mandatory as pin does not consider single mass at the
    # end-effector
    """Phi_l=model.inertias[2].toDynamicParameters()

    Phi_l[0]= model.inertias[2].mass+param['Mass_load']
    Phi_l[1:4]=model.inertias[2].lever*Phi_l[0]"""

    """
    Phi1=np.append(Phi1,model.inertias[1].lever*model.inertias[1].mass)
    Phi1=np.append(Phi1,model.inertias[1].inertia[0,0])# Ixx
    Phi1=np.append(Phi1,model.inertias[1].inertia[0,1])#, Ixy
    Phi1=np.append(Phi1,model.inertias[1].inertia[1,1])# Iyy
    Phi1=np.append(Phi1,model.inertias[1].inertia[0,2])# Ixz
    Phi1=np.append(Phi1,model.inertias[1].inertia[1,2])#Iyz
    Phi1=np.append(Phi1,model.inertias[1].inertia[2,2])#I zz"""

    """model.inertias[2]=model.inertias[2].FromDynamicParameters(Phi_l)


    #params_sip_l = standardParameters(model, param)
    #print("Standard inertial parameters: ", params_sip_l)

    # Generate model and simualated data
    #  additionnal load _l case
    tau_l, W_l = iden_model(model, data, q, dq,ddq, param)

    I_l=tau_l[0:NbSample]*param['K_1']
    I_l=np.append(I_l,tau_l[NbSample:]*param['K_2'] )
    I_l=I_l+ np.random.normal(0,0.1, size=(model.nq*NbSample))


    W_e_l, params_r_l = eliminateNonAffecting(W_l, params_sip, 1e-6)
    W_b_l, numrank_W_l, params_rsorted_l,phi_b_l = double_QR(tau_l, W_e_l, params_r_l)
    """

    """Phi_b_ref[6]=Phi_b[6]
    Phi_b_ref[7]=Phi_b[7]
    Phi_b_ref[8]=Phi_b[8]#Phi2[3]
    Phi_b_ref[9]=Phi2[0]"""

    # params_tot = [
    # "mx1",
    # "mz1R",
    # "Iyy1R",
    # "mx2",
    # "mz2",
    # "Iyy2",
    # "mxE",
    # "mzE",
    # "IyyE",
    # "ME",
    # ]
    # params_tot = ['mx1', 'mz1R',  'mx2', 'mz2', 'mxE', 'mzE', 'ME']

    # W_b_tot, numrank_W_tot, params_rsorted_tot,phi_b_l = double_QR(tau, W_tot,
    # params_tot)

    # plt.plot(tau,'k')
    # plt.plot(np.matmul(W_tot,Phi_b),'--r')
    # plt.plot(np.matmul(W_tot,Phi_b_ref),'--g')
    # plt.show()

    # print("Standard inertial parameters: ", params_sip)

    # W_tot, V_norm, residue=buildTotalRegressor(W_b_u, W_b_l, W_l, I_u, I_l ,param)

    # print(V_norm)
    # print(residue)

    """cons = ({'type': 'eq', 'fun': lambda x:  x[-1] - 3 })
    x0 = [0,-0.0,0,-0.0, 1, 1,0,0,1]#[0,-0.16,0,-0.33, 1, 1,0,0,3]

    fun = lambda x: np.sum( (np.matmul(W_tot_est,x))**2)


    #bnds = ((0, None), (0, None))
    res = minimize(fun, x0, method='SLSQP',bounds=None,constraints=cons)
    print(res.x)




    fig = plt.subplots( )

    plt.plot(res_ref,'r')
    plt.plot(res_opt,'--g')
    plt.plot(res_svd)
    #plt.show() """

    # res_opt=np.matmul(W_tot,res.x)
    # res_svd=np.matmul(W_tot,V_norm)
    # res_ref=np.matmul(W_tot,V_ref)

    # print("Residue value is")
    # print(residue)

    # print("Residue value is")
    # print(residue_ref)

    # print( np.sqrt( np.sum(np.square(res_svd) ) )  )
    # print( np.sqrt( np.sum(np.square(res_ref) ) )  )
    # print( np.sqrt( np.sum(np.square(res_opt) ) )  )

    """U, S, V = np.linalg.svd(W_tot, full_matrices=True)
    ind_min= np.argmin(S)

    print(U.shape)
    print(V.shape)
    print(ind_min)

    #*U*Vt
    toto=S[ind_min]*np.matmul(U[:,ind_min].reshape(len(W_tot),1),
    np.transpose(V[:,ind_min].reshape(len(V),1)))

    W_tot_est=W_tot-S[ind_min]*np.matmul(U[:,ind_min].reshape(len(W_tot),1),
    np.transpose(V[:,ind_min].reshape(len(V),1)))

    #print(Vt[-1,-1])
    #W_tot_est=W_tot-np.min(np.diag(S))*U*Vt
    Vt_norm=Phi2[0]*(V[:,-1]/V[-1])
    #print(Vt[:,-1])
    print(Vt_norm)


    #res=np.matmul(W_tot,V)
    res=np.matmul(W_tot_est,V[:,-1])

    fig = plt.subplots( )
    plt.plot(res)
    #plt.show()"""

    """
    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = np.dot(M.T, M)
    q_qp = -np.dot(M.T, np.array([3., 2., 3.]))
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = np.array([3., 2., -2.]).reshape((3,))
    A = np.array([1., 1., 1.])
    b = np.array([1.])

    start_time = perf_counter()
    x = solve_qp(P, q_qp, G, h, A, b)
    end_time = perf_counter()


    print("Use a QP contrained process to get SIP")
    print("    min. || P * x - q_qp ||^2")
    print("    s.t. G * x <= h")
    print("")
    #print_matrix_vector(P, "P", q_qp, "q_qp")
    #print("")
    #print_matrix_vector(G, "G", h, "h")
    print("")
    print("QP solution: x = {}".format(x))
    print(f"Solve time: {1e6 * (end_time - start_time):.0f} [us]")
    """

    """
    print('condition number of base regressor: ', np.linalg.cond(W_b))
    U, S, VT = np.linalg.svd(W_b)
    print('singular values of base regressor:', S)
    """

    # tf_ref = pin.utils.se3ToXYZQUAT(M_ct)
    # robot.viewer.gui.addLandmark('world/ref_wrench', .5)
    """
    # numbers of samples
    N = 1000
    params_standard = standardParameters(model, isFrictionincld, njoints)
    print("Standard inertial parameters: ", params_standard)
    q, qd, qdd = generateWaypoints(N, nq, nv, -10, 10)

    tau_, W_ = iden_model(model, data, N, nq, nv, q, qd, qdd, isFrictionincld)
    W_e, params_r = eliminateNonAffecting(W_, params_standard, 1e-6)
    W_b, phi_b, numrank_W, params_rsorted = QR_pivoting(W_e, tau_, params_r)

    print('condition number of base regressor: ', np.linalg.cond(W_b))
    U, S, VT = np.linalg.svd(W_b)
    print('singular values of base regressor:', S)


def generate_input():
    robot = loadModels("2DOF_description", "2DOF_description.urdf")
    model = robot.model
    data = robot.data
    nq, nv, njoints = model.nq, model.nv, model.njoints

    N = 1000

    q = np.empty((1, nq))
    v = np.empty((1, nv))
    a = np.empty((1, nv))

    for i in range(N):
        q = np.vstack((q, np.random.uniform(
            low=-np.pi/2, high=np.pi/2, size=(nq,))))
        v = np.vstack((v, np.random.uniform(low=-10, high=10, size=(nv,))))
        a = np.vstack((a, np.random.uniform(low=-10, high=10, size=(nv,))))
    path_save_bp = join(
        dirname(dirname(str(abspath(__file__)))),
        f"identification_toolbox/src/thanh/2dof_data.csv")
    header = ['q_1', 'q_2', 'dq_1', 'dq_2',
              'ddq_1', 'ddq_2', 'torque_1', 'torque_2']

    with open(path_save_bp, "w") as output_file:
        w = csv.writer(output_file)
        w.writerow(header)

        for i in range(N):
            torque = pin.rnea(model, data, q[i, :], v[i, :], a[i, :])
            data_row = np.append(
                np.append(np.append(q[i, :], v[i, :]), a[i, :]), torque)
            w.writerow(data_row.tolist())
    """
'''


if __name__ == "__main__":
    # generate_input()
    main()
