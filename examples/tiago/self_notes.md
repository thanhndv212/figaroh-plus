self_notes

To select the cutoff frequency of a Butterworth low-pass filter, you need to consider the characteristics of your signal and the desired filtering effect. In your case, you have a signal with a sampling rate of 100 Hz and a frequency of interest (input signal) at 0.2 Hz. Here are the steps to determine the cutoff frequency:

1. Normalize the cutoff frequency:
   In digital signal processing, it's common to work with normalized frequencies, which are expressed as a fraction of the sampling rate. To do this, divide the frequency of interest (0.2 Hz) by the sampling rate (100 Hz):

   Normalized Cutoff Frequency (ωc) = 0.2 Hz / 100 Hz = 0.002

2. Select a filter order (n):
   The filter order (n) determines how sharply the filter rolls off the frequency response curve. A higher filter order provides a steeper roll-off but may introduce more phase distortion. The choice of filter order depends on your specific application and trade-offs.

3. Calculate the cutoff frequency in radians per sample (ωc_radians):
   To calculate the cutoff frequency in radians per sample, multiply the normalized cutoff frequency (ωc) by π (pi):

   ωc_radians = 0.002 * π ≈ 0.00628 radians/sample

4. Calculate the actual cutoff frequency in Hertz (f_c):
   To get the actual cutoff frequency in Hertz, multiply the ωc_radians by the sampling rate (Fs):

   f_c = 0.00628 * 100 Hz ≈ 0.628 Hz

So, in your case, you would select a cutoff frequency of approximately 0.628 Hz for your Butterworth low-pass filter. Keep in mind that the filter order will affect the exact shape of the filter's frequency response curve. Higher filter orders will have steeper roll-offs but may introduce more phase distortion, so you should choose an appropriate filter order based on your application's requirements.

def update_forward_kinematics_2(model, data, var, q, param, verbose=0):
    """ Update jointplacements with offset parameters, recalculate forward kinematics
        to find end-effector's position and orientation.
    """
    # read param['param_name'] to allocate offset parameters to correct SE3
    # convert translation: add a vector of 3 to SE3.translation
    # convert orientation: convert SE3.rotation 3x3 matrix to vector rpy, add
    #  to vector rpy, convert back to to 3x3 matrix
    axis_tpl = ['d_px', 'd_py', 'd_pz', 'd_phix', 'd_phiy', 'd_phiz']
    elas_tpl = ['kx', 'ky', 'kz']
    pee_tpl = ['pEEx', 'pEEy', 'pEEz', 'phiEEx', 'phiEEy','phiEEz']
    # order of joint in variables are arranged as in param['actJoint_idx']
    assert len(var) == len(param['param_name']), "Length of variables != length of params"
    param_dict = dict(zip(param['param_name'], var))
    origin_model = model.copy()

    updated_params = []

    # base placement
    # TODO: add axis of base
    base_placement = cartesian_to_SE3(var[0: 6]).inverse()
    updated_params = param['param_name'][0:6]
    # update model.jointPlacements
    for j_id in param['actJoint_idx']:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy with kinematic errors
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose==1:
                            print("Updating [{}] joint placement at axis {} with [{}]".format(j_name, axis, key))
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)
        model = update_joint_placement(model, j_id, xyz_rpy)
    PEE = np.zeros((param['NbMarkers']*param['calibration_index'], param['NbSample']))

    # get transform
    q_ = np.copy(q)
    for i in range(param['NbSample']):

        # update model.jointPlacements with joint elastic error
        if param['non_geom']:
            tau = pin.computeGeneralizedGravity(model, data, q_[i,:]) # vector size of 32 = nq < njoints
            # update xyz_rpy with joint elastic error
            for j_id in param['actJoint_idx']:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1] # nq = njoints -1 
                if j_name in key:
                    for elas_id, elas in enumerate(elas_tpl):
                        if elas in key:
                            param_dict[key] = param_dict[key]*tau_j
                            xyz_rpy[elas_id + 3] += param_dict[key] # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, xyz_rpy)
            # get relative transform with updated model
            pin.framesForwardKinematics(model, data, q_[i, :])
            pin.updateFramePlacements(model, data)
            oMee = get_rel_transform(model, data, param['start_frame'], param['end_frame'])
            # revert model back to origin from added joint elastic error
            for j_id in param['actJoint_idx']:
                xyz_rpy = np.zeros(6)
                j_name = model.names[j_id]
                tau_j = tau[j_id - 1] # nq = njoints -1 
                if j_name in key:
                    for elas_id, elas in enumerate(elas_tpl):
                        if elas in key:
                            param_dict[key] = param_dict[key]*tau_j
                            xyz_rpy[elas_id + 3] += param_dict[key] # +3 to add only on orienation
                            updated_params.append(key)
                model = update_joint_placement(model, j_id, -xyz_rpy)

        else:
            pin.framesForwardKinematics(model, data, q_[i, :])
            pin.updateFramePlacements(model, data)
            # oMee = get_rel_transform(model, data, param['start_frame'], param['end_frame'])
            oMee = data.oMf[param['IDX_TOOL']]

        # update last frame if there is 
        if len(updated_params) < len(param_dict):
            # pee = np.zeros(6)
            # for n_id in range(len(updated_params), len(param_dict)):
            #     for axis_id, axis in enumerate(pee_tpl):
            #         if axis in param['param_name'][n_id]:
            #             if verbose==1:
            #                 print("Updating last frame with [{}]".format(param['param_name'][n_id]))
            #             pee[axis_id] = var[n_id]
            #             updated_params.append(param['param_name'][n_id])

            
            for marker_idx in range(1,param['NbMarkers']+1):
                pee = np.zeros(6)
                ee_name = 'EE'
                for key in param_dict.keys():
                    if ee_name in key and str(marker_idx) in key:
                        # update xyz_rpy with kinematic errors
                        for axis_pee_id, axis_pee in enumerate(pee_tpl):
                            if axis_pee in key:
                                if verbose==1:
                                    print("Updating [{}_{}] joint placement at axis {} with [{}]".format(ee_name, str(marker_idx), axis_pee, key))
                                pee[axis_pee_id] += param_dict[key]
                                updated_params.append(key)

                eeMf = cartesian_to_SE3(pee)
                oMf = base_placement*oMee*eeMf
                # final transform
                trans = oMf.translation.tolist()
                orient = pin.rpy.matrixToRpy(oMf.rotation).tolist()
                loc = trans + orient
                measure = []
                for mea_id, mea in enumerate(param['measurability']):
                    if mea:
                        measure.append(loc[mea_id])
                # PEE[(marker_idx-1)*param['calibration_index']:marker_idx*param['calibration_index'], i] = np.array(measure)
                PEE[:, i] = np.array(measure)
            
            assert len(updated_params) == len(param_dict), "Not all parameters are updated"
            
        # else:
        #     oMf = oMee
        
        
    PEE = PEE.flatten('C')
    # revert model back to original 
    for j_id in param['actJoint_idx']:
        xyz_rpy = np.zeros(6)
        j_name = model.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        xyz_rpy[axis_id] = param_dict[key]
        model = update_joint_placement(model, j_id, -xyz_rpy)

    # check if model.jointPlacements is reverted back to original state
    for i in range(model.njoints):
        if not origin_model.jointPlacements[i].isApprox(model.jointPlacements[i], 1e-4):
            print("original model", origin_model.names[i], origin_model.jointPlacements[i])
            print("updated model", model.names[i], model.jointPlacements[i])
            assert origin_model.jointPlacements[i].isApprox(model.jointPlacements[i], 1e-6), \
                'model.jointPlacements of joint [{}] was not reverted back to original state.'.format(model.names[i])
        
    return PEE