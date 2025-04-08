# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pinocchio as pin
import cyipopt
from scipy.optimize import approx_fprime
from os.path import dirname, join, abspath
from pinocchio.robot_wrapper import RobotWrapper

def Rquat(x, y, z, w):
    """_Returns the rotation matrix for a quaternion given as input_

    Args:
        x (_float_): _Imaginary part of the quaternion_
        y (_float_): _Imaginary part of the quaternion_
        z (_float_): _Imaginary part of the quaternion_
        w (_float_): _Real part of the quaternion_

    Returns:
        _array_: _Rotation matrix associated to the quaternion_
    """

    q = pin.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix()  

def place(viz,name, M):
    """_Sets an object "name" in the viewer viz using the transformation matrix M_

    Args:
        viz (_Visualiser_): _Pinocchio visualiser_
        name (_str_): _The name of the object_
        M (_SE3_): _The transformation matrix to which we want the object to be_
    """
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

def closest_point_on_line(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result

def chord_joint_centre(HalfJoint, JointMarker, TopJoint, StickMarker, beta = 0 ):
    """Calculate the joint centre according the Vicon Clinical Manager method
    ( ie chord function )
    Args:
        HalfJoint (double): radius of the joint
        JointMarker (array(1,3)): joint marker trajectory at a specific frame
        TopJoint (type): proximal joint centre trajectory at a specific frame
        StickMarker (type): lateral marker trajectory at a specific frame
        beta (double,Optional[0]): rotation angle offset.
    **Reference**
    Kabada, M., Ramakrishan, H., & Wooten, M. (1990). Measurement of lower extremity kinematics during level walking. Journal of Orthopaedic Research, 8, 383–392.
    """

    OffsetAngle = np.deg2rad(beta)

    if np.all(JointMarker==0) or np.all(TopJoint==0) or np.all(StickMarker==0):
        return np.zeros((3))

    X = TopJoint - JointMarker
    T = StickMarker - JointMarker
    P = np.cross( X, T )
    E = np.cross( X, P )

    E =np.divide(E,np.linalg.norm(E))

    x2 = np.dot( X, X )
    l = HalfJoint / x2
    m = 1 - HalfJoint * l


    if m > 0 :
        if np.abs( OffsetAngle ) > np.spacing(np.single(1)) :
            cosTheta = np.cos( OffsetAngle )
            r2 = HalfJoint * HalfJoint
            r2cos2th = cosTheta * cosTheta * r2
            r2cos2th_h2 = r2cos2th / (x2-r2)
            TdotX = np.dot( T, X )
            EdotT = np.dot( E, T )

            P =np.divide(P,np.linalg.norm(P))

            # solve quadratic
            a = 1 + r2cos2th_h2
            b= 2*(r2cos2th-TdotX*r2cos2th_h2-r2)
            c= r2*r2+r2cos2th_h2*TdotX*TdotX-r2cos2th*( np.dot( T, T )+r2 )

            disc=b*b-4*a*c
            if disc < 0 :
                if disc < -np.abs(b)/1e6:
                    Solutions = []
                  # return;
                else:
                    disc = 0

            else:
              disc = np.sqrt( disc )

            Solutions = [ -b+disc, -b-disc ]
            if np.abs( a ) * 1e8 <= np.abs(c) and  np.min( np.abs(Solutions)) < 1e-8*np.max( np.abs( Solutions ) ) :
                Solutions = -c/b
            else:
                a = a*2
                Solutions = np.divide(Solutions,a)

            JointCentre = X * l * HalfJoint
            lxt = l * HalfJoint * TdotX
            r1l = r2 * m

            n = len( Solutions )
            while( n > 0 ):
                if ( Solutions[n-1] < r2 ) == ( cosTheta > 0 ) :
                    mu = ( Solutions[n-1] - lxt ) / EdotT
                    nu = r1l - mu*mu
                    if nu > np.spacing(np.single(1)) :
                      nu = np.sqrt( nu )
                      if np.sin( OffsetAngle ) > 0 :
                        nu = -nu

                      R = JointCentre + E*mu + P*nu
                      return  R + JointMarker
                n = n - 1

            # if no solutions to the quadratic equation...
            E = X*l + E*(np.sqrt(m)*cosTheta) - P*np.sin(OffsetAngle)
            return JointMarker + E*HalfJoint
        else:
            return JointMarker + X*(l*HalfJoint) + E*(np.sqrt(m)*HalfJoint)
    else:
        return JointMarker + E * HalfJoint

def make_markers_dict_notime(PMarkers,markers_name,joints_width=None):
    """_Creates a dictionnary linking markers names and their positions (it also includes joints_width for elbow,knee,wrist and ankle if required)_

    Args:
        PMarkers (_array_): _array containing the 3D positions (as sub array) of all markers in global frame_
        markers_name (_list(str)_): _a list giving the names of all markers (here the 35 markers of plug in gait have been chosen)_
        joints_width (_list(float)_, optional): _a list giving the width of knee, ankle, elbow, wrist (for each joint the value is specified 2 times for left and right)_. Defaults to None.

    Returns:
        _dict_: _a dictionary containing markers positions and names_
    """

    values=[]
    markers_name_copy = markers_name.copy()
    for ii in range(len(PMarkers)):
        values.append(PMarkers[ii])
    if (joints_width is not None):
        markers_name_copy.append('LKNEW')
        values.append(joints_width[0])
        markers_name_copy.append('RKNEW')
        values.append(joints_width[1])
        markers_name_copy.append('LANKW')
        values.append(joints_width[2])
        markers_name_copy.append('RANKW')
        values.append(joints_width[3])
        markers_name_copy.append('LELBW')
        values.append(joints_width[4])
        markers_name_copy.append('RELBW')
        values.append(joints_width[5])
        markers_name_copy.append('LWRIW')
        values.append(joints_width[6])
        markers_name_copy.append('RWRIW')
        values.append(joints_width[7])
    return dict(zip(markers_name_copy,values))

def compute_joint_centers(DMarkers):
    """_Calculates the position of joint centers from a given dictionnary of markers positions_

    Args:
        DMarkers (_dict_): _a dictionnary giving markers positions in global frame (from make_markers_dict_notime for instance)_

    Returns:
        _dict_: _a dictionnary containing joint centers names and positions in global frame_
    """

    names=['PELC','THOC','LHIPC','LKNEC','LANKC','RHIPC','RKNEC','RANKC','LSHOC','LELBC','LWRC','RSHOC','RELBC','RWRC']
    positions=[]

    #Def of pelvis frame 
    PELC = ((DMarkers['RASI']+DMarkers['LASI'])/2+(DMarkers['RPSI']+DMarkers['LPSI'])/2)/2

    pel_xaxis = (DMarkers['RASI']+DMarkers['LASI'])/2-(DMarkers['RPSI']+DMarkers['LPSI'])/2
    pel_zaxis = DMarkers['RASI']-DMarkers['LASI']

    pel_xaxis = pel_xaxis/np.linalg.norm(pel_xaxis)
    pel_zaxis = pel_zaxis/np.linalg.norm(pel_zaxis)

    pel_yaxis = np.cross(pel_zaxis,pel_xaxis)

    pel_xaxis = pel_xaxis/np.linalg.norm(pel_xaxis)
    pel_yaxis = pel_yaxis/np.linalg.norm(pel_yaxis)
    pel_zaxis = pel_zaxis/np.linalg.norm(pel_zaxis)

    R_pelvis = np.array([[pel_xaxis[0],pel_yaxis[0],pel_zaxis[0]],[pel_xaxis[1],pel_yaxis[1],pel_zaxis[1]],[pel_xaxis[2],pel_yaxis[2],pel_zaxis[2]]])

    ASIS_ASIS= np.linalg.norm(DMarkers['RASI']-DMarkers['LASI'])

    offset_lasis = R_pelvis@np.array([-0.22*ASIS_ASIS,-0.3*ASIS_ASIS,0.14*ASIS_ASIS])
    offset_rasis = R_pelvis@np.array([-0.22*ASIS_ASIS,-0.3*ASIS_ASIS,-0.14*ASIS_ASIS])

    positions.append(PELC)

    THOC = (DMarkers['C7']+DMarkers['CLAV'])/2
    positions.append(THOC) 

    LHIPC = DMarkers['LASI']+offset_lasis
    positions.append(LHIPC)

    LKNEC = chord_joint_centre(DMarkers['LKNEW'],DMarkers['LKNE'],LHIPC,DMarkers['LTHI'])
    positions.append(LKNEC)

    LANKC = chord_joint_centre(DMarkers['LANKW'],DMarkers['LANK'],LKNEC,DMarkers['LTIB'])
    positions.append(LANKC)

    RHIPC = DMarkers['RASI']+offset_rasis
    positions.append(RHIPC)

    RKNEC = chord_joint_centre(DMarkers['RKNEW'],DMarkers['RKNE'],LHIPC,DMarkers['RTHI'])
    positions.append(RKNEC)

    RANKC = chord_joint_centre(DMarkers['RANKW'],DMarkers['RANK'],LKNEC,DMarkers['RTIB'])
    positions.append(RANKC)

    LSHO_RSHO=np.linalg.norm(DMarkers['RSHO']-DMarkers['LSHO'])
    offset_rshoulder=np.array([0,0,-0.17*LSHO_RSHO])
    offset_lshoulder=np.array([0,0,-0.17*LSHO_RSHO])

    LSHOC = DMarkers['LSHO']+offset_lshoulder
    positions.append(LSHOC)

    lmid_wrist = (DMarkers['LWRA']+DMarkers['LWRB'])/2

    V1 = LSHOC - DMarkers['LELB'] 
    V2 = lmid_wrist - DMarkers['LELB']

    V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
    V2 = V2 / np.linalg.norm(V2)

    # Take the cross product
    cons_vector = np.cross(V1, V2)

    LELBC = DMarkers['LELB'] - cons_vector*DMarkers['LELBW']
    positions.append(LELBC)

    # V1 = (DMarkers['LWRA']-DMarkers['LWRB'])
    # V2 = lmid_wrist- LELBC

    # V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
    # V2 = V2 / np.linalg.norm(V2)

    # # Take the cross product
    # cons_vector = np.cross(V1, V2)

    LWRC = lmid_wrist #+ DMarkers['LWRIW']*cons_vector
    positions.append(LWRC)

    RSHOC = DMarkers['RSHO']+offset_rshoulder
    positions.append(RSHOC)

    rmid_wrist = (DMarkers['RWRA']+DMarkers['RWRB'])/2

    V1 = RSHOC - DMarkers['RELB'] 
    V2 = rmid_wrist - DMarkers['RELB']

    V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
    V2 = V2 / np.linalg.norm(V2)

    # Take the cross product
    cons_vector = np.cross(V1, V2)

    RELBC = DMarkers['RELB'] + cons_vector*DMarkers['RELBW']
    positions.append(RELBC)

    # V1 = (DMarkers['RWRA']-DMarkers['RWRB'])
    # V2 = rmid_wrist- RELBC

    # V1 = V1 / np.linalg.norm(V1)  # Normalise vectors
    # V2 = V2 / np.linalg.norm(V2)

    # # Take the cross product
    # cons_vector = np.cross(V1, V2)

    RWRC = rmid_wrist #- DMarkers['RWRIW']*cons_vector
    positions.append(RWRC)

    
    return dict(zip(names,positions))

def compute_mean_joints_centers(joints_centers_list):
    """_Calculates the mean joint centers when the joint centers have been calculated on several samples_

    Args:
        joints_centers_list (_list_): _A list of dictionnary containing the joint centers positions for each sample_

    Returns:
        _dict_: _a dictionnary containing joint centers names and mean positions in global frame_
    """
    names=['PELC','THOC','LHIPC','LKNEC','LANKC','RHIPC','RKNEC','RANKC','LSHOC','LELBC','LWRC','RSHOC','RELBC','RWRC']
    positions=[]

    PELC=np.zeros((3,))
    THOC=np.zeros((3,))
    LHIPC=np.zeros((3,))
    LKNEC=np.zeros((3,))
    LANKC=np.zeros((3,))
    RHIPC=np.zeros((3,))
    RKNEC=np.zeros((3,))
    RANKC=np.zeros((3,))
    LSHOC=np.zeros((3,))
    LELBC=np.zeros((3,))
    LWRC=np.zeros((3,))
    RSHOC=np.zeros((3,))
    RELBC=np.zeros((3,))
    RWRC=np.zeros((3,))

    for ii in range(len(joints_centers_list)):
        PELC+=joints_centers_list[ii]['PELC']
        THOC+=joints_centers_list[ii]['THOC']
        LHIPC+=joints_centers_list[ii]['LHIPC']
        LKNEC+=joints_centers_list[ii]['LKNEC']
        LANKC+=joints_centers_list[ii]['LANKC']
        RHIPC+=joints_centers_list[ii]['RHIPC']
        RKNEC+=joints_centers_list[ii]['RKNEC']
        RANKC+=joints_centers_list[ii]['RANKC']
        LSHOC+=joints_centers_list[ii]['LSHOC']
        LELBC+=joints_centers_list[ii]['LELBC']
        LWRC+=joints_centers_list[ii]['LWRC']
        RSHOC+=joints_centers_list[ii]['RSHOC']
        RELBC+=joints_centers_list[ii]['RELBC']
        RWRC+=joints_centers_list[ii]['RWRC']
    positions.append(PELC/len(joints_centers_list))
    positions.append(THOC/len(joints_centers_list))
    positions.append(LHIPC/len(joints_centers_list))
    positions.append(LKNEC/len(joints_centers_list))
    positions.append(LANKC/len(joints_centers_list))
    positions.append(RHIPC/len(joints_centers_list))
    positions.append(RKNEC/len(joints_centers_list))
    positions.append(RANKC/len(joints_centers_list))
    positions.append(LSHOC/len(joints_centers_list))
    positions.append(LELBC/len(joints_centers_list))
    positions.append(LWRC/len(joints_centers_list))
    positions.append(RSHOC/len(joints_centers_list))
    positions.append(RELBC/len(joints_centers_list))
    positions.append(RWRC/len(joints_centers_list))
    return(dict(zip(names,positions)))

def scale_human_model_mocap(model,DMarkers):
    """_Scales the lengths of the human model to the lengths calculated thanks to joints centers positions_

    Args:
        model (_model_): _Pinocchio model to be scaled_
        DMarkers (_dict_): _a dictionnary containing joints centers positions and their names_

    Returns:
        _model_: _the model that is properly scaled_
        _data_: _the data associated_
    """

        # Get the joint to scale ids 

    IDX_JLANKZ = model.getJointId('left_ankle_Z')
    IDX_JRANKZ = model.getJointId('right_ankle_Z')
    IDX_JRKNEE = model.getJointId('right_knee')
    IDX_JLKNEE = model.getJointId('left_knee')
    IDX_JLELBZ = model.getJointId('left_elbow_Z')
    IDX_JRELBZ = model.getJointId('right_elbow_Z')
    IDX_JLWRZ = model.getJointId('left_wrist_Z')
    IDX_JRWRZ = model.getJointId('right_wrist_Z')
    IDX_JLSHOY = model.getJointId('left_shoulder_Z')
    IDX_JRSHOY = model.getJointId('right_shoulder_Z')

    # Retrieve the segments lengths from measures

    l_fem_lenght=np.linalg.norm(DMarkers['LKNEC']-DMarkers['LHIPC'])
    l_tib_lenght= np.linalg.norm(DMarkers['LANKC']-DMarkers['LKNEC'])
    l_forearm_lenght= np.linalg.norm(DMarkers['LWRC']-DMarkers['LELBC'])
    l_upperarm_lenght= np.linalg.norm(DMarkers['LELBC']-DMarkers['LSHOC'])


    r_fem_lenght= np.linalg.norm(DMarkers['RKNEC']-DMarkers['RHIPC'])
    r_tib_lenght= np.linalg.norm(DMarkers['RANKC']-DMarkers['RKNEC'])
    r_forearm_lenght= np.linalg.norm(DMarkers['RWRC']-DMarkers['RELBC'])
    r_upperarm_lenght= np.linalg.norm(DMarkers['RELBC']-DMarkers['RSHOC'])

    l_trunk=np.linalg.norm(DMarkers['LSHOC']-DMarkers['THOC'])
    r_trunk=np.linalg.norm(DMarkers['RSHOC']-DMarkers['THOC'])

    # KNEES
    model.jointPlacements[IDX_JRKNEE].translation=np.array([0,-r_fem_lenght,0])
    model.jointPlacements[IDX_JLKNEE].translation=np.array([0,-l_fem_lenght,0])

    # ELBOWS
    model.jointPlacements[IDX_JRELBZ].translation=np.array([0,-r_upperarm_lenght,0])
    model.jointPlacements[IDX_JLELBZ].translation=np.array([0,-l_upperarm_lenght,0])

    # ANKLES
    model.jointPlacements[IDX_JRANKZ].translation=np.array([0,-r_tib_lenght,0])
    model.jointPlacements[IDX_JLANKZ].translation=np.array([0,-l_tib_lenght,0])

    # WRISTS
    model.jointPlacements[IDX_JRWRZ].translation=np.array([0,-r_forearm_lenght,0])
    model.jointPlacements[IDX_JLWRZ].translation=np.array([0,-l_forearm_lenght,0])

    # SHOULDERS
    model.jointPlacements[IDX_JRSHOY].translation[2]=r_trunk
    model.jointPlacements[IDX_JLSHOY].translation[2]=-l_trunk

    data=model.createData()

    return model,data


class calibration_mocap(object):
    """ A class that instantiate the optimisation problem that we solve to calibrate the model with a static pose. Here ipopt is used
    """
    def __init__(self,  model,data,DMarkers):
        
        self.DMarkers=DMarkers
        self.model=model
        self.data=data
        
    def objective(self, x):
    # callback for objective 
        q_tPose=0.0001*np.ones((self.model.nq,))

        q_tPose[23]=np.pi/2
        q_tPose[25]=0.1
        q_tPose[26]=4*np.pi/9
        q_tPose[34]=np.pi/2
        q_tPose[36]=0.1
        q_tPose[37]=4*np.pi/9
      
        Goal=np.empty(shape=[0,3])

        for cle,valeur in self.DMarkers.items():
            Goal=np.concatenate((Goal,np.reshape(np.array(valeur),(1,3))),axis=0)

        ids=[] # retrieve the index of joints of interest to fit
        ids.append(self.model.getJointId('middle_lumbar_X')) # PELC
        ids.append(self.model.getJointId('middle_cervical_Y')) # THOC
        ids.append(self.model.getJointId('left_hip_Y')) # LHIPC
        ids.append(self.model.getJointId('left_knee')) # LKNEC
        ids.append(self.model.getJointId('left_ankle_X')) # LANKC
        ids.append(self.model.getJointId('right_hip_Y')) # RHIPC
        ids.append(self.model.getJointId('right_knee')) # RKNEC
        ids.append(self.model.getJointId('right_ankle_X')) # RANKC
        ids.append(self.model.getJointId('left_shoulder_Y')) # LSHOC
        ids.append(self.model.getJointId('left_elbow_Y')) # LELBC
        ids.append(self.model.getJointId('left_wrist_X')) # LWRC
        ids.append(self.model.getJointId('right_shoulder_Y')) # RSHOC
        ids.append(self.model.getJointId('right_elbow_Y')) # RELBC
        ids.append(self.model.getJointId('right_wrist_X')) # RWRC

        pin.forwardKinematics(self.model,self.data,x)

        Mposition_joints=np.empty(shape=[0, 3])

        for ii in range(len(ids)): 
            Mposition_joints=np.concatenate((Mposition_joints,np.reshape(self.data.oMi[ids[ii]].translation,(1,3))),axis=0)

        J=np.sum((Goal-Mposition_joints)**2)+1e-2*np.sum((q_tPose-x)**2)

        return  J 

    def constraints(self, x):
        """Returns the constraints."""
        return np.linalg.norm([x[3],x[4],x[5],x[6]]) #norm of the freeflyer quaternion equal to 1

    def gradient(self, x):
        # callback for gradient

        G=approx_fprime(x, self.objective, 1e-5)

        return G

    def jacobian(self, x):
    # callback for jacobian of constraints
        jac=approx_fprime(x, self.constraints, 1e-5)

        return jac

def calibrate_human_model_mocap(model,data,DMarkers,q0):
    """ Set the optimisation problem : min || joint_centers_positions_measured - joint_center_of_model_positions ||² with q being the vector of optim
    """

    lb = model.lowerPositionLimit # lower joint limits
    ub = model.upperPositionLimit # upper joint limits
    cl=cu=[1]

    nlp = cyipopt.Problem(
        n=len(q0),
        m=len(cl),
        problem_obj=calibration_mocap(model,data,DMarkers),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        )

    nlp.add_option('tol',1e-3)
    nlp.add_option('print_level',0)
    q_opt, info = nlp.solve(q0)

    return q_opt

def get_local_markers(model,data,q0,markers_global):
    """_Creates a dictionnary that list the local positions of all markers with their respective names_

    Args:
        model (_model_): _Pinocchio model_
        data (_data_): _Pinocchio data_
        q0 (_array_): _Angular configuration corresponding to angular configuration of the static pose retrieved by the optimisation problem_
        markers_global (_dict_): _Dictionnary containing positions of markers in global frame _

    Returns:
        _dict_: _Dictionnary that lists the local positions of all markers with their respective names_
    """

    local_values=[]
    names=['LFHD','RFHD','LBHD','RBHD','C7','T10','CLAV','STRN','RBAK','LSHO','LELB','LWRA','LWRB','LFIN','RSHO','RELB','RWRA','RWRB','RFIN','LASI','RASI','LPSI','RPSI','LTHI','LKNE','LTIB','LANK','LHEE','LTOE','RTHI','RKNE','RTIB','RANK','RHEE','RTOE']
    pin.forwardKinematics(model,data,q0)
    pin.updateFramePlacements
    oMi=dict(zip(model.names,data.oMi))
    local_values.append(oMi['middle_cervical_Y'].inverse().rotation@(markers_global['LFHD']-oMi['middle_cervical_Y'].translation))
    local_values.append(oMi['middle_cervical_Y'].inverse().rotation@(markers_global['RFHD']-oMi['middle_cervical_Y'].translation))
    local_values.append(oMi['middle_cervical_Y'].inverse().rotation@(markers_global['LBHD']-oMi['middle_cervical_Y'].translation))
    local_values.append(oMi['middle_cervical_Y'].inverse().rotation@(markers_global['RBHD']-oMi['middle_cervical_Y'].translation))
    local_values.append(oMi['middle_thoracic_Y'].inverse().rotation@(markers_global['C7']-oMi['middle_thoracic_Y'].translation))
    local_values.append(oMi['middle_abdomen_Y'].inverse().rotation@(markers_global['T10']-oMi['middle_abdomen_Y'].translation))
    local_values.append(oMi['middle_thoracic_Y'].inverse().rotation@(markers_global['CLAV']-oMi['middle_thoracic_Y'].translation))
    local_values.append(oMi['middle_abdomen_Y'].inverse().rotation@(markers_global['STRN']-oMi['middle_abdomen_Y'].translation))
    local_values.append(oMi['middle_thoracic_Y'].inverse().rotation@(markers_global['RBAK']-oMi['middle_thoracic_Y'].translation))
    local_values.append(oMi['left_shoulder_Y'].inverse().rotation@(markers_global['LSHO']-oMi['left_shoulder_Y'].translation))
    local_values.append(oMi['left_elbow_Y'].inverse().rotation@(markers_global['LELB']-oMi['left_elbow_Y'].translation))
    local_values.append(oMi['left_wrist_X'].inverse().rotation@(markers_global['LWRA']-oMi['left_wrist_X'].translation))
    local_values.append(oMi['left_wrist_X'].inverse().rotation@(markers_global['LWRB']-oMi['left_wrist_X'].translation))
    local_values.append(oMi['left_wrist_X'].inverse().rotation@(markers_global['LFIN']-oMi['left_wrist_X'].translation))
    local_values.append(oMi['right_shoulder_Y'].inverse().rotation@(markers_global['RSHO']-oMi['right_shoulder_Y'].translation))
    local_values.append(oMi['right_elbow_Y'].inverse().rotation@(markers_global['RELB']-oMi['right_elbow_Y'].translation))
    local_values.append(oMi['right_wrist_X'].inverse().rotation@(markers_global['RWRA']-oMi['right_wrist_X'].translation))
    local_values.append(oMi['right_wrist_X'].inverse().rotation@(markers_global['RWRB']-oMi['right_wrist_X'].translation))
    local_values.append(oMi['right_wrist_X'].inverse().rotation@(markers_global['RFIN']-oMi['right_wrist_X'].translation))
    local_values.append(oMi['middle_lumbar_X'].inverse().rotation@(markers_global['LASI']-oMi['middle_lumbar_X'].translation))
    local_values.append(oMi['middle_lumbar_X'].inverse().rotation@(markers_global['RASI']-oMi['middle_lumbar_X'].translation))
    local_values.append(oMi['middle_lumbar_X'].inverse().rotation@(markers_global['LPSI']-oMi['middle_lumbar_X'].translation))
    local_values.append(oMi['middle_lumbar_X'].inverse().rotation@(markers_global['RPSI']-oMi['middle_lumbar_X'].translation))
    local_values.append(oMi['left_hip_Y'].inverse().rotation@(markers_global['LTHI']-oMi['left_hip_Y'].translation))
    local_values.append(oMi['left_knee'].inverse().rotation@(markers_global['LKNE']-oMi['left_knee'].translation))
    local_values.append(oMi['left_knee'].inverse().rotation@(markers_global['LTIB']-oMi['left_knee'].translation))
    local_values.append(oMi['left_ankle_X'].inverse().rotation@(markers_global['LANK']-oMi['left_ankle_X'].translation))
    local_values.append(oMi['left_ankle_X'].inverse().rotation@(markers_global['LHEE']-oMi['left_ankle_X'].translation))
    local_values.append(oMi['left_ankle_X'].inverse().rotation@(markers_global['LTOE']-oMi['left_ankle_X'].translation))
    local_values.append(oMi['right_hip_Y'].inverse().rotation@(markers_global['RTHI']-oMi['right_hip_Y'].translation))
    local_values.append(oMi['right_knee'].inverse().rotation@(markers_global['RKNE']-oMi['right_knee'].translation))
    local_values.append(oMi['right_knee'].inverse().rotation@(markers_global['RTIB']-oMi['right_knee'].translation))
    local_values.append(oMi['right_ankle_X'].inverse().rotation@(markers_global['RANK']-oMi['right_ankle_X'].translation))
    local_values.append(oMi['right_ankle_X'].inverse().rotation@(markers_global['RHEE']-oMi['right_ankle_X'].translation))
    local_values.append(oMi['right_ankle_X'].inverse().rotation@(markers_global['RTOE']-oMi['right_ankle_X'].translation))
    return dict(zip(names,local_values))

def markers_local_for_df(markers_local):
    """ Not very relevant, used for the statistic study
    """
    names=['LFHDx','LFHDy','LFHDz','RFHDx','RFHDy','RFHDz','LBHDx','LBHDy','LBHDz','RBHDx','RBHDy','RBHDz','C7x','C7y','C7z','T10x','T10y','T10z','CLAVx','CLAVy','CLAVz','STRNx','STRNy','STRNz','RBAKx','RBAKy','RBAKz','LSHOx','LSHOy','LSHOz','LELBx','LELBy','LELBz','LWRAx','LWRAy','LWRAz','LWRBx','LWRBy','LWRBz','LFINx','LFINy','LFINz','RSHOx','RSHOy','RSHOz','RELBx','RELBy','RELBz','RWRAx','RWRAy','RWRAz','RWRBx','RWRBy','RWRBz','RFINx','RFINy','RFINz','LASIx','LASIy','LASIz','RASIx','RASIy','RASIz','LPSIx','LPSIy','LPSIz','RPSIx','RPSIy','RPSIz','LTHIx','LTHIy','LTHIz','LKNEx','LKNEy','LKNEz','LTIBx','LTIBy','LTIBz','LANKx','LANKy','LANKz','LHEEx','LHEEy','LHEEz','LTOEx','LTOEy','LTOEz','RTHIx','RTHIy','RTHIz','RKNEx','RKNEy','RKNEz','RTIBx','RTIBy','RTIBz','RANKx','RANKy','RANKz','RHEEx','RHEEy','RHEEz','RTOEx','RTOEy','RTOEz']
    values=[]
    for ii in ['LFHD','RFHD','LBHD','RBHD','C7','T10','CLAV','STRN','RBAK','LSHO','LELB','LWRA','LWRB','LFIN','RSHO','RELB','RWRA','RWRB','RFIN','LASI','RASI','LPSI','RPSI','LTHI','LKNE','LTIB','LANK','LHEE','LTOE','RTHI','RKNE','RTIB','RANK','RHEE','RTOE']:
        values.append(markers_local[ii][0])
        values.append(markers_local[ii][1])
        values.append(markers_local[ii][2])
    return dict(zip(names,values))

def mean_local_markers(mean):
    """ Not very relevant, Transcription of the statistical study
    """
    names=['LFHD','RFHD','LBHD','RBHD','C7','T10','CLAV','STRN','RBAK','LSHO','LELB','LWRA','LWRB','LFIN','RSHO','RELB','RWRA','RWRB','RFIN','LASI','RASI','LPSI','RPSI','LTHI','LKNE','LTIB','LANK','LHEE','LTOE','RTHI','RKNE','RTIB','RANK','RHEE','RTOE']
    values=[]
    values.append(np.array([mean['LFHDx'],mean['LFHDy'],mean['LFHDz']]))
    values.append(np.array([mean['RFHDx'],mean['RFHDy'],mean['RFHDz']]))
    values.append(np.array([mean['LBHDx'],mean['LBHDy'],mean['LBHDz']]))
    values.append(np.array([mean['RBHDx'],mean['RBHDy'],mean['RBHDz']]))
    values.append(np.array([mean['C7x'],mean['C7y'],mean['C7z']]))
    values.append(np.array([mean['T10x'],mean['T10y'],mean['T10z']]))
    values.append(np.array([mean['CLAVx'],mean['CLAVy'],mean['CLAVz']]))
    values.append(np.array([mean['STRNx'],mean['STRNy'],mean['STRNz']]))
    values.append(np.array([mean['RBAKx'],mean['RBAKy'],mean['RBAKz']]))
    values.append(np.array([mean['LSHOx'],mean['LSHOy'],mean['LSHOz']]))
    values.append(np.array([mean['LELBx'],mean['LELBy'],mean['LELBz']]))
    values.append(np.array([mean['LWRAx'],mean['LWRAy'],mean['LWRAz']]))
    values.append(np.array([mean['LWRBx'],mean['LWRBy'],mean['LWRBz']]))
    values.append(np.array([mean['LFINx'],mean['LFINy'],mean['LFINz']]))
    values.append(np.array([mean['RSHOx'],mean['RSHOy'],mean['RSHOz']]))
    values.append(np.array([mean['RELBx'],mean['RELBy'],mean['RELBz']]))
    values.append(np.array([mean['RWRAx'],mean['RWRAy'],mean['RWRAz']]))
    values.append(np.array([mean['RWRBx'],mean['RWRBy'],mean['RWRBz']]))
    values.append(np.array([mean['RFINx'],mean['RFINy'],mean['RFINz']]))
    values.append(np.array([mean['LASIx'],mean['LASIy'],mean['LASIz']]))
    values.append(np.array([mean['RASIx'],mean['RASIy'],mean['RASIz']]))
    values.append(np.array([mean['LPSIx'],mean['LPSIy'],mean['LPSIz']]))
    values.append(np.array([mean['RPSIx'],mean['RPSIy'],mean['RPSIz']]))
    values.append(np.array([mean['LTHIx'],mean['LTHIy'],mean['LTHIz']]))
    values.append(np.array([mean['LKNEx'],mean['LKNEy'],mean['LKNEz']]))
    values.append(np.array([mean['LTIBx'],mean['LTIBy'],mean['LTIBz']]))
    values.append(np.array([mean['LANKx'],mean['LANKy'],mean['LANKz']]))
    values.append(np.array([mean['LHEEx'],mean['LHEEy'],mean['LHEEz']]))
    values.append(np.array([mean['LTOEx'],mean['LTOEy'],mean['LTOEz']]))
    values.append(np.array([mean['RTHIx'],mean['RTHIy'],mean['RTHIz']]))
    values.append(np.array([mean['RKNEx'],mean['RKNEy'],mean['RKNEz']]))
    values.append(np.array([mean['RTIBx'],mean['RTIBy'],mean['RTIBz']]))
    values.append(np.array([mean['RANKx'],mean['RANKy'],mean['RANKz']]))
    values.append(np.array([mean['RHEEx'],mean['RHEEy'],mean['RHEEz']]))
    values.append(np.array([mean['RTOEx'],mean['RTOEy'],mean['RTOEz']]))
    return dict(zip(names,values))

def add_plug_in_gait_markers(model,data,markers_local):
    """_Adds the markers corresponding to the Vicon plug in gait template to the model as new frames_

    Args:
        model (_model_): _Pinocchio model_
        data (_data_): _Pinocchio data_
        markers_local (_dict_): _Dictionnary containing all the local poses of markers_

    Returns:
        _data_: _Data refreshed with markers frames_
        _list_: _List of index of markers frames_
        Note : the model has also been updated at this point
    """

    i_markers=[]

    # Get the index of frames of interest
    IDX_LF = model.getFrameId('left_foot')
    IDX_LH = model.getFrameId('left_hand')
    IDX_LLA = model.getFrameId('left_lowerarm')
    IDX_LLL = model.getFrameId('left_lowerleg')
    IDX_LUA = model.getFrameId('left_upperarm')
    IDX_LUL = model.getFrameId('left_upperleg')
    IDX_MH = model.getFrameId('middle_head')
    IDX_MP = model.getFrameId('middle_pelvis')
    IDX_MT = model.getFrameId('middle_thorax')
    IDX_MA = model.getFrameId('middle_abdomen')
    IDX_RF = model.getFrameId('right_foot')
    IDX_RH = model.getFrameId('right_hand')
    IDX_RLA = model.getFrameId('right_lowerarm')
    IDX_RLL = model.getFrameId('right_lowerleg')
    IDX_RUA = model.getFrameId('right_upperarm')
    IDX_RUL = model.getFrameId('right_upperleg')

    # Get the index of joints of interest
    
    IDX_JLF = model.getJointId('left_ankle_X')
    IDX_JLH = model.getJointId('left_wrist_X')
    IDX_JLLA = model.getJointId('left_elbow_Y')
    IDX_JLLL = model.getJointId('left_knee')
    IDX_JLUA = model.getJointId('left_shoulder_Y')
    IDX_JLUL = model.getJointId('left_hip_Y')
    IDX_JMH = model.getJointId('middle_cervical_Y')
    IDX_JMP= model.getJointId('middle_lumbar_X')
    IDX_JMT = model.getJointId('middle_thoracic_Y')
    IDX_JMA = model.getJointId('middle_abdomen_Y')
    IDX_JRF = model.getJointId('right_ankle_X')
    IDX_JRH = model.getJointId('right_wrist_X')
    IDX_JRLA = model.getJointId('right_elbow_Y')
    IDX_JRLL = model.getJointId('right_knee')
    IDX_JRUA = model.getJointId('right_shoulder_Y')
    IDX_JRUL = model.getJointId('right_hip_Y')
   
    # 35 MARKERS FRAMES TO ADD
   
    inertia = pin.Inertia.Zero()
    
    F1=pin.Frame('LASI',IDX_JMP,IDX_MP, pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LASI'][0],markers_local['LASI'][1],markers_local['LASI'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i1=model.addFrame(F1,False)
    i_markers.append(i1)
    F2=pin.Frame('RASI',IDX_JMP,IDX_MP,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RASI'][0],markers_local['RASI'][1],markers_local['RASI'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i2=model.addFrame(F2,False)
    i_markers.append(i2)
    F3=pin.Frame('LPSI',IDX_JMP,IDX_MP,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LPSI'][0],markers_local['LPSI'][1],markers_local['LPSI'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i3=model.addFrame(F3,False)
    i_markers.append(i3)
    F4=pin.Frame('RPSI',IDX_JMP,IDX_MP,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RPSI'][0],markers_local['RPSI'][1],markers_local['RPSI'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i4=model.addFrame(F4,False)
    i_markers.append(i4)

    F5=pin.Frame('LTHI',IDX_JLUL,IDX_LUL,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LTHI'][0],markers_local['LTHI'][1],markers_local['LTHI'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i5=model.addFrame(F5,False)
    i_markers.append(i5)
    F6=pin.Frame('LKNE',IDX_JLLL,IDX_LLL,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LKNE'][0],markers_local['LKNE'][1],markers_local['LKNE'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i6=model.addFrame(F6,False)
    i_markers.append(i6)
    F7=pin.Frame('LTIB',IDX_JLLL,IDX_LLL,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LTIB'][0],markers_local['LTIB'][1],markers_local['LTIB'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i7=model.addFrame(F7,False)
    i_markers.append(i7)
    F8=pin.Frame('LANK',IDX_JLF,IDX_LF,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LANK'][0],markers_local['LANK'][1],markers_local['LANK'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i8=model.addFrame(F8,False)
    i_markers.append(i8)
    F9=pin.Frame('LHEE',IDX_JLF,IDX_LF,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LHEE'][0],markers_local['LHEE'][1],markers_local['LHEE'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i9=model.addFrame(F9,False)
    i_markers.append(i9)
    F10=pin.Frame('LTOE',IDX_JLF,IDX_LF,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LTOE'][0],markers_local['LTOE'][1],markers_local['LTOE'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i10=model.addFrame(F10,False)
    i_markers.append(i10)

    F11=pin.Frame('RTHI',IDX_JRUL,IDX_RUL,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RTHI'][0],markers_local['RTHI'][1],markers_local['RTHI'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i11=model.addFrame(F11,False)
    i_markers.append(i11)
    F12=pin.Frame('RKNE',IDX_JRLL,IDX_RLL,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RKNE'][0],markers_local['RKNE'][1],markers_local['RKNE'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i12=model.addFrame(F12,False)
    i_markers.append(i12)
    F13=pin.Frame('RTIB',IDX_JRLL,IDX_RLL,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RTIB'][0],markers_local['RTIB'][1],markers_local['RTIB'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i13=model.addFrame(F13,False)
    i_markers.append(i13)
    F14=pin.Frame('RANK',IDX_JRF,IDX_RF,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RANK'][0],markers_local['RANK'][1],markers_local['RANK'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i14=model.addFrame(F14,False)
    i_markers.append(i14)
    F15=pin.Frame('RHEE',IDX_JRF,IDX_RF,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RHEE'][0],markers_local['RHEE'][1],markers_local['RHEE'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i15=model.addFrame(F15,False)
    i_markers.append(i15)
    F16=pin.Frame('RTOE',IDX_JRF,IDX_RF,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RTOE'][0],markers_local['RTOE'][1],markers_local['RTOE'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i16=model.addFrame(F16,False)
    i_markers.append(i16)

    F17=pin.Frame('LFHD',IDX_JMH,IDX_MH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LFHD'][0],markers_local['LFHD'][1],markers_local['LFHD'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i17=model.addFrame(F17,False)
    i_markers.append(i17)
    F18=pin.Frame('RFHD',IDX_JMH,IDX_MH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RFHD'][0],markers_local['RFHD'][1],markers_local['RFHD'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i18=model.addFrame(F18,False)
    i_markers.append(i18)
    F19=pin.Frame('LBHD',IDX_JMH,IDX_MH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LBHD'][0],markers_local['LBHD'][1],markers_local['LBHD'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i19=model.addFrame(F19,False)
    i_markers.append(i19)
    F20=pin.Frame('RBHD',IDX_JMH,IDX_MH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RBHD'][0],markers_local['RBHD'][1],markers_local['RBHD'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i20=model.addFrame(F20,False)
    i_markers.append(i20)

    F21=pin.Frame('C7',IDX_JMT,IDX_MT,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['C7'][0],markers_local['C7'][1],markers_local['C7'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i21=model.addFrame(F21,False)
    i_markers.append(i21)
    F22=pin.Frame('T10',IDX_JMA,IDX_MA,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['T10'][0],markers_local['T10'][1],markers_local['T10'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i22=model.addFrame(F22,False)
    i_markers.append(i22)
    F23=pin.Frame('CLAV',IDX_JMT,IDX_MT,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['CLAV'][0],markers_local['CLAV'][1],markers_local['CLAV'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i23=model.addFrame(F23,False)
    i_markers.append(i23)
    F24=pin.Frame('STRN',IDX_JMA,IDX_MA,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['STRN'][0],markers_local['STRN'][1],markers_local['STRN'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i24=model.addFrame(F24,False)
    i_markers.append(i24)
    F25=pin.Frame('RBAK',IDX_JMT,IDX_MT,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RBAK'][0],markers_local['RBAK'][1],markers_local['RBAK'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i25=model.addFrame(F25,False)
    i_markers.append(i25)

    F26=pin.Frame('LSHO',IDX_JLUA,IDX_LUA,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LSHO'][0],markers_local['LSHO'][1],markers_local['LSHO'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i26=model.addFrame(F26,False)
    i_markers.append(i26)
    F27=pin.Frame('LELB',IDX_JLLA,IDX_LLA,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LELB'][0],markers_local['LELB'][1],markers_local['LELB'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i27=model.addFrame(F27,False)
    i_markers.append(i27)
    F28=pin.Frame('LWRA',IDX_JLH,IDX_LH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LWRA'][0],markers_local['LWRA'][1],markers_local['LWRA'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i28=model.addFrame(F28,False)
    i_markers.append(i28)
    F29=pin.Frame('LWRB',IDX_JLH,IDX_LH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LWRB'][0],markers_local['LWRB'][1],markers_local['LWRB'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i29=model.addFrame(F29,False)
    i_markers.append(i29)
    F30=pin.Frame('LFIN',IDX_JLH,IDX_LH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['LFIN'][0],markers_local['LFIN'][1],markers_local['LFIN'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i30=model.addFrame(F30,False)
    i_markers.append(i30)

    F31=pin.Frame('RSHO',IDX_JRUA,IDX_RUA,pin.SE3(Rquat(1, 0, 0, 0), np.matrix([markers_local['RSHO'][0],markers_local['RSHO'][1],markers_local['RSHO'][2]]).T), pin.FrameType.OP_FRAME, inertia)
    i31=model.addFrame(F31,False)
    i_markers.append(i31)
    F32=pin.Frame('RELB',IDX_JRLA,IDX_RLA,pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local['RELB']).T), pin.FrameType.OP_FRAME, inertia)
    i32=model.addFrame(F32,False)
    i_markers.append(i32)
    F33=pin.Frame('RWRA',IDX_JRH,IDX_RH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local['RWRA']).T), pin.FrameType.OP_FRAME, inertia)
    i33=model.addFrame(F33,False)
    i_markers.append(i33)
    F34=pin.Frame('RWRB',IDX_JRH,IDX_RH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local['RWRB']).T), pin.FrameType.OP_FRAME, inertia)
    i34=model.addFrame(F34,False)
    i_markers.append(i34)
    F35=pin.Frame('RFIN',IDX_JRH,IDX_RH,pin.SE3(Rquat(1, 0, 0, 0), np.matrix(markers_local['RFIN']).T), pin.FrameType.OP_FRAME, inertia)
    i35=model.addFrame(F35,False)
    i_markers.append(i35)

    data = model.createData()

    return data,i_markers

