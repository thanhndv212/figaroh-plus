
import pinocchio as pin
from gepetto import Color

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os
from os.path import dirname, join, abspath
from figaroh.tools.robot import Robot
from figaroh.calibration.calibration_tools import load_data, update_joint_placement


def add_joint_red(gui, name, joint):
    def rotation(q):
        gui.addCylinder(name+"/axis", 0.015, 0.15, Color.red)
        gui.applyConfiguration(name+"/axis", q)
    gui.createGroup(name)
    gui.addToGroup(name, "world")
    from math import sqrt
    jointtype = joint.shortname()
    if jointtype == "JointModelRX": # Z->X
        rotation([0,0,0,0,sqrt(2)/2,0,sqrt(2)/2])
    elif jointtype == "JointModelRY": # Y->Z
        rotation([0,0,0,-sqrt(2)/2,0,0,sqrt(2)/2])
    elif jointtype == "JointModelRZ":
        rotation([0,0,0,0,0,0,1])
    #elif jointtype == "JointModelFreeFlyer":
        # gui.addXYZaxis(name+"/axis", Color.blue, 0.01, 0.1)


def add_joint_blue(gui, name, joint):
    # print(name)
    def rotation(q):
        gui.addCylinder(name+"/axis", 0.015, 0.15, Color.blue)
        gui.applyConfiguration(name+"/axis", q)
    gui.createGroup(name)
    gui.addToGroup(name, "world")
    from math import sqrt
    jointtype = joint.shortname()
    if jointtype == "JointModelRX": # Z->X
        rotation([0,0,0,0,sqrt(2)/2,0,sqrt(2)/2])
    elif jointtype == "JointModelRY": # Y->Z
        rotation([0,0,0,-sqrt(2)/2,0,0,sqrt(2)/2])
    elif jointtype == "JointModelRZ":
        rotation([0,0,0,0,0,0,1])
    #elif jointtype == "JointModelFreeFlyer":
    #    gui.addXYZaxis(name+"/axis", Color.blue, 0.01, 0.1)


def display(robot, gui, model, data, q):
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    robot.display(q)
    # for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
    #     gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    # gui.refresh()



def add_axis_to_frame(gui, data, model, frame, axis_name=None, axis_boldness=0.15):
    gui.createGroup(frame)
    gui.addToGroup(frame, "world")
    placement = pin.SE3ToXYZQUATtuple(data.oMf[model.getFrameId(frame)])
    if axis_name is None:
        axis_name = frame + "/XYZaxis"
    else:
        axis_name = str(axis_name)
        axis_name = frame + axis_name

    # print(axis_name, placement)
    gui.addXYZaxis(axis_name, Color.blue, 0.005, axis_boldness)
    gui.applyConfiguration(axis_name, placement)
    gui.refresh()


def update_model(model, param_dict, joint_list, verbose=0):
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
    model_ = model.copy()

    # update model.jointPlacements
    updated_params = []

    for joint_ in joint_list:
        j_id = model_.getJointId(joint_)
        xyz_rpy = np.zeros(6)
        j_name = model_.names[j_id]
        for key in param_dict.keys():
            if j_name in key:
                # update xyz_rpy with kinematic errors
                for axis_id, axis in enumerate(axis_tpl):
                    if axis in key:
                        if verbose==1:
                            print("Updating [{}] joint placement at axis {} with [{}]".format(j_name, axis, key))
                        xyz_rpy[axis_id] += param_dict[key]
                        updated_params.append(key)
        model_ = update_joint_placement(model_, j_id, xyz_rpy)

    return model_

# 1/ Load robot model and create a dictionary containing reserved constants
ros_package_path = os.getenv('ROS_PACKAGE_PATH')
package_dirs = ros_package_path.split(':')
robot = Robot(
    'data/tiago_schunk.urdf',
    package_dirs= package_dirs,
    # isFext=True  # add free-flyer joint at base
)
q = pin.randomConfiguration(robot.model)
q = robot.q0
arm_links = ['arm_1_link',
            'arm_2_link',
            'arm_3_link',
            'arm_4_link',
            'arm_5_link',
            'arm_6_link',
            'arm_7_link']

robot.initViewer(loadModel=True)
gui = robot.viewer.gui

model1 = robot.model
data = robot.data
viz1 = robot.viz
# viz1.initViewer(viz1.viewer)
# viz1.loadViewerModel(rootNodeName = 'pinocchio1')
robot.display(q)
pin.framesForwardKinematics(model1, data, q)
pin.updateFramePlacements(model1, data)

for arm_link in arm_links:
    add_axis_to_frame(gui, data, model1, arm_link, axis_name="/XYZaxis_1", axis_boldness=0.1)


param_dict = {'d_px_torso_lift_joint': 0.0114, 'd_py_torso_lift_joint': 0.1951, 'd_pz_torso_lift_joint': -0.3314, 'd_phix_torso_lift_joint': 0.0185, 'd_phiy_torso_lift_joint': 0.0059, 'd_phiz_torso_lift_joint': -0.0077, 'd_phix_arm_1_joint': -0.005, 'd_phiy_arm_1_joint': -0.0157, 'd_px_arm_2_joint': 0.0017, 'd_pz_arm_2_joint': 0.0, 'd_phix_arm_2_joint': 0.0088, 'd_phiz_arm_2_joint': 0.0, 'd_px_arm_3_joint': 0.0025, 'd_pz_arm_3_joint': 0.0033, 'd_phix_arm_3_joint': 0.0049, 'd_phiz_arm_3_joint': -0.0078, 'd_py_arm_4_joint': 0.0015, 'd_pz_arm_4_joint': -0.0029, 'd_phiy_arm_4_joint': -0.0466, 'd_phiz_arm_4_joint': 0.0063, 'd_py_arm_5_joint': -0.0178, 'd_pz_arm_5_joint': 0.0015, 'd_phiy_arm_5_joint': 0.0308, 'd_phiz_arm_5_joint': -0.0239, 'd_py_arm_6_joint': 0.0008, 'd_pz_arm_6_joint': 0.0069, 'd_phiy_arm_6_joint': -0.0065, 'd_phiz_arm_6_joint': -0.0308, 'd_pz_arm_7_joint': 0.0015, 'pEEx_1': 0.0579, 'pEEy_1': -0.0015, 'pEEz_1': 0.0703}
robot.model = update_model(model1, param_dict, ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'], verbose=1)
# viz2 = robot.viz
# viz2.initViewer(viz2.viewer)
# viz2.loadViewerModel(rootNodeName = 'pinocchio2')
# viz2.display(q)

# pin.framesForwardKinematics(robot.model, data, q)
# pin.updateFramePlacements(robot.model, data)
# for arm_link in arm_links:
#     add_axis_to_frame(gui, data, robot.model, arm_link,  axis_name="/XYZaxis_2", axis_boldness=0.1)

gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 0.7)
# gui.setFloatProperty("world/pinocchio2/visuals", "Alpha", 0.9)

gui.setBackgroundColor1('python-pinocchio',[1.,1,1,1])
gui.setBackgroundColor2('python-pinocchio',[1.,1,1,1])



# # # draw kinematic tree
# for jid in range(2,model.njoints):
#     name, joint = model.names[jid], model.joints[jid]
#     pid = model.parents[jid]
#     pname, pjoint = model.names[pid], model.joints[pid]
#     if pid == 1: print(name, pname)
#     pMi = model.jointPlacements[jid]
#     if name.find('arm')==0:
#         gui.addLine(pname + "/to_" + name, [0,0,0], pMi.translation.tolist(), Color.blue)

# #q = np.array([ 0.0907,0.136,-0.493,1.08,-0.545,0.0676,0.0887,0.133,-0.416,0.862,-0.425,0.0736,0,0.00676,0.258,0.573,-0.0002,-1.36,0,0,0.1,0,-0.258,-0.573,0.0002,-1.36,0,0,0.1,0,0,0 ])
# # q = np.array([ 0,0,-0.4114 ,0.8594,-0.4480,-0.0017,0,0,-0.4114,0.8594,-0.4480,-0.0017 ,0,0.00676,0.258,0.573,-0.0002,-1.36,0,0,0.1,0,-0.258,-0.573,0.0002,-1.36,0,0,0.1,0,0,0 ])



# data_path = '/home/thanhndv212/Cooking/raw_data/Calibration/Talos/contact-calibration/complied_csv/'
# file_name = 'compiled_measurements_right_1107.csv'
# path_q = data_path + file_name
# import time
# q_r = read_data(model, path_q)

# q = q_r[20, :]
# pin.forwardKinematics(model, data, q)
# pin.updateFramePlacements(model, data)
# add_axis_to_frame(data, model, "gripper_right_fingertip_1_link")
# add_axis_to_frame(data, model, "right_sole_link")
# # add_axis_to_frame(data, model, "gripper_left_fingertip_1_link")
# # add_axis_to_frame(data, model, "left_sole_link")
# add_axis_to_frame(data, model, "desk_link_1")
 
# # contact points placements
# z_contact = 0.
# x_contact = [-0.32 + 0.075*i for i in range(6)]
# y_contact = [-0.4 + 0.2*i for i in range(6)]
# contacts = []
# contact_names = []
# count = 0
# for x_c in x_contact:
#     for y_c in y_contact:
#         count += 1
#         contact = [x_c, y_c, z_contact]
#         contacts.append(contact)
#         contact_names.append("contact_%s" % count)
# rm_contact_rightarm = np.array([6, 12, 18, 19, 23, 24,
#                        25, 26, 27, 28, 29, 30,
#                        31, 32, 33, 34, 35, 36]) - 1
# # add to gui
# place_frame = "desk_link_1"
# plcmt = data.oMf[model.getFrameId(place_frame)].translation
# rot_cntct = data.oMf[model.getFrameId(place_frame)].rotation
# rot_0 = [0,0,0,1]
# gui.createGroup("contact")
# gui.addToGroup("contact", "world")
# for ij in range(len(contacts)):
#     if ij not in rm_contact_rightarm.tolist():
#         name = "contact/" + contact_names[ij]
#         cntct_plcmnt = plcmt + rot_cntct.dot(np.array(contacts[ij]))
#         gui.deleteNode(name, True)
#         gui.addSphere(name, 0.01, Color.yellow)
#         gui.applyConfiguration(name, cntct_plcmnt.tolist() + rot_0)
#         gui.refresh()
# display(q)

# gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 0.7)
# # gui.setBackgroundColor1('python-pinocchio',[1.,1,1,1])
# # gui.setBackgroundColor2('python-pinocchio',[1.,1,1,1])



# view=[4,0.0,0.0]

# r = R.from_euler('zyx', [0,90,0], degrees=True)
# R=np.matrix('0 1 0; 1 0 0; 0 0 1') 
# quat=pin.Quaternion(r.as_matrix()).coeffs().tolist()
# view[len(view):] = quat
# #print(view)

# gui.setCameraTransform('python-pinocchio',view)


#robot.viewer.gui.setBackgroundColor2('python-pinocchio',[1.,0.0,0.,0.0])
#robot.viewer.gui.deleteNode("world/pinocchio/visuals",True)
#robot.viewer.gui.setCameraToBestFit('python-pinocchio')
  
#robot.viewer.gui.addSphere("world/sphere_1",0.1, [1., 0., 0., 1.])     
#robot.viewer.gui.applyConfiguration("world/sphere_1",[0, 0, 0, 1])

 # Print out the placement of each joint of the kinematic tree
# for name, oMi in zip(model.frames, data.oMf):
#     print(name.name, oMi.translation)

    # print(name)
    
    #print(*oMi.homogeneous)

# Pl_1=np.zeros((3,34))
# Pl_2=np.zeros((3,34))
# radius_1=[0.0] * 34
# radius_2=[0.0] * 34

# # Left_leg
# Pl_1[:,0]=[0.05,0.075,-0.05]
# radius_1[0]=0.15
 
# Pl_1[:,3]=[0.0,0.05,-0.05]
# radius_1[3]=0.1

# Pl_1[:,4]=[0.075,0.0,-0.075]
# radius_1[4]=0.075
# Pl_2[:,0]=[0.1,0.05,-0.225]
# radius_2[0]=0.1
 
# Pl_2[:,3]=[0.02,0.05,-0.2]
# radius_2[3]=0.1

# Pl_2[:,4]=[-0.07,0.0,-0.075]
# radius_2[4]=0.075

# #Pl_2[:,5]=[0.05,0.0,-0.075]
# #radius_2[5]=0.05

# Pl_2[:,0]=[0.1,0.05,-0.225]
# radius_2[0]=0.1
 
# Pl_2[:,3]=[0.02,0.05,-0.2]
# radius_2[3]=0.1

# Pl_2[:,4]=[-0.07,0.0,-0.075]
# radius_2[4]=0.075

# #Pl_2[:,5]=[0.05,0.0,-0.075]
# #radius_2[5]=0.05

# # Right leg

# Pl_1[:,0+6]=[0.05,-0.075,-0.05]
# radius_1[0+6]=0.15
 
# Pl_1[:,3+6]=[0.0,-0.05,-0.05]
# radius_1[3+6]=0.1

# Pl_1[:,4+6]=[0.075,0.0,-0.075]
# radius_1[4+6]=0.075

# Pl_2[:,0+6]=[0.1,-0.05,-0.225]
# radius_2[0+6]=0.1
 
# Pl_2[:,3+6]=[0.02,-0.05,-0.2]
# radius_2[3+6]=0.1

# Pl_2[:,4+6]=[-0.07,0.0,-0.075]
# radius_2[4+6]=0.075

# # torso
# Pl_1[:,12]=[-0.0725,0.0,0.15]
# radius_1[12]=0.225


# # left arm_ 2
# Pl_1[:,15]=[0.0,0.0,0.0]
# radius_1[15]=0.1

# Pl_1[:,16]=[0.0,0.0,-0.125]
# radius_1[16]=0.0875

# Pl_1[:,17]=[0.0,0.0,0.0]
# radius_1[17]=0.0825

# Pl_1[:,18]=[0.0,0.0,0.1]
# radius_1[18]=0.0725

# Pl_1[:,19]=[0.0,0.0,0.05]
# radius_1[19]=0.0725

# Pl_1[:,20]=[0.0,0.0,-0.15]
# radius_1[20]=0.1

# # right arm_ 2
# Pl_1[:,22]=[0.0,0.0,0.0]
# radius_1[22]=0.1

# Pl_1[:,23]=[0.0,0.0,-0.125]
# radius_1[23]=0.0875

# Pl_1[:,24]=[0.0,0.0,0.0]
# radius_1[24]=0.0825

# Pl_1[:,25]=[0.0,0.0,0.1]
# radius_1[25]=0.0725

# Pl_1[:,26]=[0.0,0.0,0.05]
# radius_1[26]=0.0725

# Pl_1[:,27]=[0.0,0.0,-0.15]
# radius_1[27]=0.1

# # head
# Pl_1[:,30]=[0.0,0.0,0.175]
# radius_1[30]=0.125

# ii=0
# for name in model.names[1:32]:
#     print(name)
 
#     name_sph_1="world/SPH_"+name+"_1"
#     name_sph_2="world/SPH_"+name+"_2"

#     print(ii)
#     Id=model.getJointId(name)
    
#     P_1=np.array(data.oMi[Id].rotation).dot(Pl_1[:,ii])+data.oMi[Id].translation
#     P_2=np.array(data.oMi[Id].rotation).dot(Pl_2[:,ii])+data.oMi[Id].translation
#     print(data.oMi[Id].translation)

#     gui.deleteNode(name_sph_1,True)
#     gui.addSphere(name_sph_1, radius_1[ii], [1., 0., 0., 0.75])
#     gui.applyConfiguration(name_sph_1, P_1.tolist() + [0, 0, 0, 1])

#     gui.deleteNode(name_sph_2,True)
#     gui.addSphere(name_sph_2, radius_2[ii], [1., 0., 0., 0.75])
#     gui.applyConfiguration(name_sph_2, P_2.tolist() + [0, 0, 0, 1])
#     ii=ii+1
#     gui.refresh()


# ii=33
# # display waist
# name_sph_1="world/SPH_waist_1"
# name_sph_2="world/SPH_waist_2"

# radius_1[33]=0.125
# radius_2[33]=0.125
    
# P_1=[-0.0,0.15,-0.05]
# P_2=[-0.0,-0.15,-0.05]

# gui.deleteNode(name_sph_1,True)
# gui.addSphere(name_sph_1, radius_1[ii], [1., 0., 0., 0.75])
# gui.applyConfiguration(name_sph_1, P_1 + [0, 0, 0, 1])

# gui.deleteNode(name_sph_2,True)
# gui.addSphere(name_sph_2, radius_2[ii], [1., 0., 0., 0.75])
# gui.applyConfiguration(name_sph_2, P_2 + [0, 0, 0, 1])
# gui.refresh()
'''
Id=model.getFrameId("leg_left_1_joint")
 
P_sph_leg_left_1_1=np.array(data.oMi[Id].rotation).dot(np.array([0, 0.1, -0.05]))+data.oMi[Id].translation
P_sph_leg_left_1_2=np.array(data.oMi[Id].rotation).dot(np.array([0.0, 0.05, -0.3]))+data.oMi[Id].translation



gui.deleteNode("world/P_sph_leg_left_1_1",True)
gui.deleteNode("world/P_sph_leg_left_1_2",True)

gui.deleteNode("world/P_sph_leg_left_1_1",True)
gui.deleteNode("world/P_sph_leg_left_1_2",True)

gui.addSphere("world/P_sph_leg_left_1_1", 0.14, [1., 0., 0., 1])
gui.applyConfiguration("world/P_sph_leg_left_1_1", P_sph_leg_left_1_1.tolist() + [0, 0, 0, 1])
gui.addSphere("world/P_sph_12", 0.105, [1., 0., 0., 1])
gui.applyConfiguration("world/P_sph_leg_left_1_2", P_sph_leg_left_1_2.tolist() + [0, 0, 0, 1])
'''

#for name, oMi in zip(model.names, data.oMi):
#    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
#          .format( name, *oMi.orientation )))


#print(pin.log(data.oMi[1]).angular)
#print(data.oMi[1].rotation)

''' 
## Display workspace boxes
Id_ankle=model.getJointId("leg_left_6_joint")
Id_hand=model.getJointId("arm_left_7_joint")

pose_foot=data.oMi[6]
pose_hand=data.oMi[Id_hand]




foot_pose=pin.SE3ToXYZQUATtuple(pose_foot)
box_pose_1=list(foot_pose[0:3])
box_pose_1[len(box_pose_1):] = [0, 0, 0, 1]

box_pose_1[0]=box_pose_1[0]+0.1

box_pose_1[1]=box_pose_1[1]+0.125
box_pose_1[2]=box_pose_1[2]+0.1
 
hand_pose=pin.SE3ToXYZQUATtuple(pose_hand)
box_pose_2=list(hand_pose[0:3])
box_pose_2[len(box_pose_2):] = [0, 0, 0, 1]

box_pose_2[0]=box_pose_2[0]+0.225

box_pose_2[1]=box_pose_2[1]-0.2
box_pose_2[2]=box_pose_2[2]+0.15



gui.deleteNode("world/box_1", True)
gui.deleteNode("world/box_2", True) 
gui.addBox("world/box_1",0.6, 0.425,0.4,[0, 1, 0, 0.4])     
gui.applyConfiguration("world/box_1",box_pose_1)

gui.addBox("world/box_2",0.6, 0.6,0.6,[0, 1, 0, 0.4])     
gui.applyConfiguration("world/box_2",box_pose_2)

#gui.setCameraTransform('python-pinocchio',toto)
display(q)
gui.refresh()
'''  

#toto=gui.getNodeList()
 
#pose_foot=pose_foot.translation.T.flat
#popo=*pose_foot
#print(*pose_foot)
#group_visuals=gui.getNodeList()
#print(len(group_visuals))
#for i in range(len(group_visuals)):
#    print(group_visuals[i])
#    gui.deleteNode(group_visuals[i], True)
