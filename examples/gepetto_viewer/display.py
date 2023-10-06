
import pinocchio as pin
from gepetto import Color

import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os
from os.path import dirname, join, abspath
from figaroh.tools.robot import Robot
from figaroh.calibration.calibration_tools import read_data


def add_joint_red(name, joint):
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


def add_joint_blue(name, joint):
    # print(name)
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
    #    gui.addXYZaxis(name+"/axis", Color.blue, 0.01, 0.1)


def display(q):
    robot.display(q)
    for name, oMi in zip(model.names[1:], robot.viz.data.oMi[1:]):
        gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(oMi))
    gui.refresh()



def add_axis_to_frame(data, model, frame):
    gui.createGroup(frame)
    gui.addToGroup(frame, "world")
    placement = pin.SE3ToXYZQUATtuple(data.oMf[model.getFrameId(frame)])
    axis_name = frame + "/axis"
    # print(axis_name, placement)
    gui.addXYZaxis(axis_name, Color.blue, 0.02, 0.2)
    gui.applyConfiguration(axis_name, placement)

robot = Robot(
    "talos_data/robots",
    "talos_reduced_with_desk.urdf",
)

data = robot.model.createData()
robot.initViewer(loadModel=True)

model = robot.model
gui = robot.viewer.gui

# draw joint axis
for name, joint in zip(model.names[1:], model.joints[1:]):
    if  name.find('gripper')==-1 and name.find('head')==-1  and name.find('arm_right_6_joint')==-1 and name.find('arm_right_7_joint')==-1  and name.find('arm_left_6_joint')==-1 and name.find('arm_left_7_joint')==-1:
        add_joint_red(name, joint)
        # print(name)
        # print(name.find('head'))
    elif name.find('arm_right_6_joint')==0 or name.find('arm_right_7_joint')==0 or name.find('arm_left_6_joint')==0 or name.find('arm_left_7_joint')==0: 
        add_joint_blue(name, joint)

# draw kinematic tree
for jid in range(2,model.njoints):
    name, joint = model.names[jid], model.joints[jid]
    pid = model.parents[jid]
    pname, pjoint = model.names[pid], model.joints[pid]
    if pid == 1: print(name, pname)
    pMi = model.jointPlacements[jid]
    if name.find('head')==-1:
        gui.addLine(pname + "/to_" + name, [0,0,0], pMi.translation.tolist(), Color.blue)

#q = np.array([ 0.0907,0.136,-0.493,1.08,-0.545,0.0676,0.0887,0.133,-0.416,0.862,-0.425,0.0736,0,0.00676,0.258,0.573,-0.0002,-1.36,0,0,0.1,0,-0.258,-0.573,0.0002,-1.36,0,0,0.1,0,0,0 ])
# q = np.array([ 0,0,-0.4114 ,0.8594,-0.4480,-0.0017,0,0,-0.4114,0.8594,-0.4480,-0.0017 ,0,0.00676,0.258,0.573,-0.0002,-1.36,0,0,0.1,0,-0.258,-0.573,0.0002,-1.36,0,0,0.1,0,0,0 ])



data_path = '/home/thanhndv212/Cooking/raw_data/Calibration/Talos/contact-calibration/complied_csv/'
file_name = 'compiled_measurements_right_1107.csv'
path_q = data_path + file_name
import time
q_r = read_data(model, path_q)

q = q_r[20, :]
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
add_axis_to_frame(data, model, "gripper_right_fingertip_1_link")
add_axis_to_frame(data, model, "right_sole_link")
# add_axis_to_frame(data, model, "gripper_left_fingertip_1_link")
# add_axis_to_frame(data, model, "left_sole_link")
add_axis_to_frame(data, model, "desk_link_1")
 
# contact points placements
z_contact = 0.
x_contact = [-0.32 + 0.075*i for i in range(6)]
y_contact = [-0.4 + 0.2*i for i in range(6)]
contacts = []
contact_names = []
count = 0
for x_c in x_contact:
    for y_c in y_contact:
        count += 1
        contact = [x_c, y_c, z_contact]
        contacts.append(contact)
        contact_names.append("contact_%s" % count)
rm_contact_rightarm = np.array([6, 12, 18, 19, 23, 24,
                       25, 26, 27, 28, 29, 30,
                       31, 32, 33, 34, 35, 36]) - 1
# add to gui
place_frame = "desk_link_1"
plcmt = data.oMf[model.getFrameId(place_frame)].translation
rot_cntct = data.oMf[model.getFrameId(place_frame)].rotation
rot_0 = [0,0,0,1]
gui.createGroup("contact")
gui.addToGroup("contact", "world")
for ij in range(len(contacts)):
    if ij not in rm_contact_rightarm.tolist():
        name = "contact/" + contact_names[ij]
        cntct_plcmnt = plcmt + rot_cntct.dot(np.array(contacts[ij]))
        gui.deleteNode(name, True)
        gui.addSphere(name, 0.01, Color.yellow)
        gui.applyConfiguration(name, cntct_plcmnt.tolist() + rot_0)
        gui.refresh()
display(q)

gui.setFloatProperty("world/pinocchio/visuals", "Alpha", 0.7)
# gui.setBackgroundColor1('python-pinocchio',[1.,1,1,1])
# gui.setBackgroundColor2('python-pinocchio',[1.,1,1,1])



view=[4,0.0,0.0]

r = R.from_euler('zyx', [0,90,0], degrees=True)
R=np.matrix('0 1 0; 1 0 0; 0 0 1') 
quat=pin.Quaternion(r.as_matrix()).coeffs().tolist()
view[len(view):] = quat
#print(view)

gui.setCameraTransform('python-pinocchio',view)


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
