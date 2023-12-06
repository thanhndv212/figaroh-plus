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

import csv
import numpy as np
import pinocchio as pin
from figaroh.tools.robot import Robot
import sys
from human_calibration_tools import (
    place,
    make_markers_dict_notime,
    compute_joint_centers,
    compute_mean_joints_centers,
    scale_human_model_mocap,
    calibrate_human_model_mocap,
    get_local_markers,
    markers_local_for_df,
    mean_local_markers,
    add_plug_in_gait_markers)
import pandas as pd
from pinocchio.visualize import GepettoVisualizer

# Loading the model

robot_mocap = Robot('models/others/robots/human_description/urdf/human.urdf','models/others/robots',True,np.array([[1,0,0],[0,0,-1],[0,1,0]])) 
model_mocap = robot_mocap.model
data_mocap = robot_mocap.data

joints_width=[0.124 / 2,
            0.124 / 2,
            0.089 / 2,
            0.089 / 2,
            0.104 / 2,
            0.104 / 2,
            0.074 / 2,
            0.074 / 2,]  #KNEE ANKLE ELBOW WRIST

path_mocap_data = 'examples/human/data/Calib.csv'

with open(path_mocap_data, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    no_lines= len(list(spamreader))-1

markers_names = ["LFHD","RFHD","LBHD","RBHD","C7","T10","CLAV","STRN","RBAK","LSHO","LELB","LWRA","LWRB","LFIN","RSHO","RELB","RWRA","RWRB","RFIN","LASI","RASI","LPSI","RPSI","LTHI","LKNE","LTIB","LANK","LHEE","LTOE","RTHI","RKNE","RTIB","RANK","RHEE","RTOE"]

markers_trajectories = np.zeros((no_lines,3*len(markers_names)))

# Reading data

c=0

with open(path_mocap_data, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if 'time' in row[0]:
            print('First')
        else:
            new_row=row[0].split(',')
            for ii in range(1,3*len(markers_names)+1):
                markers_trajectories[c,ii-1]=float(new_row[ii])
            c+=1

# Reading static pose csv and calibrate over all samples

q_calib=[]
dataframe=[]
joints_centers_list=[]

q_tPose=0.0001*np.ones((model_mocap.nq,))

q_tPose[23]=np.pi/2
q_tPose[25]=0.1
q_tPose[26]=4*np.pi/9
q_tPose[34]=np.pi/2
q_tPose[36]=0.1
q_tPose[37]=4*np.pi/9

q_init = q_tPose

for ii in range(10):
    Position_markers_calib =np.zeros((len(markers_names),3))
    for jj in range(len(markers_names)):
        Position_markers_calib[jj,:]=markers_trajectories[ii,3*jj:3*(jj+1)]

    dictio=make_markers_dict_notime(Position_markers_calib,markers_names,joints_width)

    joints_centers= compute_joint_centers(dictio)
    joints_centers_list.append(joints_centers)      
        
    #Change body segments sizes
    model_mocap,data_mocap = scale_human_model_mocap(model_mocap,joints_centers)

    q0=calibrate_human_model_mocap(model_mocap,data_mocap,joints_centers,q_init)
    q_init = q0

    q_calib.append(q0)

    markers_local=get_local_markers(model_mocap,data_mocap,q0,dictio)

    markers_for_df=markers_local_for_df(markers_local)

    dataframe.append(markers_for_df)

df=pd.DataFrame(dataframe)

for ii in ['LFHDx','LFHDy','LFHDz','RFHDx','RFHDy','RFHDz','LBHDx','LBHDy','LBHDz','RBHDx','RBHDy','RBHDz','C7x','C7y','C7z','T10x','T10y','T10z','CLAVx','CLAVy','CLAVz','STRNx','STRNy','STRNz','RBAKx','RBAKy','RBAKz','LSHOx','LSHOy','LSHOz','LELBx','LELBy','LELBz','LWRAx','LWRAy','LWRAz','LWRBx','LWRBy','LWRBz','LFINx','LFINy','LFINz','RSHOx','RSHOy','RSHOz','RELBx','RELBy','RELBz','RWRAx','RWRAy','RWRAz','RWRBx','RWRBy','RWRBz','RFINx','RFINy','RFINz','LASIx','LASIy','LASIz','RASIx','RASIy','RASIz','LPSIx','LPSIy','LPSIz','RPSIx','RPSIy','RPSIz','LTHIx','LTHIy','LTHIz','LKNEx','LKNEy','LKNEz','LTIBx','LTIBy','LTIBz','LANKx','LANKy','LANKz','LHEEx','LHEEy','LHEEz','LTOEx','LTOEy','LTOEz','RTHIx','RTHIy','RTHIz','RKNEx','RKNEy','RKNEz','RTIBx','RTIBy','RTIBz','RANKx','RANKy','RANKz','RHEEx','RHEEy','RHEEz','RTOEx','RTOEy','RTOEz']:
    q1=df[ii].quantile(q=0.25)
    q3=df[ii].quantile(q=0.75)
    IQR=q3-q1
    borne_inf = q1-1.5*IQR
    borne_sup = q3 +1.5*IQR
    df= df[df[ii]<borne_sup]
    df=df[df[ii]>borne_inf]

mean=df.mean()
joints_centers_glo=compute_mean_joints_centers(joints_centers_list)

markers_local=mean_local_markers(mean)
model_mocap,data_mocap=scale_human_model_mocap(model_mocap,joints_centers_glo)
data_mocap,i_markers=add_plug_in_gait_markers(model_mocap,data_mocap,markers_local)

#DISPLAYS THE MARKERS
viz = GepettoVisualizer(robot_mocap.model, robot_mocap.collision_model, robot_mocap.visual_model)

try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)
 
try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)

viz.display(pin.neutral(model_mocap))
pin.forwardKinematics(model_mocap,data_mocap,pin.neutral(model_mocap))
pin.updateFramePlacements(model_mocap,data_mocap)

for ii in range(len(i_markers)):
        Mposition_goal_temp = data_mocap.oMf[i_markers[ii]]
        viz.viewer.gui.addXYZaxis('world/markersframe'+str(ii), [0, 0., 1., 1.], 0.0115, 0.15)
        place(viz,'world/markersframe'+str(ii), Mposition_goal_temp)

