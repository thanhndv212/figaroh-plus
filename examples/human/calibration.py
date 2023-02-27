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
import eigenpy
import hppfcl
import pinocchio as pin
# import pinocchio.casadi
import sys
from human_calibration_tools import (
    Robot,
    place,
    make_markers_dict_notime,
    compute_joint_centers,
    scale_human_model_mocap,
    calibrate_human_model_mocap,
    get_local_markers,
    markers_local_for_df,
    mean_local_markers,
    add_markers_frames)
import pandas as pd
from pinocchio.visualize import GepettoVisualizer

# Loading the model

robot = Robot('models/others/robots/human_description/urdf/human.urdf','models/others/robots',isFext=True)

model=robot.model
data=robot.data

# Reading data

list_calib=[]
Pos_markers=[]
markers_name=['LFHD','RFHD','LBHD','RBHD','C7','T10','CLAV','STRN','RBAK','LSHO','LELB','LWRA','LWRB','LFIN','RSHO','RELB','RWRA','RWRB','RFIN','LASI','RASI','LPSI','RPSI','LTHI','LKNE','LTIB','LANK','LHEE','LTOE','RTHI','RKNE','RTIB','RANK','RHEE','RTOE']

with open('examples/human/data/Calib01.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if 'time' in row[0]:
            print('First')
        else:
            new_row=row[0].split(',')
            for ii in range(len(markers_name)):
                Pos_markers.append(1e-3*np.array([float(new_row[3*ii+1]),float(new_row[3*ii+2]),float(new_row[3*ii+3])]))
            list_calib.append(Pos_markers)
            Pos_markers=[]

# Reading static pose csv and calibrate over all samples

q_calib=[]
dataframe=[]
joints_centers_list=[]

for ii in range(len(list_calib)):
    Position_markers_calib=list_calib[ii]
    
    joints_width=[0.1075/2,0.1075/2,0.0925/2,0.0925/2,0.0625/2,0.0625/2,0.0725/2,0.0725/2] #KNEE ANKLE ELBOW WRIST

    dictio=make_markers_dict_notime(Position_markers_calib,markers_name,joints_width)

    joints_centers= compute_joint_centers(dictio)
    joints_centers_list.append(joints_centers)

    #Change body segments sizes (example, the left lowerleg)
    model=scale_human_model_mocap(model,joints_centers)

    q_tPose=np.zeros((43,))
    q_tPose[20]=-np.pi/2
    q_tPose[31]=-np.pi/2
            
    q0=calibrate_human_model_mocap(robot,joints_centers,q_tPose)
    q_calib.append(q0)
    markers_local=get_local_markers(model,data,q0,dictio)
    markers_for_df=markers_local_for_df(markers_local)
    dataframe.append(markers_for_df)

# Statistical study over all samples (maybe not necessary but done still)

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
markers_local=mean_local_markers(mean)
print(markers_local)

# Adding markers to the pin model

data,i_markers=add_markers_frames(model,data,markers_local)

#DISPLAYS THE MARKERS
viz = GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)

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

viz.display(pin.neutral(model))
pin.forwardKinematics(model,data,pin.neutral(model))
pin.updateFramePlacements(model,data)

for ii in range(len(i_markers)):
        Mposition_goal_temp = data.oMf[i_markers[ii]]
        viz.viewer.gui.addXYZaxis('world/markersframe'+str(ii), [0, 0., 1., 1.], 0.0115, 0.15)
        place(viz,'world/markersframe'+str(ii), Mposition_goal_temp)

