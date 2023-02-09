import pinocchio as pin
import numpy as np
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import build_regressor_basic, get_index_eliminate, build_regressor_reduced
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import get_param_from_yaml,calculate_first_second_order_differentiation, calculate_standard_parameters, low_pass_filter_data
import matplotlib.pyplot as plt 
import pprint
import csv
import yaml
from yaml.loader import SafeLoader

robot = Robot('models/others/robots/human_description/urdf/human.urdf','models/others/robots',isFext=True) 
model = robot.model
data = robot.data

with open('examples/human/config/human_config.yaml', 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)
    pprint.pprint(config)
    
identif_data = config['identification']
params_settings = get_param_from_yaml(robot, identif_data)
print(params_settings)


# Print out the placement of each joint of the kinematic tree
print("\nJoint placements:")
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format( name, *oMi.translation.T.flat )))

# generate a list containing the full set of standard parameters
params_standard = robot.get_standard_parameters(params_settings)

# 1. First we build the structural base identification model, i.e. the one that can
# be observed, using random samples

q_rand = np.random.uniform(low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nq))

dq_rand = np.random.uniform(
     low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nv)
)

ddq_rand = np.random.uniform(
    low=-6, high=6, size=(10 * params_settings["nb_samples"], model.nv)
)

W = build_regressor_basic(robot, q_rand, dq_rand, ddq_rand, params_settings)

# remove zero cols and build a zero columns free regressor matrix
idx_e, params_r = get_index_eliminate(W, params_standard, 1e-6)
W_e = build_regressor_reduced(W, idx_e)

# Calulate the base regressor matrix, the base regroupings equations params_base and
# get the idx_base, ie. the index of base parameters in the initial regressor matrix
_, params_base, idx_base = get_baseParams(W_e, params_r, params_standard)

print("The structural base parameters are: ")
for ii in range(len(params_base)):
    print(params_base[ii])

# gather data from .csv

q_nofilt=[]
q_ii=[]
Fx=[]
Fy=[]
Fz=[]
Mx=[]
My=[]
Mz=[]

with open('examples/human/data/q+Forces_sampled_30Hz_filtered.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        if 'q_0' in row[0]:
            print('First')
        else:
            new_row=row[0].split(',')
            for ii in range(model.nq):
                q_ii.append(float(new_row[ii]))
            q_nofilt.append(np.array(q_ii))
            F = pin.Force(np.array([float(new_row[-6]),float(new_row[-5]),float(new_row[-4]),float(new_row[-3]),float(new_row[-2]),float(new_row[-1])]))
            Fx.append(F.linear[0])
            Fy.append(F.linear[1])
            Fz.append(F.linear[2])
            Mx.append(F.angular[0])
            My.append(F.angular[1])
            Mz.append(F.angular[2])
            # /!\ FORCES ARE EXPRESSED IN GLOBAL FRAME AND TORQUES EXPRESSED WRT PLATFORM ORIGIN. THEY ARE FILTERED AT 5HZ.
            q_ii = []

# Filtering the q at 10 Hz

params_settings['cut_off_frequency_butterworth']=10

q_nofilt=np.array(q_nofilt)

for ii in range(model.nq):
    if ii == 0:
        q= low_pass_filter_data(q_nofilt[:,ii], params_settings,5)
    else:
        q = np.column_stack((q,low_pass_filter_data(q_nofilt[:,ii], params_settings,5)))

q, dq, ddq = calculate_first_second_order_differentiation(model, q, params_settings)

W = build_regressor_basic(robot, q, dq, ddq, params_settings)
# select only the columns of the regressor corresponding to the structural base
# parameters
W_base = W[:, idx_base]
print("When using all trajectories the cond num is", int(np.linalg.cond(W_base)))

# simulation of the measured joint torques
Fx_simu=[]
Fy_simu=[]
Fz_simu=[]
Mx_simu=[]
My_simu=[]
Mz_simu=[]

nb_samples = len(q)

# simulation of the measured joint torques
tau_simu = np.empty(nb_samples*6)

for ii in range(nb_samples):
    pin.forwardKinematics(model,data,q[ii,:])
    pin.updateFramePlacements(model,data)
    M_pelvis = data.oMi[model.getJointId('root_joint')]
    tau_temp = pin.Force(np.array(pin.rnea(model, data, q[ii,:], dq[ii,:], ddq[ii,:])[:6]))#force de réaction
    tau_temp = tau_temp.se3Action(M_pelvis) 
    Fx_simu.append(tau_temp.linear[0])
    Fy_simu.append(tau_temp.linear[1])
    Fz_simu.append(tau_temp.linear[2])
    Mx_simu.append(tau_temp.angular[0])
    My_simu.append(tau_temp.angular[1])
    Mz_simu.append(tau_temp.angular[2])
    for j in range(6):
        tau_temp_sub = np.array([tau_temp.linear[0],tau_temp.linear[1],tau_temp.linear[2],tau_temp.angular[0],tau_temp.angular[1],tau_temp.angular[2]])
        tau_simu[j * nb_samples + ii] = tau_temp_sub[j]

# removing samples due to filtering and differentiation
Fx_filtd=Fx[25:-27]
Fy_filtd=Fy[25:-27]
Fz_filtd=Fz[25:-27]
Mx_filtd=Mx[25:-27]
My_filtd=My[25:-27]
Mz_filtd=Mz[25:-27]

tau_meas=np.array(Fx_filtd+Fy_filtd+Fz_filtd+Mx_filtd+My_filtd+Mz_filtd)

#Forces transported

tau_meast = np.empty(nb_samples*6)

for ii in range(nb_samples):
    pin.forwardKinematics(model,data,q[ii,:])
    pin.updateFramePlacements(model,data)
    M_pelvis = data.oMi[model.getJointId('root_joint')]
    tau_temp = pin.Force(np.array([Fx_filtd[ii],Fy_filtd[ii],Fz_filtd[ii],Mx_filtd[ii],My_filtd[ii],Mz_filtd[ii]]))
    tau_temp = tau_temp.se3ActionInverse(M_pelvis)
    for j in range(6):
        tau_temp_sub = np.array([tau_temp.linear[0],tau_temp.linear[1],tau_temp.linear[2],tau_temp.angular[0],tau_temp.angular[1],tau_temp.angular[2]])
        tau_meast[j * nb_samples + ii] = tau_temp_sub[j]

# Least-square identification process
phi_base = np.matmul(np.linalg.pinv(W_base), tau_meast)

tau_identif = W_base@phi_base

#Forces visualisation

Fx_identif=[]
Fy_identif=[]
Fz_identif=[]
Mx_identif=[]
My_identif=[]
Mz_identif=[]


for ii in range(int(len(tau_identif)/6)):
    Fx_identif.append(tau_identif[ii])
    Fy_identif.append(tau_identif[int(1*(len(tau_identif)/6))+ii])
    Fz_identif.append(tau_identif[int(2*(len(tau_identif)/6))+ii])
    Mx_identif.append(tau_identif[int(3*(len(tau_identif)/6))+ii])
    My_identif.append(tau_identif[int(4*(len(tau_identif)/6))+ii])
    Mz_identif.append(tau_identif[int(5*(len(tau_identif)/6))+ii])

tau_identif_proj = np.empty(nb_samples*6)

for ii in range(nb_samples):
    pin.forwardKinematics(model,data,q[ii,:])
    pin.updateFramePlacements(model,data)
    M_pelvis = data.oMi[model.getJointId('root_joint')]
    tau_temp = pin.Force(np.array([Fx_identif[ii],Fy_identif[ii],Fz_identif[ii],Mx_identif[ii],My_identif[ii],Mz_identif[ii]]))#force de réaction
    tau_temp = tau_temp.se3Action(M_pelvis) 
    for j in range(6):
        tau_temp_sub = np.array([tau_temp.linear[0],tau_temp.linear[1],tau_temp.linear[2],tau_temp.angular[0],tau_temp.angular[1],tau_temp.angular[2]])
        tau_identif_proj[j * nb_samples + ii] = tau_temp_sub[j]

Fx_identif_proj=[]
Fy_identif_proj=[]
Fz_identif_proj=[]
Mx_identif_proj=[]
My_identif_proj=[]
Mz_identif_proj=[]

for ii in range(int(len(tau_identif_proj)/6)):
    Fx_identif_proj.append(tau_identif_proj[ii])
    Fy_identif_proj.append(tau_identif_proj[int(1*(len(tau_identif_proj)/6))+ii])
    Fz_identif_proj.append(tau_identif_proj[int(2*(len(tau_identif_proj)/6))+ii])
    Mx_identif_proj.append(tau_identif_proj[int(3*(len(tau_identif_proj)/6))+ii])
    My_identif_proj.append(tau_identif_proj[int(4*(len(tau_identif_proj)/6))+ii])
    Mz_identif_proj.append(tau_identif_proj[int(5*(len(tau_identif_proj)/6))+ii])

plt.plot(tau_meas,'r',label='measured')
plt.plot(tau_simu,'b',label='Simu with AT')
plt.plot(tau_identif_proj,'g',label='identified')
plt.legend()
plt.title('Overall wrench (measured, simulated with AT and identified')
plt.show()

plt.plot(Fx_filtd,'r',label='measures')
plt.plot(Fx_simu,'b',label='AT')
plt.plot(Fx_identif_proj,'g',label='identified')
plt.legend()
plt.grid()
plt.title('Fx (N)')
plt.show()

plt.plot(Fy_filtd,'r',label='measures')
plt.plot(Fy_simu,'b',label='AT')
plt.plot(Fy_identif_proj,'g',label='identified')
plt.legend()
plt.grid()
plt.title('Fy (N)')
plt.show()

plt.plot(Fz_filtd,'r',label='measures')
plt.plot(Fz_simu,'b',label='AT')
plt.plot(Fz_identif_proj,'g',label='identified')
plt.legend()
plt.grid()
plt.title('Fz (N)')
plt.show()

plt.plot(Mx_filtd,'r',label='measures')
plt.plot(Mx_simu,'b',label='AT')
plt.plot(Mx_identif_proj,'g',label='identified')
plt.legend()
plt.grid()
plt.title('Mx (N.m) ')
plt.show()

plt.plot(My_filtd,'r',label='measures')
plt.plot(My_simu,'b',label='AT')
plt.plot(My_identif_proj,'g',label='identified')
plt.legend()
plt.grid()
plt.title('My (N.m) ')
plt.show()

plt.plot(Mz_filtd,'r',label='measures')
plt.plot(Mz_simu,'b',label='AT')
plt.plot(Mz_identif_proj,'g',label='identified')
plt.legend()
plt.grid()
plt.title('Mz (N.m) ')
plt.show()

id_inertias=[]
id_virtual=[]

for jj in range(1,len(robot.model.inertias.tolist())):
    if robot.model.inertias.tolist()[jj].mass !=0:
        id_inertias.append(jj-1)
    else:
        id_virtual.append(jj-1)

COM_max = np.ones((1,3*len(id_inertias))) # subject to be more adaptated
COM_min = -np.ones((1,3*len(id_inertias))) # subject to be more adaptated


phi_standard, phi_ref = calculate_standard_parameters(robot, W, tau_meas, COM_max[0], COM_min[0], params_settings)

print(phi_standard)
print(phi_ref)

# # TODO : LOOK AT SIP AND HOW TO HANDLE VIRTUAL INERTIAS, modify the model with SIP ?