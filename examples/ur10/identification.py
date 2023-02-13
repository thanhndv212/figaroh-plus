import pinocchio as pin
import numpy as np
from figaroh.tools.robot import Robot
from figaroh.tools.regressor import build_regressor_basic, get_index_eliminate, build_regressor_reduced
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.identification.identification_tools import get_param_from_yaml,calculate_first_second_order_differentiation, base_param_from_standard, calculate_standard_parameters
import matplotlib.pyplot as plt 
import pprint
import yaml
from yaml.loader import SafeLoader

robot = Robot('models/others/robots/ur_description/urdf/ur10_robot.urdf','models/others/robots',False)
model = robot.model
data = robot.data

with open('examples/ur10/config/ur10_config.yaml', 'r') as f:
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

# simulating a sinus trajectory on joints shoulder_lift_joint, elbow_joint, wrist_2_joint

delta_t=0.01
f1 = 1
f2 = 10
f3 = 0.1

samples=100
w1 = 2.*np.pi*f1
w2= 2*np.pi*f2
w3= 2*np.pi*f3

t = np.linspace(0, int(samples*delta_t), samples)

q=np.zeros((samples,model.nq))

for ii in range(samples):
    q[ii,0]= np.sin(w1*t[ii])
    q[ii,2]= np.sin(w2*t[ii])
    q[ii,4]= np.sin(w3*t[ii])
    # viz.display(q[ii,:])
    # time.sleep(0.5)

q, dq, ddq = calculate_first_second_order_differentiation(model, q, params_settings)

W = build_regressor_basic(robot, q, dq, ddq, params_settings)
# select only the columns of the regressor corresponding to the structural base
# parameters
W_base = W[:, idx_base]
print("When using all trajectories the cond num is", int(np.linalg.cond(W_base)))

# simulation of the measured joint torques
tau_simu = np.empty(len(q)*model.nq)

for ii in range(len(q)):
    tau_temp = pin.rnea(model, data, q[ii, :], dq[ii, :], ddq[ii, :])
    for j in range(model.nq):
        tau_simu[j * len(q) + ii] = tau_temp[j]

# Noise to add to the measure to make them more realistic
noise = np.random.normal(0,10,len(tau_simu))

tau_noised = tau_simu+noise

# Least-square identification process
phi_base = np.matmul(np.linalg.pinv(W_base), tau_noised)

phi_base_real = base_param_from_standard(params_standard,params_base) 

tau_identif = W_base@phi_base

plt.plot(tau_noised,label='simulated+noised')
plt.plot(tau_identif,label='identified')
plt.legend()
plt.show()

COM_max = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # subject to be more adaptated
COM_min = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] # subject to be more adaptated

phi_standard, phi_ref = calculate_standard_parameters(robot, W, tau_noised, COM_max, COM_min, params_settings)

print(phi_standard)
print(phi_ref)

plt.plot(phi_standard,label='SIP Identified')
plt.plot(phi_ref,label='SIP URDF')
plt.legend()
plt.show()

# TODO : adapt constraints on COM, verify SIP, modify the model with SIP ?