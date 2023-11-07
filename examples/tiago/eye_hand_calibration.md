
### 1. Check Eye-Hand

- Open **terminal 1** on machine , ssh to the robot (replace `tiago-48c` with your robot), i.e:

```bash
ssh pal@tiago-48c
```

- Make the offer motion to raise the arm in front of camera, stop head moving. 
```bash
rosrun play_motion run_motion offer
pal-stop head_manager
```

- Check calibration motion by making the robot move to a specific position.
```bash
rosparam load `rospack find pal_eye_hand_calibration_tiago`/config/tiago_calibration_motions.yaml
rosrun play_motion run_motion check_calibration
```

- Open **terminal 2** on machine, open RVIZ to check the visuality of camera.

```bash
rosrun rviz rviz -d `rospack find tiago_bringup`/config/tiago.rviz
```

### 2. Mounting chessing

Put the chessboard on the end-effector. Remember to put the screws from the bottom side.
If you’re using a Hey5 end effector, it goes like this:
**![](https://lh7-us.googleusercontent.com/NSRLWOhGsI2W6rZRJZfL_RYdtW5j2xkkQgUDrVisz55kVxyvfnJkBpDtgyn9Isv5u7BJemmazaJixmFrfprly2d5S6dbV0yjt2j1ExbTl1GNG5t1AWdIxJolqTxU4fsx39DsaL330wMIJ7RFQFuLcA)**
**![](https://lh7-us.googleusercontent.com/o3PxXb4vHfw45lKAfMgEJMP3HPF_s1-gclHpz_kwK-wb5LQOGL_aurdZBinGQGKZLas6YmEpS_Otnw3Ze5aBihe022uo-f1GkVoLe7A1bw9vI4BhcDAxFN5zzVBWlE2HYROOukGGIn7YJNwZM5cpJw)**
**![](https://lh7-us.googleusercontent.com/YEkvcQIE4pnGNCi60o0ge003UwA4cxeSGVlRzK96Qy0K8Ii0aEMgKQ-qpjcjhULczc1sCrBTNCHwKNKK6lUv_0_46FHkZGjzDirz7EOU54HH6npPgfTTskKtAJSC00bUio4BKnIknIsMSER30zr2Ug)**

### 3. Launch Eye-hand

On **terminal 1**:
Copy configuration file from FIGAROH:

```bash
roslaunch pal_eye_hand_calibration_tiago tiago_calibration_recorder.launch recorder_name:=test_1 end_effector:=hey5 base_type:=pmb2 joint_configurations_file:=/home/pal/deployed_ws/share/pal_eye_hand_calibration_tiago/config/tiago_optimal_configurations.yaml
```

Note: If `tiago_optimal_configurations.yaml` is not in `pal_eye_hand_calibration_tiago/config/`, then copy `figaroh/examples/tiago/data/tiago_optimal_configurations.yaml` to `pal_eye_hand_calibration_tiago/config/tiago_optimal_configuraitons.yaml`.

Keep monitoring on rviz, if the projected image of the hand is over-layed correctly with the actual image of the hand. On tiago-48c, the  `arm_5_joint_absolute_encoder` might give incorrect values by a large margin like this.


### 4. Convert serialized data to csv file:
On **terminal 1** on robot:
```bash
roslaunch pal_eye_hand_calibration_tiago tiago_convert_to_csv.launch recorder_name:=test_1 end_effector:=hey5 filename:=/home/pal/.ros/test_1/eye_hand_calibration_recorded_data
```
#### Copy csv file from robot to the machine:

Open **terminal 3**  on machine (replace `tiago-48c` with your robot):
```bash
cd {path_to_figaroh}/figaroh/examples/tiago/data
scp tiago-48c:/home/pal/.ros/test_1/eye_hand_calibration_recorded_data.csv .
```

### 5. Run figaroh:

- Check figaroh/examples/tiago/config/tiago_config.yaml to see all arguments have been defined correctly, i.e. data path.
- Run the calibration on **terminal 3** in the directory`figaroh/examples/tiago/` :
```bash
cd ..
python3 calibration.py
```


- Copy calibrated parameters from figaroh/examples/tiago/data/calibrated_parameters/tiago_master_calibration.yaml to master_calibration.yaml
    

Expected output, for examples:

Input data: /data/eye_hand_calibration_recorded_data_401.csv

- Stored in /data/calibration_parameters/: An output file containing calibrated parameters tiago_master_calibration_401.yaml
- Printed out on screen: root-mean-square error and mean-absolute error


	`position root-mean-squared error of end-effector: 0.0074516538959767`
	`position mean absolute error of end-effector: 0.005621146867707006`
- Stored in /data/media/: Figures of absolute distance errors expressed in camera frame. For example, you could see most of the distance errors are less than 2 cm, which is acceptable given the condition of the robot (back lashes at joints, camera calibration accuracy). The algorithm has a threshold to consider any sample which has error larger than [x] cm to be likely an outlier (in this example, it is 5 cm and there is no outlier), and then remove and re-run the algorithm. It is a good practice to check and justify these errors before copying the calibration parameters file.

Eye-Hand doesn't work properly:
If the eye-hand fails and the parameters does not correspond with the real view on rviz:

You can check again the hard limits of the hand:

- Right Arm limit Estimator Check  
- Left Arm limit Estimator Check  

If it did not change anything, delete the eye-hand configuration of the master calibration.

Reboot the robot

Do the hard limits check again of both arms so you have good initial parameters and do the eye hand calibration again.

If it still fails: please calibrate the cameras again and start the eye-hand process with the new calibration.