# Obstacle based locomotion 

**Preview file**: ctrl+shift+V

## Experiment tracking

Date: 5 Jan 2023
Folder Date: 2022-12-22

|runName					|RandomPoseMotorPos | RandomOscPhase 	| sCaudalWeight | Range | fileName											|
|---------------------------|-------------------|-------------------|---------------|-------|---------------------------------------------------|
| obstacle_np_rPoseOsc_w0	|&#x2611;	**Y**	| &#x2611; **Y**	| 0.0			| 30	|sCaudalNP_Rall_Obs10_Act9_randomInitPoseOsc		|
| obstacle_np_rPoseOsc_w30	|&#x2611;	**Y**	| &#x2611; **Y**	| -30.0			| 30	|sCaudalNP_Rall_Obs10_Act9_randomInitPoseOsc_w30	|
| obstacle_np_rPose_w30		|&#x2611;	**Y**	| &#x2612; **N**	| -30.0			| 30	|sCaudalNP_Rall_Obs10_Act9_randomInitPose_w30		|
| obstacle_np_rPose_w0		|&#x2611;	**Y**	| &#x2612; **N**	| 0.0			| 30	|sCaudalNP_Rall_Obs10_Act9_randomInitPose			|
| obstacle_np_rPose_w0_r60	|&#x2611;	**Y**	| &#x2612; **N**	| 0.0			| 60	|sCaudalNP_Rall_Obs10_Act9_randomInitPose_range60	|

```
tensorboard --logdir_spec obstacle_np_rPoseOsc_w0:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_randomInitPoseOsc/.,obstacle_np_rPoseOsc_w30:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_randomInitPoseOsc_w30/.,obstacle_np_rPose_w30:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_randomInitPose_w30/.,obstacle_np_rPose_w0:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_randomInitPose/.,obstacle_np_rPose_w0_r60:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_randomInitPose_range60/.
```


## AgnathaX

- Robot AgnathaX 


## Run Instruction

- Create python venv with `python3 -m venv obstacle`. Name of the venv is `obstacle`
- Install farms with `cd farms` & `python3 farms-install.py`
- Generate the config files `cd config`, `python3 farms-config-gen.py`
- Change the settings of vscode if using vscode 

```
	"settings": {
		"python.defaultInterpreterPath": "$PATH_TO_HIGH_LEVEL_DIR/obstacle/bin/python",
		"terminal.integrated.cwd": "$PATH_TO_HIGH_LEVEL_DIR"
	}
}
```

## Folders
- farms: Where farms repo are clone and installed 
- config: Where AganathaX configurations files are stored & generated for the system 
- logs: log folder 
- models: SDF model for environment 
- scripts: Scripts for the project


## Tensorboard use 

```
tensorboard --logdir_spec <run_name_id>:<exact_path_tensorbord_data>/.,<run_name_id>:<exact_path_tensorbord_data>/.
```

for instance:
```
tensorboard --logdir_spec obstacle_sin_p:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_ideal_P/.,obstacle_np:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_ideal_P_correct/.
```

```tensorboard --logdir_spec obstacle_np_rd_esp:/data/asgupta/Projects/rl-obstacle-based-locomotion/logs/2022-12-22/sCaudalNP_Rall_Obs10_Act9_randomInitPoseOsc/.
```


