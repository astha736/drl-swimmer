# Obstacle based locomotion 

**Preview file**: ctrl+shift+V

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

- Packages you need to install additionally:
	- gym@0.21.0
	- sb3-contrib@1.8.0
	- stable-baselines3@1.8.0
	- tensorboard@2.13.0

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


