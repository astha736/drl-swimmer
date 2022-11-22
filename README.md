# Obstacle based locomotion 

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
