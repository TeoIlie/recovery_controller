# Recovery_controller
ROS2 package enabling sim-to-real transfer of learned recovery policies

## Usage 
1. Set the controller type in `/config/recovery.yaml` to "learned" for RL policy or "stanley" for Stanley controller
2. For RL policies
    1. Save the model learned model in onnx format to `/models` and update the path in `/config/recovery.yaml`
    2. Update normalization bounds if necessary in `/config/norm_bounds.yaml`

## Useful commands
To pipe output for a file for inspection use:
```bash
ros2 launch f1tenth_stack recovery_bringup_launch.py > src/F1TENTH_System/scratch/test1.txt 2>&1
```

## Testing
- Run tests with `python -m pytest test/ -v`
