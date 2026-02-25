# Sim-to-real transfer

## Goal

This simulator is currently used to train RL policies in simulation to learn how to achieve two related goals:

1. Race around a track, using drifting as needed
2. Recover from an out-of-control situation

Several policies have been trained successfully in simulation. The next task is to test the learned policies on a real 1/10 scale F1TENTH autonomous vehicle. This will determine whether the policies are capable of accurate sim-to-real transfer. In particular, I am only testing the learned recovery policies in real life.

## Real-life setup

To train or test the recovery in the simulation, a vehicle is initialized on a straight-line segment of the IMS track with an initial state that is out-of-control, created using combinations of velocity (v), yaw (heading error), beta (sideslip), and r (yaw rate) values. The controller then learns how to use steering and acceleration inputs to bring the vehicle back into control, as defined by `F1TENTH_Gym/f1tenth_gym/envs/f110_env.py:1067`

```py
def _check_recovery_success(self) -> bool:
```

If recovery is unsuccessful after reaching a point on the track, or the car exits the lateral bounds, the episode ends.

To port this solution to the real car, some modifications need to be made, and some features must be implemented. The real car receives control inputs through a set of ROS drivers and nodes described in `F1TENTH_SYSTEM.md`. Basically, it requires `AckermannDriveStamped` messages to be published to `/drive` topic. The full system exists in another repository, `https://github.com/TeoIlie/F1TENTH_System.git`. This topic expects a target velocity and target steering angle output. The output from the RL model is of type (1) target steering angle and (2) acceleration. Therefore, when using the model to output the action, the acceleration must be integrated over time, using a known discreet timestep. The steering angle can be passed as-is.

The learned model needs to be passed the correct inputs in order to output the drive command. This is a an observation vector of type "drift", described in `F1TENTH_Gym/f1tenth_gym/envs/observation.py:878`, which includes attributes:

```python
features = [
    "linear_vel_x",  # vx - longitudinal velocity, vehicle frame
    "linear_vel_y",  # vy - lateral velocity, vehicle frame
    "frenet_u",  # u - angle between car heading, track heading, in Frenet coords
    "frenet_n",  # n - lateral distance from centerline, in Frenet coords
    "ang_vel_z",  # r - yaw rate
    "delta",  # δ - measured steering angle
    "beta",  # β - slip angle (vehicle velocity angle relative to body axis)
    "prev_steering_cmd",  # δ_ref - previous commanded steering angle
    "prev_accl_cmd",  # ω_dot_ref - last control input (acceleration)
    "prev_avg_wheel_omega",  # ω - previous measured wheel speed
    "curr_vel_cmd",  # ω_ref - current commanded velocity (integrated from acceleration)
    "lookahead_curvatures",  # c - track curvatures
    "lookahead_widths",  # w - track widths
]
```

Unlike the simulator, which already computes all these values internally at each timestep, the implementation in real life will need to calculate these using information from the vehicle drive topic, and from a Vicon motion capture system which outputs pose/state information. Track-specific attributes - lookahead curvatures, lookahead widths, frenet_u, frenet_n - must be calculated, possibly by implementing a virtual track trajectory. It is important to note that not all topics will be processing and outputting at the same Hz rate. Furthermore, unlike the simulator, the real car will not wait for the model to perform inference and output a drive command. This needs to be taken into account.

### Initial state and safety boundaries in real-life

Unlike the simulator, creating the inital out-of-control state is not as straightforward as simply intializing a vehicle with a given pose/state. The proposed solution is to manually drive at high speed from outside the Vicon motion-capture space, heading towards it, and perform a sudden cornering manoeuver on a slippery strip of plastic as it enters the motion capture space. This will generate the desired heading error, beta, and r values.

As soon as the car passes a threshold, and enters the motion capture space, the autonomous recovery controller takes over. It combines throttle, braking, and steering to attempt to regain control of the car, and continue travelling in a straight line.

If the policy is successful in regaining control, it will still be travelling forwards. To avoid collision, it must apply full braking after another threshold line. Alternativley, if it is unsuccessful at regaining control and veers off to either side of the motion capture space, it also must apply full brakes. In other words, the coordinates from the motion capture space must be used to define a rectangular region inside the motion capture space where the recover controller is active, and outside of that, emergency braking is deployed.

### RL sim-to-real transfer possible challenges

It is possible that even given a correct setup, the vehicle will not be able to recover successfully as it did in simulation. This is called the sim-to-real gap, and is a common issue when training RL policies. Several solutions can be implemented if this is the case.

First, it may be useful to implement parameter randomization, particularly with respect to the tire/road friction parameters. Second, it may be necessary to perform system identification of the vehicle parameters, and particularly, of the tire parameters, which are currently set in `F1TENTH_Gym/f1tenth_gym/envs/f110_env.py:636`

```py
def f1tenth_std_vehicle_params(cls) -> dict:
```

Performing these two tasks should make the policy more generalizable, but first I will test the policy without these. I will only implement them if performance is unsatisfactory.

## Open questions

Several questions must be addressed to plan out this task:

1. How should the system be designed to enable communication? It is necessary to coordinate running the basic system driver nodes, pass information from the car and the ViCon to compute observations and pass it to the learned model, process that output and pass it to the vehicle. How can all of this communication be achieved in a clear, testable, efficient manner using ROS2?
2. How will the "virtual" centerline be implemented in real life, allowing ViCon state information to be pre-processed into observations before being fed into the model for inference?
3. How will the coordinate system from the ViCon be used to generate a virtual region inside which the autonomous control is active, and apply emergency braking outside of that region
4. In what order should the system be built up and tested to ensure all the components are working correctly?
