# Simulation Guide

OpenRobotDemo uses [MuJoCo](https://mujoco.org/) as the default physics simulator.
The simulation stack is designed to mirror the real-robot control flow as closely
as possible, so code you write for simulation can be reused on hardware with
minimal changes.

## Simulation Model

The main simulation scene is defined in:

```text
sim/mujoco/franka_rgb_scene.xml
```

It contains:

- A simplified 7-DOF Franka Emika Panda arm.
- A parallel-jaw gripper with two sliding fingers.
- A wrist-mounted RGB camera (`wrist_cam`).
- A top-down reference camera (`top`).
- A table and a yellow cube target object.

## Key Design Decisions

### Direct Kinematic Control in Demos

The minimal demo (`examples/sim_pick_place.py`) uses **direct kinematic control**:
it solves IK for each target pose and sets the joint positions directly via
`data.qpos`. This avoids the complexity of low-level actuator tuning and makes
the demo fast and deterministic.

For physical realism (contacts, forces, object manipulation), use the full-stack
demo or implement a closed-loop controller that steps the physics simulation.

### Stable Physics Parameters

The scene uses:

- `timestep="0.001"` for stable integration with the position actuators.
- Position actuators with moderate `kp`/`kv` gains.
- Joint damping to prevent oscillations.

If you modify the model and see instability (NaN, flying objects), first try:

1. Reducing the timestep.
2. Reducing actuator `kp` gains.
3. Increasing joint damping.

### IK Solver

The demo uses a damped least-squares numerical IK solver implemented in
`openrobot_demo/hardware/mujoco_franka_adapter.py`. It supports random restarts
and returns a best-effort solution when exact convergence is not possible.

## Coordinate Frames

- The robot base is at `(0, 0, 0.4)`.
- The default home pose TCP position is approximately `(0.61, 0, 0.44)`.
- The cube starts at `(0.55, 0.15, 0.45)`.
- The gripper points down when its orientation quaternion is
  `[0, 0.707, 0, 0.707]` (scipy `[x, y, z, w]` format).

## Adding Custom Objects

To add a new object to the scene, edit `sim/mujoco/franka_rgb_scene.xml`:

```xml
<body name="my_object" pos="0.5 0.0 0.45">
  <freejoint/>
  <geom name="my_object_geom" type="box" size="0.02 0.02 0.02"
        rgba="0 1 0 1" mass="0.05"/>
</body>
```

Then update the `CUBE_POS` constant in your demo script or use the vision
pipeline to detect it at runtime.

## Running Headless

On macOS, the interactive MuJoCo viewer requires `mjpython`. If it is not
available, the demos automatically fall back to headless mode and save rendered
frames to disk.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `mujoco.viewer.launch_passive` error on macOS | `mjpython` not available | Use headless mode; demos auto-fallback. |
| IK failures | Target pose outside workspace | Use poses closer to the home pose. |
| Simulation explodes (NaN, flying objects) | Timestep too large or gains too high | Reduce timestep or actuator `kp`. |
| Cube moves on its own at startup | Arm collides with cube during settling | Move cube farther from arm or adjust keyframe. |
