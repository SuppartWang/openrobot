import numpy as np
from openrobot_control.motion_planning.interpolator import JointSpaceInterpolator, GripperTrajectory


def test_joint_interpolator_length():
    interp = JointSpaceInterpolator(num_steps=10)
    current = np.zeros(3)
    target = np.ones(3)
    traj = interp.plan(current, target)
    assert len(traj) == 11
    assert np.allclose(traj[0].values, current)
    assert np.allclose(traj[-1].values, target)


def test_gripper_trajectory():
    gt = GripperTrajectory(num_steps=5)
    traj = gt.plan(0.02)
    assert len(traj) == 5
    assert traj[0].type == "gripper"
    assert np.allclose(traj[0].values, np.array([0.02, 0.02]))
