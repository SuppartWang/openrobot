"""Tests for the minimal MuJoCo pick-and-place demo."""

import os
import sys
import shutil
import tempfile

import pytest

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(os.path.dirname(project_root), "openrobot_core"))

from examples import sim_pick_place


class TestSimPickPlace:
    def test_load_model(self):
        model, data = sim_pick_place.load_model()
        assert model is not None
        assert data is not None
        assert model.nq == 16

    def test_ik_solves_for_home_pose(self):
        model, data = sim_pick_place.load_model()
        qpos_adr = sim_pick_place.get_arm_joints(model, data)
        solver = sim_pick_place.FrankaMujocoKinematics(model, data)
        home_joints = [0.0, 1.57, 0.0, 0.3, 0.0, 0.0, 0.0]
        pose = solver.forward_quat(home_joints)
        assert len(pose) == 7
        result = solver.inverse_quat(pose, home_joints)
        assert result is not None

    def test_demo_runs_successfully(self):
        # Use a temporary directory for output so the test is hermetic.
        out_dir = tempfile.mkdtemp(prefix="openrobot_test_")
        try:
            success = sim_pick_place.run(out_dir=out_dir)
            assert success, "Pick-and-place demo did not complete successfully"
            assert os.path.isdir(out_dir)
            frames = [f for f in os.listdir(out_dir) if f.endswith(".png")]
            assert len(frames) > 0, "No frames were saved"
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
