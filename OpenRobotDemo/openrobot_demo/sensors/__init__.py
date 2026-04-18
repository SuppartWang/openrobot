"""Sensor channels for OpenRobotDemo."""

from openrobot_demo.sensors.base import SensorChannel, PerceptionData
from openrobot_demo.sensors.vision_rgb import VisionRGBSensor
from openrobot_demo.sensors.vision_depth import VisionDepthSensor
from openrobot_demo.sensors.pointcloud import PointCloudSensor
from openrobot_demo.sensors.proprioception import ProprioceptionSensor
from openrobot_demo.sensors.tactile import TactileSensor
from openrobot_demo.sensors.realsense_rgb import RealSenseRGBSensor
from openrobot_demo.sensors.realsense_depth import RealSenseDepthSensor
from openrobot_demo.sensors.realsense_vlm import RealSenseVLMSensor
from openrobot_demo.sensors.imu import IMUSensor
from openrobot_demo.sensors.wrench import WrenchSensor
from openrobot_demo.sensors.audio import AudioSensor
from openrobot_demo.sensors.lidar import LidarSensor
from openrobot_demo.sensors.ultrasonic import UltrasonicSensor
from openrobot_demo.sensors.odometry import OdometrySensor
from openrobot_demo.sensors.registry import (
    register_sensor,
    create_sensor,
    list_sensors,
)

__all__ = [
    # Base
    "SensorChannel",
    "PerceptionData",
    # Vision
    "VisionRGBSensor",
    "VisionDepthSensor",
    "PointCloudSensor",
    # Proprioception / Tactile
    "ProprioceptionSensor",
    "TactileSensor",
    # RealSense
    "RealSenseRGBSensor",
    "RealSenseDepthSensor",
    "RealSenseVLMSensor",
    # New sensors (P1-2)
    "IMUSensor",
    "WrenchSensor",
    "AudioSensor",
    "LidarSensor",
    "UltrasonicSensor",
    "OdometrySensor",
    # Registry
    "register_sensor",
    "create_sensor",
    "list_sensors",
]
