"""Sensor channels for OpenRobotDemo."""

from .base import SensorChannel, PerceptionData
from .registry import register_sensor, create_sensor, list_sensors
from .vision_rgb import VisionRGBSensor
from .vision_depth import VisionDepthSensor
from .pointcloud import PointCloudSensor
from .proprioception import ProprioceptionSensor
from .tactile import TactileSensor
from .realsense_rgb import RealSenseRGBSensor
from .realsense_depth import RealSenseDepthSensor
from .realsense_vlm import RealSenseVLMSensor

# Auto-register built-in sensors
register_sensor("vision_rgb", VisionRGBSensor)
register_sensor("vision_depth", VisionDepthSensor)
register_sensor("pointcloud", PointCloudSensor)
register_sensor("proprioception", ProprioceptionSensor)
register_sensor("tactile", TactileSensor)
register_sensor("realsense_rgb", RealSenseRGBSensor)
register_sensor("realsense_depth", RealSenseDepthSensor)
register_sensor("realsense_vlm", RealSenseVLMSensor)

__all__ = [
    "SensorChannel",
    "PerceptionData",
    "register_sensor",
    "create_sensor",
    "list_sensors",
    "VisionRGBSensor",
    "VisionDepthSensor",
    "PointCloudSensor",
    "ProprioceptionSensor",
    "TactileSensor",
    "RealSenseRGBSensor",
    "RealSenseDepthSensor",
    "RealSenseVLMSensor",
]
