#!/usr/bin/env python3
"""
Capture ALL streams from every camera reachable by OpenRobotDemo.

Maps OpenRobotDemo device names to RealSense serial numbers:
  - main_camera  -> /dev/main_camera (D435I, serial=135122077817)
  - left_camera  -> /dev/left_camera  (D405,  serial=409122273228)
  - right_camera -> /dev/right_camera (D405,  serial=409122273564)

For each connected camera, captures:
  - RGB Color
  - Aligned Depth (with post-processing filters from OpenRobotDemo)
  - Infrared 1
  - Infrared 2 (D435I only)
"""

import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import pyrealsense2 as rs
import numpy as np
import cv2
import sys

# Add OpenRobotDemo to path so we can use its classes
sys.path.insert(0, '/home/kemove/openrobot/OpenRobotDemo')


def add_label(img, label, color=(0, 255, 0)):
    h, w = img.shape[:2]
    label_h = 30
    padded = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    padded[label_h:, :] = img
    cv2.putText(padded, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return padded


def capture_device(name, serial, enable_ir2=False):
    """Open one device, capture color/depth/ir, apply filters like OpenRobotDemo does."""
    print(f"[{name}] Opening serial={serial} ...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    if enable_ir2:
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Warmup
    for _ in range(10):
        frames = pipeline.wait_for_frames(timeout_ms=5000)

    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    ir1_frame = frames.get_infrared_frame(1)
    ir2_frame = frames.get_infrared_frame(2) if enable_ir2 else None

    # Apply same post-processing filters as OpenRobotDemo
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole.process(depth_frame)

    results = {}

    if color_frame:
        results['color'] = np.asanyarray(color_frame.get_data())
        print(f"[{name}] Color: {results['color'].shape}")

    if depth_frame:
        depth_img = np.asanyarray(depth_frame.get_data())
        # Same visualization as OpenRobotDemo (0.5-1.5m focused JET)
        depth_m = depth_img.astype(np.float32) * 0.001
        depth_m = np.clip(depth_m, 0.5, 1.5)
        depth_norm = ((depth_m - 0.5) / 1.0 * 255).astype(np.uint8)
        results['depth'] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        print(f"[{name}] Depth: {results['depth'].shape}")

    if ir1_frame:
        ir1 = np.asanyarray(ir1_frame.get_data())
        results['ir1'] = cv2.cvtColor(ir1, cv2.COLOR_GRAY2BGR)
        print(f"[{name}] IR1: {results['ir1'].shape}")

    if enable_ir2 and ir2_frame:
        ir2 = np.asanyarray(ir2_frame.get_data())
        results['ir2'] = cv2.cvtColor(ir2, cv2.COLOR_GRAY2BGR)
        print(f"[{name}] IR2: {results['ir2'].shape}")

    pipeline.stop()
    print(f"[{name}] Done.")
    return results


def main():
    output_dir = "/home/kemove/openrobot/camera_captures/openrobot_all"
    os.makedirs(output_dir, exist_ok=True)

    # Map OpenRobotDemo device names to RealSense serials
    devices = [
        ("main_camera (D435I)", "135122077817", True),
        ("left_camera (D405)", "409122273228", False),
        ("right_camera (D405)", "409122273564", False),
    ]

    all_frames = {}
    for name, serial, ir2 in devices:
        try:
            all_frames[name] = capture_device(name, serial, enable_ir2=ir2)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            all_frames[name] = {}

    # Build composite image: 3 rows, one per device
    rows = []
    for name, serial, ir2 in devices:
        frames = all_frames.get(name, {})
        if not frames:
            continue

        cols = []
        if 'color' in frames:
            cols.append(add_label(frames['color'], f"{name} - Color"))
        if 'depth' in frames:
            cols.append(add_label(frames['depth'], f"{name} - Depth (filtered)"))
        if 'ir1' in frames:
            cols.append(add_label(frames['ir1'], f"{name} - IR1"))
        if 'ir2' in frames:
            cols.append(add_label(frames['ir2'], f"{name} - IR2"))

        # Resize to same height
        target_h = 480
        resized = []
        for col in cols:
            h, w = col.shape[:2]
            if h != target_h:
                col = cv2.resize(col, (int(w * target_h / h), target_h))
            resized.append(col)

        if resized:
            row_img = cv2.hconcat(resized)
            rows.append(row_img)

    # Add black padding between rows
    if rows:
        max_w = max(r.shape[1] for r in rows)
        final_rows = []
        for i, row in enumerate(rows):
            h, w = row.shape[:2]
            if w < max_w:
                pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
                row = cv2.hconcat([row, pad])
            final_rows.append(row)
            if i < len(rows) - 1:
                final_rows.append(np.zeros((20, max_w, 3), dtype=np.uint8))

        composite = cv2.vconcat(final_rows)
        output_path = os.path.join(output_dir, "openrobot_all_cameras.jpg")
        cv2.imwrite(output_path, composite)
        print(f"\nComposite saved: {output_path}")
        print(f"Image size: {composite.shape[1]}x{composite.shape[0]}")

    # Also save individual frames
    for name, frames in all_frames.items():
        for stream_type, img in frames.items():
            fname = name.replace(' ', '_').replace('(', '').replace(')', '') + f"_{stream_type}.jpg"
            path = os.path.join(output_dir, fname)
            cv2.imwrite(path, img)
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
