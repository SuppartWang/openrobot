#!/usr/bin/env python3
"""
Capture all streams from 3 RealSense cameras and composite into a single image.
Devices:
  - main_camera:   D435I (serial=135122077817)
  - left_camera:   D405  (serial=409122273228)
  - right_camera:  D405  (serial=409122273564)
Streams per device:
  - Color (RGB)
  - Depth (pseudo-colored)
  - Infrared 1
  - Infrared 2 (D435I only)
"""

import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import pyrealsense2 as rs
import numpy as np
import cv2


def normalize_depth(depth_frame):
    """Convert depth frame to uint8 for visualization."""
    depth_image = np.asanyarray(depth_frame.get_data())
    # Clip and normalize
    depth_image = depth_image.astype(np.float32)
    depth_image = np.clip(depth_image, 0, 5000)  # 0-5m range
    depth_image = (depth_image / 5000.0 * 255).astype(np.uint8)
    # Apply colormap
    depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return depth_colormap


def capture_device_frames(name, serial, enable_ir2=False):
    """Open a single device, capture one frame from each stream, then close."""
    print(f"[{name}] Opening device {serial}...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    
    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    if enable_ir2:
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    
    profile = pipeline.start(config)
    
    # Warmup: skip first 10 frames
    for _ in range(10):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
    
    # Get aligned frames (depth -> color)
    align = rs.align(rs.stream.color)
    aligned = align.process(frames)
    
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    ir1_frame = frames.get_infrared_frame(1)
    ir2_frame = frames.get_infrared_frame(2) if enable_ir2 else None
    
    results = {}
    
    if color_frame:
        color_img = np.asanyarray(color_frame.get_data())
        results['color'] = color_img
        print(f"[{name}] Color: {color_img.shape}")
    else:
        print(f"[{name}] Color: FAILED")
    
    if depth_frame:
        depth_img = normalize_depth(depth_frame)
        results['depth'] = depth_img
        print(f"[{name}] Depth: {depth_img.shape}")
    else:
        print(f"[{name}] Depth: FAILED")
    
    if ir1_frame:
        ir1_img = np.asanyarray(ir1_frame.get_data())
        ir1_img = cv2.cvtColor(ir1_img, cv2.COLOR_GRAY2BGR)
        results['ir1'] = ir1_img
        print(f"[{name}] IR1: {ir1_img.shape}")
    else:
        print(f"[{name}] IR1: FAILED")
    
    if enable_ir2 and ir2_frame:
        ir2_img = np.asanyarray(ir2_frame.get_data())
        ir2_img = cv2.cvtColor(ir2_img, cv2.COLOR_GRAY2BGR)
        results['ir2'] = ir2_img
        print(f"[{name}] IR2: {ir2_img.shape}")
    
    pipeline.stop()
    print(f"[{name}] Done.")
    return results


def add_label(img, label):
    """Add a text label at top-left of image."""
    h, w = img.shape[:2]
    # Dark overlay for text background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def main():
    output_dir = "/home/kemove/openrobot/camera_captures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture each device independently to avoid USB bandwidth issues
    devices = [
        ("main_camera (D435I)", "135122077817", True),
        ("left_camera (D405)", "409122273228", False),
        ("right_camera (D405)", "409122273564", False),
    ]
    
    all_frames = {}
    for name, serial, ir2 in devices:
        all_frames[name] = capture_device_frames(name, serial, enable_ir2=ir2)
    
    # Build composite image
    # Layout: 3 rows (one per device), up to 4 columns (color, depth, ir1, ir2)
    rows = []
    for name, serial, ir2 in devices:
        frames = all_frames[name]
        cols = []
        
        if 'color' in frames:
            cols.append(add_label(frames['color'], f"{name} - Color"))
        if 'depth' in frames:
            cols.append(add_label(frames['depth'], f"{name} - Depth"))
        if 'ir1' in frames:
            cols.append(add_label(frames['ir1'], f"{name} - IR1"))
        if 'ir2' in frames:
            cols.append(add_label(frames['ir2'], f"{name} - IR2"))
        
        # Resize all cols to same height
        target_h = 480
        resized_cols = []
        for col in cols:
            h, w = col.shape[:2]
            if h != target_h:
                new_w = int(w * target_h / h)
                col = cv2.resize(col, (new_w, target_h))
            resized_cols.append(col)
        
        if resized_cols:
            row_img = cv2.hconcat(resized_cols)
            rows.append(row_img)
    
    # Add black padding between rows
    padded_rows = []
    for i, row in enumerate(rows):
        padded_rows.append(row)
        if i < len(rows) - 1:
            h, w = row.shape[:2]
            pad = np.zeros((20, w, 3), dtype=np.uint8)
            padded_rows.append(pad)
    
    if padded_rows:
        # Make all rows same width
        max_w = max(r.shape[1] for r in padded_rows)
        final_rows = []
        for row in padded_rows:
            h, w = row.shape[:2]
            if w < max_w:
                pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
                row = cv2.hconcat([row, pad])
            final_rows.append(row)
        
        composite = cv2.vconcat(final_rows)
        
        # Save composite
        output_path = os.path.join(output_dir, "all_cameras_composite.jpg")
        cv2.imwrite(output_path, composite)
        print(f"\nComposite image saved to: {output_path}")
        print(f"Image size: {composite.shape[1]}x{composite.shape[0]}")
        
        # Also save individual frames
        for name, frames in all_frames.items():
            for stream_type, img in frames.items():
                path = os.path.join(output_dir, f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_{stream_type}.jpg")
                cv2.imwrite(path, img)
                print(f"  Individual: {path}")
    else:
        print("No frames captured!")


if __name__ == "__main__":
    main()
