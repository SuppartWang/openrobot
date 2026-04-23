#!/usr/bin/env python3
"""
Real-time depth info display for main_camera (D435I).
Runs for N seconds, printing depth statistics and saving current depth frames.
"""

import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import pyrealsense2 as rs
import numpy as np
import cv2
import time


def main(duration_sec=10):
    serial = "135122077817"
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    output_dir = "/home/kemove/openrobot/camera_captures/realtime"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Real-time Depth: main_camera (D435I) ===")
    print(f"Running for {duration_sec} seconds...")
    print(f"{'Time':>6} | {'Center(m)':>10} | {'Min(m)':>8} | {'Max(m)':>8} | {'Mean(m)':>8} | {'Valid%':>6}")
    print("-" * 65)

    start_time = time.time()
    frame_count = 0

    try:
        while time.time() - start_time < duration_sec:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            # Convert to meters (float)
            depth_m = depth_image.astype(np.float32) * 0.001

            # Statistics on valid depth pixels (depth > 0)
            valid = depth_m[depth_m > 0]
            valid_pct = len(valid) / depth_m.size * 100 if depth_m.size > 0 else 0

            if len(valid) > 0:
                d_min = valid.min()
                d_max = valid.max()
                d_mean = valid.mean()
            else:
                d_min = d_max = d_mean = 0.0

            # Center point
            cx, cy = 320, 240
            center_dist = depth_frame.get_distance(cx, cy)

            elapsed = time.time() - start_time
            print(f"{elapsed:6.2f} | {center_dist:10.3f} | {d_min:8.3f} | {d_max:8.3f} | {d_mean:8.3f} | {valid_pct:6.1f}", flush=True)

            # Save depth colormap and overlay
            depth_vis = np.clip(depth_image, 0, 5000)
            depth_vis = (depth_vis / 5000.0 * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Overlay center distance text
            color_img = np.asanyarray(color_frame.get_data()) if color_frame else np.zeros((480, 640, 3), dtype=np.uint8)
            overlay = color_img.copy()
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(overlay, f"Center: {center_dist:.3f}m", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, f"Min:{d_min:.2f} Max:{d_max:.2f} Mean:{d_mean:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Composite: color + depth side by side
            composite = cv2.hconcat([overlay, depth_colormap])

            # Save files (overwrite for realtime view)
            cv2.imwrite(f"{output_dir}/depth_color.jpg", overlay)
            cv2.imwrite(f"{output_dir}/depth_colormap.jpg", depth_colormap)
            cv2.imwrite(f"{output_dir}/depth_composite.jpg", composite)

            frame_count += 1
            time.sleep(0.3)  # ~3 fps update rate

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

    print(f"\nCaptured {frame_count} frames.")
    print(f"Saved to: {output_dir}/")
    print(f"  - depth_color.jpg     (RGB with center overlay)")
    print(f"  - depth_colormap.jpg  (JET pseudo-colored depth)")
    print(f"  - depth_composite.jpg (side-by-side)")


if __name__ == "__main__":
    import sys
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(duration_sec=duration)
