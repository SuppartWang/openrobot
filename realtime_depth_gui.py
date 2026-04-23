#!/usr/bin/env python3
"""
Real-time GUI display of main_camera (D435I) RGB + Depth streams.
Shows a live window with color image and pseudo-colored depth side-by-side.
Auto-exits after N seconds.
"""

import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import pyrealsense2 as rs
import numpy as np
import cv2
import time


def main(duration_sec=20):
    serial = "135122077817"
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Create window
    window_name = "main_camera (D435I) - Real-time RGB + Depth"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 520)
    cv2.moveWindow(window_name, 100, 100)

    print(f"[GUI] Opening real-time depth window for {duration_sec}s...")
    print(f"[GUI] Window: '{window_name}'")

    start_time = time.time()
    frame_count = 0
    fps_start = time.time()

    try:
        while time.time() - start_time < duration_sec:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Depth visualization
            depth_vis = np.clip(depth_image, 0, 5000)
            depth_vis = (depth_vis / 5000.0 * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Statistics
            depth_m = depth_image.astype(np.float32) * 0.001
            valid = depth_m[depth_m > 0]
            if len(valid) > 0:
                d_min, d_max, d_mean = valid.min(), valid.max(), valid.mean()
            else:
                d_min = d_max = d_mean = 0.0

            # Center point
            cx, cy = 320, 240
            center_dist = depth_frame.get_distance(cx, cy)

            # Overlay on color
            overlay = color_img.copy()
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(overlay, f"Center: {center_dist:.3f}m",
                        (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(overlay, f"Min:{d_min:.2f} Max:{d_max:.2f} Mean:{d_mean:.2f}m",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(overlay, f"FPS:{fps:.1f}",
                        (overlay.shape[1] - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add labels
            h, w = overlay.shape[:2]
            label_h = 30
            overlay_padded = np.zeros((h + label_h, w, 3), dtype=np.uint8)
            overlay_padded[label_h:, :] = overlay
            cv2.putText(overlay_padded, "RGB Color",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            depth_padded = np.zeros((h + label_h, w, 3), dtype=np.uint8)
            depth_padded[label_h:, :] = depth_colormap
            cv2.putText(depth_padded, "Depth (JET colormap)",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Composite
            composite = cv2.hconcat([overlay_padded, depth_padded])

            # Show
            cv2.imshow(window_name, composite)

            # Exit on 'q' key or timeout
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[GUI] User pressed 'q', exiting.")
                break

    except KeyboardInterrupt:
        print("[GUI] Interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"[GUI] Finished. Captured {frame_count} frames, avg FPS: {frame_count / (time.time() - fps_start):.1f}")


if __name__ == "__main__":
    import sys
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(duration_sec=duration)
