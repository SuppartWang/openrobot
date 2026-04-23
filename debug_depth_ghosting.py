#!/usr/bin/env python3
"""
Debug depth "ghosting" effect on D435I.
Compares:
  1. Raw depth (from depth sensor directly, no align)
  2. Aligned depth (depth mapped to color FOV)
  3. Different colormap ranges
"""

import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import pyrealsense2 as rs
import numpy as np
import cv2


def depth_to_colormap(depth_image, min_dist=0.1, max_dist=5.0):
    """Convert raw depth (uint16 mm) to pseudo-colored visualization."""
    depth_m = depth_image.astype(np.float32) * 0.001
    # Clip to range
    depth_m = np.clip(depth_m, min_dist, max_dist)
    # Normalize to 0-255
    depth_norm = ((depth_m - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET), depth_m


def main():
    serial = "135122077817"
    output_dir = "/home/kemove/openrobot/camera_captures/depth_debug"
    os.makedirs(output_dir, exist_ok=True)

    print("Capturing raw vs aligned depth frames...")

    # --- Capture 1: WITH align ---
    pipeline1 = rs.pipeline()
    config1 = rs.config()
    config1.enable_device(serial)
    config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile1 = pipeline1.start(config1)
    align1 = rs.align(rs.stream.color)

    for _ in range(10):
        frames1 = pipeline1.wait_for_frames(timeout_ms=5000)
    aligned1 = align1.process(frames1)
    depth_aligned = np.asanyarray(aligned1.get_depth_frame().get_data())
    color_img = np.asanyarray(aligned1.get_color_frame().get_data())
    pipeline1.stop()

    # --- Capture 2: WITHOUT align (raw depth sensor output) ---
    pipeline2 = rs.pipeline()
    config2 = rs.config()
    config2.enable_device(serial)
    config2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile2 = pipeline2.start(config2)

    for _ in range(10):
        frames2 = pipeline2.wait_for_frames(timeout_ms=5000)
    depth_raw = np.asanyarray(frames2.get_depth_frame().get_data())
    pipeline2.stop()

    # --- Visualize with different ranges ---
    results = []

    # 1. Raw depth, full range 0-5m
    vis_raw_5m, _ = depth_to_colormap(depth_raw, 0.0, 5.0)
    results.append(("1_RawDepth_0-5m", vis_raw_5m))

    # 2. Raw depth, focused range 0.1-2m
    vis_raw_2m, _ = depth_to_colormap(depth_raw, 0.1, 2.0)
    results.append(("2_RawDepth_0.1-2m", vis_raw_2m))

    # 3. Aligned depth, full range 0-5m
    vis_aligned_5m, depth_m = depth_to_colormap(depth_aligned, 0.0, 5.0)
    results.append(("3_AlignedDepth_0-5m", vis_aligned_5m))

    # 4. Aligned depth, focused range 0.1-2m
    vis_aligned_2m, _ = depth_to_colormap(depth_aligned, 0.1, 2.0)
    results.append(("4_AlignedDepth_0.1-2m", vis_aligned_2m))

    # 5. Aligned depth with hole-filling filter
    hole_filter = rs.hole_filling_filter()
    depth_holefilled = np.asanyarray(
        hole_filter.process(aligned1.get_depth_frame()).get_data()
    )
    vis_holefilled, _ = depth_to_colormap(depth_holefilled, 0.1, 2.0)
    results.append(("5_AlignedDepth_HoleFilled_0.1-2m", vis_holefilled))

    # 6. Side-by-side: Color + Aligned depth with overlay
    overlay = color_img.copy()
    # Blend depth colormap onto color at 50%
    vis_aligned_2m_resized = cv2.resize(vis_aligned_2m, (640, 480))
    blended = cv2.addWeighted(overlay, 0.5, vis_aligned_2m_resized, 0.5, 0)
    results.append(("6_Color+Depth_Blended", blended))

    # --- Save all ---
    for name, img in results:
        path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

    # --- Print analysis ---
    print("\n=== Depth Ghosting Analysis ===")
    print(f"Raw depth shape: {depth_raw.shape}")
    print(f"Aligned depth shape: {depth_aligned.shape}")

    # Check valid pixels
    raw_valid = np.sum(depth_raw > 0)
    aligned_valid = np.sum(depth_aligned > 0)
    print(f"Raw valid pixels: {raw_valid} / {depth_raw.size} ({raw_valid/depth_raw.size*100:.1f}%)")
    print(f"Aligned valid pixels: {aligned_valid} / {depth_aligned.size} ({aligned_valid/depth_aligned.size*100:.1f}%)")

    # Check for "duplicated edges" by looking at depth gradients
    grad_raw = np.abs(np.diff(depth_raw.astype(np.float32), axis=1))
    grad_aligned = np.abs(np.diff(depth_aligned.astype(np.float32), axis=1))
    print(f"Raw depth mean horizontal gradient: {np.mean(grad_raw):.1f} mm")
    print(f"Aligned depth mean horizontal gradient: {np.mean(grad_aligned):.1f} mm")

    # Depth sensor intrinsics vs Color intrinsics
    ctx = rs.context()
    dev = ctx.query_devices()[2]  # D435I is device 2
    for sensor in dev.query_sensors():
        sname = sensor.get_info(rs.camera_info.name)
        if "Stereo" in sname or "RGB" in sname:
            print(f"\nSensor: {sname}")
            for prof in sensor.get_stream_profiles():
                if prof.stream_type() == rs.stream.depth and "Stereo" in sname:
                    vp = prof.as_video_stream_profile()
                    intr = vp.get_intrinsics()
                    print(f"  Depth intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}")
                    print(f"  Depth FoV: {np.degrees(2 * np.arctan(intr.ppx / intr.fx)):.1f} x {np.degrees(2 * np.arctan(intr.ppy / intr.fy)):.1f} degrees")
                elif prof.stream_type() == rs.stream.color and "RGB" in sname:
                    vp = prof.as_video_stream_profile()
                    intr = vp.get_intrinsics()
                    print(f"  Color intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}")
                    print(f"  Color FoV: {np.degrees(2 * np.arctan(intr.ppx / intr.fx)):.1f} x {np.degrees(2 * np.arctan(intr.ppy / intr.fy)):.1f} degrees")

    print("\n=== Key Insight ===")
    print("If Color FoV > Depth FoV, align() must EXTRAPOLATE depth at edges.")
    print("This creates 'ghost' edges where depth values are projected beyond")
    print("the original depth sensor's field of view.")


if __name__ == "__main__":
    main()
