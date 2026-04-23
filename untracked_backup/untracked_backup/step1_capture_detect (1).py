import pyrealsense2 as rs
import numpy as np
import cv2
import base64
import re
import os
from openai import OpenAI

# 批量清除系统代理，确保 API 直连
for p in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(p, None)

def detect_center(frame, target_name, api_key):
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 按比例缩放图片以减少 API 传输延时
    h, w = frame.shape[:2]
    scale = 640 / w
    resized_frame = cv2.resize(frame, (640, int(h * scale)))
   
    _, buffer = cv2.imencode('.jpg', resized_frame)
    b64_img = base64.b64encode(buffer).decode("utf-8")

    prompt = f"精确寻找“{target_name}”的中心点。仅返回纯JSON数组：[中心X, 中心Y]。找不到返回[0, 0]，绝对禁止输出其他文字。"
    
    try:
        res = client.chat.completions.create(
            model="qwen-vl-max-latest", 
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                {"type": "text", "text": prompt}
            ]}],
            temperature=0.1
        )
        
        # 解析返回的文本，提取前两个数字作为坐标
        text = res.choices[0].message.content.strip()
        nums = re.findall(r'\d+', text)
        
        if len(nums) >= 2:
            x, y = map(int, nums[:2])
            if x == 0 and y == 0: 
                return None
            return int(x / scale), int(y / scale) # 映射回原图坐标
            
    except Exception as e:
        print(f"API 请求或解析出错: {e}")
    return None

def main():
    API_KEY = "sk-xxxx" # ⚠️ 请填入你的 API Key
    TARGET_OBJ = "布料的下边沿"

    print("初始化 RealSense D435i...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print(f"\n相机就绪！寻找目标: {TARGET_OBJ}\n操作: 按 'd' 侦测 3D 坐标 | 按 'q' 退出\n")

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            display = img.copy()

            cv2.putText(display, "Ready | 'd' Detect | 'q' Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("RealSense", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                print("大模型识别中...")
                pt = detect_center(img, TARGET_OBJ, API_KEY)
                
                if pt:
                    cx, cy = pt
                    dist = depth_frame.get_distance(cx, cy)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    
                    if dist > 0:
                        p3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                        print(f"成功! 2D:({cx}, {cy}), 距离:{dist:.3f}m")
                        print(f"3D坐标: X={p3d[0]:.3f}m, Y={p3d[1]:.3f}m, Z={p3d[2]:.3f}m")
                        cv2.putText(img, f"Z: {dist:.2f}m", (cx, max(cy - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        print("目标位置的深度数据丢失 (可能太近或反光)")
                else:
                    print("未能找到目标。")
                
                cv2.imshow("RealSense", img)
                print("按任意键恢复实时视频...")
                cv2.waitKey(0)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()