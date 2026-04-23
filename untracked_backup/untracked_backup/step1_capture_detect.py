import pyrealsense2 as rs
import numpy as np
import cv2
from openai import OpenAI
import base64
import json
import re
import os

# 屏蔽代理环境变量
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('all_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('ALL_PROXY', None)

def detect_frame_from_memory(frame, target_name, api_key):
    # 1. 替换为阿里云百炼的 Base URL
    client = OpenAI(
        api_key=api_key, # 使用传入的 API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
    )

    h, w, _ = frame.shape
    scale_ratio = 640 / w
    detect_w, detect_h = 640, int(h * scale_ratio)
    resized_frame = cv2.resize(frame, (detect_w, detect_h))
   
    _, buffer = cv2.imencode('.jpg', resized_frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # 【修改这里 1】：更改提示词，要求模型只返回中心点的 [X, Y] 坐标
    prompt_text = f"""你是一个精确的机器视觉定位系统。
    请在宽{detect_w}，高{detect_h}的图片中，精确找到“{target_name}”的中心点。
    要求：
    1. 必须精准定位该物体中心位置的坐标。
    2. 绝对禁止输出任何额外解释文字！
    3. 格式只能是包含两个数字的纯JSON数组：[中心X, 中心Y]
    如果找不到目标，请返回 [0, 0]"""
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-max-latest", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"\n[Debug] 模型的真实回复是: {result_text}\n")
        
        # 1. 先把结果里可能混入的字母全部清理掉
        clean_text = re.sub(r'[a-zA-Z"\'{}]+', '', result_text)
        
        # 2. 提取里面所有的纯数字
        nums = re.findall(r'\d+', clean_text)
        
        # 【修改这里 2】：将原先的 4 个数字校验改为 2 个数字校验
        if len(nums) >= 2:
            x, y = map(int, nums[:2])
            
            # 如果模型觉得没找到，返回了两个 0
            if x == 0 and y == 0:
                print("[Debug] 模型判定画面中不存在该物体。")
                return None
                
            # 3. 还原缩放比例，映射回原始高清摄像头的尺寸
            real_x = int(x / scale_ratio)
            real_y = int(y / scale_ratio)
            
            return (real_x, real_y)
        else:
            print("[Debug] 模型返回的数据里连两个数字都凑不齐！")
            return None

    except Exception as e:
        print(f"API 请求或解析出错: {e}")
        return None

if __name__ == "__main__":
    # 填入你的大模型 API Key 和要找的物体
    API_KEY = "sk-03b0ef30a9bf496ebeebf89523e7c668" 
    TARGET_OBJ = "粉色布料的下边沿"

    print("正在初始化 RealSense D435i 相机...")
    # 1. 配置 RealSense 数据流
    pipeline = rs.pipeline()
    config = rs.config()
    # 开启深度流 (640x480, 30fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # 开启彩色流 (640x480, 30fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动相机
    profile = pipeline.start(config)

    # 2. 极其重要的一步：深度与彩色对齐 (Align)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取相机的真实内参
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print("=====================================")
    print("相机启动成功！")
    print(f"目标物体: {TARGET_OBJ}")
    print("按 'd' 键 -> 截取画面并计算真实 3D 坐标")
    print("按 'q' 键 -> 退出")
    print("=====================================")

    try:
        while True:
            # 3. 等待并获取一帧数据
            frames = pipeline.wait_for_frames()
            
            # 将深度帧对齐到彩色帧
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 将彩色帧转换为 OpenCV 图像
            color_image = np.asanyarray(color_frame.get_data())
            display_frame = color_image.copy()

            cv2.putText(display_frame, "RealSense Active | 'd' Detect | 'q' Quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("RealSense Feed", display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                print("\n[抓拍] 正在呼叫大模型识别目标...")
                
                # 给大模型发送当前的彩色图
                point = detect_frame_from_memory(color_image, TARGET_OBJ, API_KEY)
                
                result_frame = color_image.copy()
                
                # 【修改这里 3】：接收单个点 (cx, cy) 并去除矩形框绘制逻辑
                if point:
                    cx, cy = point
                    
                    # 4. 获取真实深度 (Z 值)，单位为米 (m)
                    distance_m = aligned_depth_frame.get_distance(cx, cy)
                    
                    if distance_m > 0:
                        # 5. API 自带的逆投影：将二维像素点结合深度转换成 3D 点
                        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance_m)
                        
                        print(f"----------------------------------------")
                        print(f"[成功] 找到目标！")
                        print(f"2D 像素坐标: u={cx}, v={cy}")
                        print(f"目标距离镜头的直线距离: {distance_m:.3f} 米")
                        print(f"相机坐标系下绝对 3D 坐标: X={point_3d[0]:.3f}m, Y={point_3d[1]:.3f}m, Z={point_3d[2]:.3f}m")
                        print(f"----------------------------------------")

                        # 可视化结果：由于没有边界框，这里仅在中心点画一个红点
                        cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # 在图上打印 3D 坐标信息 (位置锚定在红点上方)
                        text_3d = f"Z: {distance_m:.2f}m"
                        cv2.putText(result_frame, text_3d, (cx, max(cy - 10, 0)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        print("目标位置的深度数据丢失 (可能太近或物体反光)")
                        cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    print("大模型未能找到目标。")
                
                cv2.imshow("RealSense Feed", result_frame)
                print("按任意键恢复实时视频...")
                cv2.waitKey(0)

    finally:
        # 优雅地关闭相机
        pipeline.stop()
        cv2.destroyAllWindows()