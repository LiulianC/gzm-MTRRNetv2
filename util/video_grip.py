import cv2
import numpy as np
import os

def create_video_grid(input_paths, output_path, fps=30):
    """
    将多个视频拼接成网格形式
    
    参数:
        input_paths: 输入视频路径列表 (16个)
        output_path: 输出视频路径
        fps: 输出视频帧率
    """
    # 验证输入视频数量
    if len(input_paths) != 16:
        raise ValueError("需要恰好16个输入视频")
    
    # 初始化视频读取器
    caps = [cv2.VideoCapture(path) for path in input_paths]
    
    # 获取第一个视频的参数作为参考
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    total_frames = max([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    
    # 创建输出视频
    output_width = width * 4
    output_height = height * 4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID' 用于avi格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # 逐帧处理
    for frame_idx in range(total_frames):
        # 读取所有视频的当前帧
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                # 如果某个视频结束，用黑色帧代替
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)
        
        # 创建4x4网格
        grid = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 将帧放置到网格中
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                y_start = i * height
                y_end = (i + 1) * height
                x_start = j * width
                x_end = (j + 1) * width
                
                grid[y_start:y_end, x_start:x_end] = frames[idx]
        
        # 写入输出视频
        out.write(grid)
    
    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"视频拼接完成，保存至: {output_path}")

# 示例使用
if __name__ == "__main__":
    # 假设有16个视频文件路径
    input_videos = [
        "video/1.mp4",  "video/2.mp4",  "video/3.mp4",  "video/4.mp4",
        "video/5.mp4",  "video/6.mp4",  "video/7.mp4",  "video/8.mp4",
        "video/9.mp4",  "video/10.mp4", "video/11.mp4", "video/12.mp4",
        "video/13.mp4", "video/14.mp4", "video/15.mp4", "video/16.mp4"
    ]

    # 确保所有视频存在
    for path in input_videos:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到视频文件: {path}")
    
    # 创建网格视频
    create_video_grid(input_videos, "output_grid.mp4", fps=30)