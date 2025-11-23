import cv2
import numpy as np
import os

def create_video_grid(input_paths, output_path, fps=30):
    """
    将多个视频拼接成网格形式
    
    参数:
        input_paths: 输入视频路径列表 (32个)
        output_path: 输出视频路径
        fps: 输出视频帧率
    """
    # 验证输入视频数量
    if len(input_paths) != 32:
        raise ValueError("需要恰好32个输入视频")
    
    # 初始化视频读取器
    caps = [cv2.VideoCapture(path) for path in input_paths]
    
    # 获取第一个视频的参数作为参考
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 获取最大帧数（确保所有视频都能处理）
    total_frames = max([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    
    # 创建输出视频
    output_width = width * 8  # 8列
    output_height = height * 4  # 4行
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        
        # 创建4行8列的网格
        grid = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 将帧放置到网格中
        for i in range(4):  # 4行
            for j in range(8):  # 8列
                idx = i * 8 + j
                y_start = i * height
                y_end = (i + 1) * height
                x_start = j * width
                x_end = (j + 1) * width
                
                # 确保索引在范围内
                if idx < len(frames):
                    grid[y_start:y_end, x_start:x_end] = frames[idx]
        
        # 写入输出视频
        out.write(grid)
        
        # 显示进度
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} 帧")
    
    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"视频拼接完成，保存至: {output_path}")
    print(f"输出视频尺寸: {output_width}x{output_height}")

# 示例使用
if __name__ == "__main__":
    # 32个视频文件路径
    # input_videos = [
    #     "video/000-out.mp4", "video/001-out.mp4", "video/002-out.mp4", "video/003-out.mp4",
    #     "video/005-out.mp4", "video/006-out.mp4", "video/008-out.mp4", "video/009-out.mp4",
    #     "video/011-out.mp4", "video/012-out.mp4", "video/013-out.mp4", "video/014-out.mp4",
    #     "video/019-out.mp4", "video/020-out.mp4", "video/021-out.mp4", "video/022-out.mp4",
    #     "video/023-out.mp4", "video/024-out.mp4", "video/025-out.mp4", "video/026-out.mp4",
    #     "video/027-out.mp4", "video/028-out.mp4", "video/029-out.mp4", "video/030-out.mp4",
    #     "video/031-out.mp4", "video/032-out.mp4", "video/033-out.mp4", "video/035-out.mp4",
    #     "video/036-out.mp4", "video/037-out.mp4", "video/038-out.mp4", "video/040-out.mp4",
    # ]
    input_videos = [
        "video/000-in.mp4", "video/001-in.mp4", "video/002-in.mp4", "video/003-in.mp4",
        "video/005-in.mp4", "video/006-in.mp4", "video/008-in.mp4", "video/009-in.mp4",
        "video/011-in.mp4", "video/012-in.mp4", "video/013-in.mp4", "video/014-in.mp4",
        "video/019-in.mp4", "video/020-in.mp4", "video/021-in.mp4", "video/022-in.mp4",
        "video/023-in.mp4", "video/024-in.mp4", "video/025-in.mp4", "video/026-in.mp4",
        "video/027-in.mp4", "video/028-in.mp4", "video/029-in.mp4", "video/030-in.mp4",
        "video/031-in.mp4", "video/032-in.mp4", "video/033-in.mp4", "video/035-in.mp4",
        "video/036-in.mp4", "video/037-in.mp4", "video/038-in.mp4", "video/040-in.mp4",
    ]

    # 确保所有视频存在
    for path in input_videos:
        if not os.path.exists(path):
            print(f"警告: 找不到视频文件: {path}")
            # 可以选择跳过或退出
    
    # 创建网格视频
    create_video_grid(input_videos, "output_grid.mp4", fps=30)