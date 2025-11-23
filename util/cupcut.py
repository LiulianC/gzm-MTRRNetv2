import cv2
import numpy as np

def create_transition_video(video1_path, video2_path, output_path, 
                           start_time=0, move_duration=5, stay_duration=3, stay_duration2=3,
                           glow_intensity=0.3):
    """
    创建带有来回移动蒙版效果的视频过渡
    
    参数:
        video1_path: 上方视频路径（原始视频）
        video2_path: 下方视频路径（处理后的视频）
        output_path: 输出视频路径
        start_time: 效果开始的时间（秒）
        move_duration: 蒙版移动的持续时间（秒）
        stay_duration: 完全显示下方视频的持续时间（秒）
        glow_intensity: 发光效果强度（0-1）
    """
    # 打开视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # 检查视频是否成功打开
    if not cap1.isOpened() or not cap2.isOpened():
        print("错误：无法打开视频文件")
        return
    
    # 获取视频属性
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算开始帧
    start_frame = int(start_time * fps)
    
    # 计算各阶段帧数
    move_frames = int(move_duration * fps)
    stay_frames = int(stay_duration * fps)
    stay_frames2 = int(stay_duration2 * fps)
    
    # 设置视频读取位置
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 添加科技感效果：创建光效蒙版
    def create_glow_mask(width, height, position):
        mask = np.zeros((height, width), dtype=np.uint8)
        # 在分割线位置创建一个光带
        start = max(0, position - 15)
        end = min(width, position + 15)
        cv2.rectangle(mask, (start, 0), (end, height), 255, -1)
        # 添加高斯模糊创建发光效果
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 第1阶段：完全显示上方视频（停留）
    print("第二阶段：完全显示处理后的视频")
    for frame_idx in range(stay_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # 完全显示下方视频
        combined_frame = frame2.copy()
        
        # 添加分割线指示器（在最右侧）
        cv2.line(combined_frame, (width-1, 0), (width-1, height), 
                (0, 255, 255), 3)
        
        # 添加文本标签
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 添加时间戳
        current_time = start_time + (frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 写入输出视频
        out.write(combined_frame)
        
        # 显示进度
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{stay_frames} 帧")

    # 第2阶段：从左向右移动（显示下方视频）
    print("第一阶段：蒙版从左向右移动")
    for frame_idx in range(move_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # 计算当前分割线位置（从左向右移动）
        split_position = int(width * frame_idx / move_frames)
        
        # 创建基础合成帧
        combined_frame = frame2.copy()
        
        # 将上方视频的左侧部分复制到合成帧
        combined_frame[:, :split_position] = frame1[:, :split_position]
        
        # 创建光效蒙版
        glow_mask = create_glow_mask(width, height, split_position)
        
        # 添加发光分割线
        combined_frame = cv2.addWeighted(combined_frame, 1.0, glow_mask, glow_intensity, 0)
        
        # 添加分割线指示器
        cv2.line(combined_frame, (split_position, 0), (split_position, height), 
                (0, 255, 255), 3)
        
        # 添加文本标签
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 添加时间戳
        current_time = start_time + (stay_frames+frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 写入输出视频
        out.write(combined_frame)
        
        # 显示进度
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{move_frames} 帧")

    
    # 第三阶段：从右向左移动（回到原始视频）
    print("第三阶段：蒙版从右向左移动")
    for frame_idx in range(move_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # 计算当前分割线位置（从右向左移动）
        split_position = width - int(width * frame_idx / move_frames)
        
        # 创建基础合成帧
        combined_frame = frame2.copy()
        
        # 将上方视频的左侧部分复制到合成帧
        combined_frame[:, :split_position] = frame1[:, :split_position]
        
        # 创建光效蒙版
        glow_mask = create_glow_mask(width, height, split_position)
        
        # 添加发光分割线
        combined_frame = cv2.addWeighted(combined_frame, 1.0, glow_mask, glow_intensity, 0)
        
        # 添加分割线指示器
        cv2.line(combined_frame, (split_position, 0), (split_position, height), 
                (0, 255, 255), 3)
        
        # 添加文本标签
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 添加时间戳
        current_time = start_time + (move_frames + stay_frames + frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 写入输出视频
        out.write(combined_frame)
        
        # 显示进度
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{move_frames} 帧")
    
    # 第四阶段：完全显示上方视频（停留）
    print("第四阶段：完全显示处理后的视频")
    for frame_idx in range(stay_frames2):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # 完全显示下方视频
        combined_frame = frame2.copy()
        
        # 添加分割线指示器（在最右侧）
        cv2.line(combined_frame, (width-1, 0), (width-1, height), 
                (0, 255, 255), 3)
        
        # 添加文本标签
        cv2.putText(combined_frame, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, "Processed", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 添加时间戳
        current_time = start_time + (move_frames*2 + stay_frames + frame_idx) / fps
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(combined_frame, time_text, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 写入输出视频
        out.write(combined_frame)
        
        # 显示进度
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{stay_frames} 帧")

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    print(f"视频处理完成！已保存至: {output_path}")
    total_effect_frames = move_frames + stay_frames + move_frames
    print(f"处理了 {total_effect_frames} 帧，时长: {total_effect_frames/fps:.2f} 秒")

# 使用示例
if __name__ == "__main__":
    # 替换为您的实际视频路径
    original_video = "output.mp4"
    processed_video = "input.mp4"
    output_video = "transition_effect.mp4"
    
    create_transition_video(
        original_video, 
        processed_video, 
        output_video,
        start_time=0,         # 从第3秒开始效果
        stay_duration=3,       # 停留3秒
        move_duration=7,      # 每次移动持续5秒
        stay_duration2=3,
        glow_intensity=0.05,
    )