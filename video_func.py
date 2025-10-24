import os
import cv2
import numpy as np
from PIL import Image
import pathlib
import torch
import importlib
import argparse
from datetime import datetime
import time
from tqdm import tqdm
from util.color_enhance import histogram_equalization_lab,hist_match_batch_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='/home/gzm/gzm-compare/dataset/JPEGImages/hyperK_202.zip', help="输入文件路径")
# parser.add_argument("--input", type=str, default='./data/med1.mp4', help="输入文件路径")
parser.add_argument("--output", type=str, default='./test_results', help="输出文件夹")
parser.add_argument("--model", type=str, default='MTRRNet', help="模型文件名称")
parser.add_argument("--ckptpath", type=str, default='./model_fit/model_62_best1.pth', help="权重文件路径")
parser.add_argument("--batchsize", type=int, default=16, help="batch大小")
args = parser.parse_args()
default_fps = 30

def read_frames_from_video(video_path, target_size=None):
    """从视频读取所有帧，并可选统一大小"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB
        img = Image.fromarray(frame)
        if target_size is not None:
            img = img.resize(target_size, Image.BICUBIC)  # 统一大小
        frames.append(img)
    cap.release()
    return frames

def read_frames_from_zip(zip_path, zip_reader, target_size=None):
    """从zip压缩包读取帧，并可选统一大小"""
    zfilelist = zip_reader.filelist(zip_path)
    frames = []
    for zfile in zfilelist:
        img = zip_reader.imread(zip_path, zfile).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.BICUBIC)
        frames.append(img)
    return frames, zfilelist

def save_video(frames, save_path, fps=24):
    """保存帧列表为视频"""
    if len(frames) == 0:
        raise ValueError("没有帧可保存")

    w, h = frames[0].size   # PIL.Image.size -> (宽, 高)
    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for f in tqdm(frames, desc="Saving video"):
        arr = np.array(f).astype(np.uint8)
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    writer.release()


def evaluate_batch_model(frames, model, device, output_dir, fps=24, batch_size=8):
    """支持任意 batchsize 的逐帧推理并保存结果视频"""
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)

    frame_result_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frame_result_dir, exist_ok=True)
    
    results = []
    data = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(frames), batch_size), desc="Processing frames"):
            end = min(start + batch_size, len(frames))
            batch_frames = frames[start:end]

            # 转换为 tensor batch
            inp = []
            for frame in batch_frames:
                arr = np.array(frame).astype(np.float32) / 255.0
                arr = torch.from_numpy(arr).permute(2,0,1)  # [3,H,W]
                inp.append(arr)
            inp = torch.stack(inp, dim=0).to(device)  # [B,3,H,W]

            # 送进模型
            data = {
                'input': inp,
                'target_t': torch.zeros_like(inp),
                'target_r': torch.zeros_like(inp)
            }
            model.set_input(data)
            model.inference()
            visual_result = model.get_current_visuals()
            out = visual_result['fake_Ts'][-1]
            out = hist_match_batch_tensor(out,inp)

            # 后处理成图像列表
            out = out.permute(0,2,3,1).cpu().numpy()  # [B,H,W,3]
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

            for idx, arr in enumerate(out):
                img = Image.fromarray(arr)
                results.append(img)

                # 保存单帧
                frame_path = os.path.join(frame_result_dir, f"frame_{start+idx:05d}.png")
                img.save(frame_path)

    # 保存最终视频
    video_path = os.path.join(output_dir, "result.mp4")
    save_video(results, video_path, fps=fps)



def main_worker():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 动态导入模型
    net = importlib.import_module(args.model)
    model = net.MTRREngine(args, device=device)
    model_state = torch.load(args.ckptpath, map_location=device, weights_only=False)
    model.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in model_state['netG_T'].items()})
    model.eval()

    # 读取输入
    if args.input.endswith(".mp4"):
        frames = read_frames_from_video(args.input, target_size=(256,256))
    elif args.input.endswith(".zip"):
        from core.utils import ZipReader
        frames, _ = read_frames_from_zip(args.input, ZipReader, target_size=(256,256))
    else:
        raise ValueError("输入必须是视频mp4或帧zip")

    time1 = time.time()
    # 这里 batch_size 可以是任意数值，不需要和模型写死
    evaluate_batch_model(frames, model, device, args.output, fps=default_fps, batch_size=args.batchsize)
    time2 = time.time()
    print(f"总耗时: {time2-time1:.2f} 秒, 平均每帧: {(time2-time1)/len(frames):.4f} 秒")

if __name__ == "__main__":
    main_worker()
