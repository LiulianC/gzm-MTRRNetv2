import os
import argparse
import importlib
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from video_func import read_frames_from_zip, save_video
from util.color_enhance import hist_match_batch_tensor
from core.utils import ZipReader

import datetime
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch infer HyperK zip sequences to MP4s (input and output videos)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=str, default="/home/gzm/gzm-compare/dataset/JPEGImages", help="Root folder containing hyperK_XXX.zip")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive), e.g. 0 for hyperK_000.zip")
    parser.add_argument("--end", type=int, default=342, help="End index (inclusive), e.g. 342 for hyperK_342.zip")
    parser.add_argument("--frames", type=int, default=600, help="Number of leading frames to use; skip zip if fewer than this")
    parser.add_argument("--output", type=str, default=f"./test_results/videos/{current_time}", help="Output directory to store NNN-in.mp4 and NNN-out.mp4")
    parser.add_argument("--model", type=str, default="MTRRNet", help="Model module name (with MTRREngine class)")
    parser.add_argument("--ckptpath", type=str, default="./model_fit/model_62_best1.pth", help="Checkpoint .pth path")
    parser.add_argument("--batchsize", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the saved mp4 videos")
    parser.add_argument("--resize", type=str, default="256,256", help="Resize WxH for frames; must be consistent with training/inference")
    return parser.parse_args()


def _parse_size(s: str) -> Tuple[int, int]:
    try:
        w_str, h_str = s.split(",")
        return int(w_str), int(h_str)
    except Exception:
        return 256, 256


def _to_tensor_batch(frames: List[Image.Image]) -> torch.Tensor:
    arrs = []
    for f in frames:
        arr = np.array(f).astype(np.float32) / 255.0
        arr = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
        arrs.append(arr)
    return torch.stack(arrs, dim=0)


def _infer_frames(frames: List[Image.Image], model, device: torch.device, batch_size: int) -> List[Image.Image]:
    """Run model on frames and return processed frames as PIL images."""
    results: List[Image.Image] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(frames), batch_size), desc="Inferring", ncols=120):
            end = min(start + batch_size, len(frames))
            batch_frames = frames[start:end]

            inp = _to_tensor_batch(batch_frames).to(device)

            feed = {
                'input': inp,
                'target_t': torch.zeros_like(inp),
                'target_r': torch.zeros_like(inp),
            }
            model.set_input(feed)
            model.inference()

            visuals = model.get_current_visuals()
            out = visuals['fake_Ts'][-1]
            # Optional color histogram match for better visual consistency vs input
            out = hist_match_batch_tensor(out, inp)

            out = out.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            out = (out * 255.0).astype(np.uint8)
            for i in range(out.shape[0]):
                results.append(Image.fromarray(out[i]))
    return results


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    W, H = _parse_size(args.resize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    net = importlib.import_module(args.model)
    model = net.MTRREngine(args, device=device)
    if not os.path.exists(args.ckptpath):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckptpath}")
    model_state = torch.load(args.ckptpath, map_location=device, weights_only=False)
    model.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in model_state['netG_T'].items()})
    model.eval()

    # Iterate over zips
    for idx in range(args.start, args.end + 1):
        zip_path = os.path.join(args.input_root, f"hyperK_{idx:03d}.zip")
        if not os.path.exists(zip_path):
            print(f"[skip] not found: {zip_path}")
            continue

        try:
            frames, _ = read_frames_from_zip(zip_path, ZipReader, target_size=(W, H))
        except Exception as e:
            print(f"[error] read zip failed: {zip_path} -> {e}")
            continue

        if len(frames) < args.frames:
            print(f"[skip] {zip_path} has only {len(frames)} frames (< {args.frames})")
            continue

        frames = frames[: args.frames]

        # Save input video
        in_path = os.path.join(args.output, f"{idx:03d}-in.mp4")
        print(f"[save] input -> {in_path}")
        save_video(frames, in_path, fps=args.fps)

        # Run model and save output video
        print(f"[infer] {zip_path} ...")
        out_frames = _infer_frames(frames, model, device, args.batchsize)
        out_path = os.path.join(args.output, f"{idx:03d}-out.mp4")
        print(f"[save] output -> {out_path}")
        save_video(out_frames, out_path, fps=args.fps)

    print("[✓] All done.")


if __name__ == "__main__":
    main()
