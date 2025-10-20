import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader

# ========= 你的工程内导入 =========
from MTRRNet import MTRREngine
from dataset.new_dataset1 import DSRTestDataset, TestDataset, VOCJsonDataset, HyperKDataset

# ---------------- DataLoader：与训练脚本一致的测试集合 ----------------
def build_test_loader(
    img_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
    test_size = (200, 0, 0, 0, 200, 0)
):
    # 按你的脚本拼接六段数据（size 按 test_size 控制）
    t1 = DSRTestDataset(
        datadir="/home/gzm/gzm-MTRRVideo/data/tissue_real",
        fns="/home/gzm/gzm-MTRRVideo/data/tissue_real_index/eval1.txt",
        enable_transforms=False, if_align=True, real=True,
        HW=[img_size, img_size], size=test_size[0], SamplerSize=False
    )

    t2 = TestDataset(
        datadir="/home/gzm/gzm-MTRRVideo/data/hyperK_000",
        fns="/home/gzm/gzm-MTRRVideo/data/hyperK_000_list.txt",
        enable_transforms=False, if_align=True, real=True,
        HW=[img_size, img_size], size=test_size[1]
    )

    t3 = DSRTestDataset(
        datadir="/home/gzm/gzm-RDNet1/dataset/laparoscope_gen",
        fns="/home/gzm/gzm-RDNet1/dataset/laparoscope_gen_index/eval1.txt",
        enable_transforms=False, if_align=True, real=True,
        HW=[img_size, img_size], size=test_size[2]
    )

    voc = VOCJsonDataset(
        "/home/gzm/gzm-RDNet1/dataset/VOC2012",
        "/home/gzm/gzm-RDNet1/dataset/VOC2012/VOC_results_list.json",
        size=test_size[3], enable_transforms=True, HW=[img_size, img_size]
    )

    hk1 = HyperKDataset(
        root="/home/gzm/gzm-MTRRNetv2/data/EndoData",
        json_path="/home/gzm/gzm-MTRRNetv2/data/EndoData/test.json",
        start=369, end=372, size=test_size[4],
        enable_transforms=False, unaligned_transforms=False,
        if_align=True, HW=[img_size, img_size], flag=None,
        SamplerSize=False, color_jitter=False
    )

    hk2 = HyperKDataset(
        root="/home/gzm/gzm-MTRRNetv2/data/EndoData",
        json_path="/home/gzm/gzm-MTRRNetv2/data/EndoData/test.json",
        start=371, end=372, size=test_size[5],
        enable_transforms=False, unaligned_transforms=False,
        if_align=True, HW=[img_size, img_size], flag=None,
        SamplerSize=False, color_jitter=False
    )

    test_data = ConcatDataset([t1, t2, t3, voc, hk1, hk2])
    loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    return loader


# -------------- 吞吐量：严格按你的计时逻辑（首个 batch） --------------
@torch.no_grad()
def measure_throughput(engine: MTRREngine, loader: DataLoader, warmup: int = 50, iters: int = 30):
    """
    与你的伪代码完全一致，但将 `model(images)` 替换为真实推理路径：
      engine.set_input(batch); engine.inference()
    一次迭代处理 batch_size 张图，因此吞吐量 = iters * batch_size / 耗时。
    """
    assert torch.cuda.is_available(), "需要 CUDA 设备进行吞吐量测试。"
    device = torch.device("cuda")
    cudnn.benchmark = True

    # 仅取首个 batch
    for batch in loader:
        # 预热
        for _ in range(warmup):
            engine.set_input(batch)   # 与训练/验证一致：传入 dict/tensors，内部自行搬运到 device
            engine.inference()
        torch.cuda.synchronize(device=device)

        print(f"throughput averaged with {iters} times")
        bs = None
        t1 = time.perf_counter()
        for _ in range(iters):
            engine.set_input(batch)
            engine.inference()
            # 从 batch 推断本轮 batch_size（兼容不同返回结构）
            if bs is None:
                # 你的数据集 batch 通常包含 'fn' 或 'input'，任选其一统计样本数
                if isinstance(batch, dict):
                    if 'input' in batch and torch.is_tensor(batch['input']):
                        bs = batch['input'].shape[0]
                    elif 'fn' in batch:
                        try:
                            bs = len(batch['fn'])
                        except TypeError:
                            bs = 1
                if bs is None:  # 兜底
                    # 尝试从任一 Tensor 值推断
                    for v in (batch.values() if isinstance(batch, dict) else batch):
                        if torch.is_tensor(v):
                            bs = v.shape[0]
                            break
                if bs is None:
                    raise RuntimeError("无法从 batch 推断 batch_size，请检查 DataLoader 返回结构。")
        torch.cuda.synchronize(device=device)
        t2 = time.perf_counter()

        thr = iters * bs / (t2 - t1)
        print(f"batch_size {bs} throughput {thr}")
        return thr, bs

    raise RuntimeError("DataLoader 未产出任何 batch，请检查数据与 batch_size。")


def main():
    ap = argparse.ArgumentParser(description="MTRREngine Inference Throughput Benchmark")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=30)
    # 可选：测试集各段 size，默认与你脚本一致 [200,0,0,0,200,0]
    ap.add_argument("--test_sizes", type=int, nargs=6, default=[200, 0, 0, 0, 200, 0])
    # 可选：权重路径（如要加载 ./model_fit/model_latest.pth，可在 engine 内部自行处理）
    ap.add_argument("--model_path", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA 不可用，无法进行 GPU 吞吐量测试。"

    # 构造一个最小 opts，仅用于初始化引擎（其余参数用默认或由引擎内部处理）
    class Opts:
        pass
    opts = Opts()
    # 你脚本里大量训练参数此处都不需要；只需确保必要路径/开关存在（视引擎实现而定）
    opts.model_path = args.model_path

    device = torch.device("cuda")
    engine = MTRREngine(opts, device)

    # 若需要加载权重：engine 内部通常会在 set_input/inference 前使用到 netG_T 的参数
    if args.model_path and os.path.isfile(args.model_path):
        try:
            # 若你的引擎有封装好的加载接口，优先使用（例如 engine.load_checkpoint(...) 等）
            # 这里演示最简方式：让引擎内部在 __init__ 或首次调用时自行 load
            print(f"[info] model_path set to: {args.model_path}")
        except Exception as e:
            print(f"[warn] failed to load weights via engine: {e}")

    # DataLoader（与你的验证数据管线一致）
    loader = build_test_loader(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=tuple(args.test_sizes)
    )

    # 测吞吐量
    with torch.no_grad():
        thr, bs = measure_throughput(engine, loader, warmup=args.warmup, iters=args.iters)

    # 摘要输出（严格匹配你的格式）
    print("=" * 60)
    print(f"[Summary] device=cuda | mode=FP32 | img={args.img_size}x{args.img_size} | batch_size={bs} | warmup={args.warmup} | iters={args.iters}")
    print(f"[Summary] Throughput: {thr:.2f} images/s")


if __name__ == "__main__":
    main()
