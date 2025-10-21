import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.new_dataset1 import *
from torch.utils.data import ConcatDataset
import os
from PIL import Image

def adaptive_gamma_match_tensor(src_tensor, ref_tensor):
    """
    自适应 Gamma 校正 (Tensor 版本)
    输入: src_tensor (C,H,W) [0,1], ref_tensor (C,H,W) [0,1]
    返回: 校正后的 Tensor (C,H,W) [0,1]
    """
    # 确保输入有正确的维度
    if src_tensor.dim() == 2:
        src_tensor = src_tensor.unsqueeze(0)
    if ref_tensor.dim() == 2:
        ref_tensor = ref_tensor.unsqueeze(0)
    
    # 将输入从[0,1]转换到[0,255]用于亮度计算
    src_uint8 = (src_tensor * 255).byte()
    ref_uint8 = (ref_tensor * 255).byte()
    
    # ITU-R BT.709 亮度权重
    weights = torch.tensor([0.2126, 0.7152, 0.0722], 
                          device=src_tensor.device, 
                          dtype=src_tensor.dtype)
    
    # 使用 einsum 计算亮度
    src_lum = torch.einsum('chw,c->hw', src_uint8.float(), weights)
    ref_lum = torch.einsum('chw,c->hw', ref_uint8.float(), weights)
    
    # 计算中位亮度 (添加极小值防止除零)
    src_med = torch.median(src_lum.flatten()).clamp_min(1e-6)
    ref_med = torch.median(ref_lum.flatten()).clamp_min(1e-6)
    
    # 计算 Gamma 值 (限制在合理范围)
    # 添加安全检查防止 NaN
    if src_med <= 1e-6 or ref_med <= 1e-6:
        gamma = 1.0
    else:
        gamma = torch.log(ref_med) / torch.log(src_med)
        gamma = torch.clamp(gamma, 0.5, 2.0)
    
    # 应用 Gamma 校正
    corrected = torch.pow(src_tensor.float(), gamma)
    
    # 确保值在[0,1]范围内
    corrected = torch.clamp(corrected, 0, 1)
    
    # 添加调试输出
    # print(f"Source median luminance: {src_med.item()}")
    # print(f"Reference median luminance: {ref_med.item()}")
    # print(f"Gamma value: {gamma.item()}")
    # print(f"Corrected min: {corrected.min().item()}, max: {corrected.max().item()}")

    return corrected

def hist_match_channel_tensor(src_channel, ref_channel, bins=256):
    """
    单通道直方图匹配 (Tensor 版本) - 简化实现
    输入: 单通道 Tensor (H,W) [0,1]
    返回: 匹配后的单通道 Tensor (H,W) [0,1]
    """
    # 确保输入在[0,1]范围内
    src_channel = torch.clamp(src_channel, 0, 1)
    ref_channel = torch.clamp(ref_channel, 0, 1)
    
    # 将输入从[0,1]转换到[0,255]
    src_uint8 = (src_channel * 255).byte()
    ref_uint8 = (ref_channel * 255).byte()
    
    device = src_channel.device
    
    # 计算直方图
    src_hist = torch.histc(src_uint8.float(), bins=256, min=0, max=255)
    ref_hist = torch.histc(ref_uint8.float(), bins=256, min=0, max=255)
    
    # 添加平滑项防止除零
    eps = 1e-8
    src_hist += eps
    ref_hist += eps
    
    # 计算累积分布函数 (CDF)
    src_cdf = src_hist.cumsum(0) / src_hist.sum()
    ref_cdf = ref_hist.cumsum(0) / ref_hist.sum()
    
    # 创建LUT：对于源图像的每个灰度级，找到其在参考图像中对应的灰度级
    lut = torch.zeros(256, device=device)
    for i in range(256):
        # 找到参考CDF中大于等于源CDF的最小索引
        j = torch.searchsorted(ref_cdf, src_cdf[i], right=True)
        j = torch.clamp(j, 0, 255)
        lut[i] = j
    
    # 应用LUT
    matched_uint8 = lut[src_uint8.long()].clamp(0, 255)
    
    # 转换回[0,1]范围
    return matched_uint8.float() / 255.0

def hist_match_rgb_tensor(src_tensor, ref_tensor, bins=256):
    """
    RGB 直方图匹配 (Tensor 版本)
    输入: src_tensor (C,H,W) [0,1], ref_tensor (C,H,W) [0,1]
    返回: 匹配后的 Tensor (C,H,W) [0,1]
    """
    # 确保输入在[0,1]范围内
    src_tensor = torch.clamp(src_tensor, 0, 1)
    ref_tensor = torch.clamp(ref_tensor, 0, 1)
    
    # 1. 自适应 Gamma 校正
    gamma_corrected = adaptive_gamma_match_tensor(src_tensor, ref_tensor)
    
    # 2. 逐通道直方图匹配
    matched_channels = []
    for c in range(3):
        matched = hist_match_channel_tensor(
            gamma_corrected[c], 
            ref_tensor[c], 
            bins=bins
        )
        matched_channels.append(matched)
    
    return torch.stack(matched_channels)

def hist_match_batch_tensor(src_batch, ref_batch, bins=256):
    """
    批量 RGB 直方图匹配 (Tensor 版本)
    输入: src_batch (N,C,H,W) [0,1], ref_batch (N,C,H,W) [0,1]
    返回: 匹配后的 Tensor (N,C,H,W) [0,1]
    """
    matched_batch = []
    for i in range(src_batch.size(0)):
        matched = hist_match_rgb_tensor(
            src_batch[i], 
            ref_batch[i], 
            bins=bins
        )
        matched_batch.append(matched)
    return torch.stack(matched_batch)















# 测试代码
if __name__ == "__main__":
    # 创建保存图像的目录
    save_dir = "/home/gzm/gzm-MTRRNetv2/hist_match_results"
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集
    test_data_dir1 = '/home/gzm/gzm-MTRRVideo/data/tissue_real'
    test_data1 = DSRTestDataset(
        datadir=test_data_dir1, 
        fns='/home/gzm/gzm-MTRRVideo/data/tissue_real_index/eval1.txt', 
        enable_transforms=False, 
        if_align=True, 
        real=True, 
        HW=[256,256], 
        size=200, 
        SamplerSize=False, 
        color_match=True
    )    
    train_loader = torch.utils.data.DataLoader(
        test_data1, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0, 
        drop_last=False, 
        pin_memory=True
    )

    def save_image(tensor, path):
        """
        将张量保存为图像文件
        输入: 
            tensor: (C,H,W) 图像张量 [0,1]
            path: 保存路径
        """
        # 转换为PIL图像
        tensor = tensor.mul(255).byte().cpu()
        if tensor.dim() == 3 and tensor.size(0) == 3:
            img = Image.fromarray(tensor.permute(1, 2, 0).numpy(), 'RGB')
        elif tensor.dim() == 3 and tensor.size(0) == 1:
            img = Image.fromarray(tensor.squeeze(0).numpy(), 'L')
        else:
            img = Image.fromarray(tensor.numpy())
        
        # 保存图像
        img.save(path)

    # 测试单张图像匹配
    src_tensor = test_data1[0]['input']  # 模拟源图像
    ref_tensor = test_data1[0]['target_t']  # 模拟参考图像
    
    print(f"Source min: {src_tensor.min().item()}, max: {src_tensor.max().item()}")
    print(f"Reference min: {ref_tensor.min().item()}, max: {ref_tensor.max().item()}")
    
    matched_tensor = hist_match_rgb_tensor(src_tensor, ref_tensor)
    
    print(f"Matched min: {matched_tensor.min().item()}, max: {matched_tensor.max().item()}")

    # 保存单张图像匹配结果
    save_image(src_tensor, os.path.join(save_dir, "source.png"))
    save_image(ref_tensor, os.path.join(save_dir, "reference.png"))
    save_image(matched_tensor, os.path.join(save_dir, "matched.png"))

    # 测试批量处理
    # 注意：DataLoader 需要通过迭代获取批次数据
    for batch_idx, batch in enumerate(train_loader):
        src_batch = batch['input']  # 模拟批量源图像 (N,C,H,W)
        ref_batch = batch['target_t']  # 模拟批量参考图像 (N,C,H,W)
        
        print(f"Batch {batch_idx} Source min: {src_batch.min().item()}, max: {src_batch.max().item()}")
        print(f"Batch {batch_idx} Reference min: {ref_batch.min().item()}, max: {ref_batch.max().item()}")
        
        # 处理当前批次
        matched_batch = hist_match_batch_tensor(src_batch, ref_batch)
        print(f"Batch {batch_idx} Matched min: {matched_batch.min().item()}, max: {matched_batch.max().item()}")
        print(f"Batch {batch_idx} Matched Shape:", matched_batch.shape)
        
        # 为当前批次创建目录
        batch_dir = os.path.join(save_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # 保存当前批次的所有图像
        for i in range(src_batch.size(0)):
            save_image(src_batch[i], os.path.join(batch_dir, f"source_{i}.png"))
            save_image(ref_batch[i], os.path.join(batch_dir, f"reference_{i}.png"))
            save_image(matched_batch[i], os.path.join(batch_dir, f"matched_{i}.png"))
        
        # 只处理第一个批次（根据需要可以移除这个break以处理所有批次）
        break

    print(f"所有图像已保存到: {save_dir}")