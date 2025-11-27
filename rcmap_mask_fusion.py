import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

def apply_rcmap_mask_and_skip(test_imgs, test_fake_Ts, test_rcmaps, threshold=10):
    """
    应用RCMap掩码并将结果覆盖到原图像上
    
    Args:
    数字0~1范围内
        test_imgs: 原始图像, 形状为 [B, 3, H, W]
        test_fake_Ts: 预测结果列表, test_fake_Ts 形状为 [B, 3, H, W]
        test_rcmaps: RCMap灰度图, 形状为 [B, 1, H, W]
        threshold: 二值化阈值, 默认30/255
    
    Returns:
        AdditionSkip: 最终结果, 形状为 [B, 3, H, W]
    """
    # 确保输入张量在相同的设备上
    device = test_imgs.device
    test_fake_T3 = test_fake_Ts.to(device)
    test_rcmaps = test_rcmaps.to(device)
    
    # 创建二值掩码 (高于阈值=1, 低于阈值=0)
    mask = (test_rcmaps > threshold/255).float()  # [B, 1, H, W]
    
    # 将掩码扩展到3个通道以匹配RGB图像
    mask_rgb = mask.repeat(1, 3, 1, 1)  # [B, 3, H, W]
    
    # 应用掩码: test_fake_Ts[3] * mask + test_imgs * (1 - mask)
    AdditionSkip = test_fake_T3 * mask_rgb + test_imgs * (1 - mask_rgb)

    # print("mask 统计信息:")
    # print(f"  形状: {mask.shape}")
    # print(f"  均值: {mask.mean().item():.6f}")
    # print(f"  最大值: {mask.max().item():.6f}")
    # print(f"  最小值: {mask.min().item():.6f}")
    # print(f"  标准差: {mask.std().item():.6f}")
    # print("-" * 50)

    return AdditionSkip, mask

# 使用示例
if __name__ == "__main__":
    # 模拟数据 (假设batch_size=4, 图像尺寸256x256)
    batch_size = 4
    H, W = 256, 256
    
    # 创建模拟数据
    test_imgs = torch.randn(batch_size, 3, H, W)  # 原始图像
    test_fake_Ts = [
        torch.randn(batch_size, 3, H, W) for _ in range(4)  # 4个预测结果
    ]
    test_rcmaps = torch.randn(batch_size, 1, H, W) * 50  # RCMap, 值在-50到50之间
    
    # 应用函数
    AdditionSkip = apply_rcmap_mask_and_skip(test_imgs, test_fake_Ts, test_rcmaps, threshold=30)
    
    print(f"输入图像形状: {test_imgs.shape}")
    print(f"输出图像形状: {AdditionSkip.shape}")
    print(f"掩码应用完成!")