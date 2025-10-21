import torch
from torchmetrics import PeakSignalNoiseRatio
from pytorch_msssim import ssim
import torch.nn.functional as F
import torchvision.models as models

class SSIMLoss(torch.nn.Module):
    """可微分的SSIM损失"""
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)        
        ssim_value = ssim(img1, img2,
                        data_range=self.data_range,
                        size_average=self.size_average)
        return 1 - ssim_value

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.ssim_metric = SSIMLoss(data_range=1.0).to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        
        # 1. 预先加载VGG模型并固定参数，避免重复加载
        self.vgg = models.vgg19(pretrained=True).features[:30].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.spr_loss = SpecularPriorityReconLoss(vgg=None, feature_layers={1,6,11,20,29},
                                             mse_weight=1.0, vgg_weight=0.1,
                                             alpha=2.0, smooth_ks=7).to('cuda')            
            
        # 存储VGG网络中我们关心的层索引
        self.feature_layers = {1, 6, 11, 20, 29}
        
        # 2. 调整损失权重，避免过大权重
        self.ssim_loss_weight = 1.0
        self.vgg_loss_weight = 0.1
        self.mse_loss_weight = 1.0
        self.color_loss_weight = 0
        
        self.fake_R_weight = 0.3
        self.fake_T_weight = 1.0
        self.Rcmaps_weight = 0
        self.all_img_weight = 0

        # 对于残留反光区域的权重
        self.spr_loss_weight = 1.0

        # fake_Ts范围软约束正则项权重
        self.fake_Ts_range_weight = 0.2

    def forward(self, fake_Ts, label1, input_image, rcmaps, fake_Rs, label2):

        loss_spr = self.spr_loss(fake_Ts, label1, label2)

        # 计算fake_Rs的损失
        fake_R_mse_loss = F.mse_loss(fake_Rs, label2) * self.mse_loss_weight
        fake_R_vgg_loss = self.compute_perceptual_loss(fake_Rs, label2) * self.vgg_loss_weight
        fake_R_ssim_loss = (self.ssim_metric(fake_Rs, label2)) * self.ssim_loss_weight
        fake_R_color_loss = color_mean_loss(fake_Rs, label2) * self.color_loss_weight


        # 计算fake_Ts的损失
        fake_T_mse_loss = F.mse_loss(fake_Ts, label1) * self.mse_loss_weight
        fake_T_vgg_loss = self.compute_perceptual_loss(fake_Ts, label1) * self.vgg_loss_weight
        fake_T_ssim_loss = (self.ssim_metric(fake_Ts, label1)) * self.ssim_loss_weight
        fake_T_color_loss = color_mean_loss(fake_Ts, label1) * self.color_loss_weight

        # fake_Ts范围软约束正则项：惩罚超出[0,1]范围的值
        fake_Ts_range_penalty = (
            torch.mean(F.relu(-fake_Ts)) +  # 惩罚小于0的值
            torch.mean(F.relu(fake_Ts - 1))  # 惩罚大于1的值
        ) * self.fake_Ts_range_weight


        # Rcmaps检测，使用更安全的方式计算
        # 5. 使用torch.clamp的同时注意梯度流
        # I_R_diff = torch.clamp(input_image - label1, 0.0, 1.0)
        I_R_diff = (input_image - label1)
        RCMap_test_img = ((rcmaps) * input_image)
        # RCMap_test_img = torch.clamp((rcmaps) * input_image, 0.0, 1.0)
        
        Rcmaps_mse_loss = F.mse_loss(RCMap_test_img, I_R_diff) * self.mse_loss_weight
        Rcmaps_vgg_loss = self.compute_perceptual_loss(RCMap_test_img, I_R_diff) * self.vgg_loss_weight
        Rcmaps_ssim_loss = (self.ssim_metric(RCMap_test_img, I_R_diff)) * self.ssim_loss_weight
        Rcmaps_color_loss = color_mean_loss(RCMap_test_img, I_R_diff) * self.color_loss_weight
        
        Rcmaps_l1_loss = torch.mean(torch.abs(rcmaps)) /3 * 0.2

        
        # 总和检测
        # all_img = torch.clamp(fake_Ts * rcmaps + fake_Rs, 0.0, 1.0)
        all_img = (fake_Ts * rcmaps + fake_Rs)
        all_img_mse_loss = F.mse_loss(all_img, input_image) * self.mse_loss_weight
        all_img_vgg_loss = self.compute_perceptual_loss(all_img, input_image) * self.vgg_loss_weight
        all_img_ssim_loss = (self.ssim_metric(all_img, input_image)) * self.ssim_loss_weight
        all_img_color_loss = color_mean_loss(all_img, input_image) * self.color_loss_weight


        # 6. 分别计算损失并应用权重，避免中间项过大

        loss_spr = loss_spr * self.spr_loss_weight

        mse_loss = (
            fake_R_mse_loss * self.fake_R_weight + 
            fake_T_mse_loss * self.fake_T_weight + 
            Rcmaps_mse_loss * self.Rcmaps_weight + 
            all_img_mse_loss * self.all_img_weight +
            Rcmaps_l1_loss
        )
        
        # 7. VGG损失除以5
        vgg_loss = (
            fake_R_vgg_loss * self.fake_R_weight + 
            fake_T_vgg_loss * self.fake_T_weight + 
            Rcmaps_vgg_loss * self.Rcmaps_weight + 
            all_img_vgg_loss * self.all_img_weight
        )
        
        ssim_loss = (
            fake_R_ssim_loss *  self.fake_R_weight + 
            fake_T_ssim_loss *  self.fake_T_weight + 
            Rcmaps_ssim_loss *  self.Rcmaps_weight + 
            all_img_ssim_loss * self.all_img_weight
        )

        color_loss = (
            fake_R_color_loss   * self.fake_R_weight + 
            fake_T_color_loss   * self.fake_T_weight +
            Rcmaps_color_loss   * self.Rcmaps_weight +
            all_img_color_loss  * self.all_img_weight
        )
        
        # 8. 对vgg_loss除以10而非5，进一步降低其影响
        total_loss = mse_loss + vgg_loss + ssim_loss + color_loss + fake_Ts_range_penalty + loss_spr
        
        # 9. 确保最终损失没有NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN or Inf detected in loss calculation!")
            # 回退到简单的MSE损失
            # total_loss = F.mse_loss(fake_Ts, label1) + F.mse_loss(fake_Rs, label2)

        loss_table = {
            'fake_R_mse_loss': fake_R_mse_loss,
            'fake_T_mse_loss': fake_T_mse_loss,
            'Rcmaps_mse_loss': Rcmaps_mse_loss,
            'all_img_mse_loss': all_img_mse_loss,
            'mse_loss': mse_loss,
            'fake_R_vgg_loss': fake_R_vgg_loss,
            'fake_T_vgg_loss': fake_T_vgg_loss,
            'Rcmaps_vgg_loss': Rcmaps_vgg_loss,
            'all_img_vgg_loss': all_img_vgg_loss,
            'vgg_loss': vgg_loss,
            'fake_R_ssim_loss': fake_R_ssim_loss,
            'fake_T_ssim_loss': fake_T_ssim_loss,
            'Rcmaps_ssim_loss': Rcmaps_ssim_loss,
            'all_img_ssim_loss': all_img_ssim_loss,
            'ssim_loss': ssim_loss,
            'fake_R_color_loss':fake_R_color_loss,
            'fake_T_color_loss':fake_T_color_loss,
            'Rcmaps_color_loss':Rcmaps_color_loss,
            'all_img_color_loss':all_img_color_loss,
            'color_loss':color_loss,
            'fake_Ts_range_penalty': fake_Ts_range_penalty,
            'spr_loss': loss_spr,
            'total_loss': total_loss,
        }
        
        # 返回的仍然是相同格式，但vgg_loss已经除以10了
        return loss_table, mse_loss, vgg_loss, ssim_loss, fake_Ts_range_penalty, total_loss

    def compute_perceptual_loss(self, x, y):
        """
        稳定改进的感知损失计算：
        1. 使用预加载的VGG模型
        2. 对每一层特征进行归一化
        3. 使用渐进式权重增加深层特征的重要性
        4. 添加数值稳定性措施
        """
        # 10. 确保输入在有效范围内 vgg预训练的条件必须得满足
        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)
        x = (x - self.vgg_mean) / self.vgg_std
        y = (y - self.vgg_mean) / self.vgg_std
        loss = 0.0
        # 11. 保存中间层特征，避免重复计算
        x_features = []
        y_features = []
        
        # 提取特征但不计算梯度
        with torch.no_grad():
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                y = layer(y)
                
                if i in self.feature_layers:
                    x_features.append(x)
                    y_features.append(y)
                    
                if i >= max(self.feature_layers):
                    break
        
        # 12. 设置递增权重，使深层特征权重更大
        weights = [0.1, 0.2, 0.4, 0.8, 1.0]
        
        # 13. 使用MSE而非L1损失，提高稳定性
        for idx, (x_feat, y_feat) in enumerate(zip(x_features, y_features)):
            # 重新启用梯度计算
            x_feat = x_feat
            y_feat = y_feat
            
            # 使用均方误差并添加数值稳定性
            feat_diff = F.mse_loss(x_feat, y_feat)
            
            # 14. 对每层损失进行裁剪，避免极端值
            # feat_loss = torch.clamp(feat_diff, 0.0, 10.0)
            feat_loss = (feat_diff)
            loss = loss + weights[idx] * feat_loss
            
        # 15. 归一化损失，使其不会因为层数增加而过大
        return loss / sum(weights)
    

def color_mean_loss(pred, target):
    # 计算RGB三个通道均值之间的差距（颜色引导）
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    return F.l1_loss(pred_mean, target_mean)


# ==========================
# 新增：反光优先重建损失（只做 MSE + VGG）
# ==========================

import torch.nn as nn
class SpecularPriorityReconLoss(nn.Module):
    """
    SPRLoss: Specular Priority Reconstruction Loss
    - 用 label2(反光GT) 构造平滑灰度权重 w ∈ [1, alpha]，在反光区域更关注 fake_Ts ≈ label1
    - 组成：加权 MSE（像素域） + 特征域加权 VGG 感知损失（方案B）
    - 直接返回标量损失（不改你现有任何接口）
    """

    def __init__(self,
                 vgg: nn.Module | None = None,
                 feature_layers: set[int] = {1, 6, 11, 20, 29},
                 mse_weight: float = 1.0,
                 vgg_weight: float = 0.1,
                 alpha: float = 2.0,
                 smooth_ks: int = 7):
        """
        Args:
            vgg: 可选外部注入的 VGG 特征提取器（如 vgg19.features[:30]，需 eval() 且冻结）
                 传 None 则内部自动创建并冻结
            feature_layers: 参与感知损失的 VGG 层索引集合
            mse_weight: 像素域加权 MSE 的系数
            vgg_weight: 特征域加权 VGG 的系数
            alpha: 反光区权重上限（权重范围 [1, alpha]），典型 1.5~3.0
            smooth_ks: 平滑核大小（AvgPool2d），0/1 表示不平滑；常用 3/5/7
        """
        super().__init__()
        self.mse_w = float(mse_weight)
        self.vgg_w = float(vgg_weight)
        self.alpha = float(alpha)
        self.smooth_ks = int(smooth_ks)

        # VGG 归一化常量（注册到 buffer 便于 to(device) 与保存/加载）
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # VGG 特征网络
        if vgg is None:
            vgg = models.vgg19(pretrained=True).features[:30]
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
        self.vgg = vgg
        self.feature_layers = set(feature_layers)

        # 分层权重（由浅到深）
        self.layer_weights = [0.1, 0.2, 0.4, 0.8, 1.0]

    # --- 工具：RGB → 亮度 ---
    @staticmethod
    def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> [B,1,H,W]
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1,3,1,1)
        return (x * w).sum(1, keepdim=True)

    # --- 从 label2 生成平滑灰度权重 [1, alpha] ---
    def _build_weight(self, label2: torch.Tensor) -> torch.Tensor:
        luma = torch.clamp(self._rgb_to_luma(label2), 0.0, 1.0)  # [B,1,H,W]
        if self.smooth_ks and self.smooth_ks > 1:
            k = self.smooth_ks
            pad = k // 2
            pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=pad, count_include_pad=False)
            luma = pool(luma)
            luma = torch.clamp(luma, 0.0, 1.0)
        # 权重：暗处≈1，高亮≈alpha
        return 1.0 + (self.alpha - 1.0) * luma  # [B,1,H,W]

    # --- 加权 MSE（像素域）---
    @staticmethod
    def _weighted_mse(pred: torch.Tensor, tgt: torch.Tensor, w1c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # w1c: [B,1,H,W] 或 [B,C,H,W]；扩到通道维
        if w1c.dim() == 4 and w1c.size(1) == 1:
            w1c = w1c.expand_as(pred)
        num = (w1c * (pred - tgt) ** 2).sum()
        den = w1c.sum() * pred.size(1)
        return num / (den + eps)

    # --- 特征域加权 VGG 感知损失 ---
    def _masked_perceptual(self, x: torch.Tensor, y: torch.Tensor, w1c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)
        x = (x - self.vgg_mean) / self.vgg_std
        y = (y - self.vgg_mean) / self.vgg_std

        loss = 0.0
        li = 0
        cur_x, cur_y = x, y
        for i, layer in enumerate(self.vgg):
            cur_x = layer(cur_x)
            cur_y = layer(cur_y)

            if i in self.feature_layers:
                # 将像素权重映射到当前特征尺度，并扩到通道维
                mask = F.interpolate(w1c, size=cur_x.shape[2:], mode='bilinear', align_corners=False)  # [B,1,h,w]
                mask = mask.expand_as(cur_x)  # [B,C,h,w]

                num = (mask * (cur_x - cur_y) ** 2).sum()
                den = mask.sum() * cur_x.size(1)
                l = num / (den + eps)
                loss = loss + self.layer_weights[li] * l
                li += 1

            if i >= max(self.feature_layers):
                break

        return loss / (sum(self.layer_weights) + eps)

    def forward(self,
                fake_Ts: torch.Tensor,  # 预测清晰图
                label1:  torch.Tensor,  # GT 清晰图
                label2:  torch.Tensor   # 反光 GT（用于构造权重）
                ) -> torch.Tensor:
        """
        返回：标量损失（mse_weight * weighted_MSE + vgg_weight * weighted_VGG）
        """
        w = self._build_weight(label2)  # [B,1,H,W], ∈ [1, alpha]
        loss_mse = self._weighted_mse(fake_Ts, label1, w)
        loss_vgg = self._masked_perceptual(fake_Ts, label1, w)
        return self.mse_w * loss_mse + self.vgg_w * loss_vgg
