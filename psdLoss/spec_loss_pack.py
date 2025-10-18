"""
SpecularityNet loss pack: unify and expose the actually used loss functions
based on the current training configuration (opt).

This module aggregates the losses used in `specularitynetModel.backward_G()`
and `backward_D()` and provides a simple, uniform interface for usage in
other scripts or models.

Interface overview
- build_spec_losses(opt, device=None, vgg=None) -> SpecularityNetLossPack
    - opt: an object or dict providing at least the following attributes/keys:
    - lambda_gan, gan_type, lambda_feat
        - lambda_vgg, lambda_ssim, lambda_detect, lambda_coarse
    - unaligned_loss, vgg_layer
  - device: torch device; if None, will infer cuda:0 if available else cpu
  - vgg: optional prebuilt Vgg19-like feature extractor; if None, will build

The returned SpecularityNetLossPack exposes methods:
- names(): list of enabled loss names
- pixel(pred, target) -> tensor
- vgg(pred, target) -> tensor or None (if disabled)
- ssim(pred, target) -> tensor or None (if disabled)
- det(pred_mask, gt_mask) -> tensor or None (if disabled)
- gan_d(netD, realA, fakeB, realB) -> (loss_D, pred_fake, pred_real) or None
- gan_g(netD, realA, fakeB, realB, return_feat=False)
    -> loss_G or (loss_G, loss_Feat) when feature-matching is enabled
- unaligned(pred, target) -> tensor or None (for CX/VGG/MSE unaligned loss)

Notes
- "Enabled" is determined by the numeric lambdas in opt (e.g., lambda_vgg>0)
  and the chosen gan_type/unaligned_loss. Even if a particular loss is only
  applied conditionally at runtime (e.g., detection requires mask/pred list,
  unaligned requires data flag), this pack prepares the criterion when the
  corresponding lambda/type indicates it may be used.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# Reuse implementations from the project
from .losses import (
    ContentLoss,
    MultipleLoss,
    GradientLoss,
    VGGLoss,
    DiscLoss,
    DiscLossR,
    DiscLossRa,
    MSSSIM,
)
from .focal_loss import BinaryFocalLoss
from .vgg import Vgg19


def _get_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _as_opt_dict(opt: Any) -> Dict[str, Any]:
    """Allow passing either a simple "opt" object with attributes or a dict."""
    if isinstance(opt, dict):
        return opt
    # Fallback: extract attributes of a simple namespace-like object
    keys = [
        "lambda_gan", "gan_type", "lambda_feat",
        "lambda_vgg", "lambda_ssim", "lambda_detect", "lambda_coarse",
        "unaligned_loss", "vgg_layer",
    ]
    d: Dict[str, Any] = {}
    for k in keys:
        if hasattr(opt, k):
            d[k] = getattr(opt, k)
    return d


class SpecularityNetLossPack(nn.Module):
    def __init__(self, opt: Any, device: Optional[torch.device] = None, vgg: Optional[nn.Module] = None):
        super().__init__()
        od = _as_opt_dict(opt)
        self.device = _get_device(device)

        # Defaults (in case keys are absent)
        self.lambda_gan: float = float(od.get("lambda_gan", 0.0))
        self.gan_type: str = str(od.get("gan_type", "rasgan")).lower()
        self.lambda_feat: float = float(od.get("lambda_feat", -1))
        self.lambda_vgg: float = float(od.get("lambda_vgg", 0.0))
        self.lambda_ssim: float = float(od.get("lambda_ssim", -1))
        self.lambda_detect: float = float(od.get("lambda_detect", 0.0))
        self.lambda_coarse: float = float(od.get("lambda_coarse", 1.0))
        self.unaligned_loss: str = str(od.get("unaligned_loss", "vgg")).lower()
        self.vgg_layer: int = int(od.get("vgg_layer", 31))

        # Pixel loss (always constructed; used in aligned mode)
        self._pixel = ContentLoss()
        self._pixel.initialize(MultipleLoss([nn.MSELoss(), GradientLoss()], [2, 4]))

        # VGG backbone for perceptual / CX losses
        self._vgg_backbone = vgg if vgg is not None else Vgg19(requires_grad=False).to(self.device)

        # Refined/coarse VGG perceptual loss
        self._vgg = None
        if self.lambda_vgg > 0:
            vggloss = ContentLoss()
            vggloss.initialize(VGGLoss(self._vgg_backbone))
            self._vgg = vggloss

        # Unaligned content loss choice
        self._unaligned: Optional[ContentLoss] = None
        if self.unaligned_loss in {"vgg", "ctx", "mse", "ctx_vgg"}:
            cx = ContentLoss()
            if self.unaligned_loss == "vgg":
                cx.initialize(VGGLoss(self._vgg_backbone, weights=[0.1], indices=[self.vgg_layer]))
            elif self.unaligned_loss == "ctx":
                # Use default indices/weights per model definition
                from .losses import CXLoss  # local import to avoid circulars
                cx.initialize(CXLoss(self._vgg_backbone, weights=[0.1, 0.1, 0.1], indices=[8, 13, 22]))
            elif self.unaligned_loss == "mse":
                cx.initialize(nn.MSELoss())
            elif self.unaligned_loss == "ctx_vgg":
                from .losses import CXLoss, CX_loss
                cx.initialize(CXLoss(
                    self._vgg_backbone,
                    weights=[0.1, 0.1, 0.1, 0.1],
                    indices=[8, 13, 22, 31],
                    criterions=[CX_loss] * 3 + [nn.L1Loss()],
                ))
            self._unaligned = cx

        # SSIM
        self._ssim = MSSSIM() if self.lambda_ssim > 0 else None

        # Detection loss (focal)
        self._det = BinaryFocalLoss() if self.lambda_detect > 0 else None

        # GAN
        self._gan = None
        if self.lambda_gan > 0:
            if self.gan_type in ("sgan", "gan"):
                self._gan = DiscLoss()
            elif self.gan_type == "rsgan":
                self._gan = DiscLossR()
            elif self.gan_type == "rasgan":
                self._gan = DiscLossRa()
            else:
                raise ValueError(f"Unsupported gan_type: {self.gan_type}")
            # Initialize GAN loss with a tensor factory matching device/dtype
            # The project uses a Tensor factory; here we pass torch.FloatTensor on device
            tensor = torch.cuda.FloatTensor if self.device.type == "cuda" else torch.FloatTensor
            self._gan.initialize(opt, tensor)

    # --------- public API ---------
    def names(self) -> List[str]:
        """Return names of losses prepared based on current opt."""
        names: List[str] = ["pixel"]
        if self._vgg is not None:
            names.append("vgg")
        if self._ssim is not None:
            names.append("ssim")
        if self._det is not None:
            names.append("detect")
        if self._gan is not None:
            names.append(f"gan:{self.gan_type}")
        if self._unaligned is not None:
            names.append(f"unaligned:{self.unaligned_loss}")
        return names

    # Individual loss calls (return None if not enabled)
    def pixel(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._pixel.get_loss(pred, target)

    def vgg(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if self._vgg is None:
            return None
        return self._vgg.get_loss(pred, target)

    def ssim(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if self._ssim is None:
            return None
        return self._ssim(pred, target)

    def det(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if self._det is None:
            return None
        return self._det(pred_mask, gt_mask)

    def unaligned(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if self._unaligned is None:
            return None
        return self._unaligned.get_loss(pred, target)

    # GAN helpers
    def gan_d(self, netD: nn.Module, realA: Optional[torch.Tensor], fakeB: Optional[torch.Tensor], realB: Optional[torch.Tensor]):
        """Discriminator loss; returns (loss_D, pred_fake, pred_real) or None if GAN disabled."""
        if self._gan is None:
            return None
        return self._gan.get_loss(netD, realA, fakeB, realB)

    def gan_g(self, netD: nn.Module, realA: Optional[torch.Tensor], fakeB: torch.Tensor, realB: Optional[torch.Tensor], return_feat: bool = False):
        """Generator loss; for RaSGAN + feature matching, can return (loss_G, loss_Feat)."""
        if self._gan is None:
            return None
        # Feature matching only if lambda_feat > 0 and gan supports it
        if return_feat and self.gan_type == "rasgan" and self.lambda_feat > 0 and hasattr(self._gan, "get_g_feat_loss"):
            return self._gan.get_g_feat_loss(netD, realA, fakeB, realB)
        return self._gan.get_g_loss(netD, realA, fakeB, realB)

    # --------- high-level aggregator ---------
    def compute_total(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor],
        *,
        aligned: bool = True,
        # for GAN
        netD: Optional[nn.Module] = None,
        realA: Optional[torch.Tensor] = None,
        # optional heads
        coarse_list: Optional[List[torch.Tensor]] = None,
        detect_list: Optional[List[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        include_d_loss: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and a dict of components following this project's logic.

        Parameters
        - pred: 模型 refined 输出
        - target: 对齐标签图（aligned=True 时需要；unaligned 时也用于感知/内容比较）
        - aligned: True 走对齐分支（pixel/vgg/ssim/detect/coarse），False 走 unaligned 分支
        - netD/realA: 可选，用于 GAN（realA 通常就是输入图像）
        - coarse_list: 可选，coarse 输出列表
        - detect_list/mask: 可选，检测分支输出与对应 mask
        - include_d_loss: 如为 True 且提供 netD/target，则额外返回 D 损失（不做 backward）

        Returns
        - total_loss: 聚合后的总损失（tensor）
        - loss_dict: 各组件损失的字典（只包含实际计算的项）
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Aligned branch: pixel (+ optional vgg/ssim), coarse, detect
        if aligned:
            # pixel (refined)
            if target is None:
                raise ValueError("aligned=True 需要提供 target")
            l_pixel = self.pixel(pred, target)
            loss_dict["pixel"] = l_pixel
            total = total + l_pixel

            # coarse list
            if coarse_list:
                closs = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                for c in coarse_list:
                    # pixel on coarse
                    cl = self.pixel(c, target)
                    # add vgg for coarse if enabled
                    if self._vgg is not None and self.lambda_vgg > 0:
                        cv = self.vgg(c, target)
                        if cv is not None:
                            cl = cl + cv * self.lambda_vgg
                    closs = closs + cl
                closs = closs / float(len(coarse_list))
                # weight by lambda_coarse (来自 opt)
                closs_w = closs * float(self.lambda_coarse)
                loss_dict["coarse"] = closs_w
                total = total + closs_w

            # refined VGG
            if self._vgg is not None and self.lambda_vgg > 0:
                lvgg = self.vgg(pred, target)
                if lvgg is not None:
                    lvgg_w = lvgg * self.lambda_vgg
                    loss_dict["vgg"] = lvgg_w
                    total = total + lvgg_w

            # SSIM
            if self._ssim is not None and self.lambda_ssim > 0:
                lssim = self.ssim(pred, target)
                if lssim is not None:
                    lssim_w = lssim * self.lambda_ssim
                    loss_dict["ssim"] = lssim_w
                    total = total + lssim_w

            # Detect head
            if detect_list and mask is not None and mask.numel() > 0 and self._det is not None and self.lambda_detect > 0:
                dloss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                for d in detect_list:
                    dloss = dloss + self.det(d, mask)
                dloss = dloss / float(len(detect_list))
                dloss_w = dloss * self.lambda_detect
                loss_dict["detect"] = dloss_w
                total = total + dloss_w
        else:
            # Unaligned branch: use configured content loss (e.g., VGG/CX/MSE)
            if self._unaligned is None:
                raise ValueError("unaligned=True 但未配置 unaligned 损失 (opt.unaligned_loss)")
            if target is None:
                raise ValueError("unaligned=True 需要提供 target 用于内容损失")
            l_u = self.unaligned(pred, target)
            loss_dict["unaligned"] = l_u
            total = total + l_u

        # GAN generator loss
        if self._gan is not None and self.lambda_gan > 0 and netD is not None:
            g_out = self.gan_g(netD, realA, pred, target)
            if isinstance(g_out, tuple):
                l_g, l_feat = g_out
                loss_dict["gan_feat"] = l_feat
            else:
                l_g = g_out
            l_g_w = l_g * self.lambda_gan
            loss_dict["gan_g"] = l_g_w
            total = total + l_g_w

            # Optional D loss (not applied to total)
            if include_d_loss:
                d_out = self.gan_d(netD, realA, pred, target)
                if d_out is not None:
                    l_d, pred_fake, pred_real = d_out
                    loss_dict["gan_d"] = l_d * self.lambda_gan

        return total, loss_dict


def build_spec_losses(opt: Any, device: Optional[torch.device] = None, vgg: Optional[nn.Module] = None) -> SpecularityNetLossPack:
    """Factory function to construct the loss pack for SpecularityNet.

    Example
        pack = build_spec_losses(opt)
        total = 0
        total += pack.pixel(pred, target)
        if (l := pack.vgg(pred, target)) is not None:
            total += l * opt.lambda_vgg
        if (g := pack.gan_g(netD, None, pred, target)) is not None:
            total += g * opt.lambda_gan
    """
    return SpecularityNetLossPack(opt, device=device, vgg=vgg)
