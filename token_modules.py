# Token-based Modules for MTRRNet
# 实现Token-only多尺度编码→融合→统一解码架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import SwinTransformerBlock,SwinTransformerStage
from timm.layers import LayerNorm2d
import math
from timm.models.layers import DropPath


def init_all_weights(model: nn.Module):
    """
    统一初始化策略（更新后）：
    1. Conv / ConvTranspose / Linear -> Xavier Uniform (近似 GELU 用 relu gain)
    2. 不再额外缩放 Mamba 的 dt_proj / x_proj / out_proj （先观察真实梯度/激活；若后续爆炸再考虑 LayerScale 或梯度裁剪）
    3. Norm 层 weight=1 bias=0
    4. PReLU weight=0.08
    5. PatchEmbed 的 proj 卷积 gain=2.0
    6. 最后裁剪异常值到 [-3,3]
    """
    gelu_gain = nn.init.calculate_gain('relu')

    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.PReLU):
            with torch.no_grad():
                m.weight.fill_(0.08)

        # PatchEmbed 特殊初始化
        if isinstance(m, PatchEmbed):
            if hasattr(m, 'proj') and hasattr(m.proj, 'weight') and m.proj.weight is not None:
                nn.init.xavier_uniform_(m.proj.weight, gain=2.0)
                if m.proj.bias is not None:
                    nn.init.zeros_(m.proj.bias)

    model.apply(_init)

    # 第二阶段：只做必要的均值校正（不再乘 0.5）
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and 'head.weight' in name.lower():
            param.data -= param.data.mean()

    # 新增：特殊初始化线性层
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # Xavier初始化 + 缩小增益
            nn.init.xavier_normal_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)  # 避免死神经元
                
    # 新增：Mamba层特殊初始化
    for name, param in model.named_parameters():
        if 'mamba' in name and 'weight' in name:
            if param.dim() == 2:  # 线性层权重
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')


class FrequencySplit(nn.Module):
    """频带分离：输入图像分离为低频（模糊）和高频部分"""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        # 使用可学习的高斯核进行低频提取
        self.register_parameter('blur_weight', nn.Parameter(torch.ones(1)))
        
    def forward(self, x):
        # 创建高斯模糊核
        kernel = self._create_gaussian_kernel(self.kernel_size, x.device)
        kernel = kernel.expand(x.size(1), 1, self.kernel_size, self.kernel_size)
        
        # 低频：高斯模糊
        padding = self.kernel_size // 2
        low_freq = F.conv2d(x, kernel, padding=padding, groups=x.size(1))
        low_freq = low_freq * self.blur_weight
        
        # 高频：原图减去低频
        high_freq = x - low_freq
        
        return low_freq, high_freq
    
    def _create_gaussian_kernel(self, kernel_size, device):
        """创建高斯核"""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        grid = coords.unsqueeze(0).expand(kernel_size, -1)
        kernel = torch.exp(-(grid ** 2 + grid.T ** 2) / (2 * (kernel_size / 6) ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)


class TokenPatchEmbed(nn.Module):
    """将图像patches转换为tokens（新增2D sin/cos位置编码，非学习型、无绝对位置参数）"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim, use_sincos_pos=True, pos_init_scale=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.use_sincos_pos = use_sincos_pos

        # patch投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        # 位置编码缩放门（可学习，防止一开始位置占太大比重）
        if self.use_sincos_pos:
            self.pos_alpha = nn.Parameter(torch.tensor(float(pos_init_scale)))
            # 预生成与默认网格匹配的2D sin/cos位置编码（注册为buffer，不参与训练）
            pos = self._build_2d_sincos_pos_embed(self.grid_size, self.grid_size, embed_dim, device='cpu')
            self.register_buffer('pos_embed', pos, persistent=False)  # shape: (1, N, C)
        else:
            self.register_buffer('pos_embed', None, persistent=False)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//ps, W//ps)
        x = self.proj(x)
        B, C, Ht, Wt = x.shape  # token网格大小（可能与预设一致，也可能不同）
        # -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # 注入2D sin/cos位置编码（解析式，支持动态尺寸）
        if self.use_sincos_pos:
            if (Ht == self.grid_size) and (Wt == self.grid_size) and (self.pos_embed is not None):
                x = x + self.pos_alpha * self.pos_embed  # 直接使用预生成buffer
            else:
                # 动态尺寸：按当前(Ht, Wt)重新生成，避免插值带来的伪差异
                pos_dyn = self._build_2d_sincos_pos_embed(Ht, Wt, self.embed_dim, device=x.device)  # (1, Ht*Wt, C)
                x = x + self.pos_alpha * pos_dyn

        return x  # (B, num_patches, embed_dim)

    @staticmethod
    def _get_1d_sincos_pos_embed(embed_dim, length, device):
        """生成1D sin/cos编码（长度=length、维度=embed_dim，embed_dim需为偶数）"""
        if embed_dim % 2 != 0:
            # 若为奇数，尾部补1维0编码，保证拼接维度一致（尽量不中断训练）
            extra = 1
            d = embed_dim - 1
        else:
            extra = 0
            d = embed_dim

        # 坐标从0到length-1，标准频率范围（与Transformer常用实现一致）
        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))  # (d/2,)

        # 计算 sin/cos
        sinusoid = position * div_term  # (L, d/2)
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=1)  # (L, d)

        if extra == 1:  # 奇数维度时补零列
            pad = torch.zeros(length, 1, device=device, dtype=torch.float32)
            emb = torch.cat([emb, pad], dim=1)  # (L, d+1)

        return emb  # (L, embed_dim)

    @classmethod
    def _build_2d_sincos_pos_embed(cls, H, W, embed_dim, device='cpu'):
        """
        生成2D sin/cos位置编码，按行优先展平为 (1, H*W, C)
        思路：将总维度分成两半，前一半给Y方向，后一半给X方向，再拼接
        """
        # 将维度一分为二（尽量均分），独立给H和W
        dim_h = embed_dim // 2
        dim_w = embed_dim - dim_h  # 避免奇数时丢1维

        # 先生成1D编码
        pos_h = cls._get_1d_sincos_pos_embed(dim_h, H, device)  # (H, dim_h)
        pos_w = cls._get_1d_sincos_pos_embed(dim_w, W, device)  # (W, dim_w)

        # 网格展开：对每个(y,x)拼接 [pos_h[y], pos_w[x]]
        # 为了可读性，这里明确地两步repeat，再拼接
        pos_h_broadcast = pos_h[:, None, :].repeat(1, W, 1)  # (H, W, dim_h)
        pos_w_broadcast = pos_w[None, :, :].repeat(H, 1, 1)  # (H, W, dim_w)
        pos_2d = torch.cat([pos_h_broadcast, pos_w_broadcast], dim=2)  # (H, W, embed_dim)

        # 展平成 (1, H*W, C)
        pos_2d = pos_2d.view(1, H * W, embed_dim)
        return pos_2d


class MambaTokenBlock(nn.Module):
    """Mamba处理token序列，采用x = x + f(LN(x))残差连接"""
    def __init__(self, dim, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                Mamba(dim)
            ))
    
    def forward(self, x):
        # x: (B, N, C)
        for block in self.blocks:
            x = x + block(x)  # 逐层残差: x = x + f(LN(x))
        return x

from vmamba import VSSBlock,SS2D
from collections import OrderedDict
class VSSTokenMambaModule(nn.Module):
    def __init__(
        self,  
        depths=[9], 
        dims=[192], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2", 
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        use_checkpoint=False,  
        # =========================
        posembed=False,
        _SS2D=SS2D,
        # =========================
        **kwargs,        
    ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.channel_first = False
        self.dims = dims
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):


            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            # print(f'VSS Stage {i} Input:', x.shape)  # (B, H, W, C)
            x = layer(x)
        return x
    
    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1], 
        use_checkpoint=False, 
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        # ===========================
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                # forward_type='v052d'
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))    



class SwinTokenBlock(nn.Module):
    """Swin Transformer处理token，采用x = x + f(LN(x))残差连接"""
    def __init__(self, dim, input_resolution, num_heads, window_size, num_blocks=1):
        super().__init__()
        self.input_resolution = input_resolution
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=4.0
                )
            ))
    
    def forward(self, x):
        # x: (B, N, C) 需要reshape为(B, H, W, C)用于Swin
        B, N, C = x.shape
        H, W = self.input_resolution
        
        # 转换为Swin格式
        x = x.view(B, H, W, C).contiguous()
        
        for block in self.blocks:
            residual = x
            x = block(x) + residual  # 逐层残差: x = x + f(LN(x))
        
        # 转换回token格式
        x = x.view(B, N, C)
        return x




class TokenStage(nn.Module):
    """单个尺度的Token处理阶段：频带分离→分别编码→融合（支持随机失活分支）"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim, mamba_blocks=2, swin_blocks=2, window_size=8, drop_branch_prob=0.1, training=True):   # 新增参数：丢失概率
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.drop_branch_prob = drop_branch_prob
        self.training = training
        
        # 低频和高频的patch embedding
        self.low_embed = TokenPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.high_embed = TokenPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Mamba处理低频，Swin处理高频
        self.mamba_processor = VSSTokenMambaModule(dims=[embed_dim], depths=[mamba_blocks]) \
            if mamba_blocks > 0 else None
        if swin_blocks > 0:
            input_resolution = (self.grid_size, self.grid_size)
            num_heads = max(1, embed_dim // 32)
            self.swin_processor = SwinTokenBlock(
                embed_dim, input_resolution, num_heads, window_size, swin_blocks)
        else:
            self.swin_processor = None
        
        # 融合模块
        if self.mamba_processor is not None and self.swin_processor is not None:
            self.fusion_type = 'weighted'  # 或 'concat'
            if self.fusion_type == 'weighted':
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            else:
                self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        # 转换为tokens
        low_tokens = self.low_embed(x)   # (B, N, C)
        high_tokens = self.high_embed(x)  # (B, N, C)

        # 分别处理
        if self.mamba_processor is not None:
            low_tokens = self.mamba_processor(low_tokens)
        else:
            low_tokens = torch.zeros_like(low_tokens)
        if self.swin_processor is not None:
            high_tokens = self.swin_processor(high_tokens)
        else:
            high_tokens = torch.zeros_like(high_tokens)

        # -------------------------
        # 随机失活分支 (类似 DropPath)
        # -------------------------
        if self.training and self.mamba_processor is not None and self.swin_processor is not None:
            rand_val = torch.rand(1, device=x.device)
            if rand_val < self.drop_branch_prob:  
                # 丢掉 mamba
                return high_tokens.detach() + high_tokens - high_tokens.detach()
            elif rand_val < 2*self.drop_branch_prob and rand_val > self.drop_branch_prob:
                # 丢掉 swin
                return low_tokens.detach() + low_tokens - low_tokens.detach()
        
        # -------------------------
        # 正常融合（训练时没丢分支 / 测试时）
        # -------------------------
        if self.mamba_processor is not None and self.swin_processor is not None:
            if self.fusion_type == 'weighted':
                w = torch.sigmoid(self.fusion_weight)
                fused_tokens = w * low_tokens + (1 - w) * high_tokens
            else:
                concat_tokens = torch.cat([low_tokens, high_tokens], dim=-1)
                fused_tokens = self.fusion_proj(concat_tokens)
        elif self.mamba_processor is not None:
            fused_tokens = low_tokens
        else:
            fused_tokens = high_tokens

        return fused_tokens



class MultiScaleTokenEncoder(nn.Module):
    """多尺度Token编码器：处理4个尺度得到token表示"""
    def __init__(self, embed_dims=[96, 96, 96, 96], 
                 mamba_blocks=[2, 2, 2, 2], swin_blocks=[2, 2, 2, 2],training=True):
        super().__init__()
        self.scales = [256, 128, 64, 32]  # 对应encoder0~3的分辨率
        self.patch_sizes = [4, 4, 4, 2]  # 每个尺度的patch size
        
        self.stages = nn.ModuleList()
        for i, (scale, patch_size, embed_dim, mb, sb) in enumerate(
            zip(self.scales, self.patch_sizes, embed_dims, mamba_blocks, swin_blocks)):
            
            stage = TokenStage(
                img_size=scale,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
                mamba_blocks=mb,
                swin_blocks=sb,
                window_size=8,
                drop_branch_prob=0.1,
                training=training
            )
            self.stages.append(stage)
    
    def forward(self, x_in):
        # x_in: (B, 3, 256, 256)
        tokens_list = []
        aux_preds = {}
        
        for i, stage in enumerate(self.stages):
            # 下采样到对应尺度
            scale = self.scales[i]
            if scale != 256:
                x_scale = F.interpolate(x_in, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                x_scale = x_in
            
            # 获得该尺度的token表示
            tokens, aux_pred = stage(x_scale)
            tokens_list.append(tokens)
            
            if aux_pred is not None:
                aux_preds[f'aux_s{i}'] = aux_pred
        
        return tokens_list, aux_preds


class TokenSubNet(nn.Module):
    """Token融合/细化模块：多尺度token交互融合"""
    def __init__(self, ref_resolution=64, embed_dim=192, num_blocks=3):
        super().__init__()
        self.ref_resolution = ref_resolution
        self.embed_dim = embed_dim
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 融合后的细化处理
        self.refinement_blocks = nn.ModuleList()
        for _ in range(len(self.projectors)):
            self.refinement_blocks.append(nn.Sequential(
                VSSTokenMambaModule(dims=[embed_dim], depths=[num_blocks]),
            ))

        alpha_init_value = 0.5  # 融合权重初始值
        channels = [embed_dim]*4
        self.alpha0 = nn.Parameter(alpha_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha1 = nn.Parameter(alpha_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha2 = nn.Parameter(alpha_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha3 = nn.Parameter(alpha_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None            
    
    def forward(self, tokens_list):
        # tokens_list: [t0, t1, t2, t3] 每个是 (B, N_i, C_i)
        if tokens_list[0].ndim == 3:  # 3维张量
            tokens_spatial = []
            for i, tokens in enumerate(tokens_list):
                B, N, C = tokens.shape
                H = W = int(math.sqrt(N))
                tokens_spatial.append(tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())

        
        # 融合
        self._clamp_abs(self.alpha3.data, 1e-3)        
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha0.data, 1e-3) 

        f0,f1,f2,f3 = tokens_spatial[0],tokens_spatial[1],tokens_spatial[2],tokens_spatial[3]  
        f0 = f0*self.alpha0 + self.refinement_blocks[0](self.upsample(f1))  
        f1 = f1*self.alpha1 + self.refinement_blocks[1](self.upsample(f2)+self.downsample(f0))  
        f2 = f2*self.alpha2 + self.refinement_blocks[2](self.upsample(f3)+self.downsample(f1))  
        f3 = f3*self.alpha3 + self.refinement_blocks[3](self.downsample(f2))  # f3最浅 f0最深
        tokens_spatial_list = [f0,f1,f2,f3]

        
        return tokens_spatial_list  # (B, H_i, W_i, self.embed_dim)
    

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign() # ​​符号保留​​
            data.abs_().clamp_(value) # 将输入张量 data 的每个元素的绝对值限制在 [value, +∞) 范围内
            data *= sign    

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()

        # 深度卷积（仅空间特征提取）
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel, padding_mode='reflect') # depthwise conv
        # 层归一化（channels_last 模式）
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        # 1x1卷积（通过线性层实现）
        self.pwconv1 = nn.Linear(in_channel, hidden_dim) # 升维# pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)# 降维
        # 层缩放参数 类似 Transformer 的 ​​可学习缩放因子​​，调整各通道的重要性
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # 随机路径丢弃
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)              # 深度卷积 [B, C, H, W]
                                        # 在 channels_last 模式下，nn.Linear 等价于 1x1 卷积，但实现更高效
        x = x.permute(0, 2, 3, 1)       # 转为 channels_last [B, H, W, C] # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)                # 层归一化
        x = self.pwconv1(x)             # 升维至 hidden_dim
        x = self.act(x)                 # GELU 激活
        x = self.pwconv2(x)             # 降维至 out_channel
        if self.gamma is not None:   
            x = self.gamma * x          # 缩放 
        x = x.permute(0, 3, 1, 2)       # (N, H, W, C) -> (N, C, H, W) # 转回 channels_first [B, C, H, W]
        x = input + self.drop_path(x)   # 残差连接 + DropPath

        return x                        # (B,C,H,W)

class UnifiedTokenDecoder(nn.Module):
    """统一Token解码器：从token一次性解码为6通道(T,R)"""
    def __init__(self, token_dim=192, base_scale_init=0.3):
        super().__init__()
        self.token_dim = token_dim
        
        self.proj23 = nn.Sequential(
            nn.conv(96,96,1,1),
            nn.InstanceNorm2d(96),
        )
        self.convblock23 = nn.Sequential(
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            )
        
        self.proj12 = nn.Sequential(
            nn.conv(96,96,1,1),
            nn.InstanceNorm2d(96),
        )
        self.convblock12 = nn.Sequential(
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            )

        self.proj01 = nn.Sequential(
            nn.conv(96,96,1,1),
            nn.InstanceNorm2d(96),
        )
        self.convblock01 = nn.Sequential(
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            ConvNextBlock(96, 192, 96, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.05),
            )

        # Base缩放因子
        self.base_scale = nn.Parameter(torch.tensor(base_scale_init))
        
        
        # 上采样和卷积解码层
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(96),
            
            # 128x128 -> 256x256  
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            
            # 最终输出层
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 6, kernel_size=1)  # 6通道输出 (T, R)
        )
        
    def forward(self, tokens_list, x_in):
        # tokens: (B, ref_H,ref_W, token_dim)
        # x_in: (B, 3, 256, 256) 原始输入
        
        f0,f1,f2,f3 = tokens_list[0],tokens_list[1],tokens_list[2],tokens_list[3]
        # 转换tokens到feature map
        f0 = f0.permute(0, 3, 1, 2).contiguous()
        f1 = f1.permute(0, 3, 1, 2).contiguous()
        f2 = f2.permute(0, 3, 1, 2).contiguous()
        f3 = f3.permute(0, 3, 1, 2).contiguous()
         
        f3 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        o23 = self.convblock23(self.proj23(f2 + f3))

        o23 = F.interpolate(o23, size=f1.shape[2:], mode='bilinear', align_corners=False)
        o12 = self.convblock12(self.proj12(f1 + o23))

        o12 = F.interpolate(o12, size=f0.shape[2:], mode='bilinear', align_corners=False)
        o01 = self.convblock01(self.proj01(f0 + o12))

        # 解码
        delta = self.decoder(o01)  # (B, 6, 256, 256)
        
        # Base residual: 输入图像的residual base
        base_input = x_in.repeat(1, 2, 1, 1)  # (B, 6, 256, 256) 复制为T,R base
        base = self.base_scale * base_input
        
        # 最终输出 = base + delta
        output = base + delta
        
        return output  # (B, 6, 256, 256)