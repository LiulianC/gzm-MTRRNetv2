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
from typing import List


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
        # 保护 Mamba2 的初始化 每个模块是独立调用的，在 Mamba2 中 return 不会影响后续的 Linear、LayerNorm 等
        if isinstance(m, Mamba2):
            print(f"跳过 Mamba2 初始化: {m}")
            return        
        if isinstance(m, Mamba2Simple):
            print(f"跳过 Mamba2Simple 初始化: {m}")
            return        
        if isinstance(m, Mamba):
            print(f"跳过 Mamba 初始化: {m}")
            return        
        if isinstance(m, VSSTokenMambaModule):
            print(f"跳过 VSSTokenMambaModule 初始化: {m}")
            return        
        if isinstance(m, Mamba2Blocks_Standard):
            print(f"跳过 Mamba2Blocks_Standard 初始化: {m}")
            return        
        
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
            if m.bias is not None:
                # nn.init.zeros_(m.bias)
                nn.init.uniform_(m.bias, -0.1, 0.1)

        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
            if m.bias is not None:
                # nn.init.zeros_(m.bias)
                nn.init.uniform_(m.bias, -0.1, 0.1)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                # nn.init.zeros_(m.bias)
                nn.init.uniform_(m.bias, -0.1, 0.1)
                
        elif isinstance(m, nn.PReLU):
            with torch.no_grad():
                m.weight.fill_(0.08)

        # PatchEmbed 特殊初始化
        elif isinstance(m, TokenPatchEmbed):
            nn.init.normal_(m.weight, std=0.02)

    model.apply(_init)





class AAF(nn.Module):
    """
    输入: List[Tensor]，每个 shape 为 [B, L, C] 或 [B, C, H, W]，长度为 num_inputs
    输出: Tensor，shape 为 [B, L, C] 或 [B, C, H, W]（与输入形状一致）
    """
    def __init__(self, in_channels: int, num_inputs: int):
        super(AAF, self).__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs

    @torch.no_grad()
    def _check(self, features: List[torch.Tensor]):
        assert isinstance(features, (list, tuple)) and len(features) == self.num_inputs, \
            f"Expect {self.num_inputs} inputs, got {len(features)}."
        shapes = [tuple(x.shape) for x in features]
        assert all(s == shapes[0] for s in shapes), \
            f"All inputs must share the same shape, got {shapes}."
        x = features[0]
        assert x.dim() in (3, 4), \
            f"Each input must be 3D [B,L,C] or 4D [B,C,H,W], got dim={x.dim()}."
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == self.in_channels, \
                f"in_channels={self.in_channels} but got C={C}."
        else:  # 3D [B, L, C]
            B, L, C = x.shape
            assert C == self.in_channels, \
                f"in_channels={self.in_channels} but got C={C}."

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: List[Tensor]
            - 每个为 [B, C, H, W]，或 [B, L, C]
        返回:
            - 与单个输入同形状
        """
        self._check(features)

        x0 = features[0]
        if x0.dim() == 4:
            # 4D case: [B, C, H, W] -> stack: [B, N, C, H, W]
            x = torch.stack(features, dim=1)
            # 在输入维度N上进行softmax，每个(b,c,h,w)位置的N个值互斥归一化
            weights = torch.softmax(x, dim=1)           # [B, N, C, H, W]
            out = (weights * x).sum(dim=1)              # [B, C, H, W]
            # 你也可以取出每个输入对应的权重张量：weights[:, i] -> [B, C, H, W]
            return out
        else:
            # 3D case: [B, L, C] -> stack: [B, N, L, C]
            x = torch.stack(features, dim=1)
            weights = torch.softmax(x, dim=1)           # [B, N, L, C]
            out = (weights * x).sum(dim=1)              # [B, L, C]
            return out



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
            # 限制 pos_alpha 的梯度，避免位置编码权重把表达“吃死”
            self.pos_alpha.register_hook(self._pos_alpha_grad_hook)

            # 预生成与默认网格匹配的2D sin/cos位置编码（注册为buffer，不参与训练）
            pos = self._build_2d_sincos_pos_embed(self.grid_size, self.grid_size, embed_dim, device='cpu')
            self.register_buffer('pos_embed', pos, persistent=False)  # shape: (1, N, C)
        else:
            self.register_buffer('pos_embed', None, persistent=False)

    def _pos_alpha_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        # 标量梯度裁剪：幅度不超过0.1（足够学习，又不至于抖动过大）
        gmax = 0.1
        return grad.clamp(min=-gmax, max=gmax)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//ps, W//ps)
        x = self.proj(x)
        B, C, Ht, Wt = x.shape
        # -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # 注入2D sin/cos位置编码（解析式，支持动态尺寸）
        if self.use_sincos_pos:
            if (Ht == self.grid_size) and (Wt == self.grid_size) and (self.pos_embed is not None):
                x = x + self.pos_alpha * self.pos_embed  # 直接使用预生成buffer
            else:
                # 动态尺寸：按当前(Ht, Wt)重新生成
                pos_dyn = self._build_2d_sincos_pos_embed(Ht, Wt, self.embed_dim, device=x.device)  # (1, Ht*Wt, C)
                x = x + self.pos_alpha * pos_dyn

        return x  # (B, num_patches, embed_dim)

    @staticmethod
    def _get_1d_sincos_pos_embed(embed_dim, length, device):
        if embed_dim % 2 != 0:
            extra = 1
            d = embed_dim - 1
        else:
            extra = 0
            d = embed_dim

        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))  # (d/2,)

        sinusoid = position * div_term  # (L, d/2)
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=1)  # (L, d)

        if extra == 1:
            pad = torch.zeros(length, 1, device=device, dtype=torch.float32)
            emb = torch.cat([emb, pad], dim=1)

        return emb

    @classmethod
    def _build_2d_sincos_pos_embed(cls, H, W, embed_dim, device='cpu'):
        dim_h = embed_dim // 2
        dim_w = embed_dim - dim_h

        pos_h = cls._get_1d_sincos_pos_embed(dim_h, H, device)  # (H, dim_h)
        pos_w = cls._get_1d_sincos_pos_embed(dim_w, W, device)  # (W, dim_w)

        pos_h_broadcast = pos_h[:, None, :].repeat(1, W, 1)  # (H, W, dim_h)
        pos_w_broadcast = pos_w[None, :, :].repeat(H, 1, 1)  # (H, W, dim_w)
        pos_2d = torch.cat([pos_h_broadcast, pos_w_broadcast], dim=2)  # (H, W, embed_dim)

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
        channel_first=False,
        # =========================
        **kwargs,        
    ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]   # dpr=[0 0.05 0.1]
        self.channel_first = channel_first
        self.dims = dims
        self.num_layers = 1
        

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            self.layers.append(self._make_layer(    # VSSblock就是mamba堆叠  make_layer堆叠了多个VSSBlock  这里期望make_layer只堆叠一个
                dim = self.dims[0],
                depth = depths[0],
                drop_path = dpr[1],
                # drop_path = 0.0,
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
        # x(B, H, W, C)
        # print('VSS Input:', x.shape)  # (B, H, W, C)
        # 自带channel first优化
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x  
    
    @staticmethod
    def _make_layer(
        dim=96, 
        depth=9,
        drop_path=0.1, 
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
        ssm_drop_rate=0.05, 
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
        depth = depth
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path,
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
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))    

from mamba_ssm.modules.mamba2 import Mamba2  # 默认有残差连接 后置归一化
from mamba_ssm.modules.mamba_simple import Mamba  # 默认有残差连接 后置归一化
from mamba_ssm.modules.mamba2_simple import Mamba2Simple  # 默认有残差连接 后置归一化
class Mamba2Blocks(nn.Module):
    def __init__(self, dim, num_blocks=1, drop_path_rate=0.05, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                Mamba2(d_model=dim,d_state=64,d_conv=4,expand=2),
                # Mamba2Simple(d_model=dim,d_state=64,d_conv=4,expand=2,headdim=96),
                # Mamba(d_model=dim,d_state=16,d_conv=4,expand=2),
                nn.Dropout(drop_path_rate),
            ))
    def forward(self, x):
        # x: (B, H, W, C)

        if self.channel_first: # 变成channel last
            x = x.permute(0,2,3,1).contiguous()  # (B, H, W, C)

        B,H,W,C = x.shape
        
        # 需要 B N C
        x = x.view(B, -1, C).contiguous()  # (B, N, C)

        for block in self.blocks:
            x = block(x)  # 逐层残差: x = x + f(LN(x))

        x = x.view(B, H, W, C).contiguous()  # (B, H, W, C)

        if self.channel_first: #变回去
            x = x.permute(0,3,1,2).contiguous()  # (B, C, H, W)

        return x #(B H W C)



from functools import partial
from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba2Blocks_Standard(nn.Module):
    def __init__(
        self,
        d_model: int,# token维度
        n_layer: int,# Mamba2数量
        d_intermediate: int,# MLP维度 0表示无MLP

        # === Mamba1 配置 ===
        ssm_cfg={
            "layer": "Mamba1",    # 指定使用Mamba1
            # 其他Mamba1参数（可选）
            "d_state": 16,       # SSM状态维度
            "d_conv": 4,          # 卷积核大小
            "expand": 2,          # 扩展因子
        },
        # LayerScale：进一步抑制深堆叠残差的幅度增长
        layer_scale_init: float = None,
        layer_scale_max: float = None,

        # # === Mamba2 配置 ===（已注释）
        # ssm_cfg={
        #     "layer": "Mamba2",    # 指定使用Mamba2
        #     # 稳定性相关默认值（可被调用方覆盖）
        #     "d_state": 64,          # 从64改为16，参考Mamba1稳定配置
        #     "d_conv": 4,
        #     "expand": 2,
        #     # 归一化与gate顺序
        #     "rmsnorm": True,
        #     "norm_before_gate": True,
        #     # dt 初始化与限制（与 fused 路径匹配）
        #     "dt_min": 1e-3,
        #     "dt_max": 5e-2,
        #     "dt_init_floor": 1e-4,
        #     "dt_limit": (1e-4, 5e-1),
        #     # 线性与卷积的bias配置（贴近官方默认）
        #     "bias": False,
        #     "conv_bias": True,
        # },
        # # LayerScale：进一步抑制深堆叠残差的幅度增长
        # layer_scale_init: float = 1e-4,
        # layer_scale_max: float = 1e-2,

        # === MHA 配置 ===（交替使用MHA和Mamba）
        attn_layer_idx=None,  # 设置为None，将在内部自动设置为奇数层索引
        attn_cfg={
            "num_heads": 8,           # 注意力头数
            "num_heads_kv": None,     # key-value头数，默认同num_heads
            "head_dim": None,         # 头维度，默认使用embed_dim // num_heads
            "mlp_dim": 0,             # MLP维度，0表示无MLP
            "qkv_proj_bias": True,    # QKV投影偏置
            "out_proj_bias": True,    # 输出投影偏置
            "softmax_scale": None,    # softmax缩放因子
            "causal": False,          # 是否因果（自回归）
            "d_conv": 0,              # 卷积维度，0表示无卷积
            "rotary_emb_dim": 0,      # 旋转位置编码维度
            "rotary_emb_base": 10000.0, # 旋转位置编码基数
            "rotary_emb_interleaved": False, # 是否交错旋转位置编码
        },
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=False,
        # 关键稳定项：在训练未采用 AMP/混合精度时，保持残差在 FP32 中累加，避免数值漂移
        residual_in_fp32=True,
        device='cuda',
        # 不强制使用 bfloat16，继承全局默认（通常为 FP32）；避免与优化器/其余模块 dtype 不一致导致的不稳定
        dtype=None,
        channel_first=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.channel_first = channel_first
        self.n_layer = n_layer 

        self.Mamba_num = n_layer
        if attn_cfg is not None:
            self.Mamba_num = self.n_layer/2
        self.Trans_num = self.n_layer - self.Mamba_num




        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # 如果attn_layer_idx为None，自动设置为奇数层索引（实现交替使用）
        if attn_layer_idx is None:
            attn_layer_idx = [i for i in range(1, n_layer, 2)]  # 奇数层使用MHA
        
        # 确保attn_cfg包含必要的参数
        if attn_cfg is None:
            attn_cfg = {}
        
        # # 添加embed_dim参数到attn_cfg（MHA需要的第一个参数）
        # attn_cfg_with_embed_dim = {"embed_dim": d_model}
        # attn_cfg_with_embed_dim.update(attn_cfg)
        
        # 合并默认稳定配置与外部传入的 ssm_cfg（外部优先）
        _default_ssm = {
            "layer": "Mamba1",  # 使用Mamba1
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "bias": False,
            "conv_bias": True,
        }
        _ssm_cfg_merged = dict(_default_ssm)
        if ssm_cfg is not None:
            _ssm_cfg_merged.update(ssm_cfg)

        # 确保使用Mamba1
        _ssm_cfg_merged["layer"] = "Mamba1"

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=_ssm_cfg_merged,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,  # 使用包含embed_dim的配置
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
        # 打印配置信息
        print(f"模型配置: 总层数={n_layer}, MHA层索引={attn_layer_idx}")
        print(f"交替模式: 偶数层(0,2,4...)使用Mamba1, 奇数层(1,3,5...)使用MHA")      

        # LayerScale stabilizes deep stacks by shrinking each block's update before it enters the
        # next residual addition. When set to a tiny value (default 1e-3) it tames the gradient norm
        # growth observed in debug-grad.log without impacting representational capacity.
        self.layer_scale_max = layer_scale_max
        if layer_scale_init is not None and layer_scale_init > 0:
            scale_dtype = torch.float32 if dtype is None else dtype
            self.layer_scales = nn.ParameterList(
                [
                    nn.Parameter(layer_scale_init * torch.ones(d_model, device=device, dtype=scale_dtype))
                    for _ in range(n_layer)
                ]
            )
        else:
            self.layer_scales = None

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, inference_params=None, **mixer_kwargs):

        # x: (B, H, W, C)

        if self.channel_first: # 变成channel last
            x = x.permute(0,2,3,1).contiguous()  # (B, H, W, C)

        B,H,W,C = x.shape
        
        # 需要 B N C
        hidden_states = x.view(B, -1, C).contiguous()  # (B, N, C)

        residual = None
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            if self.layer_scales is not None:
                scale_param = self.layer_scales[idx]
                clamp_max = self.layer_scale_max if self.layer_scale_max is not None else None
                if clamp_max is not None:
                    scale_param = torch.clamp(scale_param, min=0.0, max=clamp_max)
                scale = scale_param.to(hidden_states.dtype).view(1, 1, -1)
                hidden_states = hidden_states * scale
                if residual is not None:
                    residual = residual * scale
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        hidden_states = hidden_states.view(B, H, W, C).contiguous()  # (B, H, W, C)

        if self.channel_first: #变回去
            hidden_states = hidden_states.permute(0,3,1,2).contiguous()  # (B, C, H, W)    

        return hidden_states



class SwinTokenBlock(nn.Module):
    """Swin Transformer处理token，直接复用内部残差逻辑避免重复叠加。"""
    def __init__(self, dim, input_resolution, num_heads, window_size, num_blocks=1):
        super().__init__()
        self.input_resolution = input_resolution

        blocks = []
        for i in range(num_blocks):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=4.0,
                    attn_drop=0.05,
                    drop_path=0.05,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: (B, N, C) → reshape 为 (B, H, W, C)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C).contiguous()

        for block in self.blocks:
            x = block(x)  # block 已包含 LayerNorm 与残差路径

        x = x.view(B, N, C)
        return x





from MTRR_RD_modules import FrequencyProcessor,ChannelAttention

class EncoderUnit(nn.Module):
    """单个尺度的Token处理阶段：频带分离→分别编码→融合（支持随机失活分支）"""
    def __init__(self, ori_img_size=256, embed_dim=96, mamba_blocks=2, swin_blocks=2, grid_size=64, window_size=8, drop_branch_prob=0.1, 
                 need_downsample=False, need_freqAttention=False, need_channelAttention=False):
        super().__init__()
        self.img_size = ori_img_size
        self.grid_size = grid_size
        self.drop_branch_prob = drop_branch_prob
        # self.training = training # self.training是nn.Module自带的属性 不应该自己赋值 它会在train()/eval()时自动切换
        self.need_downsample = need_downsample
        self.need_freqAttention = need_freqAttention
        self.need_channelAttention = need_channelAttention

        if self.need_freqAttention:
            self.freqatt = FrequencyProcessor(channels=embed_dim, int_size=2*embed_dim)
        if self.need_channelAttention:
            self.channelatt = ChannelAttention(dim=embed_dim, num_heads=2, bias=True)


        # 旁路梯度强度（极小）：不改变前向数值，但能给被丢分支“续一点梯度”
        self.ghost_grad_coeff = 0.02

        # 低频和高频的patch embedding
        if need_downsample is True:
            # 输入的是embed_dim//2 后续操作都要embed_dim
            self.downSample = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=2, stride=2, padding=0, bias=False)# 不重叠下采样

        # Mamba处理低频，Swin处理高频
        # self.mamba_processor = VSSTokenMambaModule(dims=[embed_dim], depths=[mamba_blocks], channel_first=False, drop_path_rate=0.05) if mamba_blocks > 0 else None
        # self.mamba_processor = nn.Identity()
        # self.mamba_processor = Mamba2Blocks(dim=embed_dim, num_blocks=mamba_blocks, drop_path_rate=0.05) if mamba_blocks > 0 else None
        self.mamba_processor = Mamba2Blocks_Standard(d_model=embed_dim, n_layer=mamba_blocks, d_intermediate=2*embed_dim) if mamba_blocks > 0 else None

        
        if swin_blocks > 0:
            input_resolution = (self.grid_size, self.grid_size) 
            num_heads = max(1, embed_dim // 32)
            self.swin_processor = SwinTokenBlock(embed_dim, input_resolution, num_heads, window_size, swin_blocks)
            # self.swin_processor = nn.Identity()
        else:
            self.swin_processor = None

        # 融合模块
        if self.mamba_processor is not None and self.swin_processor is not None:
            self.fusion = AAF(embed_dim, 2)
            self.fusion_out = nn.Identity()
        self.out = nn.Identity()
        

    def forward(self, x):
        # x: (B, N, C)

        if self.need_downsample is True:
            B, N, C = x.shape
            x = x.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))
            x = self.downSample(x)  # (B, 2C, H/2, W/2)
            B, C, H, W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(B, H*W, C)

        if self.need_freqAttention: # 用在编码的头和解码的尾
            B, N, C = x.shape
            x = x.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))            
            x = self.freqatt(x)
            B, C, H, W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(B, H*W, C)            

        if self.need_channelAttention: # 用在编码的头和解码的尾
            B, N, C = x.shape
            x = x.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))            
            x = self.channelatt(x)
            B, C, H, W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(B, H*W, C)            

        low_tokens = x.contiguous()  # (B, N, C)
        high_tokens = x.contiguous() # (B, N, C)
        B,N,C = x.shape

        # 分别处理
        if self.mamba_processor is not None:
            low_tokens = low_tokens.view(B, int(N**0.5), int(N**0.5), C).contiguous()
            low_tokens = self.mamba_processor(low_tokens)
            low_tokens = low_tokens.view(B, N, C).contiguous()
        else:
            low_tokens = torch.zeros_like(low_tokens)

        if self.swin_processor is not None:
            high_tokens = self.swin_processor(high_tokens)
        else:
            high_tokens = torch.zeros_like(high_tokens)

        # -------------------------
        # 随机失活分支 (期望不变 + 旁路梯度)
        # -------------------------
        if self.training and self.mamba_processor is not None and self.swin_processor is not None and getattr(self, 'drop_branch_prob', 0.0) > 0:
            # 期望不变缩放：保留分支除以(1-p)
            keep_scale = 1.0 / (1.0 - self.drop_branch_prob)
            rand_val = torch.rand(1, device=x.device)

            if rand_val < self.drop_branch_prob:
                # 丢 Mamba：输出仅来自 Swin；给 Mamba 注入极小“零值旁路项”以获得梯度
                # (z - z.detach()) 在前向恒为0，但能把上游梯度传给 z
                ghost = self.ghost_grad_coeff * (low_tokens - low_tokens.detach())
                return keep_scale * high_tokens + ghost

            elif (rand_val > self.drop_branch_prob) and (rand_val < 2 * self.drop_branch_prob):
                # 丢 Swin：输出仅来自 Mamba；给 Swin 注入极小“零值旁路项”以获得梯度
                ghost = self.ghost_grad_coeff * (high_tokens - high_tokens.detach())
                return keep_scale * low_tokens + ghost

        # -------------------------
        # 正常融合（训练时没丢分支 / 测试时）
        # -------------------------
        if self.mamba_processor is not None and self.swin_processor is not None:
            fused_tokens = self.fusion([low_tokens, high_tokens])
            fused_tokens = self.fusion_out(fused_tokens)
            # fused_tokens = low_tokens + high_tokens
        elif self.mamba_processor is not None:
            fused_tokens = low_tokens
        else:
            fused_tokens = high_tokens

        fused_tokens = self.out(fused_tokens)

        return fused_tokens






class Encoder(nn.Module):
    """多尺度Token编码器：处理4个尺度得到token表示"""
    def __init__(self, in_chans=3, embed_dim=96, mamba_blocks=[2, 2, 2, 2], swin_blocks=[2, 2, 2, 2], drop_branch_prob=0.2):
        super().__init__()
        
        self.patchembed = TokenPatchEmbed(img_size=256, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)

        self.encoder_unit0 = EncoderUnit(embed_dim=96, grid_size=64, ori_img_size=256, mamba_blocks=mamba_blocks[0], swin_blocks=swin_blocks[0], 
                                    window_size=8, drop_branch_prob=drop_branch_prob, need_downsample=False, need_freqAttention=True)
        
        self.encoder_unit1 = EncoderUnit(embed_dim=192, grid_size=32, ori_img_size=256, mamba_blocks=mamba_blocks[1], swin_blocks=swin_blocks[1], 
                                    window_size=8, drop_branch_prob=drop_branch_prob, need_downsample=True, need_channelAttention=True)

        # self.downSample0 = nn.Conv2d(96, 192, kernel_size=2, stride=2, padding=0, bias=False)# 不重叠下采样
        # self.downSample1 = nn.Conv2d(192, 384, kernel_size=2, stride=2, padding=0, bias=False)# 不重叠下采样

        self.encoder_unit2 = EncoderUnit(embed_dim=384, grid_size=16, ori_img_size=256, mamba_blocks=mamba_blocks[2], swin_blocks=swin_blocks[2], 
                                    window_size=4, drop_branch_prob=drop_branch_prob, need_downsample=True, need_channelAttention=True)
        
        self.encoder_unit3 = EncoderUnit(embed_dim=768, grid_size=8, ori_img_size=256, mamba_blocks=mamba_blocks[3], swin_blocks=swin_blocks[3], 
                                    window_size=4, drop_branch_prob=drop_branch_prob, need_downsample=True, need_channelAttention=True)
    
    def forward(self, x_in):
        # x_in: (B, 3, 256, 256)
        tokens_list = []
        
        x_emb1 = self.patchembed(x_in)  # (B, N, C)  N=4096 C=96
        tokens_list.append(x_emb1)
        
        tokens = self.encoder_unit0(x_emb1)  # (B, N, C)  N=64*64 C=96
        tokens_list.append(tokens)

        tokens = self.encoder_unit1(tokens)  # (B, N, C)  N=32*32 C=192
        tokens_list.append(tokens)



        # B, N, C = x_emb1.shape
        # x_emb1 = x_emb1.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))  # (B, C, H, W)  C=96 H=W=64
        # x_emb2 = self.downSample0(x_emb1)  # (B, C, H, W)  C=192 H=W=32
        # x_emb2 = self.downSample1(x_emb2)  # (B, C, H, W)  C=384 H=W=16
        # B, C, H, W = x_emb2.shape
        # x_emb2 = x_emb2.contiguous().view(B, C, H*W).permute(0,2,1)  # (B, N, C)  C=384 N=16*16
        # tokens = self.encoder_unit2(x_emb2)  # (B, N, C)  N=16*16 C=384
        
        
        tokens = self.encoder_unit2(tokens)  # (B, N, C)  N=16*16 C=384
        tokens_list.append(tokens)

        tokens = self.encoder_unit3(tokens)  # (B, N, C)  N=8*8 C=768
        tokens_list.append(tokens)
            
        return tokens_list

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
            align_corners=self.align_corners
        )

class SubNet(nn.Module):
    """Token融合/细化模块：多尺度token交互融合"""
    def __init__(self, embed_dims=[96,192,384,768], mam_blocks=[6,6,6,6], use_rev=False):
        super().__init__()
        self.embed_dims = embed_dims
        # 是否启用可逆式前向（重算反传）
        self.use_rev = bool(use_rev)

        self.upsample1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dims[1], embed_dims[1]//2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[1]//2, affine=True),
            nn.GELU(),
        )
        self.upsample2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dims[2], embed_dims[2]//2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[2]//2, affine=True),
            nn.GELU(),
        )
        self.upsample3 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dims[3], embed_dims[3]//2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[3]//2, affine=True),
            nn.GELU(),
        )
        
        self.downsample0 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0]*2, kernel_size=2, stride=2, bias=False),# 不重叠下采样
            nn.InstanceNorm2d(embed_dims[0]*2, affine=True),
            nn.GELU(),
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[1]*2, kernel_size=2, stride=2, bias=False),# 不重叠下采样
            nn.InstanceNorm2d(embed_dims[1]*2, affine=True),
            nn.GELU(),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[2]*2, kernel_size=2, stride=2, bias=False),# 不重叠下采样
            nn.InstanceNorm2d(embed_dims[2]*2, affine=True),
            nn.GELU(),
        )
        
        # 融合后的细化处理
        self.mamba_blocks = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.mamba_blocks.append(nn.Sequential(
                # VSSTokenMambaModule(dims=[embed_dims[i]], depths=[mam_blocks[i]], channel_first=True, drop_path_rate=0.05),
                # Mamba2Blocks(dim=embed_dims[i], num_blocks=mam_blocks[i], channel_first=True, drop_path_rate=0.05),
                Mamba2Blocks_Standard(d_model=embed_dims[i], n_layer=mam_blocks[i], d_intermediate=2*embed_dims[i], channel_first=True),
            ))

        

        alpha_init_value = 0.5  # 融合权重初始值
        channels = embed_dims
        self.alpha0 = nn.Parameter(alpha_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha1 = nn.Parameter(alpha_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha2 = nn.Parameter(alpha_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha3 = nn.Parameter(alpha_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None            
    
    def forward(self, tokens_list, use_eval=True):
        # tokens_list: [t0, t1, t2, t3] 每个都是 (B, N_i, C_i) 或 (B, C_i, H_i, W_i)

        if self.use_rev:
            return self._forward_reverse(tokens_list, use_eval=use_eval)
        else:
            return self._forward_noreverse(tokens_list)

    def _forward_noreverse(self, tokens_list):
        # tokens_list: [t0, t1, t2, t3] 每个都是 (B, N_i, C_i)

        if tokens_list[0].ndim == 3:  # 3维张量
            for i, tokens in enumerate(tokens_list):
                B, N, C = tokens.shape
                H = W = int(math.sqrt(N))
                tokens_list[i] = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                # (B C H W)
        else:
            tokens_list = tokens_list
            pass  # 已经是(B C H W)格式
        
        # 融合
        self._clamp_abs(self.alpha3.data, 1e-3)        
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha0.data, 1e-3) 

        x_emb,f0,f1,f2,f3 = tokens_list[0],tokens_list[1],tokens_list[2],tokens_list[3],tokens_list[4]
        
        # print('Token shapes in TokenSubNet:', f0.shape, f1.shape, f2.shape, f3.shape)  # (B, C, H_i, W_i)
        # # (64 64) (64 64) (32 32) (16 16) (8 8)

        f0 = f0*self.alpha0 + self.mamba_blocks[0](self.upsample1(f1)+x_emb)  
        f1 = f1*self.alpha1 + self.mamba_blocks[1](self.upsample2(f2)+self.downsample0(f0))  
        f2 = f2*self.alpha2 + self.mamba_blocks[2](self.upsample3(f3)+self.downsample1(f1))  
        f3 = f3*self.alpha3 + self.mamba_blocks[3](self.downsample2(f2))  # f3最浅 f0最深 
        tokens_spatial_list = [x_emb,f0,f1,f2,f3]

        # print('Token shapes out TokenSubNet:', f0.shape, f1.shape, f2.shape, f3.shape)  # (B, C, H_i, W_i)

        return tokens_spatial_list  # (B, self.embed_dim, H_i, W_i)

    def _forward_reverse(self, tokens_list, use_eval=True):
        # tokens_list: [t0, t1, t2, t3] 每个都是 (B, N_i, C_i) 或 (B, C_i, H_i, W_i)

        if tokens_list[0].ndim == 3:  # 3维张量
            for i, tokens in enumerate(tokens_list):
                B, N, C = tokens.shape
                H = W = int(math.sqrt(N))
                tokens_list[i] = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                # (B C H W)
        else:
            tokens_list = tokens_list
            pass  # 已经是(B C H W)格式

        x_emb,f0,f1,f2,f3 = tokens_list[0],tokens_list[1],tokens_list[2],tokens_list[3],tokens_list[4]

        # 使用自定义可逆前向 Function
        x_emb, y0, y1, y2, y3 = _SubNetRevFunction.apply(
            self,
            use_eval,
            x_emb,
            f0,
            f1,
            f2,
            f3,
        )

        return [x_emb, y0, y1, y2, y3]  # (B, self.embed_dim, H_i, W_i)
    

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign() # ​​符号保留​​
            data.abs_().clamp_(value) # 将输入张量 data 的每个元素的绝对值限制在 [value, +∞) 范围内
            data *= sign    


# ========================= Reversible-style forward for SubNet =========================
class _SubNetRevFunction(torch.autograd.Function):
    """
    自定义可逆前向的 autograd Function：
    - forward: 使用 SubNet 正常公式计算 y0..y3，并返回 [x_emb, y0, y1, y2, y3]
    - backward: 重新执行一遍前向（带梯度），通过 torch.autograd.grad 同时得到对
                输入 (x_emb, f0, f1, f2, f3) 和参数的梯度，参数梯度将累计到
                subnet.parameters() 的 .grad 上，输入梯度作为返回值。

    注意：这里的 backward 不是 in-place 反演，而是“重算前向”的 VJP，
    类似 revnets 的实现思路，减少存图而保证梯度正确传播。
    """

    @staticmethod
    def forward(ctx, subnet: nn.Module, use_eval: bool,
                x_emb: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor):
        # 保存必要对象供 backward 使用
        ctx.subnet = subnet
        ctx.use_eval = use_eval
        # 保存输入的轻量副本用于 backward 的重算（避免保存中间激活）
        ctx.save_for_backward(x_emb.detach(), f0.detach(), f1.detach(), f2.detach(), f3.detach())

        # 正常前向（不需要构建 autograd 图，后面会重算）
        def _forward_once():
            M0, M1, M2, M3 = subnet.mamba_blocks[0], subnet.mamba_blocks[1], subnet.mamba_blocks[2], subnet.mamba_blocks[3]
            up1, up2, up3 = subnet.upsample1, subnet.upsample2, subnet.upsample3
            dn0, dn1, dn2 = subnet.downsample0, subnet.downsample1, subnet.downsample2

            a0 = subnet.alpha0 if hasattr(subnet, 'alpha0') and subnet.alpha0 is not None else None
            a1 = subnet.alpha1 if hasattr(subnet, 'alpha1') and subnet.alpha1 is not None else None
            a2 = subnet.alpha2 if hasattr(subnet, 'alpha2') and subnet.alpha2 is not None else None
            a3 = subnet.alpha3 if hasattr(subnet, 'alpha3') and subnet.alpha3 is not None else None

            y0 = (f0 if a0 is None else f0 * a0) + M0(up1(f1) + x_emb)
            y1 = (f1 if a1 is None else f1 * a1) + M1(up2(f2) + dn0(y0))
            y2 = (f2 if a2 is None else f2 * a2) + M2(up3(f3) + dn1(y1))
            y3 = (f3 if a3 is None else f3 * a3) + M3(dn2(y2))
            return y0, y1, y2, y3

        if use_eval:
            # 临时切 eval，避免 Dropout/BN 造成不一致；这里 forward 不需要梯度
            was = [m.training for m in subnet.mamba_blocks]
            for m in subnet.mamba_blocks:
                m.eval()
            y0, y1, y2, y3 = _forward_once()
            # 恢复状态
            for m, t in zip(subnet.mamba_blocks, was):
                m.train(t)
        else:
            y0, y1, y2, y3 = _forward_once()

        return x_emb, y0, y1, y2, y3

    @staticmethod
    def backward(ctx, grad_x_emb: torch.Tensor, grad_y0: torch.Tensor, grad_y1: torch.Tensor, grad_y2: torch.Tensor, grad_y3: torch.Tensor):
        subnet: nn.Module = ctx.subnet
        use_eval: bool = ctx.use_eval
        saved_x_emb, saved_f0, saved_f1, saved_f2, saved_f3 = ctx.saved_tensors

        # 重新构造需要梯度的叶子张量
        X = saved_x_emb.detach().requires_grad_(True)
        F0 = saved_f0.detach().requires_grad_(True)
        F1 = saved_f1.detach().requires_grad_(True)
        F2 = saved_f2.detach().requires_grad_(True)
        F3 = saved_f3.detach().requires_grad_(True)

        # 收集需要梯度的参数
        params = [p for p in subnet.parameters() if p.requires_grad]

        # 为了重算一致性，必要时临时切换 eval/train 状态
        was_states = []
        if use_eval:
            for m in subnet.mamba_blocks:
                was_states.append(m.training)
                m.eval()

        with torch.enable_grad():
            y0, y1, y2, y3 = _subnet_forward_formula(subnet, X, F0, F1, F2, F3)

            # 计算对输入和参数的 VJP
            outs = (X, y0, y1, y2, y3)
            gout = (grad_x_emb, grad_y0, grad_y1, grad_y2, grad_y3)
            grads = torch.autograd.grad(
                outputs=outs,
                inputs=[X, F0, F1, F2, F3] + params,
                grad_outputs=gout,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        # 恢复模块状态
        if use_eval:
            for m, t in zip(subnet.mamba_blocks, was_states):
                m.train(t)

        # 拆分输入与参数梯度
        gX, gF0, gF1, gF2, gF3, *gparams = grads

        # 将参数梯度累加到 .grad
        for p, gp in zip(params, gparams):
            if gp is None:
                continue
            if p.grad is None:
                p.grad = gp.detach()
            else:
                p.grad.add_(gp.detach())

        # 处理 None -> 0 的输入梯度
        def _nz(g, ref):
            return g if g is not None else torch.zeros_like(ref)

        gX = _nz(gX, X)
        gF0 = _nz(gF0, F0)
        gF1 = _nz(gF1, F1)
        gF2 = _nz(gF2, F2)
        gF3 = _nz(gF3, F3)

        # 对应 forward 的参数顺序：subnet, use_eval, x_emb, f0, f1, f2, f3
        return (None,        # subnet
                None,        # use_eval
                gX,
                gF0,
                gF1,
                gF2,
                gF3)


def _subnet_forward_reverse_call(subnet: nn.Module, x_emb: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor,
                                 use_eval: bool = True):
    """包装调用 _SubNetRevFunction，实现类似 .apply(local_funs, alpha, *args) 的接口。"""
    return _SubNetRevFunction.apply(subnet, use_eval, x_emb, f0, f1, f2, f3)


def _maybe_tokens_to_spatial(tokens_list):
    # 将 (B, N, C) 列表转为 (B, C, H, W)
    if tokens_list[0].ndim == 3:
        out = []
        for tokens in tokens_list:
            B, N, C = tokens.shape
            H = W = int(math.sqrt(N))
            out.append(tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
        return out
    return tokens_list


def _maybe_spatial_to_tokens(tokens_list):
    # 将 (B, C, H, W) 列表转回 (B, N, C)
    if tokens_list[0].ndim == 4:
        out = []
        for feat in tokens_list:
            B, C, H, W = feat.shape
            out.append(feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C))
        return out
    return tokens_list


def _split_tokens_list(tokens_list):
    # 统一解包
    if len(tokens_list) != 5:
        raise ValueError("tokens_list must be length 5: [x_emb,f0,f1,f2,f3]")
    return tokens_list[0], tokens_list[1], tokens_list[2], tokens_list[3], tokens_list[4]


def _pack_tokens_list(x_emb, y0, y1, y2, y3):
    return [x_emb, y0, y1, y2, y3]


def _subnet_forward_formula(subnet: nn.Module, x_emb, f0, f1, f2, f3):
    M0, M1, M2, M3 = subnet.mamba_blocks[0], subnet.mamba_blocks[1], subnet.mamba_blocks[2], subnet.mamba_blocks[3]
    up1, up2, up3 = subnet.upsample1, subnet.upsample2, subnet.upsample3
    dn0, dn1, dn2 = subnet.downsample0, subnet.downsample1, subnet.downsample2
    a0 = subnet.alpha0 if hasattr(subnet, 'alpha0') and subnet.alpha0 is not None else None
    a1 = subnet.alpha1 if hasattr(subnet, 'alpha1') and subnet.alpha1 is not None else None
    a2 = subnet.alpha2 if hasattr(subnet, 'alpha2') and subnet.alpha2 is not None else None
    a3 = subnet.alpha3 if hasattr(subnet, 'alpha3') and subnet.alpha3 is not None else None
    y0 = (f0 if a0 is None else f0 * a0) + M0(up1(f1) + x_emb)
    y1 = (f1 if a1 is None else f1 * a1) + M1(up2(f2) + dn0(y0))
    y2 = (f2 if a2 is None else f2 * a2) + M2(up3(f3) + dn1(y1))
    y3 = (f3 if a3 is None else f3 * a3) + M3(dn2(y2))
    return y0, y1, y2, y3


# 给 SubNet 增加一个可逆式前向的 API（不破坏原 forward）
def _subnet_forward_reverse(self: SubNet, tokens_list, *, use_eval: bool = True):
    # 统一到 (B,C,H,W)
    spatials = _maybe_tokens_to_spatial(tokens_list)
    x_emb, f0, f1, f2, f3 = _split_tokens_list(spatials)
    x_emb2, y0, y1, y2, y3 = _subnet_forward_reverse_call(self, x_emb, f0, f1, f2, f3, use_eval=use_eval)
    out_spatials = _pack_tokens_list(x_emb2, y0, y1, y2, y3)
    # 返回与输入同格式
    return _maybe_spatial_to_tokens(out_spatials)


# 将方法绑定到类
setattr(SubNet, 'forward_reverse', _subnet_forward_reverse)



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
        self.pwconv1 = nn.Linear(in_channel, hidden_dim, bias=False) # 升维# pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel, bias=False)# 降维
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
    def __init__(self, embed_dims=[96,192,384,768], base_scale_init=0.1):
        super().__init__()
        
        self.upsample1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            # 1x1 Conv，去掉 bias
            nn.Conv2d(embed_dims[1], embed_dims[1] // 2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(embed_dims[1] // 2, affine=True),  # 给IN加可学习仿射参数
            nn.GELU(),

            nn.Conv2d(embed_dims[1] // 2, embed_dims[1] // 2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[1] // 2, affine=True),
            nn.GELU(),          
        )
        self.convblock01 = nn.Sequential(
            ConvNextBlock(embed_dims[1]//2, 2*embed_dims[1]//2, embed_dims[1]//2, kernel_size=3, layer_scale_init_value=1.0, drop_path=0.05),
            ChannelAttention(dim=embed_dims[1]//2,num_heads=2,bias=True),
            )        


        self.upsample2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            # 1x1 Conv，去掉 bias
            nn.Conv2d(embed_dims[2], embed_dims[2] // 2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(embed_dims[2] // 2, affine=True),  # 给IN加可学习仿射参数
            nn.GELU(),

            nn.Conv2d(embed_dims[2] // 2, embed_dims[2] // 2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[2] // 2, affine=True),
            nn.GELU(),   
        )
        self.convblock12 = nn.Sequential(
            ConvNextBlock(embed_dims[2]//2, 2*embed_dims[2]//2, embed_dims[2]//2, kernel_size=3, layer_scale_init_value=1.0, drop_path=0.05),
            ChannelAttention(dim=embed_dims[2]//2,num_heads=2,bias=True),
            )
        

        self.upsample3 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            # 1x1 Conv，去掉 bias
            nn.Conv2d(embed_dims[3], embed_dims[3] // 2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(embed_dims[3] // 2, affine=True),  # 给IN加可学习仿射参数
            nn.GELU(),

            nn.Conv2d(embed_dims[3] // 2, embed_dims[3] // 2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[3] // 2, affine=True),
            nn.GELU(),             
        )
        self.convblock23 = nn.Sequential(
            ConvNextBlock(embed_dims[3]//2, 2*embed_dims[3]//2, embed_dims[3]//2, kernel_size=3, layer_scale_init_value=1.0, drop_path=0.05),
            ChannelAttention(dim=embed_dims[3]//2,num_heads=2,bias=True),
            )
        

        # Base缩放因子
        # self.base_scale = nn.Parameter(torch.tensor(base_scale_init))
        self.base_scale_T = nn.Parameter(torch.tensor(base_scale_init))
        self.base_scale_R = nn.Parameter(torch.tensor(base_scale_init))
        
        
        # 上采样和卷积解码层
        self.decoder = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(96, affine=True),
            nn.GELU(),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.GELU(),

            FrequencyProcessor(channels=64,int_size=128),

            # 特征细化
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.GELU(),

            # 最终输出层（后面没有归一化，bias 可以保留）
            nn.Conv2d(32, 6, kernel_size=1, bias=True)  # 6通道输出 (T, R)
        )
        
    def forward(self, tokens_list, resident_tokens_list, x_in):
        # tokens_list: (B, self.embed_dim, H_i, W_i)
        # x_in: (B, 3, 256, 256) 原始输入
        
        if tokens_list[0].ndim == 3:  # 3维张量
            for i, tokens in enumerate(tokens_list):
                B, N, C = tokens.shape
                H = W = int(math.sqrt(N))
                tokens_list[i] = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                # (B C H W)
        else:
            tokens_list = tokens_list
            pass  # 已经是(B C H W)格式

        if resident_tokens_list[0].ndim == 3:  # 3维张量
            # 注意：这里应遍历 resident_tokens_list 自身，避免误用 tokens_list
            for i, res_tokens in enumerate(resident_tokens_list):
                B, N, C = res_tokens.shape
                H = W = int(math.sqrt(N))
                resident_tokens_list[i] = res_tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                # (B C H W)
        else:
            resident_tokens_list = resident_tokens_list 
            pass  # 已经是(B C H W)格式

        f0,f1,f2,f3 = tokens_list[1],tokens_list[2],tokens_list[3],tokens_list[4]
        r0,r1,r2,r3 = resident_tokens_list[1],resident_tokens_list[2],resident_tokens_list[3],resident_tokens_list[4]

        f3 = f3 + r3

        o2 = self.convblock23((f2 + self.upsample3(f3))) + r2 # (B, 384, 16, 16)

        o1 = self.convblock12((f1 + self.upsample2(o2))) + r1 # (B, 192, 32, 32)

        o0 = self.convblock01((f0 + self.upsample1(o1))) + r0 # (B, 96, 64, 64)

        # 解码
        delta = self.decoder(o0)  # (B, 6, 256, 256)
        
        # Base residual: 输入图像的 residual base（T、R 分别保留一份轻微的输入基线）
        # 这里按通道维拼接，无需在 batch 维做任何复制；形状保持为 (B,6,H,W)
        output = delta + torch.cat([
            self.base_scale_T * x_in,  # 对应 T 分支的基线
            self.base_scale_R * x_in   # 对应 R 分支的基线
        ], dim=1)  # (B, 6, 256, 256)

        
        return output  # (B, 6, 256, 256)