# MTRRNet: Mamba + Transformer for Reflection Removal in Endoscopy Images
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import PretrainedConvNext_e2e
import math
import tabulate
import torch.utils.checkpoint as cp

# padding是边缘复制 减少边框伪影
class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()

        # Replication padding （复制边缘）
        if padding > 0:
            self.add_module('pad', nn.ReplicationPad2d(padding))  # [left, right, top, bottom] 都为 padding

        # 卷积
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias))

        # 归一化
        if norm is not None:
            self.add_module('norm', norm(out_channels))

        # 激活
        if act is not None:
            self.add_module('act', act)

# Attention-Aware Fusion
class AAF(nn.Module):
    """
    输入: List[Tensor], 每个 shape 为 [B, C, H, W]
    输出: Tensor, shape 为 [B, C, H, W]
    """
    def __init__(self, in_channels, num_inputs): # in_channels 每个图像的通道 num_input 有多少个图像
        super(AAF, self).__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs
        
        # 输入 concat 后通道为 C*num_inputs
        self.attn = nn.Sequential(
            nn.Conv2d(num_inputs * in_channels, num_inputs * in_channels * 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_inputs * in_channels * 16, num_inputs, kernel_size=1, bias=False),
            nn.Softmax(dim=1)  # 对每个位置的 num_inputs 做归一化
        )

    def forward(self, features):
        # features: list of Tensors [B, C, H, W]
        B, C, H, W = features[0].shape
        x = torch.cat(features, dim=1)  # shape: [B, C*num_inputs, H, W]
        attn_weights = self.attn(x)     # shape: [B, num_inputs, H, W]
        
        # 融合：对每个尺度乘以权重后相加
        out = 0
        for i in range(self.num_inputs):
            weight = attn_weights[:, i:i+1, :, :]  # [B,1,H,W]
            out += features[i] * weight            # 广播乘法
        return out

# 多尺度拉普拉斯特征提取
class LaplacianPyramid(nn.Module):
    # filter laplacian LOG kernel, kernel size: 3.
    # The laplacian Pyramid is used to generate high frequency images.

    def __init__(self, device='cuda', dim=3):
        super(LaplacianPyramid, self).__init__()

        # 2D laplacian kernel (2D LOG operator).
        self.channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0)  # (dim, 1, 3, 3)

        # 可学习 kernel + 保存初值用于约束
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        self.register_buffer('kernel_init', torch.FloatTensor(laplacian_kernel).clone())

        # 允许偏离初值的带宽（越小越“像拉普拉斯”）
        self.epsilon = 0.05

        # 初始化时先夹一下，避免一开始就越界
        with torch.no_grad():
            self.kernel.data.clamp_(self.kernel_init - self.epsilon, self.kernel_init + self.epsilon)

        # 给 kernel 挂梯度钩子：投影/截断不合规梯度，训练时每步都会生效
        self.kernel.register_hook(self._lap_kernel_grad_hook)

        self.aaf = AAF(3, 4)

    def _lap_kernel_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        自定义反向传播（仅作用于 self.kernel 的梯度）：
        1) 阻断把 kernel 推出 [init-eps, init+eps] 带的梯度方向
        2) 去掉破坏“每个通道 3×3 权重零和”的梯度分量
        3) 做一次温和的范数裁剪，稳住训练
        """
        if grad is None:
            return grad

        # 1) 局部带约束：已经到上界的元素，禁止继续“往外推”；到下界同理
        with torch.no_grad():
            over_upper = (self.kernel - self.kernel_init) >= self.epsilon  # 已到上界
            under_lower = (self.kernel_init - self.kernel) >= self.epsilon  # 已到下界

        # 对应方向的梯度清零（只清“继续往外”的方向）
        blocked_up = over_upper & (grad > 0)
        blocked_down = under_lower & (grad < 0)
        grad = torch.where(blocked_up | blocked_down, torch.zeros_like(grad), grad)

        # 2) 拉普拉斯零和约束：去掉每个(通道×组)的均值分量，避免整体偏移
        # 形状: (C,1,3,3)，对最后两维求均值并回减
        mean_per_kernel = grad.mean(dim=(2, 3), keepdim=True)
        grad = grad - mean_per_kernel

        # 3) 温和范数裁剪（按整体 L2 范数做一次缩放，不会“硬砍”）
        total_norm = torch.linalg.vector_norm(grad)
        if torch.isfinite(total_norm) and total_norm > 1.0:
            grad = grad * (1.0 / total_norm)

        return grad

    def forward(self, x):
        # 多尺度
        x0 = F.interpolate(x, scale_factor=0.125, mode='bicubic')
        x1 = F.interpolate(x, scale_factor=0.25, mode='bicubic')
        x2 = F.interpolate(x, scale_factor=0.5, mode='bicubic')

        # 深度可分组卷积提取高频
        lap_0 = F.conv2d(x0, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x,  self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)

        # 还原到同一尺度
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bicubic')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bicubic')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bicubic')

        lap_out = torch.cat([lap_0, lap_1, lap_2, lap_3], dim=1)
        return lap_out, x0, x1, x2


class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        # self.norm = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 更稳定的初始化 + LayerNorm 替代 BatchNorm
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.LayerNorm([channel * reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * reduction, channel, 1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x = torch.clamp(x, -10.0, 10.0)  # 限制极值
        avg_output = self.fc(torch.tanh(self.avg_pool(x)) * 3)
        max_output = self.fc(torch.tanh(self.max_pool(x)) * 3)

        out = avg_output + max_output
        return self.sigmoid(out)

# 对特征图的每个空间位置（像素）分配一个权重（0~1），突出重要区域并抑制无关背景。
class SpatialAttention(nn.Module):
    # The spatial attention block.
    # Simgoid(conv([F_max^s; F_avg^s])) -> output spatial attention feature.
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = Conv2DLayer(in_channels=2, out_channels=1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B,1,H,W]

        pool_out = torch.cat([avg_out, max_out], dim=1) # [B,2,H,W]
        x = self.conv(pool_out) # 融合
        return self.sigmoid(x) # 输出

# 通道注意力+空间注意力
class CBAMlayer(nn.Module):
    # THe CBAM module(Channel & Spatial Attention feature) implement
    # reference from paper: CBAM(Convolutional Block Attention Module)
    def __init__(self, channel, reduction=1):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()

    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x

# 带有通道注意力和空间注意力的残差快
class ResidualCbamBlock(nn.Module):
    # The ResBlock which contain CBAM attention module.

    def __init__(self, channel, reduction, norm=nn.BatchNorm2d, dilation=1, bias=False, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = CBAMlayer(channel,reduction=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam_layer(x)

        out = x + res
        return out

class SElayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = ((torch.tanh(self.se(y))+1)/2).view(b, c, 1, 1)
        return x * y

class SEResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf. 
    # Which contains SE-layer implements.
    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=0.1, act=nn.GELU()):# 调用时既没有归一化 也没有激活
        super(SEResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, kernel_size=3, stride=1, padding=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se_layer = None
        self.res_scale = res_scale # res_scale 是一个缩放因子，用于对残差块的输出进行缩放。其主要目的是在训练过程中稳定网络的梯度，从而加速收敛并提高训练的稳定性。
        if se_reduction is not None: # se_reduction 通常与 Squeeze-and-Excitation (SE) 模块有关。SE 模块是一种在卷积神经网络（CNN）中的注意力机制，它通过自适应地重新校准通道特征来提升模型的表现。se_reduction 是 SE 模块中的一个参数，用于控制特征图在 Squeeze 阶段的通道缩减比例。
            self.se_layer = SElayer(channel, se_reduction)

    def forward(self, x):
        res = x # 残差
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x) # 通道注意力
        x = x * self.res_scale 
        out = x + res # 残差链接
        return out

# --------------------------
# 编码块 CSA 
# --------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super().__init__()

        self.conv = Conv2DLayer(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.csa0 = ResidualCbamBlock(channel=out_channels, reduction=reduction)
        self.csa1 = ResidualCbamBlock(channel=out_channels, reduction=reduction)
        self.out = nn.Sequential(
            Conv2DLayer(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),  # 添加 BatchNorm2d
            nn.GELU()                    
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.csa0(x)
        x = self.csa1(x)
        return self.out(x)

# --------------------------
# 解码块
# --------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            Conv2DLayer(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加 BatchNorm2d
            nn.GELU()                    
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        return self.conv(x)

# --------------------------
# RDM 模块：完整结构版
# --------------------------
class RDM(nn.Module):
    def __init__(self):
        super().__init__()

        self.Lap = LaplacianPyramid(dim=3)

        self.se0 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)
        self.se1 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)
        self.se2 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)
        self.se3 = SEResidualBlock(channel=6, se_reduction=2, res_scale=0.1)

        self.se4 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)
        self.se5 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)
        self.se6 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)
        self.se7 = SEResidualBlock(channel=18, se_reduction=6, res_scale=0.1)

        # Output
        self.out_head = Conv2DLayer(in_channels=18,out_channels=3,kernel_size=1,padding=0,stride=1,bias=False)          
        
        self.tanh = nn.Tanh()
        
        

    def forward(self, x):

        lap,xd8,xd4,xd2 = self.Lap(x) # B 12 H W 和 B,3,H,W

        x_se = torch.cat([x, x],dim=1) # B 6 256 256 扩展是因为se要压
        x_se = self.se0(x_se)
        x_se = self.se1(x_se)
        x_se = self.se2(x_se)
        x_se = self.se3(x_se)

        x_se = torch.cat([x_se, lap], dim=1) # B 6+12 256 256
        x_se = self.se4(x_se)
        x_se = self.se5(x_se)
        x_se = self.se6(x_se)
        x_se = self.se7(x_se)

        out = self.out_head(x_se) # (B,3,256,256)
        out = (self.tanh(out)+1)/2
        return out,xd8,xd4,xd2



class MTRRNet(nn.Module):
    """Token-only多尺度架构：
    多尺度编码 → Token融合 → 统一解码
    频带分工：低频→Mamba，高频→Swin
    """
    def __init__(self, use_legacy=False):
        super().__init__()
        self.use_legacy = use_legacy
        
        if use_legacy:
            # 使用旧实现的组件
            self._init_legacy()
        else:
            # 新Token-only实现
            self._init_token_only()
        
    def _init_token_only(self):
        """初始化Token-only版本的组件"""
        # 延迟导入token模块以避免循环导入和依赖问题
        from token_modules import (
            Encoder, SubNet, UnifiedTokenDecoder, init_all_weights
        )
        init_all_weights(self)
        
        # RDM保持不变，用于生成rmap
        # self.rdm = RDM()
        
        # 编码器
        self.token_encoder = Encoder(
            mamba_blocks=[10, 10, 10, 10],    # Mamba处理低频
            swin_blocks=[4, 4, 4, 4],          # Swin处理高频
            drop_branch_prob=0.2
        )
        
        # Token SubNet：多尺度token融合
        self.token_subnet1 = SubNet(
            embed_dims=[96,192,384,768],         # 融合后的token维度
            mam_blocks=[6, 6, 6, 6]           # 融合细化的block数
            # mam_blocks=[3, 3, 3, 3]           # 融合细化的block数
        )
        self.token_subnet2 = SubNet(
            embed_dims=[96,192,384,768],         # 融合后的token维度
            mam_blocks=[6, 6, 6, 6]           # 融合细化的block数
            # mam_blocks=[3, 3, 3, 3]           # 融合细化的block数
        )
        self.token_subnet3 = SubNet(
            embed_dims=[96,192,384,768],         # 融合后的token维度
            mam_blocks=[6, 6, 6, 6]           # 融合细化的block数
        )

        # 统一Token解码器
        self.token_decoder = UnifiedTokenDecoder(
            embed_dims=[96,192,384,768],         # 输入token维度
            base_scale_init=0.3    # base缩放因子初始值
        )


        
    def forward(self, x_in):
        """
        输入: x_in (B, 3, 256, 256)
        输出: (rmap, out) 其中out为6通道(T,R)
        """
        if self.use_legacy:
            return self._forward_legacy(x_in)
        else:
            return self._forward_token_only(x_in)
    
    
    def _forward_token_only(self, x_in):
        """Token-only版本的前向传播"""
        
        
        # 2. 多尺度Token编码
        tokens_list = self.token_encoder(x_in)
        # tokens_list: [t0, t1, t2, t3] 每个(B, N_i, C_i)
        
        # 3. Token SubNet融合
        # fused_tokens = self.token_subnet1(tokens_list)  # (B, ref_H*ref_W, embed_dim)

        # tokens_list = self.token_subnet1(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        # fused_tokens = self.token_subnet2(tokens_list)  # (B, ref_H*ref_W, embed_dim)

        tokens_list = self.token_subnet1(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        tokens_list = self.token_subnet2(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        fused_tokens = self.token_subnet3(tokens_list)  # (B, ref_H*ref_W, embed_dim)


        # 4. 统一解码：token → 6通道(T,R)
        out = self.token_decoder(fused_tokens, x_in)  # (B, 6, 256, 256)
        
        return out


    



class MTRREngine(nn.Module):
 
    def __init__(self, opts=None, device='cuda', net_c=None):
        super(MTRREngine, self).__init__()
        self.device = device 
        self.opts  = opts
        self.visual_names = ['fake_T', 'fake_R', 'c_map', 'I', 'Ic', 'T', 'R']
        self.netG_T = MTRRNet().to(device)  
        self.net_c = net_c  


        # print(torch.load('./pretrained/cls_model.pth', map_location=str(self.device)).keys())
        # self.net_c = PretrainedConvNext_e2e("convnext_small_in22k").cuda()
        
        # self.net_c.eval()  # 预训练模型不需要训练        



    def load_checkpoint(self, optimizer,scheduler):
        if self.opts.model_path is not None:
            model_path = self.opts.model_path
            print('Load the model from %s' % model_path)
            model_state = torch.load(model_path, map_location=str(self.device),weights_only=False)
            
            self.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in model_state['netG_T'].items()},strict=True)

            if 'optimizer_state_dict' in model_state:
                try:
                    optimizer.load_state_dict(model_state['optimizer_state_dict'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = model_state.get('lr', param_group['lr'])
                except ValueError as e:
                    print(f"Warning: Could not load optimizer state due to: {e}")
                    print("Continuing with fresh optimizer state")
                    # 只设置学习率，不加载整个state
                    if 'lr' in model_state:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = model_state['lr']

            if 'scheduler_state_dict' in model_state:
                scheduler.load_state_dict(model_state['scheduler_state_dict'])

            best_val_loss = model_state.get('best_val_loss', None)

            best_val_psnr = model_state.get('best_val_psnr', None)

            best_val_ssim = model_state.get('best_val_ssim', None)                      

            early_stopping_counter = model_state.get('early_stopping_counter', 0)  

            if self.net_c is not None:
                if 'net_c' in model_state:
                    try:
                        self.net_c.load_state_dict(model_state['net_c'])
                    except ValueError as e:
                        print(f"Warning: Could not load net_c state due to: {e}")
                        print("Continuing with existing net_c state")
                else:
                    self.net_c.load_state_dict(torch.load('/home/gzm/gzm-MTRRNetv2/cls/cls_models/clsbest.pth', map_location=str(self.device)))

            epoch = model_state.get('epoch', None)
            print('Loaded model at epoch %d' % (epoch+1) if epoch is not None else 'Loaded model without epoch info')
            return epoch,best_val_loss,best_val_psnr,best_val_ssim,early_stopping_counter
        

    def get_current_visuals(self):
        # get the current visuals results.
        visual_result = OrderedDict() # 是 Python 标准库 collections 模块中的一个类，它是一个有序的字典，记录了键值对插入的顺序。
        for name in self.visual_names: # 这里遍历 self.visual_names 列表，该列表包含了需要获取的属性名称。 ['fake_Ts', 'fake_Rs', 'rcmaps', 'I'] 都在本class有定义
            if isinstance(name, str): # 检查 name 是否是字符串
                # 使用 getattr(self, name) 函数动态地获取 self 对象中名为 name 的属性的值，并将其存储在 visual_result 字典中
                visual_result[name] = getattr(self, name)
        return visual_result # 结果从 visual_names 来


    def set_input(self, input): 
        # load images dataset from dataloader.
        self.I = input['input'].to(self.device)
        self.T = input['target_t'].to(self.device)
        self.R = input['target_r'].to(self.device)
        


    def forward(self,input=None):# 这个input参数是为了匹配测flops函数接口

        
        # with torch.no_grad():
        #     self.Ic = self.net_c(self.I)

        # self.Ic = self.net_c(self.I)

        self.Ic = self.I  # 直接设置为原始输入
        
        # 使用原始输入调用token-only模型
        self.out = self.netG_T(self.Ic)  # 改为使用self.I而非self.Ic
        self.fake_T, self.fake_R = self.out[:,0:3,:,:],self.out[:,3:6,:,:]

        self.c_map = torch.zeros_like(self.I) # 不要rdm了


        
 
    def monitor_layer_stats(self):
        """仅监控模型的一级子模块（不深入嵌套结构）"""
        hooks = []
        model = self.netG_T

        # 修正钩子函数参数（正确接收module, input, output）
        def _hook_fn(module, input, output, layer_name):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()

                is_nan = math.isnan(mean) or math.isnan(std)
                if is_nan or self.opts.always_print:
                    msg = f"{layer_name:<50} | Mean: {mean:>15.6f} | Std: {std:>15.6f} | Shape: {tuple(output.shape)}"
                    # print(msg)
                    with open('./debug/state.log', 'a') as f:
                        f.write(msg + '\n')# 修正钩子函数参数（正确接收module, input, output）
      

        # 遍历所有子模块并注册钩子
        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):  # 过滤容器类（如Sequential）
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: _hook_fn(m, inp, out, name)
                )
                hooks.append(hook)   
        



    def monitor_layer_grad(self):
        with open('./debug/grad.log', 'a') as f:
            for name, param in self.netG_T.named_parameters():

                if param.grad is not None:
                    is_nan = math.isnan(param.grad.mean().item()) or math.isnan(param.grad.std().item())
                    is_nan = False
                    if is_nan or self.opts.always_print:
                        if param.grad is not None:
                            msg = (
                                f"Param: {name:<100} | "
                                f"Grad Mean: {param.grad.mean().item():.15f} | "
                                f"Grad Std: {param.grad.std().item():.15f}"
                            )
                        else:
                            msg = f"Param: {name:<50} | Grad is None"  # 梯度未回传  
                        # print(msg)
                        f.write(msg + '\n')

    def apply_weight_constraints(self):
        """动态裁剪权重，保持在合理范围内"""
        with torch.no_grad():
            for name, param in self.netG_T.named_parameters():
                # 针对不同参数类型使用不同的裁剪策略
                
                # PReLU参数特殊约束，避免负斜率过大
                if any(x in name for x in ['proj.2.weight', '.out.2.weight']) or ('norm_act' in name and name.endswith('.weight')):
                    param.data.clamp_(min=0.01, max=0.3)
                    
                # scale_raw参数约束
                elif name.endswith('scale_raw'):
                    param.data.clamp_(min=-2.0, max=2.0)
                    
                # 普通权重参数通用约束
                elif 'weight' in name and param.dim() > 1:
                    if param.numel() == 0:
                        continue  # 跳过空张量
                    if torch.max(torch.abs(param.data)) > 10.0:
                        param.data.clamp_(min=-10.0, max=10.0)

    def eval(self):
        self.netG_T.eval()

    def inference(self):
        # with torch.no_grad():
        self.forward()             #所以启动全部模型的最高层调用

    def count_parameters(self):
        table = []
        total = 0
        for name, param in self.netG_T.named_parameters():
            if param.requires_grad:
                num = param.numel()
                table.append([name, num, f"{num:,}"])
                total += num
        print(tabulate(table, headers=["Layer", "Size", "Formatted"], tablefmt="grid"))
        print(f"\nTotal trainable parameters: {total:,}")    

    # 微调代码
    def _set_requires_grad(self, module, flag: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _split_decay(self, named_params):
        """按是否权重衰减拆分参数（与现有规则保持一致）"""
        decay, no_decay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if (p.dim() == 1 and 'weight' in n) or any(x in n.lower() for x in ['raw_gamma', 'norm', 'bn', 'running_mean', 'running_var']):
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    def build_finetune_param_groups(self, opts):
        """
        根据微调模式构建参数组（仅当 opts.enable_finetune=True 时使用）：
        - 模块：
            encoder = token_encoder
            subnet  = token_subnet
            decoder = token_decoder
            rdm     = rdm（默认不训练）
        - ft_mode:
            'decoder_only'   仅解码器
            'freeze_encoder' 冻结编码器，训练 subnet + decoder（推荐作为第一阶段）
            'freeze_decoder' 冻结解码器，训练 encoder + subnet
            'all'            全部训练（后期收敛）
        - 学习率倍率：
            lr_mult_encoder / lr_mult_subnet / lr_mult_decoder
        """
        base_lr = getattr(opts, 'base_lr', 1e-4)
        wd      = getattr(opts, 'weight_decay', 1e-4)
        ft_mode = getattr(opts, 'ft_mode', 'freeze_encoder')
        lr_me   = getattr(opts, 'lr_mult_encoder', 0.1)
        lr_ms   = getattr(opts, 'lr_mult_subnet', 0.5)
        lr_md   = getattr(opts, 'lr_mult_decoder', 1.0)
        train_rdm = bool(getattr(opts, 'train_rdm', False))

        enc = getattr(self.netG_T, 'token_encoder', None)
        sub = getattr(self.netG_T, 'token_subnet',  None)
        dec = getattr(self.netG_T, 'token_decoder', None)
        rdm = getattr(self.netG_T, 'rdm',           None)

        # 1) 冻结/解冻
        if ft_mode == 'decoder_only':
            self._set_requires_grad(enc, False)
            self._set_requires_grad(sub, False)
            self._set_requires_grad(dec, True)
        elif ft_mode == 'freeze_encoder':
            self._set_requires_grad(enc, False)
            self._set_requires_grad(sub, True)
            self._set_requires_grad(dec, True)
        elif ft_mode == 'freeze_decoder':
            self._set_requires_grad(enc, True)
            self._set_requires_grad(sub, True)
            self._set_requires_grad(dec, False)
        elif ft_mode == 'all':
            self._set_requires_grad(enc, True)
            self._set_requires_grad(sub, True)
            self._set_requires_grad(dec, True)
        else:
            # 兜底：等价 freeze_encoder
            self._set_requires_grad(enc, False)
            self._set_requires_grad(sub, True)
            self._set_requires_grad(dec, True)

        # RDM 默认不训练
        self._set_requires_grad(rdm, train_rdm)

        # 2) 组装参数组（每个模块各自拆 decay/no_decay，设置不同 lr）
        param_groups = []

        if enc is not None:
            decay, no_decay = self._split_decay(enc.named_parameters())
            if decay:
                param_groups.append({'name': 'encoder_decay', 'params': decay, 'weight_decay': wd, 'lr': base_lr * lr_me, 'initial_lr': base_lr * lr_me})
            if no_decay:
                param_groups.append({'name': 'encoder_no_decay', 'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr * lr_me, 'initial_lr': base_lr * lr_me})

        if sub is not None:
            decay, no_decay = self._split_decay(sub.named_parameters())
            if decay:
                param_groups.append({'name': 'subnet_decay', 'params': decay, 'weight_decay': wd, 'lr': base_lr * lr_ms, 'initial_lr': base_lr * lr_ms})
            if no_decay:
                param_groups.append({'name': 'subnet_no_decay', 'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr * lr_ms, 'initial_lr': base_lr * lr_ms})

        if dec is not None:
            decay, no_decay = self._split_decay(dec.named_parameters())
            if decay:
                param_groups.append({'name': 'decoder_decay', 'params': decay, 'weight_decay': wd, 'lr': base_lr * lr_md, 'initial_lr': base_lr * lr_md})
            if no_decay:
                param_groups.append({'name': 'decoder_no_decay', 'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr * lr_md, 'initial_lr': base_lr * lr_md})

        if train_rdm and rdm is not None:
            decay, no_decay = self._split_decay(rdm.named_parameters())
            if decay:
                param_groups.append({'name': 'rdm_decay', 'params': decay, 'weight_decay': wd, 'lr': base_lr * 0.05, 'initial_lr': base_lr * 0.05})
            if no_decay:
                param_groups.append({'name': 'rdm_no_decay', 'params': no_decay, 'weight_decay': 0.0, 'lr': base_lr * 0.05, 'initial_lr': base_lr * 0.05})

        # 记录初始 lr 供 warmup 使用
        self._ft_param_groups_meta = [{'name': g.get('name', ''), 'initial_lr': g.get('initial_lr', g.get('lr', base_lr))} for g in param_groups]
        return param_groups

    def progressive_unfreeze(self, epoch, opts):
        """
        渐进解冻（字符串计划，如 "10:encoder,20:all"）
        到点即打开对应模块的 requires_grad
        """
        plan = getattr(opts, 'unfreeze_plan', '')
        if not plan:
            return
        items = [x.strip() for x in plan.split(',') if ':' in x]
        for it in items:
            try:
                ep, target = it.split(':')
                ep = int(ep)
            except:
                continue
            if epoch == ep:
                if target == 'encoder':
                    self._set_requires_grad(getattr(self.netG_T, 'token_encoder', None), True)
                elif target == 'decoder':
                    self._set_requires_grad(getattr(self.netG_T, 'token_decoder', None), True)
                elif target == 'subnet':
                    self._set_requires_grad(getattr(self.netG_T, 'token_subnet',  None), True)
                elif target == 'all':
                    self._set_requires_grad(getattr(self.netG_T, 'token_encoder', None), True)
                    self._set_requires_grad(getattr(self.netG_T, 'token_subnet',  None), True)
                    self._set_requires_grad(getattr(self.netG_T, 'token_decoder', None), True)
    

# --------------------------
# 模型验证
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTRRNet().to(device)
    x = torch.randn(1,3,256,256).to(device)  # 输入一张256x256 RGB图
    y = model(x)
    print(y.shape)  # 应输出 torch.Size([1, 3, 256, 256])
