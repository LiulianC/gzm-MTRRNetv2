import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()  # super() 是一个内置函数，返回一个临时对象，该对象允许你调用父类中的方法。这里的 Conv2DLayer 是当前子类的名字。self 是当前实例对象
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2 # dilation指的是一种修改卷积操作的方法，它通过在卷积核中插入空洞（即在卷积核的元素之间增加间距）来扩大感受野
            # 用 add_module 方法将卷积层注册到模块中，命名为 'conv2d'，这样这个层可以在整个模型中被追踪和更新。
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias)) # k_size 通常是 kernel size（卷积核大小）的缩写
        if norm is not None: # "Norm" 归一化函数
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)

class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        # 即使原始输入特征图非常大，经过这个池化层之后，输出的特征图的每个通道都会缩小成一个单一的值。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 自适应池化（Adaptive Pooling）是一种特殊的池化操作，可以将输入特征图缩放到指定的输出尺寸。参数 1 指定了输出特征图的高度和宽度均为 1，即将输入特征图缩放到 1x1 的大小
        self.se = nn.Sequential( # Sequential把括号内所有层打包
            nn.Linear(channel, channel // reduction),
            nn.LayerNorm(channel//reduction),
            nn.ReLU(inplace=True), # nn.ReLU(inplace=True) 是 PyTorch 中定义 ReLU 激活函数的一种方式，其中 inplace=True 表示在原地进行操作。这意味着激活函数将直接修改输入张量，而不创建新的张量。这可以节省内存，但需要注意，它会覆盖输入张量的值
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        # "Linear" 在深度学习中一般指线性层或全连接层，其作用是对输入数据进行仿射变换：
        # 即计算 Y = XWᵀ + b，其中 W 是权重矩阵，b 是偏置向量。

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y
        # 这里的 x 是输入特征图，y 是经过 Squeeze-and-Excitation 模块处理后的特征图。最终的输出是输入特征图和经过注意力机制处理后的特征图的逐元素相乘。
    

# 残差快 但加SElayer通道注意力
class ResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf. 
    # Which contains SE-layer implements.
    def __init__(self, channel, norm=nn.InstanceNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=1, act=nn.ReLU(True)):# 调用时既没有归一化 也没有激活
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
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

# 与SE一样 也是通道注意力 但细节不相同
# 同时使用GAP和全局最大池化（GMP）
# 双路径并行（GAP+GMP → 共享FC → 相加）
# 使用1*1卷积替代线性层
class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=True)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
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

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, padding=padding_size, bias=False, kernel_size=kernel_size)
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
    def __init__(self, channel, reduction=16):
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

    def __init__(self, channel, norm=nn.InstanceNorm2d, dilation=1, bias=False, cbam_reduction=None, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = None
        if cbam_reduction is not None:
            self.cbam_layer = CBAMlayer(channel, cbam_reduction)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam_layer:
            x = self.cbam_layer(x)

        out = x + res
        return out
    
# 多尺度拉普拉斯特征提取
class LaplacianPyramid(nn.Module):
    # filter laplacian LOG kernel, kernel size: 3.
    # The laplacian Pyramid is used to generate high frequency images.

    def __init__(self, device, dim=3):
        super(LaplacianPyramid, self).__init__()

        # 2D laplacian kernel (2D LOG operator).
        self.channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0) # 变成 (dim, 1, H, W) 
        # learnable laplacian kernel


        # 让 kernel 可学习但只允许在微小范围内变动
        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        self.register_buffer('kernel_init', torch.FloatTensor(laplacian_kernel).clone())

        # 限制 kernel 在初始值±epsilon范围内
        epsilon = 0.05
        with torch.no_grad():
            self.kernel.data.clamp_(self.kernel_init - epsilon, self.kernel_init + epsilon)


    def forward(self, x):
        # print(self.kernel[0,0,:,:])
        # pyramid module for 4 scales.
        x0 = F.interpolate(x, scale_factor=0.125, mode='bilinear')# 下采样到 1/8
        x1 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        # groups=self.channel_dim：表示使用分组卷积，分组数为 self.channel_dim。当 groups 等于输入通道数时，相当于对每个通道进行独立卷积。
        lap_0 = F.conv2d(x0, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bilinear')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bilinear')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bilinear')
        # lap_0, lap_1, lap_2, lap_3 是经过不同尺度的拉普拉斯卷积和插值后的图像高频部分。
        # 这些高频部分能够捕捉图像中的细节和边缘信息，有助于图像增强、复原等任务。
        # 最终的实现应包括将这些高频部分组合起来，以便进一步处理或计算损失

        return torch.cat([lap_0, lap_1, lap_2, lap_3], 1) # 使用 torch.cat 函数沿着指定的维度（在本例中为维度1，即通道维度）将多个张量拼接在一起
        # 返回(B,6*4,256,256)

class LRM(nn.Module):

    def __init__(self, device):
        super(LRM, self).__init__()

        # Laplacian blocks
        self.lap_pyramid = LaplacianPyramid(device, dim=6) # multi-scale laplacian submodules (RDMs)
        # self.lap_single = SingleLaplacian(device, dim=6)

        self.det_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )

        # SE-resblocks(ReLU)
        self.det_conv1 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv2 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv3 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        # 网络进入分支
        self.det_conv4_1 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4_2 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)

        # Convolutional blocks for encoding laplacian features.
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
            )

        # SE-resblocks(P-ReLU)
        self.det_conv6 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv7 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv8 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv9 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv10 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv11 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())

        # Activations.
        self.p_relu = nn.PReLU()
        self.relu = nn.ReLU()

        # Convolutional block for RCMap_{i+1}
        self.det_conv_mask0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3, stride= 1, padding= 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )        

    def forward(self, I):
        # I: original image.
        # T: transmission image.
        # h, c: hidden states for LSTM block in stage_1.

        x = torch.cat([I, I], 1) # 1 表示在channel上拼接成(B,6,256,256) 0 表示在batchsize上拼接
        # get laplacian(frequency) information of [I,T].
        lap = self.lap_pyramid(x) # 下采样3次 算子卷积4次 上采样3次 得到(B,24,256,256)

        # ----- Stage1 -----
        # encode [I, T].
        x = self.det_conv0(x) # x=(B,32,256,256)
        # se-resblock layer1 for [I, T] features.
        x = F.relu(self.det_conv1(x)) # x(B,32,256,256)
        x = F.relu(self.det_conv2(x)) # x(B,32,256,256)
        x = F.relu(self.det_conv3(x)) # x(B,32,256,256)
        # se-resblock layer2 for [I, T] features.
        x = F.relu(self.det_conv4(x)) # x(B,32,256,256)
        x = F.relu(self.det_conv4_1(x)) # x(B,32,256,256)
        x = F.relu(self.det_conv4_2(x)) # x(B,32,256,256)

        # encode [I_lap, T_lap].
        lap = self.det_conv5(lap) # input 24 channel, output 32 channel  
        # se-resblock layer3 for [I_lap, T_lap] features (p-relu for activation.)
        lap = self.p_relu(self.det_conv6(lap)) # x(B,32,256,256)
        lap = self.p_relu(self.det_conv7(lap)) # x(B,32,256,256) 
        lap = self.p_relu(self.det_conv8(lap)) # x(B,32,256,256)
        # predict RCMap from laplacian features.
        c_map = self.det_conv_mask0(lap) # c_map(B,1,256,256)
        # se-resblock layer4 for [I_lap, T_lap] features (p-relu for activation.)
        lap = self.p_relu(self.det_conv9(lap))  # x(B,32,256,256)
        lap = self.p_relu(self.det_conv10(lap)) # x(B,32,256,256)
        lap = self.p_relu(self.det_conv11(lap)) # x(B,32,256,256)

        # suppress transmission features.
        lap = lap * c_map # lap(B,32,256,256)

        # concat image & laplacian feature and recurrent features.
        # 通道合成  x(B,32,256,256) lap(B,32,256,256)  -> (B,64,256,256)
        x = torch.cat([x, lap], 1)  # h是上一循环的h
        return x,c_map








import torch.fft as fft
import math
from einops import rearrange


def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real

class Toning(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(Toning, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):
        x1 = inv_mag(x)
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Mapping(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3, res=True):
        """
        Parameters:
            in_features (int): Number of input features (channels).
            hidden_features (int): Number of features in hidden layers.
            hidden_layers (int): Number of hidden layers.
            out_features (int): Number of output features (channels).
            res (bool): Whether to use residual connections.
        """
        super(Mapping, self).__init__()

        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())

        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, inp):
        original_shape = inp.shape
        inp = inp.view(-1, inp.shape[1])

        output = self.net(inp)

        if self.res:
            output = output + inp
            output = torch.clamp(output, 0., 1.)

        output = output.view(original_shape)

        return output

class FrequencyProcessor(nn.Module):
    def __init__(self, channels=3, int_size=64):
        super(FrequencyProcessor, self).__init__()
        self.identity1 = nn.Conv2d(channels, channels, 1)
        self.identity2 = nn.Conv2d(channels, channels, 1)

        self.conv_f1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.map = Mapping(in_features=channels, out_features=channels, hidden_features=int_size, hidden_layers=5)
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.tone = Toning(channels)

    def forward(self, x):
        out = self.identity1(x)

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft = F.gelu(self.conv_f1(x_fft))
        x_fft = self.map(x_fft)
        x_reconstructed = fft.ifftn(x_fft, dim=(-2, -1)).real
        x_reconstructed += self.identity2(x)

        f_out = self.fuse(torch.cat([out, x_reconstructed], dim=1))

        return self.tone(f_out)
        # 输入输出都是BCHW    
        # 用在最开头，32维度上



# Channel-Wise Contextual Attention
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
        # 输入输出都是BCHW
        # 用在高维特征上




        # 用的是 overlap patchembed

        # 1×1 conv + depthwise 3×3 conv 使用卷积 + depthwise 卷积来产生 Q/K/V，而不是全连接

        # 整图操作 而不是用token