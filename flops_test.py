import torch
from thop import profile, clever_format
from thop.vision.basic_hooks import count_convNd, count_linear
from vmamba import selective_scan_flop_jit, flops_selective_scan_fn

# ----------------------
# 1. 定义模型
# ----------------------
from MTRRNet import MTRREngine

# ----------------------
# 2. 加载模型
# ----------------------
model = MTRREngine(device='cuda', training=False)

model.eval()

# ----------------------
# 3. 注册自定义 FLOPs
# ----------------------
# 用于分别统计 Mamba / Swin
mamba_flops = 0
swin_flops = 0
mamba_params = 0
swin_params = 0

# --- Mamba ---
def count_vmamba(m, x, y):
    global mamba_flops, mamba_params
    inp = x[0]  # (B, C, H, W)
    B, C, H, W = inp.shape
    L = H * W
    D = C
    N = 16  # 内部状态维度，可根据实际改

    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)

    # total_ops 初始化
    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64, device=inp.device)
    else:
        m.total_ops = m.total_ops.to(inp.device)

    m.total_ops += torch.tensor([flops], dtype=torch.float64, device=inp.device)
    mamba_flops += flops
    mamba_params += sum(p.numel() for p in m.parameters())

# --- Swin ---
swin_flops = 0
swin_params = 0

def count_transformer_block(m, x, y):
    """
    统计 SwinTransformerBlock FLOPs 和 Params
    支持输入为 (B, C, H, W) 或 (B, N, C)
    """
    global swin_flops, swin_params

    inp = x[0]

    # 判断输入维度
    if inp.ndim == 4:  # (B, C, H, W)
        B, C, H, W = inp.shape
        N = H * W
    elif inp.ndim == 3:  # (B, N, C)
        B, N, C = inp.shape
    else:
        raise ValueError(f"Unexpected input shape {inp.shape}")

    # 粗略计算 FLOPs
    qkv = 3 * B * N * C * C          # QKV 投影
    attn = B * N * N * C             # 注意力得分
    proj = B * N * C * C             # 输出投影
    flops = qkv + attn + proj

    # 初始化 total_ops
    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64, device=inp.device)
    else:
        m.total_ops = m.total_ops.to(inp.device)

    m.total_ops += torch.tensor([flops], dtype=torch.float64, device=inp.device)

    # 累加全局统计
    swin_flops += flops
    swin_params += sum(p.numel() for p in m.parameters())
# 注册自定义 hook

from token_modules import VSSTokenMambaModule,SwinTokenBlock
import thop

custom_ops = {
    VSSTokenMambaModule: count_vmamba,
    SwinTokenBlock: count_transformer_block,
    torch.nn.Conv2d: count_convNd,
    torch.nn.Linear: count_linear,
}




# ----------------------
# 4. 统计 Params
# ----------------------
num_params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {num_params/1e6:.2f} M")

# ----------------------
# 5. 统计 FLOPs
# ----------------------
dummy_input = torch.randn(1, 3, 256, 256).to('cuda')
model.I = dummy_input
flops, params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops)
flops, params = clever_format([flops, params], "%.3f")

# ----------------------
# 统计 被调用模块
# ----------------------
hooked_modules = []

def debug_hook(m, x, y):
    hooked_modules.append(m.__class__.__name__)
    return None

for name, module in model.named_modules():
    module.register_forward_hook(debug_hook)


model.forward()


print("Forward 中被调用的模块类型:")
print(set(hooked_modules))





print(f"Total FLOPs: {flops}")
print(f"Total Params: {params}")

# ----------------------
# 6. 输出 Mamba / Swin 各自占比
# ----------------------
print(f"Mamba Params: {mamba_params/1e6:.2f} M, FLOPs: {mamba_flops/1e9:.2f} G")
print(f"Swin Params: {swin_params/1e6:.2f} M, FLOPs: {swin_flops/1e9:.2f} G")


