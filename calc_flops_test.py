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
model = MTRREngine(device='cuda')
model.eval()

# ----------------------
# 2.1 可选：排除一些不计入 FLOPs/Params 的模块
# ----------------------
EXCLUDE_DECODERS = [
    'netG_T.token_decoder0',
    'netG_T.token_decoder1',
    'netG_T.token_decoder2',
]

class NoOpDecoder(torch.nn.Module):
    def forward(self, tokens_list, resident_tokens_list, x_in):
        B, _, H, W = x_in.shape
        return torch.zeros(B, 6, H, W, device=x_in.device, dtype=x_in.dtype)

def _replace_module_by_path(root, dotted_path: str, new_mod: torch.nn.Module):
    parts = dotted_path.split('.')
    cur = root
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise AttributeError(f"Path segment '{p}' not found while resolving '{dotted_path}'")
        cur = getattr(cur, p)
    last = parts[-1]
    if not hasattr(cur, last):
        raise AttributeError(f"Target attribute '{last}' not found on '{'.'.join(parts[:-1])}'")
    setattr(cur, last, new_mod)

for path in EXCLUDE_DECODERS:
    try:
        _replace_module_by_path(model, path, NoOpDecoder())
        print(f"[FLOPs] Excluded module by replacement: {path}")
    except AttributeError as e:
        print(f"[FLOPs] Skip exclude '{path}': {e}")

# ----------------------
# 3. 注册自定义 FLOPs
# ----------------------
# ----------------------
# 3. 注册自定义 FLOPs
# ----------------------
mamba_flops = 0
swin_flops = 0
mamba_params = 0
swin_params = 0

# --- 改进的 Mamba FLOPs 计算 ---
def count_vmamba(m, x, y):
    global mamba_flops, mamba_params
    
    # 添加调试信息
    print(f"[DEBUG] Mamba module called: {m.__class__.__name__}")
    print(f"[DEBUG] Input shape: {x[0].shape if isinstance(x, (list, tuple)) else x.shape}")
    
    # 处理不同的输入格式
    if isinstance(x, (list, tuple)):
        inp = x[0]
    else:
        inp = x
    
    # 处理不同的输入维度
    if inp.ndim == 4:  # 可能是 (B, C, H, W) 或 (B, H, W, C)
        # 检查通道维度的位置
        if inp.shape[1] >= 96:  # 如果第二个维度较小，可能是通道维度
            # 格式: (B, C, H, W)
            B, C, H, W = inp.shape
            L = H * W
            D = C
        else:
            # 格式: (B, H, W, C)  
            B, H, W, C = inp.shape
            L = H * W
            D = C
    elif inp.ndim == 3:  # (B, L, C)
        B, L, D = inp.shape
    else:
        B, L, D = inp.shape[0], inp.shape[1], inp.shape[2]
    
    N = 16  # 内部状态维度
    
    # 获取实际的层数
    if hasattr(m, 'Mamba_num'):
        mamba_layers = m.Mamba_num
        mha_layers = m.Trans_num
        print(f"[DEBUG] Mamba2Blocks_Standard配置: Mamba层数={mamba_layers}, MHA层数={mha_layers}")
    else:
        # 如果没有n_layer属性，假设只有1层
        n_layer = 1
        mamba_layers = 1
        mha_layers = 0
        print(f"[DEBUG] 警告: 未找到n_layer属性，使用默认单层")
    
    print(f"[DEBUG] Mamba dimensions - B:{B}, L:{L}, D:{D}, N:{N}")
    
    # 计算Mamba层的FLOPs
    mamba_flops_total = 0
    try:
        single_mamba_flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
        mamba_flops_total = single_mamba_flops * mamba_layers
        print(f"[DEBUG] 单层Mamba FLOPs: {single_mamba_flops}")
        print(f"[DEBUG] {mamba_layers}层Mamba总FLOPs: {mamba_flops_total}")
    except Exception as e:
        print(f"[DEBUG] flops_selective_scan_fn失败: {e}")
        single_mamba_flops = 8 * B * L * D * N
        mamba_flops_total = single_mamba_flops * mamba_layers
        print(f"[DEBUG] 使用备选计算: 单层={single_mamba_flops}, 总={mamba_flops_total}")
    
    # 计算MHA层的FLOPs（使用与Swin相同的计算方法）
    mha_flops_total = 0
    if mha_layers > 0:
        # Transformer FLOPs计算
        qkv_flops = 3 * B * L * D * D
        num_heads = 8
        head_dim = D // num_heads
        attn_flops = B * num_heads * L * head_dim * L
        proj_flops = B * L * D * D
        mlp_flops = 2 * B * L * D * (4 * D)  # 假设MLP扩展4倍
        
        single_mha_flops = qkv_flops + attn_flops + proj_flops + mlp_flops
        mha_flops_total = single_mha_flops * mha_layers
        print(f"[DEBUG] 单层MHA FLOPs: {single_mha_flops}")
        print(f"[DEBUG] {mha_layers}层MHA总FLOPs: {mha_flops_total}")
    
    total_flops = mamba_flops_total + mha_flops_total
    print(f"[DEBUG] Mamba2Blocks_Standard总FLOPs: {total_flops}")

    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64, device=inp.device)
    else:
        m.total_ops = m.total_ops.to(inp.device)

    m.total_ops += torch.tensor([total_flops], dtype=torch.float64, device=inp.device)
    mamba_flops += total_flops  # 注意：这里把MHA的FLOPs也计入mamba_flops，因为Mamba2Blocks_Standard被注册为Mamba模块
    print(f"[DEBUG] Mamba FLOPs累计: {mamba_flops}\n")

# --- 改进的 Swin FLOPs 计算 ---
def count_transformer_block(m, x, y):
    global swin_flops, swin_params
    
    print(f"[DEBUG] Swin module called: {m.__class__.__name__}")

    if isinstance(x, (list, tuple)):
        inp = x[0]
    else:
        inp = x

    print(f"[DEBUG] Swin input shape: {inp.shape}")

    if inp.ndim == 4:  # (B, C, H, W)
        B, C, H, W = inp.shape
        N = H * W
    elif inp.ndim == 3:  # (B, N, C)
        B, N, C = inp.shape
    else:
        B, N, C = inp.shape[0], inp.shape[1], inp.shape[2]

    print(f"[DEBUG] Swin dimensions - B:{B}, N:{N}, C:{C}")

    # 更详细的Transformer FLOPs计算
    qkv_flops = 3 * B * N * C * C
    num_heads = 8
    head_dim = C // num_heads
    attn_flops = B * num_heads * N * head_dim * N
    proj_flops = B * N * C * C
    mlp_flops = 2 * B * N * C * (4 * C)
    
    flops = qkv_flops + attn_flops + proj_flops + mlp_flops

    print(f"[DEBUG] Swin FLOPs breakdown - QKV: {qkv_flops}, Attn: {attn_flops}, Proj: {proj_flops}, MLP: {mlp_flops}")
    print(f"[DEBUG] Swin total FLOPs: {flops}")

    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64, device=inp.device)
    else:
        m.total_ops = m.total_ops.to(inp.device)

    m.total_ops += torch.tensor([flops], dtype=torch.float64, device=inp.device)
    swin_flops += flops
    print(f"[DEBUG] Swin FLOPs accumulated: {swin_flops}\n")

# --- 扩展自定义操作注册 ---
from token_modules import VSSTokenMambaModule, SwinTokenBlock, Mamba2Blocks_Standard, SwinTransformerBlock

# 尝试导入其他可能的Mamba相关模块
try:
    from vmamba import Mamba
    MAMBA_CLASSES = (VSSTokenMambaModule, Mamba2Blocks_Standard, Mamba)
except:
    MAMBA_CLASSES = (VSSTokenMambaModule, Mamba2Blocks_Standard)

custom_ops = {
    **{cls: count_vmamba for cls in MAMBA_CLASSES},
    SwinTransformerBlock: count_transformer_block,
    torch.nn.Conv2d: count_convNd,
    torch.nn.ConvTranspose2d: count_convNd,
    torch.nn.Linear: count_linear,
}

# 添加WindowAttention（Swin的核心组件）
try:
    from timm.models.layers import WindowAttention
    custom_ops[WindowAttention] = count_transformer_block
except:
    pass

# ----------------------
# 4. 调试：检查前向传播中调用的模块
# ----------------------
hooked_modules = []

def debug_hook(m, x, y):
    hooked_modules.append(m.__class__.__name__)
    return None

# 注册调试hook
for name, module in model.named_modules():
    module.register_forward_hook(debug_hook)

# ----------------------
# 5. 统计 Params 和 FLOPs
# ----------------------
dummy_input = torch.randn(1, 3, 256, 256).to('cuda')

# 先运行一次前向传播来收集模块信息
with torch.no_grad():

    model.I = dummy_input
    model.forward()
    output = model.fake_Ts[3]

print("Forward 中被调用的模块类型:")
called_modules = set(hooked_modules)
print(called_modules)
print(f"Total unique modules: {len(called_modules)}")

# 检查是否有Mamba相关模块被调用但未注册
mamba_related = {'Mamba', 'VSSTokenMambaModule', 'Mamba2Blocks_Standard'} & called_modules
swin_related = {'SwinTokenBlock', 'WindowAttention', 'SwinTransformerBlock'} & called_modules

print(f"Mamba related modules called: {mamba_related}")
print(f"Swin related modules called: {swin_related}")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {total_params/1e6:.3f}M")

# ... 前面的代码保持不变 ...

# 统计FLOPs
flops, params = profile(
    model, 
    inputs=(dummy_input,), 
    custom_ops=custom_ops,
    verbose=True  # 添加详细输出
)

# 分别统计Mamba和Swin的参数
mamba_params = 0
swin_params = 0
other_params = 0

for name, module in model.named_modules():
    module_params = sum(p.numel() for p in module.parameters())
    
    if isinstance(module, MAMBA_CLASSES):
        mamba_params += module_params
    elif isinstance(module, (SwinTokenBlock, SwinTransformerBlock)):
        swin_params += module_params
    elif 'WindowAttention' in str(type(module)):
        swin_params += module_params
    else:
        other_params += module_params

# 计算总参数用于验证
total_calculated_params = mamba_params + swin_params + other_params
total_actual_params = sum(p.numel() for p in model.parameters())

print(f"参数统计验证:")
print(f"Mamba参数: {mamba_params/1e6:.3f}M")
print(f"Swin参数: {swin_params/1e6:.3f}M") 
print(f"其他参数: {other_params/1e6:.3f}M")
print(f"计算总参数: {total_calculated_params/1e6:.3f}M")
print(f"实际总参数: {total_actual_params/1e6:.3f}M")

flops, params = clever_format([flops, params], "%.3f")

# 按要求的格式输出
print(f"Total FLOPs: {flops}")
print(f"Total Params: {params}")
print(f"Mamba Params: {mamba_params/1e6:.2f} M, FLOPs: {mamba_flops/1e9:.2f} G")
print(f"Swin Params: {swin_params/1e6:.2f} M, FLOPs: {swin_flops/1e9:.2f} G")




# 检查FLOPs统计的合理性
if mamba_flops < 1e9:  # 如果Mamba FLOPs小于1G，可能统计有问题
    print("警告: Mamba FLOPs可能统计不完整！")
    print("建议检查:")
    print("1. Mamba模块是否正确注册到custom_ops")
    print("2. flops_selective_scan_fn函数是否正确实现")
    print("3. 输入维度处理是否正确")






# # ----------------------
# # 6. 输出详细检查信息
# # ----------------------
# print("\n" + "="*50)
# print("详细检查信息")
# print("="*50)

# # 检查Mamba模块的实际参数和调用情况
# mamba_modules = []
# swin_modules = []

# for name, module in model.named_modules():
#     if isinstance(module, MAMBA_CLASSES):
#         mamba_modules.append((name, module))
#     elif isinstance(module, (SwinTokenBlock, SwinTransformerBlock)):
#         swin_modules.append((name, module))

# print(f"找到的Mamba模块数量: {len(mamba_modules)}")
# for name, module in mamba_modules:
#     params = sum(p.numel() for p in module.parameters())
#     print(f"  {name}: {params} 参数")

# print(f"找到的Swin模块数量: {len(swin_modules)}")
# for name, module in swin_modules:
#     params = sum(p.numel() for p in module.parameters())
#     print(f"  {name}: {params} 参数")

# # 检查flops_selective_scan_fn函数
# print("\n检查flops_selective_scan_fn函数:")
# try:
#     # 测试一个典型输入
#     test_flops = flops_selective_scan_fn(B=1, L=65536, D=256, N=16, with_D=True, with_Z=False)
#     print(f"测试 flops_selective_scan_fn(B=1, L=65536, D=256, N=16): {test_flops}")
#     print(f"这相当于: {test_flops/1e9:.2f} GFLOPs")
# except Exception as e:
#     print(f"flops_selective_scan_fn测试失败: {e}")

# # 手动计算Mamba FLOPs作为对比
# print("\n手动计算Mamba FLOPs对比:")
# B, L, D, N = 1, 65536, 256, 16  # 典型值
# manual_flops = 8 * B * L * D * N  # 标准Selective Scan公式
# print(f"手动计算: 8 * {B} * {L} * {D} * {N} = {manual_flops} FLOPs")
# print(f"这相当于: {manual_flops/1e9:.2f} GFLOPs")

# print(f"\n当前统计结果:")
# print(f"Mamba FLOPs: {mamba_flops/1e9:.2f} GFLOPs")
# print(f"Swin FLOPs: {swin_flops/1e9:.2f} GFLOPs")

if mamba_flops < 1e9:  # 如果Mamba FLOPs小于1G
    print("\n⚠️  Mamba FLOPs统计确实有问题!")
    print("可能原因:")
    print("1. flops_selective_scan_fn函数计算不准确")
    print("2. Mamba模块的输入维度可能比预期的小")
    print("3. 某些Mamba模块可能没有被正确hook")