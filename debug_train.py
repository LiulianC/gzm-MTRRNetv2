import os
import torch
from MTRRNet import MTRREngine
from customloss import CustomLoss
from dataset.new_dataset1 import DSRTestDataset, HyperKDataset
from torch.utils.data import ConcatDataset
import math
import warnings

warnings.filterwarnings('ignore')

# 配置参数
class DebugOpts:
    def __init__(self):
        self.data_root = './data'
        self.model_dir = './model'
        self.save_dir = './results'
        self.batch_size = 4
        self.shuffle = True
        self.num_workers = 0
        self.enable_finetune = False
        self.model_path = './model_fit/model_latest.pth'
        self.model_path = None
        self.always_print = 1  # 总是打印所有层

opts = DebugOpts()

LearnRate = 1e-3

# 运行3个训练步骤
step = 0
max_steps = 10

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建debug目录
os.makedirs('./debug', exist_ok=True)

# 删除旧的日志文件
if os.path.exists('./debug/debug-state.log'):
    os.remove('./debug/debug-state.log')
if os.path.exists('./debug/debug-grad.log'):
    os.remove('./debug/debug-grad.log')

# 初始化模型
print("Initializing model...")
model = MTRREngine(opts, device)

# 加载权重
if opts.model_path and os.path.exists(opts.model_path):
    print(f"Loading checkpoint from {opts.model_path}")
    checkpoint = torch.load(opts.model_path, map_location=device, weights_only=False)
    model.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in checkpoint['netG_T'].items()}, strict=True)
    print("Checkpoint loaded successfully")
else:
    print("No checkpoint found, using random initialization")

# 独立的完整 state 监控（包括 Mamba 内部层）
def monitor_all_layers_independently(model):
    """
    独立监控所有层的 forward 输出，包括展开 Mamba 变体的内部层
    不依赖于 MTRRNet.py 的 monitor_layer_stats()
    """
    hooks = []

    def make_hook(layer_name):
        def hook_fn(mod, inp, output):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()
                min_val = output.min().item()
                max_val = output.max().item()
                median = output.median().item()
                l2_norm = torch.norm(output).item()

                msg = (f"{layer_name:<100} | {mean:>12.6e} | {std:>12.6e} | {min_val:>12.6e} | "
                       f"{max_val:>12.6e} | {median:>12.6e} | {l2_norm:>12.6e} | {tuple(output.shape)}")
                with open('./debug/debug-state.log', 'a') as f:
                    f.write(msg + '\n')
        return hook_fn

    # 导入所有三种 Mamba 变体
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        has_mamba = True
    except ImportError:
        has_mamba = False
        Mamba = Mamba2 = Mamba2Simple = type(None)

    mamba_count = 0
    mamba2_count = 0
    mamba2simple_count = 0
    total_layers = 0

    # 遍历所有模块
    from torch import nn
    for name, module in model.netG_T.named_modules():
        # 跳过容器类模块
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue

        # 检查是否是 Mamba 变体
        is_mamba_variant = False
        if has_mamba and isinstance(module, (Mamba, Mamba2, Mamba2Simple)):
            is_mamba_variant = True
            if isinstance(module, Mamba2):
                mamba2_count += 1
                module_type = "Mamba2"
            elif isinstance(module, Mamba2Simple):
                mamba2simple_count += 1
                module_type = "Mamba2Simple"
            elif isinstance(module, Mamba):
                mamba_count += 1
                module_type = "Mamba"

            # 为 Mamba 模块本身注册钩子
            hook = module.register_forward_hook(make_hook(f"{name} ({module_type})"))
            hooks.append(hook)
            total_layers += 1

            # 为 Mamba 内部子模块注册钩子
            for sub_name, sub_module in module.named_modules():
                if sub_name:  # 跳过模块自身（空字符串）
                    full_name = f"{name} ({module_type}).{sub_name}"
                    hook = sub_module.register_forward_hook(make_hook(full_name))
                    hooks.append(hook)
                    total_layers += 1
        else:
            # 普通模块直接注册钩子
            if name:  # 跳过根模块
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                total_layers += 1

    mamba_total = mamba_count + mamba2_count + mamba2simple_count
    print(f"Registered {total_layers} hooks for forward monitoring:")
    print(f"  - Total Mamba variants: {mamba_total}")
    print(f"    - Mamba: {mamba_count}")
    print(f"    - Mamba2: {mamba2_count}")
    print(f"    - Mamba2Simple: {mamba2simple_count}")
    return hooks


# 注册独立的 state 监控钩子
print("Registering independent forward hooks for all layers (including Mamba internals)...")
state_hooks = monitor_all_layers_independently(model)

# 准备数据集（使用少量数据）
print("Loading dataset...")
tissue_dir = '/home/gzm/gzm-MTRRVideo/data/tissue_real'
tissue_data = DSRTestDataset(
    datadir=tissue_dir,
    fns='/home/gzm/gzm-MTRRVideo/data/tissue_real_index/train1.txt',
    size=800,  # 只使用8个样本
    enable_transforms=True,
    unaligned_transforms=False,
    if_align=True,
    real=True,
    HW=[256, 256],
    SamplerSize=True
)

HyperKroot = "/home/gzm/gzm-MTRRNetv2/data/EndoData"
HyperKJson = "/home/gzm/gzm-MTRRNetv2/data/EndoData/test.json"
HyperK_data = HyperKDataset(
    root=HyperKroot,
    json_path=HyperKJson,
    start=343,
    end=369,
    size=1200,  # 只使用12个样本
    enable_transforms=True,
    unaligned_transforms=False,
    if_align=True,
    HW=[256, 256],
    flag=None,
    SamplerSize=True,
    color_jitter=True
)

train_data = ConcatDataset([tissue_data, HyperK_data])
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opts.batch_size,
    shuffle=opts.shuffle,
    num_workers=opts.num_workers,
    drop_last=False,
    pin_memory=True
)

print(f"Dataset loaded: {len(train_data)} samples")

# 损失函数
loss_function = CustomLoss().to(device)

# 优化器 - 为不同模块设置差异化学习率
# token_encoder.encoder_unit0/1/2 和 token_encoder.patchembed: 1e-3
# token_subnet1/2/3: 1e-4
# token_decoder: 1e-4
param_groups = []

# 模块名称到学习率的映射
LearnRate = 1e-4

# lr_base = 5e-3, gamma = 0.93
lr_map = {
    # ---- 头（heads）----
    'token_decoder3': 2.000000e-02,  # 最终输出头（×4）
    'token_decoder2': 1.500000e-02,  # 中间头（×3）
    'token_decoder1': 1.500000e-02,  # 中间头（×3）
    'token_decoder0': 5.000000e-03,  # 早期头（= base）

    # ---- 从输出端到输入端，按 γ=0.93 衰减（lr_base=5e-3）----
    # subnet3（最靠近输出）
    'token_subnet3.mamba_blocks.3': 5.000000e-03,  # d=0
    'token_subnet3.mamba_blocks.2': 4.650000e-03,  # d=1
    'token_subnet3.mamba_blocks.1': 4.324500e-03,  # d=2
    'token_subnet3.mamba_blocks.0': 4.021785e-03,  # d=3

    # subnet2
    'token_subnet2.mamba_blocks.3': 3.740260e-03,  # d=4
    'token_subnet2.mamba_blocks.2': 3.478442e-03,  # d=5
    'token_subnet2.mamba_blocks.1': 3.234951e-03,  # d=6
    'token_subnet2.mamba_blocks.0': 3.008504e-03,  # d=7

    # subnet1
    'token_subnet1.mamba_blocks.3': 2.797909e-03,  # d=8
    'token_subnet1.mamba_blocks.2': 2.602055e-03,  # d=9
    'token_subnet1.mamba_blocks.1': 2.419912e-03,  # d=10
    'token_subnet1.mamba_blocks.0': 2.250518e-03,  # d=11

    # —— 这些 mamba_processor 容易梯度消失：在对应 encoder_unit 基础上 ×1.5 ——    长后缀排在前面
    'token_encoder.encoder_unit3.mamba_processor': 3.139472e-03,  # 1.5 × 2.092981e-03
    'token_encoder.encoder_unit2.mamba_processor': 2.919709e-03,  # 1.5 × 1.946473e-03
    'token_encoder.encoder_unit1.mamba_processor': 2.715330e-03,  # 1.5 × 1.810220e-03
    'token_encoder.encoder_unit0.mamba_processor': 2.525256e-03,  # 1.5 × 1.683504e-03

    # encoder（越往下越小）
    'token_encoder.encoder_unit3': 2.092981e-03,  # d=12
    'token_encoder.encoder_unit2': 1.946473e-03,  # d=13
    'token_encoder.encoder_unit1': 1.810220e-03,  # d=14
    'token_encoder.encoder_unit0': 1.683504e-03,  # d=15

    # 最早的嵌入
    'token_encoder.patchembed':    1.565659e-03,  # d=16
}




# 为每个模块分别收集参数
module_params = {k: {'decay': [], 'no_decay': []} for k in lr_map.keys()}
module_params['other'] = {'decay': [], 'no_decay': []}

for n, p in model.netG_T.named_parameters():
    if not p.requires_grad:
        continue
    
    # 判断是否需要权重衰减
    need_decay = not ((p.dim() == 1 and 'weight' in n) or any(x in n.lower() for x in ['raw_gamma', 'norm', 'bn', 'running_mean', 'running_var']))
    
    # 找到参数所属的模块
    matched = False
    for module_name in lr_map.keys():
        if n.startswith(module_name):
            if need_decay:
                module_params[module_name]['decay'].append(p)
            else:
                module_params[module_name]['no_decay'].append(p)
            matched = True
            break
    
    # 其他模块使用默认学习率 LearnRate
    if not matched:
        if need_decay:
            module_params['other']['decay'].append(p)
        else:
            module_params['other']['no_decay'].append(p)

# 构建优化器参数组
for module_name, lr in lr_map.items():
    if module_params[module_name]['decay']:
        param_groups.append({
            'params': module_params[module_name]['decay'],
            'lr': lr,
            'weight_decay': 1e-4,
            'name': f'{module_name}_decay'
        })
    if module_params[module_name]['no_decay']:
        param_groups.append({
            'params': module_params[module_name]['no_decay'],
            'lr': lr,
            'weight_decay': 0.0,
            'name': f'{module_name}_no_decay'
        })

# 其他模块使用默认学习率
if module_params['other']['decay']:
    param_groups.append({
        'params': module_params['other']['decay'],
        'lr': LearnRate,
        'weight_decay':0.0,
        'name': 'other_decay'
    })
if module_params['other']['no_decay']:
    param_groups.append({
        'params': module_params['other']['no_decay'],
        'lr': LearnRate,
        'weight_decay': 0.0,
        'name': 'other_no_decay'
    })

optimizer = torch.optim.Adam(param_groups, betas=(0.5, 0.999), eps=1e-8)

if opts.model_path and os.path.exists(opts.model_path):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded successfully")

# for i, group in enumerate(optimizer.param_groups):
#     wd = group.get("weight_decay", None)
#     print(f"\n==== Param Group {i} | weight_decay = {wd} ====")
#     for p in group["params"]:
#         # 如果模型有 .named_parameters() 可反查名字：
#         for name, param in model.named_parameters():
#             if param is p:
#                 print(" ", name)
#                 break



# 打印各模块的学习率设置
print("\n" + "="*80)
print("Learning Rate Configuration:")
print("="*80)
for group in optimizer.param_groups:
    n_params = sum(p.numel() for p in group['params'])
    print(f"{group['name']:<40} LR: {group['lr']:.2e}  Params: {n_params:>10,}  WD: {group['weight_decay']}")
print("="*80 + "\n")

# 设置训练模式
model.netG_T.train()

print("\n" + "="*120)
print("Starting debug training for 3 steps...")
print("="*120 + "\n")



for batch_idx, data in enumerate(train_loader):
    if step >= max_steps:
        break

    step += 1
    print(f"\n{'='*120}")
    print(f"DEBUG STEP {step}/{max_steps}")
    print(f"{'='*120}")

    # 写入state.log的步骤头
    with open('./debug/debug-state.log', 'a') as f:
        f.write(f"\n{'='*280}\n")
        f.write(f"STEP {step} - Forward Pass Statistics (Including Mamba Internal Layers)\n")
        f.write(f"{'='*280}\n")
        f.write(f"{'Layer Name':<100} | {'Mean':>12} | {'Std':>12} | {'Min':>12} | {'Max':>12} | {'Median':>12} | {'L2Norm':>12} | {'Shape'}\n")
        f.write(f"{'-'*280}\n")

    # 设置输入
    model.set_input(data)

    # 前向传播（触发state监控钩子）
    print("Running forward pass...")
    model.inference()

    # 获取输出
    visuals = model.get_current_visuals()
    train_input = visuals['I']
    train_ipt = visuals['Ic']
    train_label1 = visuals['T']
    train_label2 = visuals['R']
    train_fake_Ts = visuals['fake_Ts']
    train_fake_Rs = visuals['fake_Rs']
    train_rcmaps = visuals['c_map']

    # 计算损失
    print("Computing loss...")
    # 计算四个阶段的损失，确保第4个输出(out3)参与反传
    _, _, _, _, _, all_loss0 = loss_function(
        train_fake_Ts[0], train_label1, train_ipt, train_rcmaps, train_fake_Rs[0], train_label2
    )
    _, _, _, _, _, all_loss1 = loss_function(
        train_fake_Ts[1], train_label1, train_ipt, train_rcmaps, train_fake_Rs[1], train_label2
    )
    _, _, _, _, _, all_loss2 = loss_function(
        train_fake_Ts[2], train_label1, train_ipt, train_rcmaps, train_fake_Rs[2], train_label2
    )
    loss_table, mse_loss, vgg_loss, ssim_loss, fake_Ts_range_penalty, all_loss3 = loss_function(
        train_fake_Ts[3], train_label1, train_ipt, train_rcmaps, train_fake_Rs[3], train_label2
    )
    # 与主训练脚本保持一致的权重设置，保证梯度覆盖到最后一阶段
    all_loss = 0.5*all_loss0 + 0.5*all_loss1 + 0.5*all_loss2 + 1.0*all_loss3

    print(f"Loss: {all_loss.item():.6f} | MSE: {mse_loss.item():.6f} | VGG: {vgg_loss.item():.6f} | SSIM: {ssim_loss.item():.6f}")

    # 反向传播
    print("Running backward pass...")
    optimizer.zero_grad()
    all_loss.backward()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

    # 写入grad.log的步骤头
    with open('./debug/debug-grad.log', 'a') as f:
        f.write(f"\n{'='*220}\n")
        f.write(f"STEP {step} - Gradient Statistics (Including Mamba Internal Parameters)\n")
        f.write(f"{'='*220}\n")
        f.write(f"{'Parameter Name':<100} | {'Grad Mean':>15} | {'Grad Std':>15} | {'Grad Min':>15} | {'Grad Max':>15} | {'Grad Norm':>15}\n")
        f.write(f"{'-'*220}\n")

    # 打印梯度统计（包含所有参数，包括 Mamba 内部参数）
    print("Collecting gradient statistics...")
    with open('./debug/debug-grad.log', 'a') as f:
        for name, param in model.netG_T.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_min = param.grad.min().item()
                grad_max = param.grad.max().item()
                grad_norm = torch.norm(param.grad).item()

                # 检测NaN或Inf
                is_nan = math.isnan(grad_mean) or math.isnan(grad_std)
                is_inf = math.isinf(grad_mean) or math.isinf(grad_std)

                status = ""
                if is_nan:
                    status = " [NaN DETECTED!]"
                elif is_inf:
                    status = " [Inf DETECTED!]"
                elif abs(grad_norm) > 100:
                    status = " [Large Gradient]"
                elif abs(grad_norm) < 1e-6:
                    status = " [Vanishing]"

                msg = (f"{name:<100} | {grad_mean:>15.8e} | {grad_std:>15.8e} | "f"{grad_min:>15.8e} | {grad_max:>15.8e} | {grad_norm:>15.8e}{status}")
                f.write(msg + '\n')                
                    



    # 更新参数
    optimizer.step()

    print(f"Step {step} completed.")

print("\n" + "="*120)
print("Debug training completed!")
print("="*120)
print(f"\nResults saved to:")
print(f"  - Forward pass statistics (with Mamba internals): ./debug/debug-state.log")
print(f"  - Gradient statistics (with Mamba internals): ./debug/debug-grad.log")
print("\nYou can review the logs to analyze layer outputs and gradients.")
print("\nNote: Mamba internal layers (in_proj, conv1d, act, norm, out_proj, x_proj, dt_proj) are expanded in-place.")
print("If some Mamba layers are not captured, it's because Mamba uses fused CUDA kernels.")
print("Consider setting use_mem_eff_path=False in Mamba initialization for full monitoring.")
