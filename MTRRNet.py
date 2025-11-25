# MTRRNet: Mamba + Transformer for Reflection Removal in Endoscopy Images
import torch
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn as nn
from classifier import PretrainedConvNext_e2e
import math
import tabulate
from MTRR_RD_modules import LRM

class MTRRNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self._init_token_only()
        
    def _init_token_only(self):
        """初始化Token-only版本的组件"""
        # 延迟导入token模块以避免循环导入和依赖问题
        from MTRR_token_modules import (
            Encoder, SubNet, UnifiedTokenDecoder, init_all_weights
        )
        init_all_weights(self)
        
        # 反射先验
        self.ref_detect = LRM('cuda')  # 输出64通道的256*256
        
        self.use_rev = True

        # 编码器
        self.token_encoder = Encoder(
            in_chans=64,
            embed_dim=96,
            mamba_blocks=[10, 10, 10, 10],    # Mamba处理低频
            swin_blocks=[4, 4, 4, 4],          # Swin处理高频
            drop_branch_prob=0.2
        )
        self.token_decoder0 = UnifiedTokenDecoder(
            embed_dims=[96,192,384,768],         # 输入token维度
            base_scale_init=0.1    # base缩放因子初始值
        )
        

        # Token SubNet：多尺度token融合
        self.token_subnet1 = SubNet(
            embed_dims=[96,192,384,768],         # 融合后的token维度
            mam_blocks=[6, 6, 6, 6],           # 融合细化的block数
            # mam_blocks=[3, 3, 3, 3],           # 融合细化的block数
            use_rev=self.use_rev
        )
        self.token_decoder1 = UnifiedTokenDecoder(
            embed_dims=[96,192,384,768],         # 输入token维度
            base_scale_init=0.1    # base缩放因子初始值
        )


        self.token_subnet2 = SubNet(
            embed_dims=[96,192,384,768],         # 融合后的token维度
            mam_blocks=[6, 6, 6, 6],           # 融合细化的block数
            # mam_blocks=[3, 3, 3, 3],           # 融合细化的block数
            use_rev=self.use_rev
        )
        self.token_decoder2 = UnifiedTokenDecoder(
            embed_dims=[96,192,384,768],         # 输入token维度
            base_scale_init=0.1    # base缩放因子初始值
        )


        self.token_subnet3 = SubNet(
            embed_dims=[96,192,384,768],         # 融合后的token维度
            mam_blocks=[6, 6, 6, 6],           # 融合细化的block数
            use_rev=self.use_rev
        )
        self.token_decoder3 = UnifiedTokenDecoder(
            embed_dims=[96,192,384,768],         # 输入token维度
            base_scale_init=0.1    # base缩放因子初始值
        )


    def forward(self, x_in):
        
        x,c_map = self.ref_detect(x_in) # 输出64通道的256*256
        
        # 2 Token编码
        tokens_list = self.token_encoder(x)
        # tokens_list: [t0, t1, t2, t3] 每个(B, N_i, C_i)
        
        resident_tokens_list = tokens_list 

        out0 = self.token_decoder0(tokens_list, resident_tokens_list, x_in)
        
        # 3. Token SubNet融合
        # fused_tokens = self.token_subnet1(tokens_list)  # (B, ref_H*ref_W, embed_dim)

        # tokens_list = self.token_subnet1(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        # fused_tokens = self.token_subnet2(tokens_list)  # (B, ref_H*ref_W, embed_dim)

        tokens_list = self.token_subnet1(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        out1 = self.token_decoder1(tokens_list, resident_tokens_list, x_in)  # (B, 6, 256, 256)

        tokens_list = self.token_subnet2(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        out2 = self.token_decoder2(tokens_list, resident_tokens_list, x_in)  # (B, 6, 256, 256)

        tokens_list = self.token_subnet3(tokens_list)  # (B, ref_H*ref_W, embed_dim)
        out3 = self.token_decoder3(tokens_list, resident_tokens_list, x_in)  # (B, 6, 256, 256)

        outs = [out0,out1,out2,out3]
        # outs = [torch.zeros_like(out3),torch.zeros_like(out3),torch.zeros_like(out3),out3]  # 测试吞吐量时不需要中间监督
        
        return outs, c_map


    



class MTRREngine(nn.Module):
 
    def __init__(self, opts=None, device='cuda', net_c=None):
        super(MTRREngine, self).__init__()
        self.device = device 
        self.opts  = opts
        self.visual_names = ['fake_Ts', 'fake_Rs', 'c_map', 'I', 'Ic', 'T', 'R']
        self.fake_Ts = [None]*4  # 存储三个尺度的去反射图
        self.fake_Rs = [None]*4  # 存储三个尺度的反射图
        self.netG_T = MTRRNet().to(device)  
        self.net_c = net_c  


        # print(torch.load('./pretrained/cls_model.pth', map_location=str(self.device)).keys())
        # self.net_c = PretrainedConvNext_e2e("convnext_small_in22k").cuda()
        
        # self.net_c.eval()  # 预训练模型不需要训练        



    def load_checkpoint(self, optimizer,scheduler):
        if self.opts.model_path is not None:
            model_path = self.opts.model_path
            print('Load the model from %s' % model_path)
            model_state = torch.load(model_path, map_location=str(self.device), weights_only=False)
            
            self.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in model_state['netG_T'].items()},strict=True)

            if 'optimizer_state_dict' in model_state:
                optimizer.load_state_dict(model_state['optimizer_state_dict'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = model_state.get('lr', param_group['lr'])

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
        
        
        self.outs,self.c_map = self.netG_T(self.Ic)  # 改为使用self.I而非self.Ic
        
        for i in range(0,len(self.outs)):
            self.fake_Ts[i] = self.outs[i][:,0:3,:,:] 

        for i in range(0,len(self.outs)):
            self.fake_Rs[i] = self.outs[i][:,3:6,:,:] 
        

        # self.c_map = torch.zeros_like(self.I) # 不要rdm了


        
 
    def monitor_layer_stats(self):
        """监控模型所有层，包括 Mamba2 内部子模块"""
        hooks = []
        model = self.netG_T

        # 修正钩子函数参数（正确接收module, input, output）
        def _hook_fn(module, input, output, layer_name):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()
                min_val = output.min().item()
                max_val = output.max().item()
                median = output.median().item()
                l2_norm = torch.norm(output).item()

                is_nan = math.isnan(mean) or math.isnan(std)
                if is_nan or self.opts.always_print:
                    msg = (f"{layer_name:<100} | {mean:>12.6e} | {std:>12.6e} | {min_val:>12.6e} | "
                           f"{max_val:>12.6e} | {median:>12.6e} | {l2_norm:>12.6e} | {tuple(output.shape)}")
                    # print(msg)
                    with open('./debug/state.log', 'a') as f:
                        f.write(msg + '\n')

        # 导入 Mamba2 用于类型检查
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
            has_mamba2 = True
        except ImportError:
            has_mamba2 = False

        # 遍历所有子模块并注册钩子
        mamba2_internal_count = 0
        for name, module in model.named_modules():
            # 跳过容器类
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                continue

            # 为所有模块注册钩子
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: _hook_fn(m, inp, out, name)
            )
            hooks.append(hook)

            # 特别处理 Mamba2：额外为其内部子模块注册钩子
            if has_mamba2 and isinstance(module, Mamba2):
                for sub_name in ['in_proj', 'conv1d', 'act', 'norm', 'out_proj']:
                    if hasattr(module, sub_name):
                        sub_module = getattr(module, sub_name)
                        sub_hook = sub_module.register_forward_hook(
                            lambda m, inp, out, fn=f"{name}.{sub_name}": _hook_fn(m, inp, out, fn)
                        )
                        hooks.append(sub_hook)
                        mamba2_internal_count += 1

        if mamba2_internal_count > 0:
            print(f"[Monitor] Registered {mamba2_internal_count} Mamba2 internal hooks")   
        



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
