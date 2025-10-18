"""
reverse_function.py

基于 token_modules.SubNet 的近似可逆函数：
- 给定 SubNet 的输出 tokens_spatial_list = [x_emb, f0_out, f1_out, f2_out, f3_out]
- 在不修改 SubNet 参数的前提下，利用 SubNet 内部算子（上/下采样与 Mamba2 blocks）
  估计恢复进入 SubNet 之前的 tokens：[x_emb, f0, f1, f2, f3]

说明
- SubNet 前向：
    f0_out = f0 * alpha0 + M0( up1(f1)           + x_emb )
    f1_out = f1 * alpha1 + M1( up2(f2)           + dn0(f0_out) )
    f2_out = f2 * alpha2 + M2( up3(f3)           + dn1(f1_out) )
    f3_out = f3 * alpha3 + M3(         dn2(f2_out)           )
  其中 Mi = mamba_blocks[i]，up* 与 dn* 分别为上/下采样。

- 逆向估计（近似）：
    由最后一层开始自上而下恢复：
    f3 = (f3_out - M3(dn2(f2_out))) / alpha3
    f2 = (f2_out - M2(up3(f3) + dn1(f1_out))) / alpha2
    f1 = (f1_out - M1(up2(f2) + dn0(f0_out))) / alpha1
    f0 = (f0_out - M0(up1(f1) + x_emb)) / alpha0

- 注意：Mamba/Dropout/Norm 等不是严格可逆；此处为“方程求解式”的近似逆，
  用于调试/可视化/辅助约束，而非严格数值逆变换。
  为避免 Dropout 干扰，内部会临时将 mamba blocks 切换到 eval() 模式，并默认 no_grad。

用法示例
    from token_modules import SubNet
    from reverse_function import reverse_subnet

    subnet: SubNet = ...
    tokens_out = [x_emb, f0_out, f1_out, f2_out, f3_out]
    tokens_in_est = reverse_subnet(subnet, tokens_out, no_grad=True)

作者：自动生成
"""
from __future__ import annotations
from typing import List, Tuple, Iterable, Any
import contextlib
import torch

try:
    # 类型注解友好：运行时不依赖
    from token_modules import SubNet  # type: ignore
except Exception:  # pragma: no cover
    SubNet = object  # 仅为静态工具友好


def _clamp_alpha(alpha: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """确保 alpha 有下界，避免除零/极小数值放大。"""
    return torch.clamp(alpha, min=eps) if alpha is not None else torch.tensor(1.0, device=alpha.device if isinstance(alpha, torch.Tensor) else 'cpu')


@contextlib.contextmanager
def _temporary_eval(modules: Iterable[torch.nn.Module]):
    """将若干模块临时切换到 eval 模式，退出后恢复原状态。"""
    states: List[Tuple[torch.nn.Module, bool]] = []
    for m in modules:
        states.append((m, m.training))
        m.eval()
    try:
        yield
    finally:
        for m, was_training in states:
            m.train(was_training)


def reverse_subnet(
    subnet: Any,
    tokens_out: List[torch.Tensor],
    no_grad: bool = True,
    detach_mamba_inputs: bool = False,
) -> List[torch.Tensor]:
    """
    近似反演 SubNet：由输出 tokens 估计进入 SubNet 之前的 tokens。

    参数：
    - subnet: SubNet 实例（已构建、含 mamba_blocks / upsample* / downsample* / alpha*）
    - tokens_out: [x_emb, f0_out, f1_out, f2_out, f3_out]，均为 (B, C, H, W)
    - no_grad:     反演时是否禁用梯度（默认 True）
    - detach_mamba_inputs: 给 mamba 的输入先 .detach()，避免污染反向（默认 False）

    返回：
    - tokens_in_est: [x_emb, f0, f1, f2, f3]，形状同输入
    """
    assert isinstance(tokens_out, (list, tuple)) and len(tokens_out) == 5, \
        f"Expect 5 tensors [x_emb, f0_out, f1_out, f2_out, f3_out], got {len(tokens_out)}"

    x_emb, f0_o, f1_o, f2_o, f3_o = tokens_out

    # 读取组件
    M0, M1, M2, M3 = subnet.mamba_blocks[0], subnet.mamba_blocks[1], subnet.mamba_blocks[2], subnet.mamba_blocks[3]
    up1, up2, up3 = subnet.upsample1, subnet.upsample2, subnet.upsample3
    dn0, dn1, dn2 = subnet.downsample0, subnet.downsample1, subnet.downsample2

    # alpha 形状 [1, C, 1, 1]
    a0 = _clamp_alpha(getattr(subnet, 'alpha0', None))
    a1 = _clamp_alpha(getattr(subnet, 'alpha1', None))
    a2 = _clamp_alpha(getattr(subnet, 'alpha2', None))
    a3 = _clamp_alpha(getattr(subnet, 'alpha3', None))

    # 反演时避免 Dropout 干扰，临时切 eval；默认禁用梯度。
    ctx_eval = _temporary_eval([M0, M1, M2, M3])
    ctx_grad = torch.no_grad() if no_grad else contextlib.nullcontext()

    with ctx_eval, ctx_grad:
        # f3: 使用 f2_out
        t3_in = dn2(f2_o)
        if detach_mamba_inputs:
            t3_in = t3_in.detach()
        m3 = M3(t3_in)
        f3 = (f3_o - m3) / a3

        # f2: 使用 f3 与 f1_out
        t2_in = up3(f3) + dn1(f1_o)
        if detach_mamba_inputs:
            t2_in = t2_in.detach()
        m2 = M2(t2_in)
        f2 = (f2_o - m2) / a2

        # f1: 使用 f2 与 f0_out
        t1_in = up2(f2) + dn0(f0_o)
        if detach_mamba_inputs:
            t1_in = t1_in.detach()
        m1 = M1(t1_in)
        f1 = (f1_o - m1) / a1

        # f0: 使用 f1 与 x_emb
        t0_in = up1(f1) + x_emb
        if detach_mamba_inputs:
            t0_in = t0_in.detach()
        m0 = M0(t0_in)
        f0 = (f0_o - m0) / a0

    return [x_emb, f0, f1, f2, f3]


__all__ = ["reverse_subnet"]


class ReverseFunction(torch.nn.Module):
    """
    面向 SubNet 的“逆/正”双向映射封装：
    - forward(tokens_out)  : 近似逆向（从 SubNet 输出估计进入前的 tokens）
    - backward(tokens_in)  : 正向重建（用 SubNet 的公式从输入 tokens 重建输出）

    注意：这里的 backward 不是 PyTorch autograd 的 backward，而是“正向公式”的实现，
    方便你像另一个工程那样通过类接口调用。
    """
    def __init__(self, subnet: Any):
        super().__init__()
        self.subnet = subnet

    def forward(
        self,
        tokens_out: List[torch.Tensor],
        no_grad: bool = True,
        detach_mamba_inputs: bool = False,
        use_eval: bool = True,
    ) -> List[torch.Tensor]:
        """
        近似逆向：由 SubNet 输出估计进入 SubNet 之前的 tokens。
        返回 [x_emb, f0, f1, f2, f3]
        """
        subnet = self.subnet
        assert isinstance(tokens_out, (list, tuple)) and len(tokens_out) == 5
        x_emb, f0_o, f1_o, f2_o, f3_o = tokens_out

        M0, M1, M2, M3 = subnet.mamba_blocks[0], subnet.mamba_blocks[1], subnet.mamba_blocks[2], subnet.mamba_blocks[3]
        up1, up2, up3 = subnet.upsample1, subnet.upsample2, subnet.upsample3
        dn0, dn1, dn2 = subnet.downsample0, subnet.downsample1, subnet.downsample2

        a0 = _clamp_alpha(getattr(subnet, 'alpha0', None))
        a1 = _clamp_alpha(getattr(subnet, 'alpha1', None))
        a2 = _clamp_alpha(getattr(subnet, 'alpha2', None))
        a3 = _clamp_alpha(getattr(subnet, 'alpha3', None))

        mods = [M0, M1, M2, M3] if use_eval else []
        ctx_eval = _temporary_eval(mods) if use_eval else contextlib.nullcontext()
        ctx_grad = torch.no_grad() if no_grad else contextlib.nullcontext()

        with ctx_eval, ctx_grad:
            t3_in = dn2(f2_o)
            if detach_mamba_inputs:
                t3_in = t3_in.detach()
            m3 = M3(t3_in)
            f3 = (f3_o - m3) / a3

            t2_in = up3(f3) + dn1(f1_o)
            if detach_mamba_inputs:
                t2_in = t2_in.detach()
            m2 = M2(t2_in)
            f2 = (f2_o - m2) / a2

            t1_in = up2(f2) + dn0(f0_o)
            if detach_mamba_inputs:
                t1_in = t1_in.detach()
            m1 = M1(t1_in)
            f1 = (f1_o - m1) / a1

            t0_in = up1(f1) + x_emb
            if detach_mamba_inputs:
                t0_in = t0_in.detach()
            m0 = M0(t0_in)
            f0 = (f0_o - m0) / a0

        return [x_emb, f0, f1, f2, f3]

    def backward(
        self,
        tokens_in: List[torch.Tensor],
        no_grad: bool = True,
        detach_mamba_inputs: bool = False,
        use_eval: bool = True,
    ) -> List[torch.Tensor]:
        """
        正向重建：根据 SubNet 正向公式，由 [x_emb, f0, f1, f2, f3] 重建 [x_emb, f0_out, f1_out, f2_out, f3_out]
        """
        subnet = self.subnet
        assert isinstance(tokens_in, (list, tuple)) and len(tokens_in) == 5
        x_emb, f0, f1, f2, f3 = tokens_in

        M0, M1, M2, M3 = subnet.mamba_blocks[0], subnet.mamba_blocks[1], subnet.mamba_blocks[2], subnet.mamba_blocks[3]
        up1, up2, up3 = subnet.upsample1, subnet.upsample2, subnet.upsample3
        dn0, dn1, dn2 = subnet.downsample0, subnet.downsample1, subnet.downsample2

        a0 = _clamp_alpha(getattr(subnet, 'alpha0', None))
        a1 = _clamp_alpha(getattr(subnet, 'alpha1', None))
        a2 = _clamp_alpha(getattr(subnet, 'alpha2', None))
        a3 = _clamp_alpha(getattr(subnet, 'alpha3', None))

        mods = [M0, M1, M2, M3] if use_eval else []
        ctx_eval = _temporary_eval(mods) if use_eval else contextlib.nullcontext()
        ctx_grad = torch.no_grad() if no_grad else contextlib.nullcontext()

        with ctx_eval, ctx_grad:
            y0_in = up1(f1) + x_emb
            if detach_mamba_inputs:
                y0_in = y0_in.detach()
            y0 = f0 * a0 + M0(y0_in)

            y1_in = up2(f2) + dn0(y0)
            if detach_mamba_inputs:
                y1_in = y1_in.detach()
            y1 = f1 * a1 + M1(y1_in)

            y2_in = up3(f3) + dn1(y1)
            if detach_mamba_inputs:
                y2_in = y2_in.detach()
            y2 = f2 * a2 + M2(y2_in)

            y3_in = dn2(y2)
            if detach_mamba_inputs:
                y3_in = y3_in.detach()
            y3 = f3 * a3 + M3(y3_in)

        return [x_emb, y0, y1, y2, y3]


__all__.extend(["ReverseFunction"])
