import torch
import torch.nn as nn
from contextlib import contextmanager

def _collect_and_zero_probs(module, store):
    """
    记录并置零所有可能的丢弃概率字段：
    - nn.Dropout{,2d,3d}, AlphaDropout: .p
    - timm.stochastic_depth.DropPath: .drop_prob
    - 自定义模块可能仍叫 DropPath/Dropout：优先按属性名兜底
    """
    # 标准 Dropout 家族
    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        store.append((module, 'p', module.p))
        module.p = 0.0

    # timm 的 DropPath（或自定义里常见的字段名）
    if hasattr(module, 'drop_prob'):
        store.append((module, 'drop_prob', float(module.drop_prob)))
        module.drop_prob = 0.0

    # 某些自定义会把注意力里的 attn_drop/proj_drop/… 也做成 nn.Dropout
    # 已在上面 isinstance 覆盖；这里再兜底一下“字段里塞 Dropout 子模块”的常见命名
    for attr in ('attn_drop', 'proj_drop', 'drop', 'drop1', 'drop2', 'drop_path',
                 'drop_path1', 'drop_path2'):
        if hasattr(module, attr):
            sub = getattr(module, attr)
            # 子模块本身是 Dropout/DropPath
            if isinstance(sub, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                store.append((sub, 'p', sub.p))
                sub.p = 0.0
            elif hasattr(sub, 'drop_prob'):
                store.append((sub, 'drop_prob', float(sub.drop_prob)))
                sub.drop_prob = 0.0

@contextmanager
def eval_no_dropout(model):
    """
    用法：
        with eval_no_dropout(model):
            validate(...)
    效果：
      - model.eval()
      - 关闭所有 Dropout / DropPath / attn/proj_drop
      - 退出时恢复原值，并把模型状态切回 train()
    """
    # 进入：切 eval 并置零
    was_training = model.training
    model.eval()

    store = []  # 记录被修改的 (module, attr, old_value)
    for m in model.modules():
        _collect_and_zero_probs(m, store)

    try:
        yield
    finally:
        # 恢复
        for m, attr, old in store:
            setattr(m, attr, old)
        if was_training:
            model.train()
