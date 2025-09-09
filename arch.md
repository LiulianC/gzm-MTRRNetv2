# MTRRNetv2 — Token-only 主干前向简图

```
Input
I: (B,3,256,256)
│
├─ MultiScaleTokenEncoder（4尺度 → tokens）
│   ├─ Stage s0: 256/patch4 → 64×64 → t0 (B,4096,96)
│   │   PatchEmbed(low/high) → VSS-Mamba(low) + Swin(high) → AAF
│   ├─ Stage s1: 128/patch4 → 32×32 → t1 (B,1024,96)
│   │   PatchEmbed(low/high) → VSS-Mamba(low) + Swin(high) → AAF
│   ├─ Stage s2:  64/patch4 → 16×16 → t2 (B,256,96)
│   │   PatchEmbed(low/high) → VSS-Mamba(low) + Swin(high) → AAF
│   └─ Stage s3:  32/patch2 → 16×16 → t3 (B,256,96)
│       PatchEmbed(low/high) → VSS-Mamba(low) + Swin(high) → AAF
│
├─ TokenSubNet（tokens → 多尺度特征交互）
│   ├─ tokens → feature maps:
│   │   t0 → f0 (B,96,64,64)
│   │   t1 → f1 (B,96,32,32)
│   │   t2 → f2 (B,96,16,16)
│   │   t3 → f3 (B,96,16,16)
│   └─ 融合细化（α为可学习；Refine=VSS-Mamba）
│       f0 = f0*α0 + Refine0( up(f1) )
│       f1 = f1*α1 + Refine1( up(f2) + down(f0) )
│       f2 = f2*α2 + Refine2( f3 + down(f1) )
│       f3 = f3*α3 + Refine3( f2 )
│
└─ UnifiedTokenDecoder（特征 → 输出6通道）
    ├─ 层间融合（ConvNeXt ×3 每层）
    │   o23 = CN×3( proj(f2 + f3) )           16→ up → 32
    │   o12 = CN×3( proj(f1 + o23) )          32→ up → 64
    │   o01 = CN×3( proj(f0 + o12) )          64
    ├─ 解码上采样
    │   o01 → Deconv 64→128 → Deconv 128→256 → Conv 32 → Conv 6 → delta (B,6,256,256)
    ├─ 残差基线
    │   base = base_scale * concat(I, I)       (B,6,256,256)
    └─ 输出
        out6 = base + delta                    (B,6,256,256)
        fake_T = out6[:, :3]                   (B,3,256,256)
        fake_R = out6[:, 3:]                   (B,3,256,256)
```