# FRONT 各向异性预训练—微调实验
## 结构

- 双路由、Mixed/Heave/Front/Deep 四专家、基础重建头和原三项损失保持原定义。
- 固定 3×3 FRONT 保留；在其隐藏特征上增加锋面引导各向异性修正。
- SST 在低分辨率网格上进行有效性加权池化，中心差分估计法向和切向。
  标准化 SST 保持梯度方向；低置信度、缺测和纯陆地位置抑制新增修正。
- 九点采样：中心、沿锋面 ±r/±2r、跨锋面 ±r/±2r；分别编码沿锋面均值、
  跨锋面对称信息和有符号差分，再与中心特征一起投影。

- 5 个几何场各有 3 个上下文相关系数，以 [1,z/zmax,(z/zmax)^2] 生成随深度平滑
  变化的两半径、旋转角、两中心位移。方向修正上限 30°，中心位移每方向上限 30 km。
  数值为待验证的初始设置，不代表已证明最优。
- 半径初始分别为沿锋面 110 km、跨锋面 40 km；训练范围为各自的 0.5–1.5 倍。
- 几何头零初始化，初始几何深度共享；投影零初始化，初始新增修正为零。
  训练后通过深度系数调整几何。前几步几何梯度可能因零投影/零残差缩放延迟出现。
- 使用向量化双线性 gather，经度循环、纬度边界复制，不使用 grid_sample。
  采样角点按粗网格海洋比例加权，纯陆地采样不贡献特征；没有实现跨陆地路径检查，
  也没有新增逐深度海底遮挡。现有标签/输出掩膜仍按原流程处理。
- FRONT checkpoint 默认开启以减少 27 层分支的训练激活内存。GPU 耗时、显存及
  CUDA 复现性仍需在实际训练环境验证；现有 deterministic 配置主要控制 cuDNN，
  不等于保证所有 CUDA 运算逐位可复现。
- 全分辨率 detail 仍是原固定卷积、深度共享细节图。这一轮消融只检验低分辨率 FRONT。

## 学习率设置思路
- 当前 IPRC 预训练从随机初始化开始，全网络架构都需要共同学习，没有“已预训练的编码器需要小步保护、新模块需要更快适应”的区别，所以采用统一的 1×10⁻⁴ 作为起点。
- GLORYS 微调则不同：此时整个网络已经过 IPRC 预训练。当前分组设置用较小的编码器学习率保留已有表征，用较大的解码器、重建头和 MoE 学习率适应 GLORYS 的目标分布。

## 正式预训练和微调

```
Set-Location 'Z:\Swin-transformer-GLORYS-loss-moe-history-pretrain-Front'
$env:PYTHONPATH = (Resolve-Path '.\src').ProviderPath
$env:PYTHONUTF8 = '1'
python -m glorys_swin_reconstruction.two_stage --pretrain-config configs/iprc_pretrain_front.yaml --finetune-config configs/glorys_finetune_grouped_lr_front.yaml --max-samples 2 --epochs 1
```


## 测试

```
Set-Location 'Z:\Swin-transformer-GLORYS-loss-moe-history-pretrain-Front'
$env:PYTHONPATH = (Resolve-Path '.\src').ProviderPath
$env:PYTHONUTF8 = '1'

python -m glorys_swin_reconstruction.reconstruct --config '.\configs\glorys_finetune_grouped_lr_front.yaml' --checkpoint '.\outputs\glorys_finetune_grouped_lr_front_anisotropic\best.pt' --split test
```

