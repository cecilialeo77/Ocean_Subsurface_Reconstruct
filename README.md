> 本仓库 FRONT 新实验已接入：锋面引导、随深度调整的各向异性 FRONT。
> 请先阅读 [FRONT各向异性预训练微调说明](docs/FRONT各向异性预训练微调.md)，使用
> `configs/iprc_pretrain_front.yaml` 和 `configs/glorys_finetune_grouped_lr_front.yaml`。
> 以下旧说明保留作固定 FRONT 参考；其中旧仓库路径不适用于本次实验。

# IPRC-Argo 预训练 → GLORYS 微调：Geo-Swin + MoE 当月海表重建

本轮分组学习率实验使用 `configs/glorys_finetune_grouped_lr.yaml`：编码部分 `1e-5`、解码/重建头/MoE `5e-5`，独立输出至 `outputs/glorys_finetune_currentmonth_grouped_lr`。从已有IPRC best.pt开始，不需要重新预训练。完整微调、测试及续训命令见 [分组学习率运行说明](docs/分组学习率微调说明.md)。原 `glorys_finetune.yaml` 保留为统一学习率 `1e-4` 对照，本文下方旧命令仍对应该对照。

本仓库位于 `Z:\Swin-transformer-GLORYS-loss-moe-history-pretrain`，独立于原始 GLORYS 实验目录。两个阶段使用相同的 11 通道输入、27 层输出、Geo-Swin、连续重建头、残差 MoE 和三项损失。第一阶段采用 IPRC 位温背景场，第二阶段迁移全部可学习权重，用 GLORYS 更新整个网络。

本版两个阶段均只读取目标月海表：SST、SSS、SLA、ussw、vssw，加3个地理通道、1个海洋掩膜和2个月份编码，共11通道。不拼接前五个月，输出仍为目标月27层位温。

本实现不会重新处理标签，不会修改共享标准化文件，也不会通过历史次表层标签构造输入。完整结构和代码解释见 [docs/中文版代码仓库解读.md](docs/中文版代码仓库解读.md)。

## 1. 数据、架构和实验约定

| 项目 | 设置 |
|---|---|
| 海表输入 | `Z:/surface/surface_input/input_8ch_1993_2024.nc` |
| IPRC标签 | `Z:/subsurface/IPRC_Argo_processed/combined_nc/iprc_argo_temperature_standardized_1993_2024_27layers_0.25deg.nc` |
| GLORYS标签 | `Z:/subsurface/GLORYS_processed/GLORYS_step2_standardized/combined_nc/glorys_temperature_standardized_1993_2024_27layers_0.25deg.nc` |
| 标签共享标准化 | `Z:/subsurface/GLORYS_processed/shared_label_norm_stats/shared_temperature_label_norm_train_only.json` |
| 海表标准化 | `Z:/surface/surface_input/surface_norm_2005_2018.json` |
| 水深掩膜 | `Z:/surface/quarter_degree/depth_valid_ocean_masks.nc` |
| 模型输入/输出 | `[B,11,512,1440]` → `[B,27,512,1440]` |
| 温度定义 | IPRC 使用 PTEMP；GLORYS 使用 thetao，均为位温 |
| 总损失 | `L_value + λ_vertical L_vertical + λ_transport L_transport` |

各配置中已有的垂向梯度和transport权重保持原值，以当前YAML为准。

两份标签已经使用同一组 GLORYS 2005—2018 年逐深度均值和标准差进行标准化。训练直接读取标准化标签，不再二次标准化；垂向损失与导出结果按共享参数恢复物理摄氏度。

| 阶段 | 配置文件 | 训练月份 | 验证月份 | 测试月份 |
|---|---|---|---|---|
| IPRC预训练 | `configs/iprc_pretrain.yaml` | 2005-01—2018-12，168月 | 2019-01—2020-04，16月 | 禁用 |
| GLORYS微调 | `configs/glorys_finetune.yaml` | 2005-01—2018-12，168月 | 2019-01—2020-12，24月 | 2021-01—2024-12，48月 |
| GLORYS从头基线 | `configs/glorys_baseline.yaml` | 同微调 | 同微调 | 同微调 |

IPRC文件虽然有384个月的时间坐标，但只有2005-01—2020-04共184个月有标签。其他月份是 `temperature=0, valid_label_mask=0, label_available=0` 的占位。本仓库会拒绝采样不可用月份，并明确禁用IPRC测试，不能把占位0当作真实温度。

## 2. 环境与代码来源

PowerShell中进入新仓库，激活现有PyTorch环境。优先使用与GPU/CUDA匹配的现有PyTorch，不要为了本项目盲目替换CUDA版本。

```powershell
Set-Location 'Z:\Swin-transformer-GLORYS-loss-moe-history-pretrain'
$env:PYTHONPATH = (Resolve-Path '.\src').ProviderPath
$env:PYTHONUTF8 = '1'
python -c "import sys,torch; import glorys_swin_reconstruction.train as t; print(sys.executable); print(t.__file__); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
```

代码来源必须指向本目录。包名沿用原仓库，因此每个新终端都应设置本目录的 `PYTHONPATH`，避免误导入旧项目。若缺少其他依赖：

```powershell
python -m pip install numpy scipy xarray netCDF4 pyyaml pytest
```

也提供 `pyproject.toml`，可在已准备好PyTorch的环境中执行 `python -m pip install -e '.[test]'`。无需安装xESMF或重新插值IPRC。

## 3. 正式训练前检查

```powershell
$pre = '.\configs\iprc_pretrain.yaml'
$fine = '.\configs\glorys_finetune.yaml'
$preOut = '.\outputs\iprc_pretrain_currentmonth'
$fineOut = '.\outputs\glorys_finetune_currentmonth_lr1e4'

python -m glorys_swin_reconstruction.inspect_data --config $pre
python -m glorys_swin_reconstruction.inspect_data --config $fine
python -m pytest -q
```

检查应显示预训练168/16月且test禁用；微调168/24/48月；首个训练样本仅使用2005-01当月海表，不需要2004年历史月份。输入11通道、输出27层。检查同时校验所有目标月份的当月输入和标签可用性，并验证标签声明的标准化SHA256与当前JSON一致。

自动化测试使用小网格临时数据，覆盖真实训练CLI、两阶段权重迁移、MoE先验重算、优化器重置、同阶段续训、最终重建和离线指标复算，不会训练正式全球模型。


## 5. 第一阶段：IPRC预训练

```powershell
python -m glorys_swin_reconstruction.train --config $pre
```

默认200轮、AdamW、学习率1e-4，全部模块参与训练。按IPRC验证集 `macro_nrmse_depth` 最小保存 `best.pt`，每轮保存 `last.pt`，每5轮及最后一轮保存编号checkpoint。两阶段均保留三项原始损失，不自动调权、不自动早停。

预训练验证重建：

```powershell
python -m glorys_swin_reconstruction.reconstruct --config $pre --checkpoint "$preOut\best.pt" --split val
```

不要对预训练配置运行 `--split test`，程序会明确拒绝。

## 6. 第二阶段：GLORYS微调

```powershell
python -m glorys_swin_reconstruction.train --config $fine --init-checkpoint "$preOut\best.pt"
```

`--init-checkpoint` 是迁移初始化，不是断点续训。程序执行以下顺序：

1. 检查源checkpoint阶段为IPRC预训练，目标为GLORYS微调；检查模型配置完全一致。
2. 检查当月输入契约、输入通道、深度、共享标签标准化、海表标准化以及输入/掩膜路径契约。
3. 严格加载全部模型权重，保留编码器、解码器、重建头、路由器、四类专家和残差缩放参数。
4. 只用GLORYS训练月份重算并覆盖MoE的气候态梯度和等温层深度；标准化参数仍共享。
5. 新建AdamW、调度器及AMP scaler，从epoch=1开始，重置最佳分数和训练历史。

微调默认全网络更新，本轮初始学习率1e-4、最低1e-6，其余架构和损失不变。本轮从同一个已有IPRC best.pt独立初始化，不需要重新预训练；完整命令见[同一IPRC权重微调说明](docs/同一IPRC权重微调_lr1e4.md)。可在YAML修改总轮数和学习率，但不要用最终测试集反复调参。输出 `initialization.json` 和checkpoint均记录迁移来源与MoE统计来源。

## 7. 可选：一条命令完成两阶段

```powershell
python -m glorys_swin_reconstruction.two_stage --pretrain-config $pre --finetune-config $fine
```

默认只训练，不自动测试。若实验方案已确定，需要训练结束后立即运行最终测试：

```powershell
python -m glorys_swin_reconstruction.two_stage --pretrain-config $pre --finetune-config $fine --run-test
```

该入口不自动寻找旧checkpoint、不自动恢复中断，也不会覆盖已有训练。若中断，请按照下一节分别续训。若已独立完成预训练，直接执行第6节，避免重复预训练。

## 8. 同阶段断点续训

```powershell
python -m glorys_swin_reconstruction.train --config $pre --resume "$preOut\last.pt"
python -m glorys_swin_reconstruction.train --config $fine --resume "$fineOut\last.pt"
```

延长微调：

```powershell
Set-Location 'Z:\Swin-transformer-GLORYS-loss-moe-history-pretrain-Front'

$env:PYTHONPATH = "$PWD\src"

python -c "import glorys_swin_reconstruction; print(glorys_swin_reconstruction.__file__)"

python -m glorys_swin_reconstruction.train --config '.\configs\glorys_finetune_grouped_lr_front.yaml' --resume '.\outputs\glorys_finetune_grouped_lr_front_anisotropic\last.pt' --epochs 300
```

`--epochs`是总轮数，不是新增轮数。

已有checkpoint目录禁止从头覆盖；新实验请更换YAML输出目录或使用 `--output-dir`。训练参数以原实验为准，续训改变学习率配置并不会自动覆盖checkpoint中恢复的优化器学习率。

## 9. 微调后的完整验证与最终测试

```powershell
python -m glorys_swin_reconstruction.reconstruct --config $fine --checkpoint "$fineOut\best.pt" --split val
python -m glorys_swin_reconstruction.reconstruct --config $fine --checkpoint "$fineOut\best.pt" --split test
```

验证覆盖24个月，最终测试覆盖48个月。输出包括：

- `reconstruction_val.nc` / `reconstruction_test.nc`：反标准化后的摄氏度位温场；静态海洋域之外为NaN。
- `metrics_val.json` / `metrics_test.json`：完整期汇总及逐深度指标。
- `metrics_*_monthly.json`：逐月及逐月逐深度记录和覆盖检查。
- `metrics_*_monthly_overall.csv`、`metrics_*_monthly_depth.csv`：便于分析的表格。

主指标沿用现有定义：在标签掩膜、静态深度掩膜和有限标签交集上评估；逐深度NRMSE使用训练期标准差；不新增面积或层厚加权。不得将预览当作完整测试。

查看测试摘要：

```powershell
$m = Get-Content "$fineOut\metrics_test.json" -Raw | ConvertFrom-Json
$m | Select-Object pooled_rmse,mean_depth_rmse,macro_nrmse_depth,month_count,is_complete
```

独立复算已保存结果，不重新运行神经网络：

```powershell
python -m glorys_swin_reconstruction.evaluate_reconstruction --config $fine --reconstruction "$fineOut\reconstruction_test.nc" --split test
```

输出 `metrics_test_reevaluated.json`，与已有汇总进行一致性检查；保留原 `metrics_test.json`。训练、重建和离线评估使用同一标签可用性检查。

## 10. GLORYS从头训练对照

```powershell
python -m glorys_swin_reconstruction.train --config '.\configs\glorys_baseline.yaml'
python -m glorys_swin_reconstruction.reconstruct --config '.\configs\glorys_baseline.yaml' --checkpoint '.\outputs\glorys_baseline_currentmonth\best.pt' --split test
```

该实验不加载IPRC权重。比较时同时考虑总训练步数和计算预算，并建议重复随机种子。IPRC插值没有增加独立0.25°信息，改善是否来自背景场预训练必须由对照判断。最终面向GLORYS的误差改善不等于已证明对独立真实观测更准确。