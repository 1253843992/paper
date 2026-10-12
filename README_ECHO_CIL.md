# ECHO-CIL 算法实现

该目录在原仓库 `cwru_preprocess.py` 的基础上，增加了可直接训练的 Python 类增量学习实现。模型输入不是重新读取 MATLAB 文件，而是使用 `echo_cil_processed` 中的固定 shard，保证训练协议与数据处理完全一致。

## 方法结构

1. **时域分支**：一维卷积 + 三个条件膨胀残差块，膨胀率为 `[1, 3, 7]`，提取冲击瞬态和多尺度周期结构。
2. **物理分支**：对三个频带的 Hilbert 包络阶次谱使用卷积编码，同时编码 9 维质量特征。
3. **门控融合**：时域特征产生物理分支注入系数，形成 128 维融合表示。
4. **低秩专家**：每个增量阶段新增一个 rank=8 的低秩专家；专家由独立 sigmoid 门控，降低新增类别对旧表示的扰动。
5. **分类头扩展**：新阶段只增加新类行，旧类行在每轮更新后恢复冻结。
6. **回放与保护**：固定 32 MB 类均衡回放；对旧样本特征做 SVD，保留前 16 个判别子空间方向，并对增量训练中的表示漂移加保护损失。
7. **蒸馏**：旧模型 logits 使用温度 2 的 KL 蒸馏，抑制旧类遗忘。
8. **验证集自动校准**：最终阶段同时训练类别平衡与普通逻辑回归头，并比较多组正则强度；只依据验证集的旧类/新类下界、宏平均和最弱类别准确率选择最终头。测试集不参与选择，最终推理仍只有一个线性分类头。

## 运行

在 `CWRU轴承数据` 目录执行：

```bash
pip install -r requirements_echo_cil.txt
python train_echocil.py --data echo_cil_processed --epochs 10 --batch-size 64
```

设备选择：

```bash
# 自动选择：优先 CUDA，其次 MPS，最后 CPU
python train_echocil.py --device auto

# 强制使用第一张 NVIDIA GPU，并开启混合精度（默认）
python train_echocil.py --device cuda:0 --amp

# 强制使用 CPU
python train_echocil.py --device cpu --no-amp
```

启动时会打印实际使用的设备。若指定 `--device cuda` 但当前 PyTorch
不是 CUDA 版本、NVIDIA 驱动不可用或没有可见 GPU，程序会立即给出明确错误，
不会悄悄退回 CPU。可用下面的命令检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

输出目录默认为 `runs/echo_cil`，包含 `stage_0.pt`、`stage_1.pt`、`stage_2.pt` 和 `results.json`。
`results.json` 的最终阶段还会记录 `calibration` 字段，包括自动选中的校准方式、正则强度以及对应验证集指标。混合精度仅用于训练，最终校准和评估固定使用 FP32，减少临界样本的数值波动。

需要关闭最终校准时可传入 `--calibration-weight-decay -1`；默认会自动比较
`1e-5、1e-4、1e-3`，通常无需手工调整。若显存足够但更看重准确率，建议保持
`--batch-size 64`；128 的大批次更快，但本数据集上可能略微降低泛化精度。

## 重要实验约束

- 阶段 0 只使用 `[0,1,2,3]`；阶段 1 只看到新增 `[4,5,6]` 和回放样本；阶段 2 只看到新增 `[7,8,9]` 和回放样本。
- 验证集始终使用当前阶段全部已见类。
- 不得在训练脚本中重新随机切窗或重新计算归一化统计量。
- `record_inventory.csv` 中的 `sha256`、`preprocess_config.json` 和 `normalization.json` 应作为实验记录的一部分保存。
