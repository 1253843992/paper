# ECHO-CIL 类增量对比实验

新增统一的类增量对照入口，不改变 `train_echocil.py` 的现有 ECHO-CIL 训练逻辑。

## 方法

| CLI 名称 | 方法 | 本实现保留的核心机制 | 推理分类器 |
|---|---|---|---|
| `finetune` | 顺序微调基线 | 只使用当前阶段数据，无回放、无蒸馏 | 线性头 |
| `er` | Experience Replay | 当前数据与小样本回放联合交叉熵 | 线性头 |
| `icarl` | iCaRL | herding、旧类 sigmoid 蒸馏、exemplar class mean | NME（同时输出线性头结果） |
| `bic` | BiC | 回放、旧类蒸馏、独立平衡校准子集、两参数新类偏置校正 | 校正后的线性头 |
| `lucir` | LUCIR | cosine classifier、less-forget、inter-class margin ranking、分类头平衡微调 | cosine 头 |
| `wa` | Weight Alignment | 回放、旧类蒸馏、阶段后新类权重范数对齐 | 无偏置线性头 |
| `podnet` | PODNet | 多 proxy cosine 头、NCA、全局特征和 1-D 中间特征 POD、平衡微调 | 多 proxy 头（另输出 NME） |
| `derpp` | DER++ | 当前数据 CE、两次独立旧样本抽取、历史 logits MSE 和旧样本 CE | 线性头 |
| `der_expand` | DER（可扩展表征） | 每阶段增加编码器、冻结旧编码器、辅助 old/new 头、权重对齐 | 拼接特征线性头 |
| `foster` | FOSTER | 特征扩展、类平衡 boosting、旧类蒸馏、权重对齐、知识压缩回单编码器 | 压缩学生线性头 |
| `echo` | 当前方法 | 原 `train_echocil.py` 完整流程 | 原 ECHO-CIL 推理流程 |

这些实现把论文机制适配到 2048 点一维振动信号。它们是同一数据、同一回放上限下的工程对照，不宣称逐参数复现论文图像实验。`derpp` 指 Dark Experience Replay++，`der_expand` 指 Dynamically Expandable Representation，两者不是同一个方法。

## 公平性约束

- 默认各对照使用同一 ECHO 编码器，但不使用保护专家、物理辅助分类损失、原型融合或 ECHO 专有损失。
- 可用 `--backbone cnn1d` 做更中性的纯一维 CNN 对比；不要把不同 backbone 的结果放进同一主表。
- 回放缓冲区最多为累计已见训练窗的 10%，还受 `--memory-bytes` 的字节上限约束。
- iCaRL/LUCIR/PODNet 用 herding；ER/DER++ 用确定性样本选择。旧类只能从前一阶段缓冲区继续缩减，不重新读取旧类训练集。
- BiC 的校准集从当前阶段允许使用的“旧回放 + 当前新类训练集”中按类等量划出，与主训练集不重叠。
- DER++ 的历史 logits 在样本第一次进入训练、该次参数更新之前保存，并随样本一起计入持久化缓冲区字节数。
- 默认每个 epoch 在全部已见类的验证集上选择最佳模型，与当前 ECHO-CIL 的验证权限一致。若要避免逐 epoch 历史验证访问，所有对照统一使用 `--selection fixed`，同时从主表排除原 ECHO 结果或另行修改其协议。
- 测试集只在阶段训练、校准、压缩和模型选择全部结束后评估；不参与超参数选择。
- `results.json` 记录实际回放数、实际回放比例、校准数、优化步数、参数量、峰值扩展参数量、持久化字节数和 CUDA 峰值显存。

## 单个方法

服务器 GPU 示例：

```bash
python train_cil_baselines.py \
  --method er \
  --data echo_cil_processed \
  --output runs/cil_comparison/er/O0/seed_0 \
  --order O0 \
  --device cuda:0 \
  --amp \
  --num-workers 4 \
  --batch-size 128 \
  --epochs 60 \
  --replay-fraction 0.10
```

将 `--method` 改为 `finetune`、`icarl`、`bic`、`lucir`、`wa`、`podnet`、`derpp`、`der_expand` 或 `foster` 即可。建议每次使用新的输出目录；程序不会覆盖非空目录。

CPU 示例：

```bash
python train_cil_baselines.py \
  --method icarl \
  --data echo_cil_processed \
  --output runs/cil_comparison/icarl/O0/seed_0 \
  --device cpu \
  --no-amp \
  --num-workers 0 \
  --cpu-threads 8
```

## 批量主实验

先跑 O0、3 个随机种子的完整方法表：

```bash
python run_cil_comparison.py \
  --data echo_cil_processed \
  --orders O0 \
  --seeds 0 1 2 \
  --methods finetune er icarl bic lucir wa podnet derpp der_expand foster echo \
  --output-root runs/cil_comparison \
  --device cuda:0 \
  --amp \
  --num-workers 4 \
  --batch-size 128 \
  --epochs 60 \
  --replay-fraction 0.10
```

多类别顺序实验需要每个顺序各自预处理，因为归一化只允许使用该顺序的基础类：

```bash
python run_cil_comparison.py \
  --input CWRU轴承数据 \
  --processed-root echo_cil_processed_suite \
  --orders O0 O1 O2 O3 O4 \
  --seeds 0 1 2 \
  --methods finetune er icarl bic lucir wa podnet derpp der_expand foster echo \
  --output-root runs/cil_comparison \
  --device cuda:0 --amp --num-workers 4
```

运行器只有在 `results.json` 明确含 3 个完整阶段，且命令与输入/源码哈希完全一致时才跳过任务；不一致或中断目录不会被自动覆盖。汇总文件为：

- `runs/cil_comparison/comparison_summary.csv`：每个 order/seed 的原始结果；
- `runs/cil_comparison/comparison_summary.json`：先在相同 order 内对随机种子求均值和标准差，不把类别顺序当随机种子混合。

只重新生成汇总：

```bash
python run_cil_comparison.py --output-root runs/cil_comparison --summarize-only
```

## 输出解释

每个方法目录包含：

- `results.json`：阶段指标、旧/新类准确率、遗忘和资源开销；
- `predictions_stage_*.csv`：测试样本级真实类与预测类；
- `stage_*.pt`：可恢复的阶段模型状态与模型结构元数据；
- `memory_stage_*.npz`：实际持久化的回放样本，DER++ 额外含历史 logits/mask；
- `configuration.json`：完整命令配置和实现源码哈希；
- `suite_receipt.json`：批量运行器的命令、依赖哈希与恢复校验信息。

主表至少报告最终 macro、old、new、Average Incremental Accuracy、Forgetting、回放样本数/字节、最终参数量和峰值扩展参数量。DER-expand 与 FOSTER 的峰值模型更大，不能只比较最终准确率。

## 建议实验顺序

先用 O0/seed0 检查全流程；再跑 O0 的 3 个种子；确认超参数后冻结设置，最后运行 5 个类别顺序。正式比较时不要根据测试结果单独调整某一种方法。`finetune` 是新增的无抗遗忘下界基线；ER 是最重要的同预算回放基线。
