# 基础消融实验

所有组运行同一 ECHO 训练入口，各组独立训练，不复用其他组模型。默认 O0、seed 0、回放上限 10%、CPU 计算线程 4，原有阶段指标和自动汇总全部保留。

| ablation 名称 | 唯一干预 |
|---|---|
| full | 完整 ECHO，作为配对参照 |
| no_evidence_loss | 物理证据辅助 CE 权重为 0，保留证据编码器和推理融合 |
| no_spectrum | 不执行频谱编码器和谱残差融合 |
| no_spectral_gate | 谱门值固定为 1，保留谱残差 dropout |
| no_spectral_dropout | 谱残差 dropout 为 0 |
| no_projection | 所有新增专家使用空保护基，保留专家与其他约束 |
| no_kd | 旧类蒸馏权重为 0 |
| no_metric_losses | 监督对比与困难样本度量损失权重均为 0 |
| no_consolidation | 跳过最终巩固阶段 |

这里先实现基础单因素实验，不包含随机同秩投影、计算预算匹配巩固、推理融合消融和回放比例网格。no_consolidation 减少训练步数，不能仅凭此组把收益归因于巩固形式。

## GPU 运行

在服务器代码和数据所在目录执行：

```bash
python run_cil_ablation.py \
  --data echo_cil_processed \
  --order O0 \
  --seeds 0 \
  --output-root runs/cil_ablation \
  --device cuda:0 --amp \
  --num-workers 4 --cpu-threads 4 \
  --batch-size 128 --epochs 10 \
  --replay-fraction 0.10
```

默认运行全部 9 组。全部完成后 comparison_summary.csv 为 27 行阶段记录，method 列表示消融组名。每阶段自动更新 CSV/JSON，无需另外合并。

三种子正式运行可设置 `--seeds 0 1 2 --epochs 60` 并使用新输出目录，最终 81 行。`--epochs` 只控制基础/增量主训练，预热与巩固沿用 `echo_cil_config.json`。所有组使用同一配置，消融预设在解析参数后覆盖其对应参数，实际值保存到 configuration.json。

CPU 功能检查（不能用于比较正式准确率）：

```bash
python run_cil_ablation.py --data echo_cil_processed \
  --output-root runs/cil_ablation_smoke \
  --ablations full no_spectrum no_projection no_consolidation \
  --device cpu --no-amp --num-workers 0 --cpu-threads 4 --smoke
```

`--smoke` 将各训练阶段轮数设为 1，并固定原型与证据推理融合权重为 0、取消相关搜索以加速功能检查；no_consolidation 仍为 0。正式实验不要传此参数。`--dry-run` 只列出命令。`--summarize-only` 仅重新整理已有实验。非空、不完整或签名不同的目录不会覆盖，不支持优化器断点续训。

## 核对方式与边界

- configuration.json 保存 ablation、spectrum_mode、projection_enabled 和所有实际权重。检查点 architecture 保存谱模式和度量权重；恢复消融模型时须按 configuration 调用 echo_cil.ablations.configure_model，不能只 load_state_dict 后使用默认谱模式。
- no_projection 的 protection.constructed_rank 记录原构造秩，rank 记录实际使用秩 0；constructed_rank 本来为 0 的阶段不能证明投影贡献。
- no_spectrum 真正跳过频谱计算，但为保留统一初始化与检查点结构，模块参数仍保留在模型中。因此 model_parameters 仍包含未执行模块，不能当作结构剪枝后的参数数目。
- 损失为 0 的消融仍可能计算该损失供日志观察，其原始值非零不代表它参与优化。无蒸馏仍可能需要教师来构造保护基。
- 所有组允许同样的验证选择和推理参数搜索；指标反映各消融模型在同一选择流程下的效果。辅助监督为 0 不等于完全移除物理分支。
- 不改动样本划分、旧样本访问权限或回放上限。按相同 order/seed 配对比较 full 与消融；先看最终 Macro-F1、旧/新类准确率、遗忘，再看实际步数和推理耗时。字段定义见 README_CIL_EXPERIMENTS.md。
