# ECHO-CIL 类增量对比实验运行与指标说明

基础消融实验使用 `run_cil_ablation.py`，分组、运行命令和核对方式见 [README_CIL_ABLATION.md](README_CIL_ABLATION.md)。

更新日期：2026-10-15。本文对应当前 `run_cil_comparison.py`、`train_cil_baselines.py`、`train_echocil.py` 和 `echo_cil/metrics.py`，覆盖目前 comparison_summary.csv 的全部 54 个字段。

## 1. 直接运行：训练过程中自动生成汇总

在服务器项目目录执行。根据当前服务器路径，工作目录为 `~/zsj`，其中应包含训练脚本和 `echo_cil_processed`。

### 1.1 当前使用的 GPU 命令：O0、seed 0、主训练 10 轮

```bash
cd ~/zsj

python run_cil_comparison.py \
  --data echo_cil_processed \
  --orders O0 \
  --seeds 0 \
  --methods finetune er icarl bic lucir wa podnet derpp der_expand foster echo \
  --output-root runs/cil_comparison_retry \
  --device cuda:0 \
  --amp \
  --num-workers 4 \
  --batch-size 128 \
  --epochs 10
```

以上命令串行运行 11 种方法，每种方法完成 3 个阶段。每完成一个阶段，就自动更新：

```text
runs/cil_comparison_retry/comparison_summary.csv
runs/cil_comparison_retry/comparison_summary.json
```

不需要再运行合并或汇总命令。全部完成时 CSV 有 33 行数据，另加 1 行表头。训练过程中终端会打印 `SUMMARY updated`、汇总文件绝对路径和当前行数。每种方法结束后刷新 `run_complete`。

如果该目录已含完整实验，只有命令和记录的源码/输入哈希一致才会跳过；中断实验或配置不一致会报错，不会覆盖。需要重新训练时，将输出目录改成一个新的名称，例如 `runs/cil_comparison_repeat`。当前不支持从中断的优化器状态续训。

### 1.2 三个随机种子：O0、主训练 60 轮

```bash
python run_cil_comparison.py \
  --data echo_cil_processed \
  --orders O0 \
  --seeds 0 1 2 \
  --methods finetune er icarl bic lucir wa podnet derpp der_expand foster echo \
  --output-root runs/cil_comparison_o0 \
  --device cuda:0 \
  --amp \
  --num-workers 4 \
  --batch-size 128 \
  --epochs 60 \
  --replay-fraction 0.10
```

共 33 次完整训练，CSV 完成后为 99 行阶段记录。此处的 60 轮是主训练设置，不是所有方法的总训练预算，详见第 3 节。

### 1.3 五种类别顺序、三个随机种子

不同顺序的基础类不同，必须分别预处理归一化，不能让五种顺序共用 O0 的归一化文件：

```bash
python run_cil_comparison.py \
  --input CWRU轴承数据 \
  --processed-root echo_cil_processed_suite \
  --orders O0 O1 O2 O3 O4 \
  --seeds 0 1 2 \
  --methods finetune er icarl bic lucir wa podnet derpp der_expand foster echo \
  --output-root runs/cil_comparison_all_orders \
  --device cuda:0 \
  --amp \
  --num-workers 4 \
  --batch-size 128 \
  --epochs 60 \
  --replay-fraction 0.10
```

缺少预处理数据时自动调用预处理脚本。已有目录不会自动重建，应保证它对应指定类别顺序。完整实验为 165 次训练、495 行阶段记录。`--input` 路径按服务器原始数据位置调整。

### 1.4 CPU 批量运行

```bash
python run_cil_comparison.py \
  --data echo_cil_processed \
  --orders O0 \
  --seeds 0 \
  --methods finetune er icarl bic lucir wa podnet derpp der_expand foster echo \
  --output-root runs/cil_comparison_cpu \
  --device cpu \
  --no-amp \
  --num-workers 0 \
  --batch-size 128 \
  --epochs 10
```

### 1.5 只运行一个方法，仍自动生成统一汇总

使用批量入口并只传一个方法即可：

```bash
python run_cil_comparison.py \
  --data echo_cil_processed \
  --orders O0 --seeds 0 \
  --methods er \
  --output-root runs/cil_er \
  --device cuda:0 --amp --num-workers 4 \
  --batch-size 128 --epochs 10
```

`--methods echo` 可以单独运行当前 ECHO-CIL。直接调用 `train_cil_baselines.py` 或 `train_echocil.py` 只写各自实验结果，不负责根目录的自动对比汇总。

### 1.6 辅助命令

仅检查将执行的命令，不训练：

```bash
python run_cil_comparison.py --data echo_cil_processed --orders O0 --seeds 0 --dry-run
```

仅重新汇总已有结果，不训练（可选，正常训练无需执行）：

```bash
python run_cil_comparison.py \
  --output-root runs/cil_comparison_retry \
  --summarize-only
```

该操作读取输出根目录下全部匹配 `方法/顺序/seed_*/results.json` 的实验，包括已经结束的部分阶段。旧结果没有测量过的延迟不会被补造。

## 2. 方法名称与实现范围

| CLI 名称 | 方法 | 当前保留的主要机制 | 汇总主指标对应的推理方式 |
|---|---|---|---|
| `finetune` | 顺序微调基线 | 当前新类训练，无回放、无蒸馏 | 线性分类头 |
| `er` | Experience Replay | 新类数据与旧类回放联合交叉熵 | 线性分类头 |
| `icarl` | iCaRL | herding、旧类 sigmoid 蒸馏、样本均值 | NME 最近类别均值，额外保存分类头结果 |
| `bic` | BiC | 回放、蒸馏、独立平衡校准集、两参数偏置校正 | 校正后的分类头 |
| `lucir` | LUCIR | cosine 头、特征保持、margin ranking、平衡微调 | cosine 分类头 |
| `wa` | Weight Alignment | 回放、蒸馏、阶段后权重范数对齐 | 无偏置线性头 |
| `podnet` | PODNet | 多 proxy、NCA、全局和一维中间特征蒸馏、平衡微调 | 多 proxy 头，额外保存 NME 结果 |
| `derpp` | Dark Experience Replay++ | 当前样本 CE、历史 logits MSE、独立抽样的回放 CE | 线性分类头 |
| `der_expand` | Dynamically Expandable Representation | 扩展编码器、冻结旧编码器、辅助头、权重对齐 | 拼接特征分类头 |
| `foster` | FOSTER | 特征扩展、boosting、蒸馏、权重对齐、压缩 | 压缩后的单编码器学生 |
| `echo` | ECHO-CIL | 当前完整方法及其验证选择、推理融合 | 实际启用的融合推理 |

这些是针对 2048 点一维振动信号的适配实现，不是原论文图像实验的逐参数复现。默认对照方法复用 ECHO 编码器结构，但不加入 ECHO 的保护专家、物理辅助分类损失、原型融合及专有训练损失，也不使用其谱残差 dropout。

ER/DER++ 使用确定性的按类选样；其他有回放的对照采用 herding。DER++ 在首次呈现当前阶段样本、该次参数更新前记录 logits，阶段末选择保留样本；这是阶段式有限回放适配，不是原始在线 reservoir 流程。DER-expand 当前未实现论文的剪枝流程。论文中应披露这些适配边界。

`--backbone cnn1d` 可用于纯 CNN 对照，但与完整 `echo` 同时指定会被拒绝。`--selection fixed` 也必须从方法列表中排除 `echo`。

## 3. 参数与实际训练预算

| 参数 | 当前含义 / 默认值 |
|---|---|
| `--data` | 单个类别顺序的预处理目录；只能与一个 `--orders` 值搭配 |
| `--input` | 原始数据目录，用于生成缺失的分顺序预处理数据 |
| `--processed-root` | 多顺序数据根目录，子目录为 O0、O1 等 |
| `--orders` | 类别顺序，默认 O0 |
| `--seeds` | 随机种子列表，默认 0 |
| `--methods` | 方法列表，默认全部 11 种 |
| `--output-root` | 实验和自动汇总的根目录 |
| `--device` | `cuda:0`、`cpu` 或默认 `auto` |
| `--amp / --no-amp` | 开启/关闭训练混合精度；CPU 实际关闭 AMP；推理计时固定 FP32 |
| `--num-workers` | 数据加载进程数，默认 0；不是 CPU 计算线程数 |
| `--batch-size` | 训练批量大小，默认 128；不是推理计时批量大小 |
| `--epochs` | 基础/增量主训练轮数，默认 60 |
| `--replay-fraction` | 回放上限占累计已见训练样本数的比例，默认 0.10，范围 (0, 0.10] |
| `--memory-bytes` | 回放序列化字节上限，默认 33554432，即 32 MiB |
| `--finetune-epochs` | LUCIR/PODNet 增量阶段额外平衡分类头微调，默认 20 |
| `--compression-epochs` | FOSTER 增量阶段额外学生压缩训练，默认 40 |
| `--calibration-steps` | BiC 增量阶段偏置校准步数，默认 200 |
| `--backbone` | 对照编码器，默认 echo，可选 cnn1d |
| `--selection` | 对照模型选择：默认 val_macro，或 fixed（最终轮） |
| `--dry-run` | 只显示命令 |
| `--summarize-only` | 只读取已有结果并更新汇总 |

### 3.1 为什么 epochs 相同，优化步数仍不同

`--epochs 10` 并不表示每种方法每个阶段总共只训练 10 轮：

- LUCIR/PODNet 还会默认执行 20 轮分类头微调。
- FOSTER 还会默认执行 40 轮压缩。
- BiC 还会执行 200 次偏置校准更新。
- ECHO 的主训练轮数由批量入口传入；头部预热、专家预热、巩固等仍由 `echo_cil_config.json` / 原训练入口控制。当前配置分别为 5、每组 3、40 轮，并受相关早停逻辑影响。
- 各方法每轮样本池、批量采样方式及每一步计算量也有区别。

批量入口的 `--finetune-epochs`、`--compression-epochs`、`--calibration-steps` 不会修改 ECHO 的额外训练阶段。请比较实际 `optimizer_steps` 和 `stage_elapsed_seconds`，不能据相同 epochs 宣称计算预算相同。

### 3.2 数据权限与回放

回放上限为 floor(比例 × 累计已见原始训练窗口数)，另受字节上限约束。当前 O0、10% 设置下各阶段最多 105、186、268 个样本；Fine-tuning 为 0。旧类只能保留/缩减上一阶段缓冲区，不重读已丢弃旧训练样本。每批重复抽样不会增加唯一回放样本数。

BiC 从允许访问的旧回放与当前新类训练数据中划出独立平衡校准集。默认验证选择允许访问全部已见类验证集：对照按验证 macro 选模型，ECHO 使用自己的综合验证准则并选择推理融合参数。权限类似，选择准则并不完全相同。测试集在该阶段训练和选择完成后用于评估。

## 4. comparison_summary.csv 全部表头解释

每行代表一个 `method × order × seed × stage`。以下按实际 CSV 列顺序解释。

统一单位约定：

- 比例类原始列取值通常为 0–1；`_pct` 列为原值 × 100，例如 0.9675 与 96.75 表示同一准确率。
- `_pp` 表示百分点，例如遗忘 0.0365 对应 3.65 个百分点，不是相对下降 3.65%。
- 空白表示不适用或源结果缺失，不等于 0。stage 0 没有旧类，旧类、双向误判、调和平均和遗忘相关列留空。
- 混淆矩阵行为真实类别、列为预测类别。所有分类指标使用当前已见类别。

### 4.1 实验标识与准确率（1–13）

| 字段 | 含义 | 单位 / 说明 |
|---|---|---|
| `method` | 方法 CLI 名称 | 见方法表 |
| `order` | 类别到达顺序编号 | O0–O4 |
| `seed` | 初始化、数据打乱等随机种子 | 整数 |
| `stage` | 当前阶段编号 | 0=基础，1/2=增量 |
| `run_complete` | 该方法、顺序、种子的整个三阶段实验是否完成 | True/False；False 行也可以是有效的已完成阶段 |
| `test_overall` | 正确测试样本数 / 全部已见类测试样本数 | 0–1，每个样本等权 |
| `test_overall_pct` | test_overall × 100 | % |
| `test_macro` | 各已见类准确率的算术平均 | 0–1，每个类别等权 |
| `test_macro_pct` | test_macro × 100 | % |
| `test_old` | 当前阶段之前引入的全部旧类别的宏平均准确率 | 0–1；stage 0 空白 |
| `test_old_pct` | test_old × 100 | % |
| `test_new` | 当前阶段新增类别的宏平均准确率 | 0–1；stage 0 的基础类均视为新类 |
| `test_new_pct` | test_new × 100 | % |

O0：stage 0 学习 [0,1,2,3]；stage 1 新增 [4,5,6]、评估 7 类；stage 2 新增 [7,8,9]、评估全部 10 类。因此 stage 2 的 old 对应 0–6 类，new 对应 7–9 类。

### 4.2 Precision、Recall、F1 与双向误判（14–27）

| 字段 | 含义 | 单位 / 说明 |
|---|---|---|
| `test_macro_precision` | 各类 Precision=TP/(TP+FP) 的等权平均 | 0–1；没有预测样本的类别取 0 |
| `test_macro_precision_pct` | test_macro_precision × 100 | % |
| `test_macro_recall` | 各类 Recall=TP/(TP+FN) 的等权平均 | 0–1；本单标签任务中与 test_macro 相同 |
| `test_macro_recall_pct` | test_macro_recall × 100 | % |
| `test_macro_f1` | 先计算各类 F1=2TP/(2TP+FP+FN)，再等权平均 | 0–1；不是对宏平均 P/R 再求一次 F1 |
| `test_macro_f1_pct` | test_macro_f1 × 100 | % |
| `test_weighted_f1` | 按各类真实测试样本数加权平均 F1 | 0–1 |
| `test_weighted_f1_pct` | test_weighted_f1 × 100 | % |
| `test_old_to_new_rate` | 真实旧类被判成任意新类的样本数 / 全部真实旧类测试样本数 | 0–1，越低越好 |
| `test_old_to_new_rate_pct` | test_old_to_new_rate × 100 | % |
| `test_new_to_old_rate` | 真实新类被判成任意旧类的样本数 / 全部真实新类测试样本数 | 0–1，越低越好 |
| `test_new_to_old_rate_pct` | test_new_to_old_rate × 100 | % |
| `test_old_new_harmonic` | H=2×test_old×test_new/(test_old+test_new) | 0–1，两者均为 0 时取 0；无旧类时空白 |
| `test_old_new_harmonic_pct` | test_old_new_harmonic × 100 | % |

双向误判不统计“旧类内部互相混淆”或“新类内部互相混淆”，不能代替完整错误率。其分母是各自组内的全部真实样本，不是误判样本数。

### 4.3 遗忘与后向迁移（28–31）

| 字段 | 含义 | 单位 / 说明 |
|---|---|---|
| `forgetting` | 对每个旧任务，取此前阶段的历史最高任务准确率减去当前任务准确率，再对旧任务等权平均 | 比例差，越低越好；stage 0 空白 |
| `forgetting_pp` | forgetting × 100 | 百分点 |
| `backward_transfer` | 对每个旧任务，取当前任务准确率减去该任务首次学完时的准确率，再等权平均 | 比例差，越高越好；负数表示遗忘 |
| `backward_transfer_pp` | backward_transfer × 100 | 百分点 |

“任务”指一个阶段引入的类别组；任务准确率是组内各类准确率的平均。因此本指标先按任务等权，不是直接对全部旧类别等权。历史最高值不包含当前阶段，当前优于全部历史时 forgetting 可为负，代码不截断。Forgetting 和 BWT 通常不互为相反数，只有历史最高值恰等于首次学完时的值才成立。

### 4.4 推理耗时、吞吐与测量设置（32–43）

| 字段 | 含义 | 单位 / 说明 |
|---|---|---|
| `inference_batch_ms` | 重复测量的每批推理耗时平均值 | ms/批，越低越好 |
| `inference_p50_ms` | 每批耗时的第 50 百分位 | ms/批，中位数 |
| `inference_p95_ms` | 每批耗时的第 95 百分位 | ms/批；不是置信区间 |
| `inference_ms_per_sample` | inference_batch_ms / inference_batch_size | ms/样本，批量摊销值，不是 batch=1 请求延迟 |
| `inference_samples_per_second` | inference_batch_size × 1000 / inference_batch_ms | 样本/秒，越高越好 |
| `inference_batch_size` | 实际计时批量大小 | 默认 min(32, 当前阶段测试样本数)，与训练 batch-size 独立 |
| `inference_warmup` | 正式计时前的预热次数 | 当前 5 次 |
| `inference_repeats` | 正式计时次数 | 当前 20 次 |
| `inference_precision` | 推理计时数值精度 | 当前 float32；与训练 AMP 开关独立 |
| `inference_scope` | 计时边界标识 | device_resident_inputs_to_argmax，详见第 5 节 |
| `inference_cpu_threads` | 计时时 PyTorch 的 CPU 计算线程数 | 不等于 num-workers |
| `inference_device_name` | 计时设备名称 | GPU 型号，或 CPU；CPU 值不是具体处理器型号 |

### 4.5 计算与回放开销（44–50）

| 字段 | 含义 | 单位 / 说明 |
|---|---|---|
| `model_parameters` | 当前阶段最终模型的全部参数数目 | 个，包含冻结参数；不含原型等 buffer 张量 |
| `peak_expanded_model_parameters` | 对照方法在压缩前等扩展状态下记录的模型参数量 | 个；不等于同时驻留的教师+学生总参数；当前 ECHO 缺失时留空 |
| `optimizer_steps` | 当前阶段各训练子阶段累计的优化更新步骤计数 | 次，包含适用的校准/微调/压缩；不是 epoch，也不是全实验累计值 |
| `stage_elapsed_seconds` | 当前阶段训练及部分评估、选样等工作到记录时的累计耗时 | 秒；不是纯训练时间，两个训练入口包含的细节略有不同 |
| `peak_cuda_allocated_bytes` | 当前阶段 CUDA 张量分配峰值 | 字节；包含教师和临时张量，不等于显存预留量或 nvidia-smi 总占用；CPU 旧入口可为 0，其他入口可为空 |
| `replay_fraction_actual` | 当前阶段结束后保留样本数 / 累计已见原始训练样本数 | 0–1；不超过 0.10，不是每批旧样本占比 |
| `prototype_tensor_bytes` | 单独统计的原型/类别均值张量大小 | 字节；ECHO 原型、标签和转速，iCaRL 类别均值；其他方法通常为 0 |

### 4.6 历程汇总、回放大小和溯源（51–54）

| 字段 | 含义 | 单位 / 说明 |
|---|---|---|
| `average_incremental_accuracy` | 从 stage 0 到当前阶段的 test_macro 算术平均 | 0–1；仅 stage 2 才是整个三阶段的 AIA |
| `memory_count` | 当前阶段结束后实际保留的回放样本数 | 个；不是本阶段训练时使用的旧样本数 |
| `memory_bytes` | 当前回放缓冲区序列化大小 | 字节；DER++ 含历史 logits/mask；不含模型权重、另记的原型或整个输出文件夹 |
| `results_path` | 对应原始 results.json 的绝对路径 | 在运行汇总的机器上解析；服务器路径复制到本地不会自动变成有效本地路径 |

1 MiB=1,048,576 字节；1 GiB=1,073,741,824 字节。样本预算一致不代表字节预算完全相同：NPY/NPZ 文件头、额外 logits、原型等均影响存储开销。ECHO 原型也属于模型 buffer，统计完整模型存储时应避免与额外列重复相加。

## 5. 推理计时的实际范围与限制

当前固定取该阶段测试集前 32 个输入（不足时取实际数量），预先放到设备上，使用 eval、inference_mode 和 FP32。预热 5 次，测量 20 次，CUDA 每次计时前后同步。

计时包括特征提取、该方法实际使用的最终评分和 argmax。iCaRL 包括 NME，ECHO 包括实际启用的推理融合。排除磁盘读取、物理特征预处理、CPU→GPU 传输及预测结果搬回 CPU。因此这里是模型推理耗时，不是原始信号到最终输出的端到端耗时。

当前统一的是计时程序和批量/精度设置，尚未自动统一所有运行环境：

- 对照入口固定默认 CPU 计算线程数为 4；ECHO 沿用运行环境默认线程数，已有实验曾出现 52。
- 批量入口没有 `--cpu-threads` 参数；该参数目前仅属于 `train_cil_baselines.py`。
- 在同一设备、相同线程设置、空闲 GPU 下重新测量后，才能严谨比较速度。GPU 型号相同并不保证运行时负载和频率相同。
- 训练额外阶段、不同模型选择准则、不同回放选样方式都影响比较。应同时报告实际步数、时间、样本和字节成本。

CPU/GPU 计时结果不要混为一组。单个种子和 20 次重复计时不代表跨训练重复的统计显著性。

## 6. 输出文件与绘图

目录示例：

```text
runs/cil_comparison_retry/
  comparison_summary.csv
  comparison_summary.json
  er/O0/seed_0/
    configuration.json
    results.json
    predictions_stage_0.csv
    predictions_stage_1.csv
    predictions_stage_2.csv
    stage_0.pt
    stage_1.pt
    stage_2.pt
    memory_stage_0.npz
    memory_stage_1.npz
    memory_stage_2.npz
    suite_receipt.json
```

- ECHO 的回放文件扩展名为 `.npy`，对照方法为 `.npz`。
- `results.json` 保存阶段指标、训练历史、适用的补充结果等；不能只看终端最后一行。
- `stage_*.pt` 保存模型及结构信息，供恢复推理使用；没有完整优化器、调度器和随机数状态，不能当作精确断点续训文件。
- `configuration.json` 保存该实验实际配置。`suite_receipt.json` 保存批量命令、被校验文件的哈希和签名，用于判断可否跳过已完成运行。
- CSV 包含原始逐阶段行；JSON 按 method/order/stage 对可用种子计算均值与样本标准差。`runs` 是该阶段可用记录数，`completed_runs` 是对应整次实验已完成数；单个值的标准差为 null。
- JSON 各字段保留原单位：准确率为比例，延迟为 ms，时间为秒，大小为字节。不要把全部 JSON 数值都乘以 100。
- 旧结果若有混淆矩阵可补算 F1/双向误判，有 task_macro 可补算遗忘；未测量的延迟保持空白。

常见绘图方式：

| 目的 | 数据选择 | 绘图字段 |
|---|---|---|
| 增量学习准确率曲线 | 固定 order/seed，各方法全部阶段 | x=stage，y=test_macro_pct，按 method 分组 |
| 最终新旧类对比 | stage=2，通常只取 run_complete=True | test_old_pct 与 test_new_pct |
| 最终整体质量 | stage=2 | test_macro_f1_pct、test_macro_pct |
| 遗忘比较 | stage=1/2 | forgetting_pp，越低越好 |
| 模型偏置分析 | stage=1/2 | test_old_to_new_rate_pct 与 test_new_to_old_rate_pct |
| 准确率与速度权衡 | 相同设备、线程和计时设置，stage=2 | x=inference_batch_ms，y=test_macro_pct |

多种子结果先在同一个 order、同一个 stage 内汇总；不要将不同阶段当作重复实验，也不要把不适用的空白值填成 0 后平均。仅一个 seed 的结果不能计算跨训练随机性的可靠标准差。
