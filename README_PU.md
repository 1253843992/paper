# PU 15 类预处理

```bash
python pu_cil_preprocess.py --input "德国帕德博恩轴承数据集" --output pu_cil_processed --skip-invalid-records
```

只核对文件清单可加 `--audit-only`。输出目录必须为空，不覆盖已有数据。
默认 `--split-seed 2026 --windows-per-record 20`；训练随机种子不能改变数据划分。
`--skip-invalid-records` 显式允许排除不可读/无效记录，并写入 excluded_records.json；
不重新分配集合、不补位。每类每工况至少保留 10 训练/3 验证/3 测试记录，否则停止。
不传此参数则遇到无效记录立即报错。本地 KA08/N15_M01_F10_KA08_2.mat 曾出现 MAT 解析错误。

类别 ID 按以下顺序：K001、KA01、KA03、KI01、KI03、KA04、KI04、KA05、KI05、KA06、KI07、KA07、KI08、KA08、KI18。
阶段为 `[0,1,2,3,4] / [5,6] / [7,8] / [9,10] / [11,12] / [13,14]`。
这是轴承试件/状态分类，不等价于 15 种故障机制，也不能证明未见试件泛化。

每类四工况、每工况 20 个完整 MAT 文件，固定分为 12 训练 / 4 验证 / 4 测试。
先分文件再切窗；不同集合不共享采集记录。读取 vibration_1 与 speed 通道及其 XIndex 时间轴。
依时间轴插值到接近原始平均采样率的均匀网格，再抗混叠降采样到 12 kHz。
每条记录首尾各避开 2048 点，最多选 20 个互不重叠的 2048 点窗口。
时间轴缺口及两侧 10ms 的窗口被排除；不足 20 个会记录警告及实际数量，不补造样本。
默认请求 20 个时若不足 10 个有效窗口则报错。各类样本数可能略有差异。
每窗转速取对应时间区间实测 speed 均值；不从文件名假定实际转速。
统计量只拟合基类训练窗口；验证、测试及未来类别不参与拟合。

输出包含 manifest.csv、record_inventory.csv（含 SHA256）、normalization.json、
class_mapping.json、preprocess_config.json、summary.json、status.json 和 shards/*.npz。
manifest 的 start/end 是目标采样率下相对于记录 time_origin 的位置，end 为开区间。
只有 status.json 的 complete=true 才表示处理完成。中间缓存由程序在完成各分片后移除。

采用近似名义 6203 物理阶次，不声称是每个试件的实测几何参数：
1、2、FTF=0.382、BSF=1.994、2BSF=3.987、BPFO=3.053、BPFI=4.947。
来源：[NTN 6203 技术表](https://docs.rs-online.com/a404/A700000014892791.pdf)。
不同制造商参数可能不同，后续需评估先验敏感性。回放重建读取 normalization 内相同阶次。

## PU 训练适配

训练器现在从数据目录读取任务顺序和物理锚点。`--config` 省略时，PU 自动选择
`echo_cil_pu_config.json`，CWRU 仍选择原配置。建议正式实验显式指定配置以便复现。
现有 PU 数据只声明 O0；请求 O1～O4 会报错，不会套用 CWRU 顺序。
初始归一化类别必须与任务初始类别一致；更换初始类别需要重新预处理。

```bash
# 仅核验协议，不训练、不创建实验目录
python train_echocil.py --data pu_cil_processed --config echo_cil_pu_config.json --device cpu --validate-only

# 服务器正式运行示例（本地未启动正式实验）
python train_echocil.py \
  --data pu_cil_processed \
  --config echo_cil_pu_config.json \
  --output runs/pu_echo/O0/seed_0 \
  --order O0 --seed 0 \
  --device cuda:0 --amp \
  --num-workers 4 --cpu-threads 4 --batch-size 128 \
  --replay-fraction 0.05
```

CPU 使用 `--device cpu --no-amp --num-workers 0`。已有非空输出目录会被拒绝，避免覆盖结果。
正式默认基类/增量各 60 epochs，头预热 5、专家预热 3、巩固 40；如需 10 epochs 试运行，
显式添加 `--base-epochs 10 --incremental-epochs 10`（不改变其他训练阶段的 epoch）。

PU 分类目标仍是 15 个试件/状态，语义只提供 healthy/outer/inner；不推断损伤尺寸或严重程度。
`prototype_severity_weights` 必须为 `[0.0]`，非零会报错；family 默认也为零，避免未经验证的层级融合。
特征预处理、回放重建、主模型、ECHO 骨干对照模型使用同一套已保存的近似 6203 锚点。
checkpoint 保存类别语义和锚点；可通过 `echo_cil.model.restore_echo_model(checkpoint)` 恢复推理模型。
该函数接收已读取的可信 checkpoint 字典，不负责恢复优化器以继续训练。

六阶段含五次增量：PU 配置每阶段最多 2 个专家、总上限 10（并不保证每阶段一定分配满）。
CWRU 原配置每阶段最多 3、总上限 6 保持不变；跨数据集资源对比需报告实际专家数和参数量。
5% 回放上限按累计实际训练窗口数计算，最终最多 floor(14393×0.05)=719 个，仍受字节预算约束。

对比/消融入口已支持读取 PU 协议，结果汇总按实际六阶段判断完成并计算新旧类指标。
使用这些入口时显式传 `--data pu_cil_processed --orders O0`（消融为 `--order O0`）
和 `--replay-fraction 0.05`。不使用旧 CWRU 自动原始数据预处理入口。

```bash
# 开发用缩小数据六阶段检查，不得将其准确率用于论文
python tests/smoke_pu_training.py --data pu_cil_processed --workspace runs/pu_smoke_check
```

本次不修改 CWRU 处理结果或默认物理先验，也不修改已有 PU 分片。
