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

**当前仅完成预处理，不要直接使用旧 CWRU 训练命令。**
训练端仍需读取 PU 的 15 类六阶段任务顺序、注入模型物理锚点并移除 CWRU 专用类别语义。
本次不修改 CWRU 原始数据、处理结果或其默认物理先验。
