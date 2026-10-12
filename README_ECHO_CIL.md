# ECHO-CIL 严格实现

本实现按《ECHO-CIL_完整技术方案》重构，旧版 `echo_cil_processed`、旧模型检查点和旧结果不可复用。

## 已实现的核心机制

1. 记录内连续区块60/20/20划分，边界两侧各删除一个目标窗口长度；正常记录按48 kHz审计，故障记录按12 kHz审计，再分区块重采样至12 kHz。
2. 2048点时域输入；三频带包络阶次谱、`log(1+S)`、0--20阶256点插值；每窗反射延拓256点。
3. 21个通用谐波查询、4头阶次距离偏置注意力，以及每频带PBR/谱熵/RPM稳定性组成的9维质量向量。
4. 三个由谐波证据和质量门控共同控制的时域尺度块，通道为64/96/128，膨胀率为1/3/7。
5. 时域Query、谐波证据Key/Value的跨域注意力，生成128维冻结基础特征。
6. 可靠旧类边、最大权匹配、分类权重差SVD，以及新增专家残差的FP32前向正交投影。
7. 训练侧旧新混淆分组；每阶段最多两个rank=8专家、总数最多四个，独立sigmoid门控。
8. 独立冻结旧分类头，新分类行使用冻结特征类均值初始化。
9. 精确旧/新各半采样、类别均匀重要性校正、旧新竞争间隔损失和残差幅度损失。
10. 512 KiB波形回放，按样本ID哈希确定性选择；保存后的NumPy文件大小必须不超过预算。

## 第一步：重新预处理

主顺序O0：

```bash
python echo_cil_preprocess.py \
  --input CWRU轴承数据/cwru_data \
  --output echo_cil_processed \
  --base-classes 0 1 2 3
```

预处理会生成新的 `normalization.json`。训练程序会拒绝缺少严格物理统计量的旧数据目录。

## 第二步：运行主顺序

GPU：

```bash
python train_echocil.py \
  --data echo_cil_processed \
  --output runs/echo_cil_strict/O0/seed_0 \
  --order O0 \
  --seed 0 \
  --device cuda:0 \
  --amp \
  --num-workers 4
```

CPU验证：

```bash
python train_echocil.py --device cpu --no-amp --num-workers 0
```

默认参数来自 `echo_cil_config.json`：基阶段60轮，新头预热4轮，专家分组预热2轮，增量联合训练30轮，batch为64。

## 完整5顺序×3种子实验

每个类别顺序需要独立的基阶段归一化统计。批量脚本会自动为每个顺序生成独立处理目录，再运行15个配对实验：

```bash
python run_echo_cil_suite.py \
  --input CWRU轴承数据/cwru_data \
  --processed-root echo_cil_processed_suite \
  --output-root runs/echo_cil_suite \
  --device cuda:0 \
  --num-workers 4
```

## 输出

每个运行目录包含：

- `configuration.json`：实际参数、类别顺序和配置哈希；
- `stage_0.pt`、`stage_1.pt`、`stage_2.pt`：模型、架构元数据和随机状态；
- `memory_stage_*.npy`：真实512 KiB预算内的回放；
- `results.json`：阶段精度、AIA、遗忘、BWT、旧新调和均值、可靠边、匹配边、保护秩、投影误差、专家分组、旧类被新类抢占率和资源记录。

`results.json` 是一个对象，阶段结果位于 `stages`，汇总指标位于 `summary`。
