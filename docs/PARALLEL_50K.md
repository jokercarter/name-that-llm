# 5 万条 × 2×T4：数据并行、显存分配与参数扩容

本文对应 `configs/kaggle_9h_50k_2gpu.yaml` + `scripts/train_parallel.py`，说明在“必须用满约 5 万条样本”的前提下，如何把两张 T4 都用起来，并据此把参数规模从 0.6B 抬到 1.7B。

## 0. 一页结论

| 项 | D 方案（单卡 0.6B） | 本方案（双卡 1.7B） | 变化 |
|---|---|---|---|
| 骨干 | 0.6B | **1.7B** | 2.8× |
| 可训练参数 | ~9 M | **~52 M** | 5.8× |
| 训练样本 | 50,003 | 50,003 | 不变 |
| 并行 | 单卡 | **2 进程数据并行** | 吞吐 ~1.9× |
| 单卡峰值显存 | ~2 GB | **~3 GB / 16 GB** | 仍 <20% |
| 墙钟（实测标定） | 2.1–4.7 h | **9.0 h（3 折，batch 8）/ 7.8–8.2 h（3 折，batch 12）** | 见 §4.1；2 折为 6.0 h |
| 每次 fit 硬上限 | — | 1.9 h（4 次 = 7.6 h） | 超时中断而非撑爆 |

一句话：**第二张卡不是用来缩小数据，而是用来买参数的**。数据量是需求（写死的 5 万），能弹的只有“每步通过多少样本”和“模型多大”，双卡把前者翻倍，于是后者可以翻 2.8 倍而总时长只涨 1.4 倍。

## 1. 并行方式

### 1.1 选型：数据并行，2 进程，1 卡/进程

```
rank 0 ── 4-bit 副本 ── 全局 batch 的前 8 行 ─┐
                                             ├─ all-reduce 梯度 → 同一个 optimizer step
rank 1 ── 4-bit 副本 ── 全局 batch 的后 8 行 ─┘
```

- 每个 rank 持有**完整的一份 4-bit 模型**（1.7B 只占 ~1.5 GB），只对各自的 micro-batch 做前向/反向；
- 每个 optimizer step 前把 LoRA + 融合头的梯度求和除以 2，两边得到完全相同的更新；
- 全局 batch = 8（每卡）× 2（卡）× 2（梯度累积）= **32**，比单卡的 16 大一倍，梯度噪声更小。

### 1.2 为什么不用 `DistributedDataParallel`

可训练张量是挂在 bitsandbytes 4-bit 模块上的 LoRA 适配器，模型还开着梯度检查点、嵌入是共享的。DDP 的 reducer 假设参数是布局稳定的普通稠密张量，这三点正好是它最容易出问题的地方（unused-parameter 报错、检查点重入后 bucket 重建、对冻结量化张量做 bucket）。手写 all-reduce 只有十几行（`dist_utils.all_reduce_grads`），对模块类型没有假设，而且 `world_size == 1` 时是严格 no-op——本地单卡路径一个字节都没变。

### 1.3 为什么不是模型并行 / ZeRO

- **模型并行**：1.7B 的 4-bit 权重只有 1.5 GB，切成两半后每层都要跨卡通信，收益为负；
- **ZeRO-2/3 + CPU offload**：可训练参数只有 52 M，优化器状态 0.83 GB，offload 到 CPU 只会把 PCIe 变成瓶颈；
- **流水线并行**：单卡放得下整个模型，切层只会引入气泡。

结论：在“单卡装得下”的区间里，数据并行是唯一划算的并行方式。

### 1.4 三个必须由 rank0 决定的同步点

| 同步点 | 为什么 |
|---|---|
| 梯度 all-reduce | 两个 rank 必须走同样多的 step，否则下一步集合通信会错位挂死 |
| 早停判定 | 只在 rank0 上算内层 BA，结果用 1 元素 `broadcast` 广播；两条 rank 各自判会一侧先退出 |
| 时间预算 | 同上，deadline 由 rank0 检查后广播，粒度 = `log_every` 步（约 20 s） |
| 最终权重 | 早停回滚只在 rank0 上做，回滚后把 52 M 参数广播一次（~208 MB，约 20 ms），否则 rank1 会用它自己那份“未回滚”的权重去预测另一半 OOF |

最后一条最容易漏：OOF 预测是**分片**的（每卡一半，再 `all_gather`），如果两张卡的权重不一致，拼出来的概率来自两个不同的模型。

### 1.5 预测也并行

5 万条里每次 fold 的验证集约 1.67 万行，前向-only 约 0.04 s/样本，单卡要 11 分钟；分片后 5.5 分钟，`all_gather` 只搬 1.2 MB。整轮 OOF 推理因此从约 35 分钟降到 18 分钟。

## 2. 显存分配策略

### 2.1 每卡预算（1.7B / batch 8 / 384 token）

| 项 | 估算 | 说明 |
|---|---:|---|
| 4-bit 线性层 | 0.78 GB | NF4 + double quant ≈ 0.55 B/param，非嵌入 1.41B |
| 词嵌入（fp16） | 0.62 GB | bnb 只量化 Linear，Embedding 保持 compute dtype |
| LoRA + 融合头 | 0.21 GB | 52 M × 4 B |
| 梯度 + AdamW 状态 | 0.83 GB | 52 M × 16 B |
| 激活（检查点） | 0.03–0.08 GB | 只存 ~6 个边界，重算一层 |
| CUDA context / 碎片 | 0.4 GB | |
| **峰值** | **~2.9 GB / 16 GB（18%）** | 实测值由 `[mem] rankN` 日志打印 |

### 2.2 四条降显存的改动

1. **砍掉 `lm_head`**（`_hide_lm_head`）：`CausalLM.forward` 一定会算 `logits = lm_head(hidden)`，而 18 路打分由融合头给出，这个 151936×2048 的投影纯属浪费——batch 8×384 token 时是 ~1.2 GB 激活，外加约 6% 的前向 FLOPs。换成 `nn.Identity` 后隐藏状态直接透传（嵌入是共享的，不损失任何参数）。
2. **梯度检查点**：只存每层输入，反向时重算；激活从 ~1.4 GB 降到几十 MB。
3. **`expandable_segments:True`**：4-bit + 变长序列最容易死于碎片而不是峰值，这个选项让分配器复用段而不是反复 cudaMalloc。
4. **阶段间 `release()`**：每折结束丢掉骨干并 `empty_cache()`，避免下一折加载时的瞬时双份权重。

### 2.3 为什么只用到 18%

“显存利用率”有两个口径：**占用率**和**计算利用率**。这里刻意压低前者、抬高后者：

- 压占用率是为了抗抖动：Kaggle 的 T4 是共享卡，4-bit 反量化 + 变长序列 + 碎片是 OOM 的经典组合，留 13 GB 余量意味着把 batch 临时调到 24 也不会炸；
- 抬计算利用率靠三件事：每卡 batch 8（而不是 1）、长度分桶（见 §3）、去掉 `lm_head`。在 1.7B 这个尺寸上，T4 的瓶颈是 kernel 启动与反量化，不是容量，把显存塞满只会增加 OOM 概率而不增加吞吐。

## 3. 吞吐：四个真正的杠杆

| 杠杆 | 效果 | 代价 |
|---|---|---|
| 数据并行 2 卡 | **×1.9** | all-reduce 每步 ~20 ms（步长约 1.4 s，约 1.4%） |
| 长度分桶 | **−52% padded token** | 无；实测随机分批 pad 304.7 token，分桶后 146.5 |
| 全局 batch 32 | 步数减半，kernel 启动开销摊薄 | 无（梯度质量反而更好） |
| 去掉 lm_head | −6% FLOPs，−1.2 GB | 无 |

长度分桶的实现要点：按字符长度排序后切成长度为 `batch × 40` 的池，池内再打乱并切 batch，最后**打乱 batch 顺序**。这样每个 batch 长度同质，但一个 epoch 内的长度分布仍是随机的，不会退化成“先短后长”的课程学习。

## 4. 参数规模：每一档的依据

| 参数 | D 方案 | 本方案 | 依据 |
|---|---|---|---|
| 骨干 | 0.6B | **1.7B** | 单样本 FLOPs ∝ 参数量；双卡把这 2.8× 抵掉一半，净 1.4× 时间 |
| LoRA rank | 16 | **32** | 50k 样本下 52 M 可训练参数仍属“每参数 1 条样本”，离过拟合还远 |
| LoRA 目标 | q/k/v/o | **全部 7 个投影** | 4B 以下模型的 MLP 也承载风格信号；1.7B 的参数预算允许 |
| 融合头 | 6 层 × 512 | **8 层 × 768** | 1.7B 有 28 层，取 [2,5,8,12,16,20,24,28] 覆盖早/中/晚；宽度按 hidden/2.7 取 |
| 序列长度 | 384 | 384 | 274 词 ≈ 360 token，再长只增加 pad |
| 折数 | 3 | 3 | 与 D 一致；折数影响 OOF 稳定性，不影响单折成本 |

可训练参数明细（1.7B）：

```
LoRA: 28 层 × 1,245,184 = 34,865,152
头  : 8 × (2048×768) + MLP(6144→768→18) ≈ 17,334,554
合计: 52,199,706  (占 1.77B 总参数的 2.9%)
```

### 4.1 为什么不是 4B / 8B（实测修正）

| 方案 | 单样本/卡 | 5 万条 × 3 折（双卡） | 判定 |
|---|---:|---:|---|
| 1.7B | 0.43 s | 9.0 h（8 折前） | 默认 |
| 4.0B | 1.01 s | 21 h | 否决 |
| 8.0B | 2.02 s | 42 h | 否决 |

想把预算填满到 8 h，正确的旋钮是**折数 × 每卡 batch**（3 折 + batch 12 = 7.8–8.2 h），不是骨干：4B 的 FLOPs 是 1.7B 的 2.35 倍，3 折直接顶到 21 h。若仍想要更大容量，把 LoRA 从 r32 提到 r64、融合头 768→1024（可训练参数 52 M → ~100 M），耗时只增加约 3%，配置见 `configs/kaggle_8h_50k_3fold.yaml` 的注释。

## 5. 训练与评估必须分开

5 万条里 **96% 是外部语料**（46,138 条 external + 3,562 条重复），而隐藏测试集 100% 是竞赛文本。如果温度、融合权重、家族专家、护盾都在全量 5 万条上拟合，优化目标就变成了“外部域的 18 分类”，与竞赛分布错位。

因此 `PACT18Pipeline.fit(..., eval_mask=...)` 把**所有被拟合的后处理参数**限制在 `origin == "competition"` 的 2,303 行上，OOF 矩阵本身仍是全量（外部行照常参与训练，只是不参与打分）。这样：

- 外部数据只影响“成员怎么学”，不影响“成员怎么组合”；
- 报告的 OOF balanced accuracy 与之前 0.4211 的口径完全可比；
- 3,562 条重复样本的 `prompt_id` 与原行相同，分组折必然把它们放在同一折，不会跨折泄漏（已实测：跨折文本 0 条）。

## 6. 验证方案

### 第 0 步：20 步标定（约 10 分钟）

```bash
python scripts/train_parallel.py --stage qwen --config configs/kaggle_9h_50k_2gpu.yaml \
  --train-file data/real/trainval_plus_ext_50k.csv \
  --model-path /kaggle/input/<qwen3-1-7b> \
  --set features.qwen.time_budget_hours=0.02
```

日志里读 `[mem] rankN ... peak=` 与每步耗时，算出 s/样本，然后：

```bash
python scripts/estimate_qlora_budget.py --rows 50003 --folds 3 --gpus 2 --sec-per-sample <实测>
```

若总时长 >8.5 h：把 `batch_size` 提到 12（更快）、或把外部配额降到 2,200/类（每次 fold 少 6k 样本）。

### 第 1 步：单折 dry-run（约 40 分钟）

```bash
--set data.n_folds=2 --set features.qwen.time_budget_hours=0.35
```

检查三件事：loss 单调下降、`peak < 6 GB`、早停不在 150 步内触发。

### 第 2 步：全量（4 次 fit，硬上限 7.6 h）

### 第 3 步：验收

| 指标 | 通过标准 | 依据 |
|---|---|---|
| OOF blended BA（2,303 行） | ≥ 0.411（相对 0.4211 降 ≤0.010） | 1.7B 表征更强，掉分说明外部域拖累 |
| test BA（396 行） | ≥ 0.588（降 ≤0.015） | 1 个标准误以内 |
| 弱类召回（fable / 2×GLM / terra） | 不低于 8B probe 版本 | 50k 的主要收益应体现在这里 |
| 单卡峰值 | < 6 GB | 留足抗抖动余量 |
| 墙钟 | < 8.5 h | 9 h 上限留 30 min 给预测与落盘 |
| 折间 BA 极差 | < 0.06 | 3 折方差比 5 折大，超过说明数据混合不稳 |

任一项不通过的回退顺序：① 配额 2,650→2,000/类；② 骨干回 4B/0.6B；③ `lr 1e-4→5e-5` + `lora_r 32→16`。

## 7. 已知风险

1. **Kaggle 镜像没有 bitsandbytes**（首跑 v1 已实证：Qwen 成员被静默关闭，只剩 4 成员栈）。必须把 cp313 manylinux 的 wheel 放进数据集，`pip install --no-index --find-links /kaggle/input/<ds> bitsandbytes`，并在 notebook 里断言 `import bitsandbytes` 成功再启动训练。
2. **T4 不支持 bf16**：`compute_dtype: auto` 会在 T4 上选 fp16、在 Ada 上选 bf16；写死 bf16 会在 4-bit kernel 里报错或退化成 fp32 仿真。
3. **NCCL 首次建链**可能因共享机卡住：`MASTER_PORT` 用随机空闲端口（脚本已做），失败时重跑即可；不要用 `gloo` 兜底，PCIe 上慢 3×。
4. **折数与双卡的对齐**：3 折时第 3 折只有一张卡在跑（约 8% 的浪费）；4 折 + 1 次重训恰好是 4 个"折"任务，可以把两折分给一张卡、另两折给另一张，负载最均衡。这也是 8 h 方案选 4 折的附带理由。
5. **`joblib.dump` 体积**：LoRA + 头以 numpy 存约 208 MB，加上流水线其余成员约 400 MB，仍在 Kaggle 输出限制内。

## 8. 命令

```bash
# 1) 5 万条训练文件（本地 base=trainval: 2,303 竞赛 + 44,138 外部 + 3,562 重复）
#    Kaggle 侧 base=train.csv 时为 2,699 + 43,874 + 3,430
python scripts/build_external_mix.py --target-total 50000 --oversample \
    --external data/external/ntl_merged_external_large.parquet \
    --out data/real/trainval_plus_ext_50k.csv

# 2) 预算
python scripts/estimate_qlora_budget.py --rows 50003 --folds 3 --gpus 2

# 3) 端到端（2 卡 QLoRA → CPU 成员 → 后处理 → submission.parquet）
python scripts/train_parallel.py --stage all \
    --config configs/kaggle_9h_50k_2gpu.yaml \
    --train-file data/real/trainval_plus_ext_50k.csv \
    --model-path /kaggle/input/<qwen3-1-7b> \
    --test-file /kaggle/input/name-that-llm/test.csv \
    --submission submission.parquet
```

单卡等价命令（`--nproc 1`）走同一份代码，用于本地对照：`dist_utils` 的所有集合通信在 `world=1` 时都是 no-op，`fit` 的 `members/oof/refit` 默认参数也保持原行为。全程 59 条单测 + 1 条端到端冒烟通过。

### 8.1 在 Kaggle notebook 里调用

```python
# 1) 解压源码数据集、装 bitsandbytes wheel（镜像默认没有）
!cd /kaggle/working && unzip -oq /kaggle/input/ntl-pact18-src/ntl-pact18-code.zip
!pip install -q --no-index --find-links /kaggle/input/<bnb-wheel-ds> bitsandbytes
import bitsandbytes  # 失败就直接抛错，不要静默退回 4 成员

# 2) 用挂载的外部语料 + 竞赛 train.csv 生成 5 万条训练文件（带 origin 列）
!cd /kaggle/working && python ntl-pact18-code/scripts/build_external_mix.py \
    --base /kaggle/input/name-that-llm/train.csv \
    --external /kaggle/input/ntl-external/ntl_merged_external_large.parquet \
    --target-total 50000 --oversample \
    --out /kaggle/working/trainval_plus_ext_50k.csv

# 3) 两个 rank 由脚本自己拉起，notebook 只跑一个前台进程
!cd /kaggle/working && python ntl-pact18-code/scripts/train_parallel.py --stage all \
    --config  ntl-pact18-code/configs/kaggle_9h_50k_2gpu.yaml \
    --train-file /kaggle/working/trainval_plus_ext_50k.csv \
    --model-path /kaggle/input/qwen3-1-7b/transformers/default/1 \
    --test-file  /kaggle/input/name-that-llm/test.csv \
    --submission /kaggle/working/submission.parquet
```

`--train-file` 必须是**带 `origin` 列**的 5 万条文件；直接喂官方 `train.csv` 也不会报错（`eval_mask=None`，退回全量拟合），但那样 96% 的外部行会参与融合层拟合，分布就错位了。
