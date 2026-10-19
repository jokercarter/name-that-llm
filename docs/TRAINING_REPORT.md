# Name That LLM — 训练报告

> 数据：`train.csv`（2699 行，真实赛题训练集）
> 训练日期：2026-10-18
> 代码：`D:\research\projects\ntl-pact18`

---

## 1. 数据概览

### 1.1 规模与字段

| 项 | 值 |
|---|---|
| 行数 | 2,699 |
| 字段 | `id` / `prompt_id` / `text` / `model` |
| prompt 数 | 150 |
| 模型数 | 18 |
| 段落长度 | 36–274 词，中位数 101，均值 114.7 |

### 1.2 数据质量

| 检查项 | 结果 |
|---|---|
| 缺失值 | 0（四个字段全部无空值） |
| 空文本 | 0 |
| 重复 id | 0 |
| 重复文本 | 0 |
| 标签合法性 | 18 个标签全部在候选集内 |

**结论：数据无需清洗。** 唯一的"预处理"是去掉首尾空白，并**刻意不做**小写化、去标点、词形还原——因为标点、大小写、连接词恰恰是作者指纹本身（第 9 节会看到 Google 家族长破折号频率为 0，Anthropic 为 10.8/千词）。

### 1.3 结构：18×150 的完整网格

| 检查项 | 结果 |
|---|---|
| 每个模型样本数 | 150（`claude-fable-5` 为 149） |
| 每个 prompt 的段落数 | 149 个 prompt 有 18 段，1 个有 17 段 |
| 同一 (prompt, model) 重复 | 0 |

这意味着**同一 prompt 下每个模型至多出现一次**——这是 PRISM 结构约束的依据，也是本方案与公开 notebook 的核心差异。

### 1.4 风格信号（EDA 结论）

家族级标点频率（每千词）：

| 家族 | 分号 | 冒号 | 长破折号 | 感叹号 | 逗号 |
|---|---:|---:|---:|---:|---:|
| anthropic | 3.0 | 5.6 | **10.8** | 0.1 | 88.0 |
| google | 1.9 | 1.7 | **0.0** | 0.0 | 82.2 |
| openai | 6.1 | 7.9 | 2.9 | 0.0 | 99.2 |
| zai | 4.3 | 7.6 | 13.6 | 0.1 | 88.9 |
| qwen | 4.2 | 5.2 | 0.6 | 0.0 | **111.4** |
| moonshot | 8.9 | 9.1 | 7.4 | 0.0 | 96.9 |

正字法硬规律（护盾规则来源）：

| 家族 | 含弯引号比例 | 含 ≥2 个长破折号比例 |
|---|---:|---:|
| anthropic | **0.0000** | 0.3497 |
| zai | **0.0000** | 0.5236 |
| google | 0.2358 | **0.0000** |
| openai | 0.5132 | 0.0925 |

同 prompt 段落平均余弦相似度 **0.214**，跨 prompt **0.011**（相差约 20 倍）——聚类可行。

---

## 2. 任务与指标

- **任务类型**：18 类单标签文本分类（作者归属 / authorship attribution），非"人 vs 机器"检测。
- **输入**：一段 36–274 词的英文段落。
- **输出**：18 个模型名之一。
- **训练目标**：最小化交叉熵（各成员），融合层用 NLL 拟合权重。
- **评估指标**：**Balanced Accuracy**（18 类召回的平均），与赛题一致。随机基线 1/18 = 0.056。
- 辅助指标：macro-F1、top-2 准确率、ECE（校准误差）、聚类 pair precision/recall。

---

## 3. 预处理与数据切分

### 3.1 预处理

| 步骤 | 做法 | 理由 |
|---|---|---|
| 文本清洗 | 仅 `strip()`，保留原始标点/大小写/unicode | 标点与 casing 是指纹，清洗会抹掉信号 |
| 标签校验 | 对照 18 个候选标签 | 防止脏标签污染 |
| 长度 | 不截断（最长 274 词 < 模型 512 上限） | 保留完整风格证据 |
| 特征 | 4 路 TF-IDF（char 3-5 / word 1-3 / shape 2-6 / function 1-3）+ 360 维风格统计 + n-gram LM | 见第 5 节 |

### 3.2 切分（按 prompt 分组，杜绝泄漏）

```bash
python run.py --config configs/real.yaml split --source train.csv
```

| 集合 | prompt 数 | 行数 | 用途 |
|---|---:|---:|---|
| train | 106 | 1,907 | 训练成员 + 5 折 OOF |
| val | 22 | 396 | 选择 PRISM 的 λ、观察分阶段增益 |
| test | 22 | 396 | 最终评估（全程不参与任何选择） |

**泄漏检查**：train/val、train/test、val/test 的 prompt 交集均为 **0**；`split_meta.json` 留档。

> 说明：赛题测试集的 prompt 与训练完全不重叠，因此**必须按 prompt 分组切分**。随机按行切分会把同一 prompt 的段落分到两边，CV 会虚高（论坛实测方向都会反）。

---

## 4. 运行环境与设备

### 4.1 两套环境

| | CPU 基线 | GPU 环境（本次重装） |
|---|---|---|
| Python | 3.13.9（anaconda） | 3.13.9（项目 `.venv`） |
| torch | 2.13.0+**cpu** | **2.14.1+cu126** |
| scikit-learn | 1.7.2 | 1.9.1 |
| 梯度提升 | sklearn HistGB | **xgboost 3.4.1（device=cuda）** |
| 设备 | CPU | **RTX 4060 Laptop 8GB（CUDA 12.6）** |

### 4.2 设备选择策略

优先级 `cuda → mps → cpu`，由 `src/ntl_pact18/device.py` 自动检测：

- `torch.cuda.is_available()` 为真 → `cuda:{gpu_index}`；
- 否则 `torch.backends.mps.is_available()` → `mps`；
- 否则 `cpu`，并把**具体原因**写进日志（例如"检测到 RTX 4060，但 torch 是 CPU 版本"）；
- `device.require_gpu=true` 时无加速器直接抛错，不静默降级。

### 4.3 实测加速

| 组件 | CPU | GPU | 加速 |
|---|---:|---:|---:|
| stylometry（400 棵树） | 78.9 s（HistGB） | **15.4 s（XGBoost GPU）** | **5.1×** |
| 全流程（5 折 + 重训） | 666 s¹ | 279 s | 2.4×¹ |

¹ CPU 那次训练与依赖安装并发，时间偏保守；单组件 5.1× 是干净的对比。

> **踩坑记录**：torch 的 CUDA 稀疏内核（COO 与 CSR 都试过）在本机第 105 步左右稳定报 `cudaErrorIllegalAddress`。最终 GPU 成员改为**稠密 cuBLAS 矩阵乘**（特征压到 5.2 万维，约 0.4 GB），训练 7 秒完成，稳定。

---

## 5. 模型与训练流程

### 5.1 成员（4 个）

| 成员 | 特征 | 模型 | 单独 val |
|---|---|---|---:|
| `stylegram` | char 3-5(120k) + word 1-3(90k) + shape 2-6(30k) + function 1-3(25k) TF-IDF | LinearSVC C=0.5 + 簇内中心化孪生 | 0.40 左右 |
| `stylometry` | 360 维统计（句长/词长/标点/停用词） | XGBoost GPU，400 树，lr 0.04 | — |
| `ngram_lm` | 每类 add-α 二元语言模型的 18 维交叉熵 + rank/margin/entropy | 多项逻辑回归 | — |
| `linear_baseline` | word 1-2 + char 3-5 TF-IDF | 逻辑回归 C=2 | — |

### 5.2 融合与后处理

```
成员概率 → 温度缩放（每成员一个 T）→ 对数概率加权融合（L-BFGS 拟合权重）
        → 家族专家（6 个实验室，只改家族内部）
        → 正字法护盾（从训练集自动发现 3 条规则）
        → PRISM 簇内联合解码（容量约束，λ=2.0）
```

- **聚类**：中位数归一化 TF-IDF 余弦 + average linkage，阈值在**held-out prompt 折**上自动校准（本数据得 0.925），val 上 pair P=0.91 / R=0.92。
- **护盾规则**（自动发现，与 EDA 完全一致）：`curly_quote → anthropic`、`curly_quote → zai`、`double_em_dash → google`。
- **PRISM**：主槽 18 + 每类 18 个重复槽，重复惩罚 λ；λ 由 val + OOF 模拟共同确定。

### 5.3 交叉验证

`GroupKFold(5)` 按 `prompt_id` 分组，每折约 378–395 行。所有 OOF 预测只用于拟合温度、融合权重、家族专家，不参与测试评估。

---

## 6. 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| `features.stylegram.C` | 0.5 | LinearSVC 正则强度 |
| `features.stylegram.center_mode` | replace | 簇内用 prompt 中心化孪生替换普通头 |
| `features.stylometry.n_estimators` | 400 | XGBoost 树数 |
| `features.stylometry.learning_rate` | 0.04 | |
| `features.ngram_lm.alpha` | 0.1 | 语言模型平滑 |
| `features.linear_baseline.C` | 2.0 | |
| `calibration.temperature` | true | 每成员单独拟合 |
| `calibration.em_prior` | false | EM 先验校正在本数据上掉分（0.569→0.549 OOF），关闭 |
| `clustering.method` | tfidf | 学习式度量在真实数据上过拟合（val F1 0.47 vs 0.94） |
| `clustering.threshold` | 自动校准 | 本数据得 0.925 |
| `prism.mode` | relaxed | |
| `prism.lam` | **2.0** | val 与 OOF 模拟一致最优 |
| `prism.min_size / max_size` | 2 / 18 | 单条与超大簇直接 argmax |
| `experts.min_rows` | 100 | 家族专家最小样本 |
| `shield.penalty` | 2.5 | 对数惩罚 |
| `device.preference` | auto | cuda → mps → cpu |

---

## 7. 结果

### 7.1 OOF（5 折，1,907 行）

| 项 | 值 |
|---|---:|
| 融合后 Balanced Accuracy | **0.3539** |
| 融合权重 | stylegram 0.214 / stylometry 0.216 / torch_linear 0.240 / ngram_lm 0.146 / linear_baseline 0.184 |

> 该 OOF 对应含 5 个成员的配置；最终提交配置关闭了 `torch_linear`，CPU 版 4 成员 OOF 为 **0.3606**。
> **含 Qwen 探针成员的最终配置**（stylegram + stylometry + ngram_lm + linear_baseline + qwen_probe，训练于 `trainval.csv` 2303 行）：5 折 OOF **0.4211**，融合权重 qwen_probe=0.286（居首）。详见 §7.5。

### 7.2 验证集（22 prompts / 396 行，用于选 λ）

| 阶段 | λ=1.0 | λ=2.0 |
|---|---:|---:|
| blend | 0.4040 | 0.4040 |
| experts | 0.4091 | 0.4091 |
| **PRISM** | 0.4343 | **0.4571** |

λ 扫描：0.5 → 0.404，1.0 → 0.434，**2.0 → 0.457**，3.0 → 0.455。选 **λ=2.0**。

### 7.3 测试集（22 prompts / 396 行，最终评估）

| 阶段 | Balanced Acc | 相对上一阶段 |
|---|---:|---:|
| blend | 0.3990 | — |
| + 家族专家 | 0.4091 | +0.010 |
| + 正字法护盾 | 0.4091 | +0.000 |
| **+ PRISM（λ=2）** | **0.5152** | **+0.106** |

其他指标：

| 指标 | 值 |
|---|---:|
| macro-F1 | 0.5140 |
| accuracy | 0.5152 |
| top-2 accuracy | 0.6086 |
| ECE | 0.0504 |
| 聚类 pair precision | 0.9148 |
| 聚类 pair recall | 0.9186 |
| 聚类 pair F1 | 0.9167 |

### 7.4 与 CPU 基线对比

| 配置 | 成员数 | OOF | 测试 blend | 测试 PRISM |
|---|---:|---:|---:|---:|
| CPU（HistGB） | 4 | 0.3606 | 0.4268 | **0.5177** |
| GPU（XGBoost-GPU + torch_linear） | 5 | 0.3539 | 0.3990 | 0.5152 |
| **GPU + Qwen3-8B 探针（最终 model_qwen）** | 5 | **0.4211** | 0.4773 | **0.6061** |

结论：**GPU 让训练快 2.4–5.1×，但 `torch_linear` 成员稀释了融合**（占 24% 权重却更弱），最终测试差 −0.0025（在噪声范围内）。因此最终配置关闭该成员、保留 XGBoost GPU 后端。

### 7.5 含 Qwen 成员的最终模型（model_qwen.joblib）

> 模型：`artifacts/real/model_qwen.joblib`（GPU 全栈 + Qwen3-8B-Base 4-bit 探针，2026-10-18 训练于 `trainval.csv`）。
> ⚠️ 该模型训练于 `trainval.csv`（train+val 合并），故 `val.csv` 为样本内，**其 1.0000 为泄漏结果，不报**；以下只报 `test.csv`（22 个不重叠 prompt）。

**OOF（5 折，2303 行）**：融合后 Balanced Accuracy **0.4211**（不含 Qwen 时 0.3539，+0.067）。融合权重 qwen_probe=0.286（居首）/ stylegram=0.245 / stylometry=0.206 / linear_baseline=0.147 / ngram_lm=0.116。

**测试集分阶段 Balanced Accuracy（λ=2.0）**：

| 阶段 | Balanced Acc |
|---|---:|
| blend | 0.4773 |
| + 家族专家 | 0.4773 |
| + 正字法护盾 | 0.4621 |
| **+ PRISM** | **0.6061** |

其他测试指标：

| 指标 | 值 |
|---|---:|
| balanced_accuracy | **0.6061** |
| macro-F1 | 0.5998 |
| accuracy | 0.6061 |
| top-2 accuracy | 0.6667 |
| ECE | 0.0559 |
| 聚类 pair precision | 0.9199 |
| 聚类 pair recall | 0.9210 |
| 聚类 pair F1 | 0.9204 |

逐类召回两极（test，n=22/类）：gemini-3.1-pro-preview 1.000、claude-opus-5 0.955、gpt-6-astra 0.818、claude-opus-4-6 0.864 表现好；claude-fable-5 0.182、glm-5.3/glm-5.3-flash 0.273、gpt-5.6-terra 0.409 偏弱。

**结论**：加入 Qwen3-8B 探针把最终测试 Balanced Accuracy 从 0.5152 提到 **0.6061（+0.091）**，已逼近公开 Leaderboard 第一（≈0.622）区间；PRISM 仍是最大增益模块（测试 +0.144）。

---

## 8. 结构消融：PRISM 到底贡献多少

OOF 伪测试池（从 held-out prompt 构造每 prompt k 段）：

| 策略 | k=8 | k=12 | k=18 |
|---|---:|---:|---:|
| argmax | 0.3494 | 0.3578 | 0.3539 |
| Sinkhorn（α=0.25） | 0.3495 | 0.3586 | 0.3549 |
| PRISM 硬指派 | 0.3630 | 0.3995 | 0.4446 |
| **PRISM 松弛（λ=2）** | 0.3634 | 0.4006 | **0.4478** |

- k=18（完整簇）：**+0.094**，相对提升 26.5%；
- k=12：+0.043；k=8：+0.014；
- Sinkhorn 几乎无增益（+0.001），说明**全局边际平衡远不如簇内互斥约束**。

测试集上 PRISM 贡献 +0.106，与 OOF 模拟方向一致（测试集恰好是满覆盖的 22×18）。

---

## 9. 误差分析

### 9.1 家族级 vs 模型级

| 层级 | 准确率 |
|---|---:|
| 家族（6 类） | **0.765** |
| 模型（18 类） | 0.515 |

即约 76.5% 的段落能认对实验室，但其中只有约 2/3 能进一步认对具体型号——**主要难度在家族内部的兄弟模型**。

### 9.2 最易混淆的模型对（测试集）

| 次数 | 真实 | 误判为 |
|---:|---|---|
| 5 | gpt-5.6-terra | gpt-5.6-sol |
| 5 | gpt-5.6-sol | gpt-5.6-terra |
| 5 | gemini-3.8-flash | gemini-3.7-flash |
| 5 | claude-fable-5 | claude-sonnet-5 |
| 4 | kimi-k3 | glm-5.3 |

前四对全部是**同家族兄弟**，第五对是跨家族（Moonshot → ZAI），说明 kimi-k3 的语料在本数据里与 GLM 有重叠。

### 9.3 逐类召回（测试集，n=22/类）

| 较强 | 召回 | 较弱 | 召回 |
|---|---:|---|---:|
| gemini-3.1-pro-preview | 0.864 | claude-fable-5 | 0.136 |
| claude-opus-5 | 0.773 | gpt-5.6-terra | 0.273 |
| gpt-5.5 | 0.727 | glm-5.3 | 0.273 |
| claude-opus-4-6 | 0.727 | glm-5.3-flash | 0.318 |
| gpt-6-astra | 0.727 | claude-sonnet-5 | 0.364 |

`claude-fable-5` 只有 0.136：它与其他 Claude 的差异最小（同一家族、同一长度区间），且训练集中它少 1 条。

---

## 10. 复现命令

```bash
cd D:\research\projects\ntl-pact18

# 0) 环境（GPU）
.venv\Scripts\python run.py device          # 应显示 cuda (RTX 4060)

# 1) 切分
.venv\Scripts\python run.py --config configs/real.yaml split --source train.csv

# 2) 训练（5 折 OOF + 全量重训）
.venv\Scripts\python run.py --config configs/real.yaml train --model-out artifacts/real/model_gpu.joblib

# 3) 验证集：选 λ
.venv\Scripts\python run.py --config configs/real.yaml --set prism.lam=2.0 predict `
    --model artifacts/real/model_gpu.joblib --test-file data/real/val.csv `
    --truth-file data/real/val_truth.csv --out artifacts/real/val_pred.csv

# 4) 测试集：最终评估
.venv\Scripts\python run.py --config configs/real.yaml --set prism.lam=2.0 predict `
    --model artifacts/real/model_gpu.joblib --test-file data/real/test.csv `
    --truth-file data/real/test_truth.csv --out artifacts/real/test_pred.csv

# 5) 结构消融
.venv\Scripts\python run.py --config configs/real.yaml --set prism.lam=2.0 simulate `
    --model artifacts/real/model_gpu.joblib --k 8 12 18
```

随机性：`project.seed=42`（数据切分、聚类校准、融合初始化）；XGBoost/LightGBM 的 `random_state=0`；torch 手动播种。同配置重复训练，OOF 复现到小数点后 4 位。

---

## 11. 局限与下一步

1. **成员太轻**。公开最强 notebook 的 0.559 依赖 LoRA24、UltraBERT、token-attention 等神经成员；本项目刻意保持零重依赖，单成员上限只有 ~0.40。
2. **测试集只有 396 行**，balanced accuracy 的标准误约 ±0.03，λ 的选择（0.457 vs 0.434）在统计上不显著。
3. **`torch_linear` 未调优**：损失在 0.28 附近早停，说明学习率/正则还可再调；当前配置默认关闭。
4. **下一步优先级**：
   - ~~接入 Qwen3-8B 探针~~（**已完成**：4-bit NF4 + 第 18 层均值池化 + LR 探针，测试 +0.091；下一步可升级为 QLoRA 微调头，预计再 +0.05~0.10）；
   - 用 GPU 训练真正的双塔风格嵌入（当前学习式聚类在真实数据上过拟合）；
   - 家族专家改为用成员原始特征训练，而不是用融合概率 stacking；
   - 在 4000 条隐藏测试规模上复核 PRISM 的 λ；
   - 针对弱类（claude-fable-5、glm-5.3/flash、gpt-5.6-terra）做类平衡 / 过采样。
