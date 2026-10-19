# ntl-pact18

**PACT-18** — *Prompt-Aware Constrained aTtribution*：面向「Name That LLM」这类 **18 类 LLM 文本作者归属**任务的完整实现。

给定一段 40–250 词的英文段落，判断它由 18 个前沿模型中的哪一个生成。项目的核心主张是：**除了把每段文本独立分类，还要利用数据的生成结构**——同一 prompt 下每个模型至多出现一次，因此在推断出的 prompt 簇内可以做带容量约束的联合解码（PRISM）。

```
文本 → 风格不变表征 → 多成员对数概率融合 → 学习式 prompt 聚类
     → 家族专家 → 正字法护盾 → PRISM 簇内指派 → 18 类标签
```

- 目标指标：balanced accuracy（每类召回的平均）
- 公开参照：Code 区最强 notebook 0.559，Leaderboard 第一 0.622
- 本项目定位：在公开融合栈之上，用结构约束与更好的聚类把上限推高

---

## 1. 快速开始

```bash
cd D:\research\projects\ntl-pact18

# 1) 核心依赖（numpy/pandas/scipy/sklearn/pyyaml/joblib/pyarrow）
python -m pip install -r requirements.txt
#    需要 Qwen 成员（probe / QLoRA）时再加 GPU 栈：
#    python -m pip install -r requirements-gpu.txt
#    Kaggle 镜像缺 bitsandbytes，离线安装方式见 docs/KAGGLE_RUNBOOK.md

# 2) 端到端自检：生成合成数据 → 训练 → 预测 → 结构消融
python run.py smoke
```

`smoke` 全程约 2–4 分钟（笔记本 CPU），会打印：

- 每个成员的 OOF 表现与融合权重、温度
- 自动发现的护盾规则
- 分阶段 balanced accuracy（blend → em → experts → shield → prism）
- 伪测试池上的结构消融（argmax / Sinkhorn / PRISM-hard / PRISM-relaxed）

产物写入 `artifacts/`：`model.joblib`、`submission.csv`、`smoke_report.json`。

> 不需要安装包：`run.py` 会把 `src/` 加入 `sys.path`。也可以用 `pip install -e .` 后直接调用 `ntl-pact18` 命令。

---

## 2. 用真实赛题数据跑

Kaggle 数据需要先下载（`train.csv` / `test.csv`），放到 `data/` 下：

```
data/
├── train.csv          # id, prompt_id, text, model
├── test.csv           # id, text
└── test_truth.csv     # 可选：id, model, prompt_id（仅用于本地评估）
```

```bash
python run.py train                     # 训练并保存 artifacts/model.joblib
python run.py predict --out submission.csv
python run.py simulate --k 4 8 12 18    # 结构消融
```

比赛要求 `submission.parquet` 且**离线运行**，转换一下即可：

```python
import pandas as pd
pd.read_csv("submission.csv").to_parquet("submission.parquet", index=False)
```

`predict` 会读取 `test_truth.csv`（如果存在）打印分阶段指标与逐类召回；不存在时只写提交文件。

### 真实数据训练结果（2,699 行，prompt 分组切分 106/22/22）

| 阶段 | 验证集（396 行） | 测试集（396 行） |
|---|---:|---:|
| 融合 | 0.4040 | 0.3990 |
| + 家族专家 | 0.4091 | 0.4091 |
| + 正字法护盾 | 0.4091 | 0.4091 |
| **+ PRISM（λ=2）** | **0.4571** | **0.5152** |

结构消融（OOF 伪测试池，k=18）：argmax 0.3539 → PRISM 0.4478（**+0.094**）；Sinkhorn 仅 +0.001。

完整的数据分析、超参数、误差分析与复现命令见 **[`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md)**。

---

## 3. 目录结构

```
ntl-pact18/
├── run.py                     # 免安装入口
├── pyproject.toml             # 包元数据与可选依赖
├── requirements.txt
├── configs/
│   ├── default.yaml           # 全量配置
│   └── smoke.yaml             # 快速自检配置
├── src/ntl_pact18/
│   ├── config.py              # 分层配置（YAML + --set + NTL_* 环境变量）
│   ├── labels.py              # 18 个模型与 6 个实验室
│   ├── data.py                # 加载/校验 + 合成数据生成器
│   ├── features/
│   │   ├── stylegram.py       # char/word/shape/function n-gram + LinearSVC（含中心化孪生）
│   │   ├── stylometry.py      # 360 维风格统计 + 梯度提升
│   │   └── ngram_lm.py        # LLMDet 式代理困惑度成员
│   ├── members.py             # 成员注册 + 官方 starter 基线
│   ├── calibration.py         # 温度缩放 / Sinkhorn / SLD-EM / ECE
│   ├── blend.py               # 对数概率融合，L-BFGS 拟合权重
│   ├── clustering.py          # TF-IDF 聚类 + 学习式对角度量聚类
│   ├── prism.py               # 容量约束松弛指派（核心）
│   ├── experts.py             # 6 家族两级专家（自适应 ρ）
│   ├── shield.py              # 正字法贝叶斯护盾（规则自动发现）
│   ├── pipeline.py            # 端到端编排、持久化
│   ├── evaluate.py            # 指标、聚类 pair 指标、prompt 级 bootstrap
│   ├── simulate.py            # 伪测试池 + 噪声注入的结构消融
│   └── cli.py                 # 命令行
├── tests/                     # pytest（PRISM/校准/指标/聚类/端到端）
├── docs/DESIGN.md             # 设计说明与推导
├── data/                      # 数据（gitignore）
└── artifacts/                 # 模型与报告（gitignore）
```

---

## 4. 命令参考

| 命令 | 作用 |
|---|---|
| `python run.py synth` | 生成合成语料（默认 150 train prompts / 60 test prompts） |
| `python run.py train` | 训练全部成员 + 校准 + 融合 + 专家 + 护盾 + 聚类，保存模型 |
| `python run.py predict` | 预测并写 `submission.csv`，有 truth 时打印分阶段指标 |
| `python run.py simulate` | 在 OOF 伪测试池上比较 argmax / Sinkhorn / PRISM |
| `python run.py smoke` | 上述全流程的自动自检 |
| `python run.py version` | 版本号 |

公共参数：

```bash
--config configs/default.yaml          # 也可以只写 default.yaml
--set prism.lam=1.5 clustering.method=tfidf experts.enabled=false
```

环境变量覆盖（优先级最高）：

```bash
set NTL_PRISM__LAM=1.5
set NTL_EXPERTS_ENABLED=false
```

---

## 5. 关键配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| `features.stylegram.center_mode` | `replace` | 簇内用中心化孪生替换普通头；`add` 为相加，`off` 关闭 |
| `clustering.method` | `learned` | `learned` 为学习式对角度量，`tfidf` 为公开基线 |
| `clustering.threshold` | `0.90` | 平均连接聚类的距离阈值（中位数归一化后） |
| `prism.mode` | `relaxed` | `relaxed` 允许重复但罚 λ；`hard` 强制每类恰好一次 |
| `prism.lam` | `1.0` | 重复槽惩罚（nats）。越小越保守，越大越依赖聚类纯度 |
| `prism.confidence_threshold` | `0.0` | 簇内平均相似度低于该值时跳过指派 |
| `calibration.em_prior` | `true` | SLD-EM 估计测试集类别先验 |
| `experts.enabled` | `true` | 6 家族两级专家 |
| `shield.penalty` | `2.5` | 违反正字法硬规则时的对数惩罚 |

**调参优先级**：先定 `clustering.threshold`（用 `simulate` 看 pair F1），再定 `prism.lam`（看含噪消融），最后动成员。

---

## 6. 输出与产物

`artifacts/train_summary.json` 记录成员、权重、温度、护盾规则与 OOF 指标：

```json
{
  "members": ["stylegram", "stylometry", "ngram_lm", "linear_baseline"],
  "weights": {"stylegram": 0.27, "stylometry": 0.30, "ngram_lm": 0.17, "linear_baseline": 0.27},
  "temperatures": {"stylegram": 0.09, "...": 1.0},
  "shield_rules": ["curly_quote -> anthropic (rate=0.0000, n=300)"],
  "oof_summary": {"oof_blend_balanced_accuracy": 0.62}
}
```

`artifacts/smoke_report.json` 额外包含结构消融表。

---

## 7. 常见问题

**Q：没有 Kaggle 数据能跑吗？**
能。`synth` / `smoke` 会生成结构与真实数据同构的合成语料（prompt 网格、未见 prompt、部分覆盖、正字法硬规则），足以验证全链路。

**Q：lightgbm 没装会怎样？**
自动降级到 `sklearn.ensemble.HistGradientBoostingClassifier`，`train_summary.json` 里 `stylometry.backend_used` 会显示 `sklearn-hist`。

**Q：PRISM 会不会因为聚类错误反而掉分？**
会，所以默认是 `relaxed` + `lam=1.0`，并支持 `confidence_threshold` 门控。用 `simulate --merge-rate 0.1` 做压力测试再决定是否开 `hard`。

**Q：训练多久？**
`smoke` 配置约 2–4 分钟；默认配置（150 prompts、2700 条、全特征）约 10–20 分钟（CPU），主要花在 stylegram 的 4 折 OOF。

**Q：为什么四个 LM 头的普通权重是 0？**
这是公开最强栈的做法：冻结 LM 头只在簇内以 prompt 中心化的孪生形式贡献证据，单条段落不参与。本项目的 `stylegram.center_mode` 保留了同样的开关。

**Q：`test.csv` 没有 `prompt_id`，怎么聚类？**
只能从文本推断。`clustering.py` 的学习式聚类器就是用训练集的 `prompt_id` 监督出来的「同题」判别器。

**Q：为什么明明有 GPU，日志却显示 `device=cpu`？**
先跑 `python run.py device` 看 `fallback reason`。最常见的原因是装的是 CPU-only 版 torch（版本号带 `+cpu`），换装 CUDA 轮子即可，见第 8 节。

---

## 8. 设备选择（GPU / MPS / CPU）

### 优先级

```
device.preference=auto  →  CUDA  →  MPS  →  CPU
```

1. **CUDA**：`torch.cuda.is_available()` 为真时选 `cuda:{gpu_index}`，并打印显卡型号与显存；
2. **MPS**：Apple Silicon 且 `torch.backends.mps.is_available()` 为真时选 `mps`；
3. **CPU**：以上都不可用，或显式 `preference=cpu`。

检测逻辑在 `src/ntl_pact18/device.py`，不依赖 torch 也能工作：即使 torch 导入失败，也会通过 `nvidia-smi` 报告物理显卡是否存在，避免"明明有卡却不知道为什么没用上"。

### 回退策略

| 情况 | 行为 |
|---|---|
| torch 未安装 / 导入失败 | 回退 CPU，`fallback_reason` 写明 `torch is not importable` |
| torch 是 CPU-only 构建（如 `2.x+cpu`） | 回退 CPU，并注明"检测到 NVIDIA GPU，但 torch 是 CPU 版本" |
| `preference=cuda` 但 CUDA 不可用 | 回退 CPU，注明是哪一步失败 |
| `device.require_gpu=true` 且无加速器 | **直接抛错并终止**，绝不静默降级 |

训练开始时会把完整设备报告写进日志和 `artifacts/*/train_summary.json` 的 `device` 字段。

### 相关参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `device.preference` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `device.gpu_index` | `0` | 多卡时选择哪一张 |
| `device.require_gpu` | `false` | `true` 时无加速器直接报错 |
| `features.torch_linear.enabled` | `auto` | `auto` = 仅在有加速器时启用 GPU 成员 |
| `features.torch_linear.epochs` | `300` | Adam 全批迭代步数 |
| `features.torch_linear.lr` | `0.05` | 学习率 |
| `features.stylometry.backend` | `auto` | 装了 LightGBM 且有加速器时自动用 `device_type=gpu` |

### 哪些计算真的能用 GPU

| 组件 | 能否用 GPU | 说明 |
|---|---|---|
| `torch_linear` 成员 | ✅ CUDA / MPS | 稀疏 n-gram 特征上的线性 softmax，Adam 训练 |
| `stylometry`（LightGBM 后端） | ✅ GPU（需 LightGBM GPU 构建） | `device_type='gpu'` |
| `stylegram`（TF-IDF + LinearSVC） | ❌ | sklearn 只支持 CPU；要 GPU 需换 RAPIDS cuML |
| `ngram_lm`、聚类、PRISM | ❌ | 稀疏统计与组合优化，CPU 已足够快（毫秒级） |

### 本机实测（RTX 4060 Laptop）

**重装前**（系统 anaconda，torch 是 CPU 版）：

```
selected device : cpu (cpu)
torch           : 2.13.0+cpu
fallback reason : torch 2.13.0+cpu is a CPU-only build although an NVIDIA GPU is present
```

**重装后**（项目内 `.venv`）：

```
selected device : cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
torch           : 2.14.1+cu126 (cuda build 12.6)
backends        : torch=yes, torch_cuda=yes, torch_mps=no, cuml=no, lightgbm=no, xgboost=yes
accelerator in use : True
torch_linear member: enabled
stylometry backend : xgboost-gpu (device=cuda)
```

安装步骤（已在本机执行，C 盘只剩 15GB，所以 venv 与临时目录都放 D 盘）：

```bash
cd D:\research\projects\ntl-pact18
set TMPDIR=%CD%\.tmp & set TMP=%CD%\.tmp & set TEMP=%CD%\.tmp
python -m venv .venv
.venv\Scripts\python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install --no-cache-dir numpy pandas scipy scikit-learn pyyaml joblib pytest xgboost
.venv\Scripts\python -m pip install --no-cache-dir -e .
.venv\Scripts\python run.py device
```

之后所有命令把 `python` 换成 `.venv\Scripts\python` 即可。`features.torch_linear.enabled=auto` 会自动把 GPU 成员加进融合，`stylometry.backend=auto` 会自动切到 `xgboost-gpu`，无需改配置。

> 想彻底回到 CPU：`python run.py --set device.preference=cpu train`（或直接改用系统 python）。

## 9. 冒烟测试实测输出

`python run.py smoke` 在本机（Python 3.13 / CPU）的实际结果：

```
[fit] oof blended balanced acc = 0.5694
[fit] weights: stylegram=0.229, stylometry=0.357, ngram_lm=0.143, linear_baseline=0.271
[fit] temperatures: stylegram=0.15, stylometry=1.32, ngram_lm=0.56, linear_baseline=0.32
[fit] shield rules: curly_quote -> anthropic | curly_quote -> zai | double_em_dash -> google
```

分阶段（合成测试集 365 条，聚类 pair precision 1.000 / recall 0.975）：

| stage | balanced accuracy |
|---|---:|
| blend | 0.5431 |
| em（默认关闭） | 0.5431 |
| experts | 0.5720 |
| shield | 0.5720 |
| **prism** | **0.5898** |

结构消融（OOF 伪测试池，每 prompt k 段）：

```
strategy            k=4      k=8     k=12     k=18
argmax           0.5931   0.5854   0.5807   0.5694
sinkhorn         0.5867   0.5826   0.5819   0.5704
prism_hard       0.6014   0.5937   0.6277   0.6565
prism_relaxed    0.5988   0.5934   0.6166   0.6380
```

两点解读：

1. **k=18 时 PRISM 比 argmax 高 0.07–0.09**，k 越小增益越小（k=4 几乎为 0）——这与设计预期一致：约束只在簇足够完整时才有信息量。
2. **Sinkhorn 几乎无增益**，说明全局边际平衡确实不如簇内互斥约束。

> ⚠️ 合成语料只复现结构性质与风格差异，**绝对分数没有意义**（不代表真实赛题的 0.59 水平），只有「相对消融」和「链路正确性」有意义。真实数据上的预期见 `docs/DESIGN.md`。

## 10. 引用与致谢
设计思路综合了 Kaggle 社区公开方案（Lucas Boesen 的融合底座、OBSTACLE DEVELOPER 的中心化 stylegram、kirilltym 的家族门、Zbigniew Kirchner 的 Claude 头）与以下论文：

- Huang, Chen, Shu. *Authorship Attribution in the Era of LLMs*. SIGKDD Explorations 2024.
- Wu et al. *LLMDet: A Third Party LLMs Generated Text Detection Tool*. EMNLP 2023 Findings.
- Ko et al. *Who's Behind the Text? Robust Model Authorship Attribution* (ATLAS). arXiv:2610.19583.
- Patel et al. *StyleDistance: Stronger Content-Independent Style Embeddings*. NAACL 2025.
- Rivera-Soto et al. *Learning Universal Authorship Representations* (LUAR). EMNLP 2021.
- *MultiGhostBench*. arXiv:2609.02379.

详见 `docs/DESIGN.md`。
