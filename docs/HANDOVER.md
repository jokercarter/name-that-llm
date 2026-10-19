# PACT-18 项目交接文档（Handover）

> 项目路径：`D:\research\projects\ntl-pact18`
> 编写时间：2026-10-18 ｜ 编写人：AI 协作助手
> 适用对象：接手本项目继续训练 / 调参 / 提交的另一位 AI（或人类）
> 目标：读完本文即可独立复现、继续训练、评估与提交，无需追问。

---

## 一、任务背景（Task Background）

**任务**：Kaggle「Name That LLM」风格的 **18 类 LLM 文本作者归属（Authorship Attribution）**。
- **输入**：一段英文段落（36–274 词，中位数 101 词），来自某个 LLM 在某个 prompt 下的输出。
- **输出**：判定它属于 18 个候选模型中的哪一个。
- **评估指标**：**Balanced Accuracy**（18 类各自召回率的算术平均，赛题一致）。随机基线 = 1/18 ≈ 0.056。
- **关键结构约束**：训练集是 `18 × 150` 的近似完整网格——**同一 prompt 下每个模型至多出现一次**。这是一个簇内互斥约束，比全局边际平衡（Sinkhorn）强得多，是本方案的核心差异化来源。

**项目主张（PACT-18 = Prompt-Aware Constrained aTtribution）**：不把每段当独立样本分类，而是先推断 prompt 簇，再在簇内做带容量约束的联合解码（PRISM）。
公开参照：Code 区最强 notebook ≈ 0.559，Leaderboard 第一 ≈ 0.622。本项目定位是用结构约束 + 更好的聚类把上限推高。

**核心推理链**：
```
文本 → 风格不变表征 → 多成员对数概率融合 → 学习式 prompt 聚类
     → 家族专家 → 正字法护盾 → PRISM 簇内指派 → 18 类标签
```

---

## 二、当前进度（Current Progress）

### 2.1 数据与切分（已完成）
| 集合 | prompt 数 | 行数 | 用途 |
|---|---:|---:|---|
| 全量 `train.csv` | 150 | 2,699 | 原始赛题训练集（含 `claude-fable-5` 少 1 条 = 149） |
| `data/real/train.csv` | 106 | 1,907 | 训练成员 + 5 折 OOF |
| `data/real/val.csv` | 22 | 396 | 选 PRISM λ、看分阶段增益 |
| `data/real/test.csv` | 22 | 396 | **最终评估，全程不参与任何选择** |
| `data/real/trainval.csv` | 128 | 2,303 | train+val 合并，用于产出最终提交模型 |

- 切分按 **prompt 分组**（GroupKFold / prompt 不交叉），train/val/test 的 prompt 交集均为 0（见 `data/real/split_meta.json`）。
- EDA 结论（见 `docs/TRAINING_REPORT.md` §1）：无需清洗；刻意保留标点/大小写（指纹本身）；同 prompt 段落余弦相似度 0.214，跨 prompt 0.011（聚类可行）。

### 2.2 已完成的训练结果（不含 Qwen 成员）
| 配置 | 成员数 | OOF | 测试 blend | 测试 PRISM(λ=2) |
|---|---:|---:|---:|---:|
| CPU（HistGB） | 4 | 0.3606 | 0.4268 | **0.5177** |
| GPU（XGBoost-GPU + torch_linear） | 5 | 0.3539 | 0.3990 | 0.5152 |

- **PRISM 是最大赢家**：测试集上 blend→PRISM 贡献 **+0.106**；OOF k=18 伪池 argmax 0.3539 → PRISM 0.4478（+0.094）。
- `torch_linear`（GPU 成员）稀释融合（占 24% 权重但更弱），故 `configs/real.yaml` 中默认 `enabled: false`。
- 结构消融、误差分析、逐类召回、最易混淆对均已写入 `docs/TRAINING_REPORT.md` §7–§9。

### 2.3 含 Qwen 成员的训练（★ 已完成，含测试集实测）
- **训练已完成**：`artifacts/real/model_qwen.joblib` 已成功保存（GPU 全栈 + **Qwen3-8B-Base probe 成员**），训练于 `data/real/trainval.csv`（2303 行 / 128 prompts）。
- **关键修复**：`QwenMember` 此前因持有 `torch` 模块对象导致 `joblib.dump` 报 `cannot pickle 'module' object`；已通过 `__getstate__/__setstate__` 剥离不可序列化引用修复（见 §4.5.3）。
- **训练结果（OOF，2303 行 5 折）**：
  - **融合后 Balanced Accuracy = 0.4211**（不含 Qwen 时为 0.3539，提升 +0.067）
  - **融合权重**：`qwen_probe=0.286`（居首）、`stylegram=0.245`、`stylometry=0.206`、`ngram_lm=0.116`、`linear_baseline=0.147`
  - **温度**：qwen_probe=0.52、stylegram=0.14、stylometry=1.19、ngram_lm=0.51、linear_baseline=0.33
  - **护盾规则**：`curly_quote -> anthropic`、`curly_quote -> zai`、`double_em_dash -> google`
- **实测分阶段 Balanced Accuracy（已完成，2026-10-18 后台评估任务 `VfXZOV`）**：
  | 评估集 | blend | em | experts | shield | **prism（最终）** |
  |---|---:|---:|---:|---:|---:|
  | `val.csv`（泄漏，勿报） | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
  | `test.csv`（留出，22 prompt） | 0.4773 | 0.4773 | 0.4621 | 0.4621 | **0.6061** |
  - **test 最终指标**：balanced_accuracy **0.6061**、macro_f1 **0.5998**、accuracy **0.6061**、top2_accuracy **0.6667**、ECE **0.0559**。
  - **亮点**：PRISM 在测试集把 0.4621→0.6061（**+0.144**），是最大增益模块；护盾/专家在 test 上未额外增益（簇相似度门控触发）。
  - **逐类召回（test）两极**：gemini-3.1-pro-preview=1.000、claude-opus-5=0.955、gpt-6-astra=0.818、claude-opus-4-6=0.864 表现好；claude-fable-5=0.182、glm-5.3/glm-5.3-flash=0.273、gpt-5.6-terra=0.409 偏弱（少数族/混淆严重）。
- ⚠️ **重要：本模型训练于 `trainval.csv`（train+val 合并），因此 `val.csv` 是样本内（in-sample）数据，其评估分数（实测为 1.0000）是泄漏结果，不可当作真实性能报告**。`data/real/test.csv`（22 个与训练完全不重叠的 prompt）才是唯一真正的留出评估集。汇报时只报 test 指标（balanced_accuracy=0.6061）。若想用 val 做选型，请改用 `train.csv`（1907 行）训练，或在 `split` 时把 val/test 都留出（但本仓库默认 `trainval` 已含 val）。

---

## 三、文件结构与依赖关系（File Structure & Dependencies）

### 3.1 目录树
```
ntl-pact18/
├── run.py                      # 免安装入口；把 src/ 加入 sys.path，调用 cli.main
├── pyproject.toml             # 包元数据 + 可选依赖（boost/gpu/dev）
├── requirements.txt           # 运行时依赖（numpy/pandas/scipy/sklearn/pyyaml/joblib）
├── README.md                  # 项目总说明（快速开始、命令、设备、FAQ）
├── docs/
│   ├── DESIGN.md              # 设计推导（PRISM 模拟、为何用结构而非更大模型）
│   ├── TRAINING_REPORT.md     # 训练报告（EDA/切分/设备/超参/结果/消融/误差/复现）
│   └── HANDOVER.md            # 本文档
├── configs/
│   ├── default.yaml           # 全量默认配置（含 features.qwen / torch_linear 段）
│   ├── real.yaml              # 真实数据配置（覆盖 default；torch_linear 关闭）
│   └── smoke.yaml             # 合成数据快速自检配置
├── src/ntl_pact18/
│   ├── __init__.py            # 版本号 + 导出 LABELS/FAMILIES
│   ├── labels.py              # 18 个模型标签 + 6 个实验室家族映射（全局常量）
│   ├── config.py             # 分层配置：YAML + --set + NTL_* 环境变量（加载/合并/覆盖）
│   ├── data.py               # 数据加载/校验 + 合成语料生成器
│   ├── device.py             # 设备检测 CUDA→MPS→CPU + require_gpu 策略
│   ├── utils.py              # 通用工具：seed_everything / normalize_rows / softmax / encode_groups
│   ├── members.py            # 成员注册表 build_members / fit_members / precompute_members
│   ├── pipeline.py           # 端到端编排（fit / predict / save / load）
│   ├── blend.py              # 对数概率加权融合（L-BFGS 拟合权重）
│   ├── calibration.py        # 温度缩放 / Sinkhorn / SLD-EM / ECE
│   ├── clustering.py         # TF-IDF 聚类 + 学习式对角度量聚类
│   ├── prism.py              # ★ PRISM 容量约束松弛指派（核心算法）
│   ├── experts.py            # 6 家族两级专家（自适应 ρ）
│   ├── shield.py             # 正字法贝叶斯护盾（规则自动发现）
│   ├── evaluate.py           # 指标（balanced_acc / macro_f1 / top2 / ECE / 聚类 pair / bootstrap）
│   ├── simulate.py           # 伪测试池 + 噪声注入的结构消融
│   ├── cli.py                # 命令行：synth/split/train/predict/simulate/smoke/device/version
│   └── features/
│       ├── __init__.py       # 导出 StylegramMember / StylometryMember / NgramLMMember
│       ├── stylegram.py      # ★ 最强单成员：4路 TF-IDF + LinearSVC + 簇内中心化孪生
│       ├── stylometry.py     # 360 维风格统计 + 梯度提升（XGBoost-GPU 优先）
│       ├── ngram_lm.py       # LLMDet 式代理困惑度成员
│       ├── torch_linear.py   # GPU 成员：稀疏 n-gram 特征上线性 softmax，Adam 训练（默认关）
│       └── qwen_member.py    # ★ 新增：Qwen3-8B-Base 4-bit 探针 / QLoRA 成员（GPU）
├── tests/                    # pytest：prism / calibration / metrics / clustering / pipeline smoke
├── scripts/check_env.py      # 依赖检查脚本
├── data/                     # 数据（gitignore；真实数据放 data/real/）
└── artifacts/                # 模型与报告（gitignore）
    └── real/                 # model_*.joblib, *_pred.csv, train_summary.json, split_meta.json
```

### 3.2 模块依赖图（谁 import 谁）
```
run.py ──> cli.py
cli.py ──> config, data, evaluate, labels, pipeline, simulate, utils
pipeline.py ──> blend, calibration, clustering, data, device, evaluate,
                experts, labels, members, prism, shield, utils
members.py ──> features.{stylegram, stylometry, ngram_lm, torch_linear, qwen_member},
               labels, device
features/stylegram.py  ──> labels, utils
features/stylometry.py ──> labels
features/ngram_lm.py   ──> labels, utils
features/torch_linear.py ──> device, labels, utils, features.stylegram(shape_text/function_text)
features/qwen_member.py  ──> device, labels, utils
blend.py          ──> utils
calibration.py    ──> utils
clustering.py     ──> utils
experts.py        ──> labels, utils
shield.py         ──> labels, utils
prism.py          ──> labels, utils
evaluate.py       ──> calibration, labels
simulate.py       ──> calibration, evaluate, prism
config.py         ──> (仅依赖 yaml)
data.py           ──> labels, utils
device.py / utils.py / labels.py ──> 无内部依赖（叶子节点）
```
**配置优先级（低→高）**：`configs/default.yaml` < `--config` 指定文件 < `--set k=v` < `NTL_*` 环境变量。
**注意**：`real.yaml` 没有 `features.qwen` 段，因此直接继承 `default.yaml` 的 `features.qwen.enabled: auto`；在 CUDA 环境下 `auto` ⇒ 启用 Qwen 成员（这正是当前训练的行为）。

---

## 四、训练方法细节（Training Method Details）

### 4.1 整体流程（`pipeline.PACT18Pipeline.fit`）
1. **设备检测**：`detect_device()` → CUDA→MPS→CPU；`enforce_policy()` 在 `require_gpu=true` 且无加速器时直接报错。
2. **按 prompt 分组 5 折**（`add_group_folds`，GroupKFold）。
3. **GPU 骨干预计算**（`precompute_members`）：Qwen 成员对全语料抽一次特征并缓存，随后 `release()` 释放显存给后续成员。
4. **5 折 OOF**：每折用训练子集 fit 所有成员 → 在验证子集预测 → 累积每成员的 OOF 概率矩阵。
5. **校准**：每成员单独拟合温度缩放 `T`（最小化 NLL）。
6. **融合**：`fit_blend_weights` 用 L-BFGS 在对数概率上拟合非负权重（softmax 参数化）。
7. **家族专家**（`FamilyExperts`）：6 家族两级，只重写已预测家族内部分布。
8. **正字法护盾**（`OrthographicShield`）：从训练集自动发现硬规则（弯引号→anthropic/zai，双长破折号→google），后置对数惩罚。
9. **聚类器**（`build_clusterer`，real 用 `tfidf`）：中位数归一化 TF-IDF 余弦 + average linkage，阈值在 held-out prompt 折上自动校准。
10. **全量重训**（`fit_members` on 全训练集）。
11. **保存**：`joblib.dump(self, path)`（注意 Qwen 成员的序列化见 §4.5）。

### 4.2 成员清单与关键参数
| 成员 | 特征 | 模型/头 | 关键参数 | 设备 |
|---|---|---|---|---|
| `stylegram` | char 3-5(120k)+word 1-3(90k)+shape 2-6(30k)+function 1-3(25k) TF-IDF（lowercase=False） | LinearSVC C=0.5 + 簇内中心化孪生 | `center_mode=replace`, `min_size=2`,`max_size=18` | CPU |
| `stylometry` | 360 维风格统计（句/词长、标点率、大小写、停用词频次…） | XGBoost-GPU（400 树, lr 0.04, max_leaf_nodes 15） ｜ 回退 LightGBM / HistGB | `backend=auto`, `use_gpu=设备是加速器` | **GPU（XGBoost）** |
| `ngram_lm` | 每类 add-α unigram+bigram LM 的 18 维交叉熵 + rank/margin/entropy | 多项逻辑回归 C=1 | `alpha=0.1`, `order=2` | CPU |
| `linear_baseline` | word 1-2 + char 3-5 TF-IDF | 逻辑回归 C=2 | — | CPU |
| `torch_linear` | 4 路 TF-IDF（较小 caps） | 线性 softmax + Adam | `lr=0.1`, `epochs=400`, `weight_decay=1e-4` | **GPU（cuBLAS 稠密）** |
| `qwen_probe` | Qwen3-8B-Base **layer 18** 隐状态均值池化（4096 维） | 逻辑回归探针 C=0.0025 | `mode=probe`, `max_len=384`, `batch_size=8`, `load_in_4bit=True` | **GPU（4-bit）** |

> 当前 `real.yaml` 启用的成员：stylegram + stylometry + ngram_lm + linear_baseline + **qwen_probe**（auto 在 CUDA 下启用）；torch_linear 默认关闭。

### 4.3 融合与后处理顺序（`pipeline.predict`）
```
成员概率 → 温度缩放(每成员 T) → 对数概率加权融合
         → [SLD-EM 先验校正，默认关] → 家族专家
         → 正字法护盾 → PRISM 簇内联合解码(最后一步，保证结构约束必成立)
```
- **PRISM（relaxed, λ=2.0）**：对大小 m 的簇，构造代价矩阵
  - 主槽 K=18 列：`cost[r,c] = -log p_r(c)`
  - 重复槽 K×K 列：`cost[r,c] = -log p_r(c mod K) + λ`
  - 匈牙利指派，`pred = 槽位 % K`。`λ=0`→近似 argmax；`λ→∞`→硬指派；`λ≈1~2` 为最优折中。
  - 只对 `2 ≤ m ≤ 18` 的簇启用；`confidence_threshold>0` 时簇内平均相似度过低则跳过。

### 4.4 设备策略（`device.py`）
- 优先级：`device.preference=auto` ⇒ **CUDA → MPS → CPU**。
- CUDA：`torch.cuda.is_available()` → `cuda:{gpu_index}`，打印显卡型号与显存。
- 回退时把**具体原因**写入日志（如「检测到 RTX 4060，但 torch 是 CPU 版」）。
- `device.require_gpu=true` 且无加速器 → **直接抛错**，绝不静默降级。
- **本机**：RTX 4060 Laptop 8GB，`torch 2.14.1+cu126`（项目内 `.venv`），XGBoost 3.4.1（device=cuda），transformers/peft/bitsandbytes（Qwen 4-bit）。

### 4.5 关键坑位与修复（必读，避免重蹈覆辙）
1. **torch CUDA 稀疏内核崩溃**：COO 与 CSR 稀疏矩阵乘在 ~105 步报 `cudaErrorIllegalAddress`。→ `torch_linear` 改用 **稠密 cuBLAS**（`_dense()`，~5.2 万维 ≈0.4GB），7 秒稳定。
2. **XGBoost 标签报错**：要求 0..K-1 整数标签 → `StylometryMember.fit` 内部手动 encode，保存 `classes_`，`predict_proba` 按位置解码。
3. **Qwen 成员无法 pickle（本次主要修复）**：`QwenMember` 持有 `torch` **模块对象**，`joblib.dump` 时报 `cannot pickle 'module' object`。→ 已加 `__getstate__/__setstate__`，序列化时剥离 `torch/model/tokenizer/device`，保留 `head`（sklearn LR）与配置；`predict()` 经 `load()` 惰性重载骨干。已用 round-trip 测试验证（`.tmp/test_qwen_pickle.py`）。
4. **跨环境 joblib 不兼容**：模型只能在训练环境（同 sklearn/torch 版本）加载；迁移机器需重训或对齐版本。
5. **临时目录权限**：曾因 `/d/tmp` 拒绝访问失败 → 已在命令中 `set TMPDIR=.tmp`（项目内目录）。

---

## 五、后续操作指引（Next Steps — 直接可执行）

> 所有命令在 `D:\research\projects\ntl-pact18` 下执行；GPU 环境用项目内 `.venv` 的 python。
> 若提示符为 PowerShell，`set` 改为 `$env:TMPDIR="..."`。

### 5.1 环境准备（仅首次 / 换机器）
```bash
cd D:\research\projects\ntl-pact18
set TMPDIR=%CD%\.tmp & set TMP=%CD%\.tmp & set TEMP=%CD%\.tmp
python -m venv .venv
.venv\Scripts\python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install --no-cache-dir numpy pandas scipy scikit-learn pyyaml joblib pytest xgboost transformers peft bitsandbytes
.venv\Scripts\python -m pip install --no-cache-dir -e .
.venv\Scripts\python run.py device        # 应显示 cuda (RTX 4060)
```

### 5.2 复现训练（按 prompt 分组切分 → 训练 → 保存）
```bash
cd D:\research\projects\ntl-pact18
set TMPDIR=%CD%\.tmp

# 0) 切分（已有 data/real/* 可跳过；重新切分需 --force）
.venv\Scripts\python run.py --config configs/real.yaml split --source train.csv

# 1) 训练（全栈 GPU + Qwen 探针，训练于 trainval 全量，输出最终模型）
.venv\Scripts\python run.py --config configs/real.yaml train ^
    --train-file data/real/trainval.csv ^
    --model-out artifacts/real/model_qwen.joblib
#   注：默认 train 命令不加 --train-file 时用 data/real/train.csv（1907 行）
```

### 5.3 评估（★ 当前待执行，训练完成后马上跑）
```bash
# 验证集（选 λ，可对比 0.5/1.0/2.0/3.0）
.venv\Scripts\python run.py --config configs/real.yaml --set prism.lam=2.0 predict ^
    --model artifacts/real/model_qwen.joblib ^
    --test-file data/real/val.csv --truth-file data/real/val_truth.csv ^
    --out artifacts/real/val_pred_qwen.csv

# 测试集（最终评估，全程不参与选择）
.venv\Scripts\python run.py --config configs/real.yaml --set prism.lam=2.0 predict ^
    --model artifacts/real/model_qwen.joblib ^
    --test-file data/real/test.csv --truth-file data/real/test_truth.csv ^
    --out artifacts/real/test_pred_qwen.csv
#   输出会打印分阶段 Balanced Accuracy：blend → em → experts → shield → prism
```

### 5.4 结构消融（验证 PRISM 贡献）
```bash
.venv\Scripts\python run.py --config configs/real.yaml --set prism.lam=2.0 simulate ^
    --model artifacts/real/model_qwen.joblib --k 8 12 18
```

### 5.5 调参优先级（来自 README / DESIGN）
1. 先定 `clustering.threshold`（用 `simulate` 看 pair F1）。
2. 再定 `prism.lam`（看含噪消融：加 `--merge-rate 0.1`）。
3. 最后动成员。
- 开启/关闭 Qwen：`configs/default.yaml` 的 `features.qwen.enabled`（auto/true/false）。
- 开启/关闭 `torch_linear`：`features.torch_linear.enabled`（默认 false，因其稀释融合）。
- 强制用 GPU 否则报错：`--set device.require_gpu=true`。

### 5.6 常见命令速查
| 命令 | 作用 |
|---|---|
| `run.py device` | 查看设备检测与后端（qwen / torch_linear / stylometry） |
| `run.py smoke` | 合成数据端到端自检（2–4 分钟） |
| `run.py synth` | 生成合成语料到 `data/` |
| `run.py train` | 训练并保存 `artifacts/.../model.joblib` |
| `run.py predict` | 预测 + 写 submission.csv + 有 truth 时打印分阶段指标 |
| `run.py simulate` | OOF 伪测试池上 argmax/Sinkhorn/PRISM 对比 |

### 5.7 接手者必须知道的三件事
1. **测试集 prompt 与训练完全不重叠** → 必须按 prompt 分组切分，随机按行切分会让 CV 虚高。
2. **PRISM 依赖聚类质量** → 聚类错了硬指派会掉分，所以默认 `relaxed` + `confidence_threshold` 门控。
3. **模型不可跨 sklearn/torch 版本加载** → 评估/提交必须在训练用的 `.venv` 内进行；换机请重训或用 `pip install -e .` 对齐依赖。

---

## 六、附录：关键超参速查表
| 参数 | 值 | 说明 |
|---|---|---|
| `features.stylegram.C` | 0.5 | LinearSVC 正则 |
| `features.stylegram.center_mode` | replace | 簇内中心化孪生 |
| `features.stylometry.n_estimators` | 400 | XGBoost 树数 |
| `features.stylometry.learning_rate` | 0.04 | |
| `features.ngram_lm.alpha` | 0.1 | 语言模型平滑 |
| `features.qwen.C` | 0.0025 | 探针逻辑回归正则 |
| `features.qwen.layers` | [18] | 取第 18 层隐状态池化 |
| `features.qwen.max_len` | 384 | 截断长度 |
| `features.qwen.load_in_4bit` | true | 4-bit NF4 |
| `calibration.temperature` | true | 每成员单独 T |
| `calibration.em_prior` | false | SLD-EM 在本数据掉分，关 |
| `clustering.method` | tfidf | learned 在真实数据过拟合 |
| `prism.mode` | relaxed | |
| `prism.lam` | **2.0** | val 与 OOF 模拟一致最优 |
| `prism.min_size / max_size` | 2 / 18 | 单条与超大簇直接 argmax |
| `experts.min_rows` | 100 | 家族专家最小样本 |
| `shield.penalty` | 2.5 | 对数惩罚 |
| `device.preference` | auto | cuda → mps → cpu |

---
*本文档与 `docs/TRAINING_REPORT.md`、`README.md`、`docs/DESIGN.md` 互为补充；若三者冲突，以本文档的「当前进度」与 `train_summary.json` 实际产物为准。*
