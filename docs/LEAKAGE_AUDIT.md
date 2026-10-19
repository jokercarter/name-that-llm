# PACT-18 数据泄露审查与外部数据集评估

> 审查对象：`D:\research\projects\ntl-pact18`（commit 状态：2026-10-18 15:40 工作区）
> 审查方式：静态代码走查（逐行定位）+ 数据实测（pandas 全量比对）+ 训练/预测日志核对
> 审查范围：`data/real/*` 切分、`src/ntl_pact18/*` 全部拟合路径、`configs/*.yaml`
> 结论口径：只区分「是否把测试信息带入训练/选择」与「是否让报告分数失真」，不把统计上的乐观偏差一律称为泄露。

---

## 0. 结论摘要

| 级别 | 问题 | 位置 | 是否影响 test 分数 |
|---|---|---|---|
| **P1** | 训练/服务不一致：OOF 阶段 stylegram 只用 plain 头，测试阶段对合格簇切换为 centered 头 | `pipeline.py:97` vs `pipeline.py:192`；`stylegram.py:168-186` | 不泄露；实测 centered 使 blend +0.025、但 PRISM 增益由 +0.154 降到 +0.126，最终 0.6035 vs 0.6061（噪声内） |
| **P1** | `val.csv` 对最终模型是样本内数据，`val=1.0000` 属评估污染 | `HANDOVER.md §2.3`；训练输入 `trainval.csv` | 不影响 test，但任何引用 val 的结论无效 |
| **P1** | `prism.lam` 在 val 上选定，而最终模型训练集已含 val，λ 的「留出最优」不再成立 | `configs/real.yaml:76`；`docs/TRAINING_REPORT.md §7.2` | 对 test 无直接泄露，但 λ 选择失去了独立验证 |
| **P2** | 聚类器 TF-IDF 在全部训练文本上 fit（含用于校准阈值的 held-out prompt 折） | `clustering.py:100` | 无监督，不传标签；轻微 transductive 乐观 |
| **P2** | 护盾/专家在完整训练集（含 val）上拟合，OOF 指标略乐观 | `pipeline.py:119-131` | 不影响 test（test prompt 与训练不交） |
| **P2** | `GroupKFold` 未 shuffle，折与 prompt 顺序绑定 | `data.py:319-322` | 不泄露，影响 OOF 方差估计 |
| **P2** | `predict` 用命令行 cfg 覆盖训练 cfg，但成员对象不会重建 | `cli.py:133` | 改成员开关会造成静默不一致 |
| **P0** | 训练/验证/测试样本重叠、重复、近重复 | 实测 | **无** |
| **P0** | 标签作为特征进入模型 | 走查 | **无** |
| **P0** | 有监督预处理器在划分前拟合（Scaler/Imputer/Vectorizer） | 走查 + grep | **无** |

**总结论：不存在把测试标签或测试样本带入训练的 P0 级泄露；存在 3 个会让「报告分数」失真的 P1 级方法学问题，其中训练/服务不一致最值得优先修。**

---

## 1. 样本重叠与重复（实测）

数据文件：`data/real/{train,val,test,trainval}.csv`，真值：`{val,test}_truth.csv`，全量：`train.csv`。
实测脚本：`.tmp/audit_leak.py`（可复跑）。

| 检查项 | 结果 |
|---|---|
| `id` 交集 train∩val / train∩test / val∩test | 0 / 0 / 0 |
| `id` 交集 trainval∩test | 0 |
| 全量 `train.csv` 重复 id | 0 |
| `prompt_id` 交集 train∩val / train∩test / val∩test | 0 / 0 / 0（与 `split_meta.json` 一致） |
| 全量 `(prompt_id, model)` 重复 | 0 |
| 每 prompt 行数 | 17–18（仅 `claude-fable-5` 少 1 条） |
| 原文完全相同（跨 split） | 0 / 0 / 0 |
| 归一化后相同（小写、去标点、压空白） | 0 / 0 / 0 |
| 前 120 字符相同（跨 split） | 0 / 0 / 0 |
| 字符 5-gram Jaccard > 0.8 的跨 split 近重复 | test 396 行中 **0 行** |
| 缺失值 / 空文本 | 0 / 0 |

**判断依据**：切分在 prompt 层完成（`cli.py:53-70`），而不是在行层做随机划分。`prompts = prompts[rng.permutation(n)]` 之后才构造 `train_p / val_p / test_p`，因此同一 prompt 的 18 段必然落在同一侧；`split_meta.json` 记录的三组交集均为 0，与实测吻合。

**结论：通过。** 不存在样本级、prompt 级、文本级或近似重复级的重叠。

---

## 2. 标签泄露

| 检查点 | 代码位置 | 判断 |
|---|---|---|
| 标签是否作为特征列进入模型 | 全仓 grep `y`/`model` 特征构造 | 否。`text` 是唯一输入列，`model` 只在 `fit` 时作为目标 |
| 类别先验是否跨集使用 | `pipeline.py:75-76` 计算 `class_prior_` | 仅在 `calibration.em_prior=true` 时用于 SLD-EM；配置为 `false`（`real.yaml:57`），实际未启用 |
| 家族专家是否用标签构造特征 | `experts.py:36-48` | 输入是 OOF 概率 `probs`，标签只用于拟合；属标准 stacking |
| 护盾规则是否用测试标签 | `shield.py:55-78` | 规则在训练文本+训练标签上发现（弯引号→anthropic 等），推理时只看文本模式 |
| 测试真值是否参与任何拟合 | `test_truth.csv` 仅在 `cli.py:143-165` 读取用于打分 | 是，仅评估，不回流 |
| 是否存在「用 val 调参后又把 val 并入训练」 | `trainval.csv` 作为最终训练输入 | **是**：λ 在 val 上选定，随后 val 进入训练集。对 test 无泄露，但 val 分数彻底失效（实测 1.0000 即为此） |

**结论：无标签泄露。** 但「val 先选 λ、后并入训练」使 val 从验证集退化为训练集，这是报告口径问题，必须在任何汇报中剔除 val 指标。

---

## 3. 特征穿越（推理期信息进入特征）

本任务无时间维度，「穿越」的等价物是**用到了推理时不可获得的信息**。逐项核对：

| 机制 | 位置 | 是否穿越 | 判断依据 |
|---|---|---|---|
| Qwen 隐状态预计算 | `pipeline.py:84` → `qwen_member.py:134-150` | 否（但跨折共享缓存） | 冻结骨干、逐样本确定性前向，无任何参数拟合；缓存只避免重复计算，不携带标签 |
| 聚类阈值校准 | `pipeline.py:134-135` → `clustering.py:107-119` | 否 | 阈值在 3 个 held-out prompt 折上取中位数，明确排除了被校准的 prompt |
| 聚类 TF-IDF 向量器 | `clustering.py:99-100` | 轻微 transductive | IDF 在全部训练文本上估计（含被 hold-out 的 prompt 折），但无监督、不传标签；且推理期只 `transform`，不再 fit |
| 测试期聚类 | `pipeline.py:181` → `clustering.py:133-136` | 否（比赛允许的 batch 推理） | 对整批 test 文本做无监督聚类，不使用任何标签；Code Competition 一次性跑全量测试集，成立 |
| 测试期簇内中心化 | `pipeline.py:192` → `stylegram.py:168-186` | 否 | 减去的是**测试簇**的均值，属 transductive 特征，不含真值；但若测试被拆成单行运行，行为会退化（见 §5.2） |
| 用真实 `prompt_id` 训练 stylegram 中心化头 | `pipeline.py:96` → `stylegram.py:154-158` | 否 | `prompt_id` 是结构标签而非目标标签；且 GroupKFold 保证整簇在训练侧，中心化均值不含验证行 |

**结论：不存在标签意义上的特征穿越。** 存在两类 transductive 依赖（聚类、中心化），在「整批推理」前提下合法，但会削弱逐行可复现性。

---

## 4. 预处理拟合时点

| 预处理 | 位置 | 拟合数据 | 是否在划分前 |
|---|---|---|---|
| `fillna("")` | `data.py:290,300` | 无参数（常量替换） | 不适用，无泄露 |
| 归一化 / 标准化 | 全仓 grep：无 `StandardScaler` / `MinMaxScaler` / `Normalizer` | — | 不存在 |
| 缺失值填充模型 | 全仓 grep：无 `SimpleImputer` | — | 不存在 |
| stylegram 4 路 TF-IDF | `stylegram.py:148` → `_build_vectorizers` | **每折训练子集** | 否，正确 |
| stylegram LinearSVC | `stylegram.py:150-158` | 每折训练子集 | 否，正确 |
| linear_baseline TF-IDF + LR | `members.py:29-43`（sklearn Pipeline） | 每折训练子集 | 否，正确 |
| stylometry 360 维统计 | `stylometry.py:183-194` | 逐行计算，无全局拟合 | 否，正确 |
| ngram_lm 每类 LM + LR | `ngram_lm.py:117-127` | 每折训练子集 | 否，正确 |
| Qwen 探针 LR | `qwen_member.py:193-213` | 每折训练子集（特征来自冻结骨干） | 否，正确 |

`grep -rn "StandardScaler|MinMaxScaler|Normalizer|SimpleImputer|fillna|fit_transform" src/` 只命中两处 `fillna("")`，说明项目里根本不存在「在划分前拟合的全局预处理器」。

**结论：通过。** 所有有监督预处理器都在 `fit_members` 内部、按折拟合。

---

## 5. 三个会让分数失真的方法学问题（非泄露，但必须修）

### 5.1 训练/服务不一致：stylegram 在 OOF 与 test 用的不是同一个头

- OOF 循环：`pipeline.py:97` 调用 `predict_members(..., groups=None)`。
- `stylegram.py:168` 判定 `groups is None` → 直接返回 `plain`，**centered 孪生头在 OOF 中从未被评估过**。
- 测试：`pipeline.py:192` 传入预测簇 → `stylegram.py:183-184` 对 `2 ≤ m ≤ 18` 的簇用 `centered` **替换** `plain`。

后果：温度（`T=0.14`）与融合权重（`0.245`）是在 plain 头上拟合的，却作用在 centered 头上；OOF 0.4211 与 test 0.6035 描述的不是同一条管线。这也是「PRISM 增益 +0.144」难以直接外推的原因之一。

**实测（`.tmp/ablate_center.py`，同一模型、只改 `center_mode`）**：

| 配置 | blend | final | PRISM 增量 |
|---|---:|---:|---:|
| `center_mode=off` | 0.4520 | **0.6061** | +0.154 |
| `center_mode=replace`（实际服务） | 0.4773 | 0.6035 | +0.126 |

说明两点：① centered 头确实带来 +0.025 的 base 提升，不是因为泄露；② 它同时压缩了 PRISM 的可改进空间，净效果 −0.003，在噪声范围内。因此该问题**没有虚高 test 分数**，但让 OOF 与 test 不可比、权重非最优。

**整改**：OOF 折内预测时传入**折内真实 prompt 组**（`groups=prompt_codes[va]`），让 centered 头参与 OOF；或在 OOF 阶段显式 `center_mode="off"`，测试阶段同样关闭，二者取其一，不能一边开一边关。

### 5.2 中心化依赖批内簇规模

`stylegram.py:173-178`：`min_size=2` 以下不中心化。交互式会话的 `test.csv` 只有样例一行 → 单行成簇 → 自动退回 plain；提交时 ~4,000 行 → 大量簇被中心化。同一份代码在两种输入下输出不同，调试结论不可迁移。

**整改**：在 notebook 中固定用「全量 test 一次推理」，并在本地用「单行 / 全量」两种输入各跑一次，确认差异只来自设计而非 bug。

### 5.3 聚类器向量器与阈值校准的样本范围不一致

`clustering.py:100` 用全部训练文本 fit IDF，`clustering.py:107-119` 只在 3 个 hold-out prompt 折上选阈值。阈值是干净的，IDF 不是。影响很小（IDF 无监督），但严格来说「阈值校准所用的特征」已经见过被 hold-out 的 prompt。

**整改**：把 `TfidfVectorizer` 移入折内 fit，或在文档中明确标注该 transductive 假设。

### 5.4 重复训练的端到端方差约 ±0.003

同配置、同数据重训两次：

| 运行 | OOF blend | test blend | test experts | test final |
|---|---:|---:|---:|---:|
| 2026-10-18 15:06（`model_qwen.joblib`） | 0.4211 | 0.4773 | 0.4773 | **0.6061** |
| 2026-10-18 15:40（`model_run2.joblib`） | 0.4211 | 0.4773 | 0.4621 | **0.6035** |

OOF 完全一致，差异只出现在 test 的 experts 阶段。最可能来源是 `stylometry` 的 XGBoost-GPU 直方图在 CUDA 上非严格确定性，导致少数边界样本翻转，进而改变家族专家的拟合。

**含义**：0.001–0.003 的差异不能当作改进信号；本项目的 test 集只有 396 行，标准误约 0.025，任何小于 0.005 的调参收益都应视为噪声。

---

## 6. 外部数据集评估（严格按「文本字段 + 18 类模型名」筛选）

判定标准：① 有 LLM 输出文本字段；② 有具体模型名称字段（或可无损转换的列名）；③ 名称必须落在项目 18 个标签内。三条同时满足才算「完全匹配」。

### 6.1 完全匹配

| 数据集 | 来源 | 规模 | 字段 | 许可 | 获取 |
|---|---|---|---|---|---|
| **prvsiyan's Extended Name That LLM Dataset** | Kaggle Dataset `prvsiyan/prvsiyans-extended-name-that-llm-dataset`（v3） | 6,460 段 / 4,584 段过质量漏斗；360 prompts；`new` split 覆盖全部 18 类，每类约 300 段 | `id`, `prompt_id`, **`model`（18 个竞赛标签之一）**, `text`, `n_words`, `target_words`, `domain`, `task_type`, `split`, `served_model_id`, 质量 flags | **CC BY 4.0** | Kaggle 挂载；`passages_clean.csv` 直接用 |

适用性：最贴合的一份外部数据，18 类齐全、prompt 分组清晰、有 `served_model_id` 可核验路由。风险：每类 clean 样本约 226，Claude/GLM 的近亲混淆不会因量增而自动缓解；底层 API 条款可能限制「用输出训练竞争模型」，需自行核对；`proxy_v1` 只覆盖 12 类，混用会破坏类别平衡。

### 6.2 部分匹配（需改造或只覆盖子集）

| 数据集 | 来源 | 规模 | 字段 | 许可 | 不满足项 |
|---|---|---|---|---|---|
| **Name That LLM: Response Corpus** | Kaggle `olegzholobov/name-that-llm-response-corpus`（2026-10-15 快照） | 70,980 宽表行 / 164,954 答案单元 / 135,171 条唯一回复；1,902 个 prompt 组 18 模型齐全 | `id`, `prompt`, **18 个模型名作为列头**（宽表）；另有 cell metadata / provenance | **Other**（混合来源，未重新授权，需逐条看 provenance） | 模型名不是字段而是列名，必须 `melt` 转换；列名与 18 个官方标签是否逐字一致需下载核对（README 提到 `gpt-5.6-luna` 隔离集，说明至少含该标签）；标签为来源自报，非 API 逐条核验 |
| **Name That LLM: Extra API Passages** | Kaggle `seyitkaangunes/ntl-extra-api-passages`（v1） | 4,998 段；1,000 prompts × 5 个 Claude 模型 | `id`, `prompt_id`, **`model`**, `text`, `n_words`, `target_words`, `domain` | **CC BY 4.0** | 只覆盖 5/18 类（全部 Claude），不满足「18 类齐全」；作者消融：风格模型 0.372→0.388（w=0.5），`claude-fable-5` 召回 0.10→0.20，但 GLM 类会掉点 |

### 6.3 不匹配（明确排除）

| 数据集 | 原因 |
|---|---|
| LLM-NodeJS（Tihanyi et al., 2026） | 代码（JavaScript）而非自然语言段落；20 个模型不在本项目 18 类内 |
| HC3 | 只有 ChatGPT 一个生成模型 + 人类文本，无法构造 18 类 |
| M4 / AuTexTification / DAIGT | 生成器集合与 18 类不重合，或只有「人 vs 机」二分类标签 |
| OpenTuringBench | 模型家族与版本均不同，标签体系不兼容 |
| `NTL Tabular Offline Assets` / `Qwen3-8B LoRA + Claude Head` | 是特征/权重资产，不含「文本 + 模型名」样本对 |

### 6.4 使用前的硬性检查

1. **列名逐字比对** `LABELS`（`labels.py`），任何大小写/版本号差异都会静默变成第 19 类。
2. **prompt 分组隔离**：外部数据的 `prompt_id` 必须与竞赛 150 个 prompt 无交集；训练时外部 prompt 只能进训练折。
3. **类别权重**：外部数据按类配额进入，避免某一 provider 的样本量淹没 18 类均衡（项目指标是 balanced accuracy）。
4. **去重**：对 `text` 做归一化后与 `train.csv` 比对；Response Corpus 作者已做 exact/normalized/substring 检查，但未覆盖改写。
5. **许可**：CC BY 4.0 的两份可商用可再分发（需署名）；Response Corpus 是混合许可，用于竞赛前需核对每条 provenance。

---

## 7. 整改建议（按优先级）

1. **统一 stylegram 的训练/服务行为**（P1，改动 1 行）：`pipeline.py:97` 改为 `groups=prompt_codes[va]`，重跑 OOF，确认 0.4211 的变化方向；若上升，说明 centered 头本就该参与融合，当前的 0.6061 是被低估而非高估。
2. **停止引用 val 指标**（P1，文档层面）：训练集含 val 后，`val=1.0000` 只能作为「管线能跑通」的证据，不能作为性能。所有汇报只保留 test。
3. **把 λ 的选择迁到嵌套协议**（P1）：在 `train.csv` 内部再做一层 prompt-grouped 折选 λ，test 只最终评一次；或直接固定 λ=2.0 并声明其为先验选择。
4. **聚类向量器折内拟合**（P2）：`clustering.py:100` 移入 `_calibrate_threshold` 的每一折，消除 transductive IDF。
5. **predict 配置校验**（P2）：`cli.py:133` 覆盖 cfg 后，比对 `pipeline.describe()["members"]` 与 cfg 的启用列表，不一致直接报错。
6. **`GroupKFold` 加 shuffle**（P2）：`data.py:319` 用 `GroupKFold` 前先按 seed 打乱 prompt 顺序，降低折与主题顺序的耦合。
7. **外部数据接入**（增量）：先接 `prvsiyans` 的 4,584 条 clean 样本做 5 折对照，只加在训练折；若 `claude-fable-5` 召回仍 <0.3，再叠加 `ntl-extra-api-passages` 的 4,998 条 Claude 专数据，并按类权重 0.5 混入。

---

## 附录 A：本次训练与评估记录（2026-10-18 15:40–16:05）

| 项 | 值 |
|---|---|
| 命令 | `run.py --config configs/real.yaml train --train-file data/real/trainval.csv --model-out artifacts/real/model_run2.joblib` |
| 数据 | 2,303 行 / 128 prompts / 18 类（`trainval`） |
| 设备 | RTX 4060 Laptop 8GB，torch 2.14.1+cu126，XGBoost GPU |
| 成员 | qwen_probe(layer 18, 4-bit)、stylegram、stylometry、ngram_lm、linear_baseline |
| 耗时 | **1,182.7 s（19.7 min）**；其中 Qwen 全语料前向约 8 min |
| OOF | balanced accuracy **0.4211** |
| 融合权重 | qwen 0.286 / stylegram 0.245 / stylometry 0.206 / linear 0.147 / ngram 0.116 |
| 温度 | qwen 0.52 / stylegram 0.14 / stylometry 1.19 / ngram 0.51 / linear 0.33 |
| 护盾规则 | `curly_quote→anthropic`、`curly_quote→zai`、`double_em_dash→google` |
| test（396 行 / 22 prompts） | blend 0.4773 → experts 0.4621 → shield 0.4621 → **PRISM 0.6035** |
| 其他 test 指标 | macro-F1 0.5974、top-2 0.6641、ECE 0.0583、聚类 pair-F1 0.9204 |

弱类（test 每类 n=22）：`claude-fable-5` 0.182、`glm-5.3` 0.273、`glm-5.3-flash` 0.273、`gpt-5.6-terra` 0.409；强类：`gemini-3.1-pro-preview` 1.000、`claude-opus-5` 0.955、`claude-opus-4-6` 0.864。

---

## 8. 复现

```bash
cd D:/research/projects/ntl-pact18
.venv/Scripts/python .tmp/audit_leak.py          # 重叠/重复实测
.venv/Scripts/python run.py --config configs/real.yaml train \
    --train-file data/real/trainval.csv --model-out artifacts/real/model_run2.joblib
.venv/Scripts/python run.py --config configs/real.yaml --set prism.lam=2.0 predict \
    --model artifacts/real/model_run2.joblib --test-file data/real/test.csv \
    --truth-file data/real/test_truth.csv --out artifacts/real/test_pred_run2.csv
```
