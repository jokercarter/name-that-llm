# 公开高分方案复盘（2026-10-23）

## 一、当前格局

公开榜（Public LB，约 40% 测试集）：

| 名次 | 队伍 | 分数 |
|---|---|---|
| 1 | Hod Carriers | 0.700 |
| 2 | Vishak K Bhat | 0.673 |
| 3 | Le Viet | 0.666 |
| 4 | Søren Ravn Andersen | 0.665 |
| 5 | kulshekhar | 0.660 |
| … | … | … |
| 20 | Nishant Dahal | 0.639 |

公开代码的天花板 **0.636**（fleongg/ntl-part-two-complement），约合第 22 名。
紧随其后：mrpicklerick 0.627、nuridudak 0.626 / 0.624、nishantdahal 0.620。
**公开代码与榜首差 0.064**，且榜首无公开实现。

## 二、骨架：两阶段 LoRA（nishantdahal，LB 0.620）

| 成员 | 训练数据 | 第一版 (0.592) | 第二版 (0.620) |
|---|---|---|---|
| n-gram SVC（4 个 TF-IDF 视图，C=0.5） | train + 公开池 | 0.435 | **0.473** |
| Qwen3-14B LoRA，前 27/40 层 | 公开池 → train | 0.569 | **0.611** |
| Qwen2.5-14B LoRA，前 32/48 层 | 公开池 → train | — | 0.569 |
| DeBERTa-v3-large 全参微调 | 公开池 → train | — | 0.546 |
| 加权 log-prob + Sinkhorn（CV） | | 0.593 | **0.642** |
| Public LB（含 centred twin） | | 0.592 | **0.620** |

**最大杠杆是数据不是模型**：

- stage 1 池子 35k → **83,827 段（26 个公开数据集）**；
- 同一个 Qwen3-14B、同一套 recipe：单模 0.569 → 0.611；
- 只看 stage 1 之后（还没见 host 数据）head 在 host 上的分数：0.493 → **0.572**；
- 连 SVC 都从 0.435 → 0.473（池子 34,940 → 75,783）；
- 旧池上 Qwen3-8B / Qwen3-14B / Qwen2.5-7B / Qwen2.5-14B **四个骨干全是 0.569**——骨干规模在 35k 数据下完全饱和。

两条新的数据清洗规则：

1. 剔除某套 claude-fable-5 的 system-prompt 语料（与 host 的 Fable 文风不像）；
2. 剔除与任一 train 段落 content-TF-IDF 余弦 ≥ 0.3 的 1,153 条公开段落——**这是让 CV 诚实，不是提分**。

**只在 Kaggle 出现的 tokenizer bug**：Qwen3-14B-Base 在 Kaggle 上的 tokenizer 配置是 chat 版的，
eos = `<|im_end|>`(151645)，而训练用的是 `<|endoftext|>`(151643)。结果每条段落结尾都是模型训练时
没见过的 token，14B 的 top-1 在约 **14%** 的段落上发生变化，且不报错不告警。
修法：用 checkpoint `meta.json` 里存的那个 id。

## 三、0.636 冠军：主集成 + 15% 互补栈 + 10% ATLAS + Sinkhorn

- 主集成：SVC 1.0 / Qwen3-14B 0.4 / Qwen2.5-14B 0.2 / DeBERTa 0.25；
- **15% 互补栈**（12 个成员，全部来自公开权重，重新跑推理）：
  - 解码器线性头：GLM4-9B(层 8,3)、Qwen3-8B(7,22)、Qwen2.5-7B(6,20)、Qwen3-4B(4,7, style 池化)；
  - 编码器线性头：DeBERTa-v3-large(24,6)、ELECTRA-large(21,24)、desklib-detector(23)；
  - LoRA：Qwen3-24 层 ×3 个不同训练（public / seed7 / claude-head）、Qwen2.5-20 层；
  - TokenAttention ×2（Qwen3.5、DeBERTa）；
  - UltraBERT-380M、Gaussian 共享 prompt 头 ×4、StyleGram、stylometry-LGB；
- **10% ATLAS**：只对"不确定行"用 source-conditioned likelihood（Joseph Adamski 的公开权重）；
- 最后 Sinkhorn 均衡 18 类。

设计原则（原文）：**权重是圆整数字，在 OOF 上定；在 2,699 行上细调权重基本是拟合噪声。**

## 四、当前增量方向：编码器成员 + 排版守卫

- **nuridudak 0.624/0.626**：在 0.620 基础上加 **ModernBERT-large** 成员，权重 0.25；
  两阶段微调（122k prompt-disjoint 外料 → 官方 train），单模 OOF **0.569**；
  再加 TypoGuard（排版守卫）。每个增件失败就跳过，保留基础融合。
- **mrpicklerick 0.627**：0.624 版本去掉硬排版守卫，编码器配比 6:1。
- 为什么编码器有效：**双向编码器原样保留大小写/标点/空格/排版**，是解码器（tokenizer 会吃掉排版）
  看不到的另一个视角；它同时是"排版特征"最自然的载体。

## 五、明确无效的方向（原文"what did not help"）

- 更大参数（14B = 8B = 7B，同一池下都是 0.569）；
- seed 平均 / LoRA 权重平均（≤0.1 分）；
- Gemma-4-12B 骨干；
- 单家族专属判别器（family gates）；
- **簇内"每个模型至多一段"**（即 PRISM 类硬约束）：模拟更好、LB 更差；
- stage 2 更多 epoch、把公开池混回 stage 2、短窗口切分；
- 用 stacker 代替固定权重。

两个最难的对：`glm-5.3` vs `glm-5.3-flash`、`gemini-3.7-flash` vs `gemini-3.8-flash`——所有成员都卡在同一处。

## 六、CV 与 LB 的差

0.642 CV → 0.620 LB（差 2.2 分；第一版只差 0.1）。原因：CV 只在 host 段落上测，而成员的知识大部
分来自"另一种设定下写的"公开段落，部分不迁移。公开 LB 又只是 40% 切片，本身有噪声。

## 七、与当前项目的对照

我们这次两阶段 run 的成员 OOF（同日实测）：

| 成员 | 我们 | 公开同尺寸/同类 |
|---|---|---|
| qwen_two_stage（Qwen3-8B，24 层） | **0.5238** | Qwen3-8B 旧池 0.569 / 14B 新池 0.611 |
| stylegram | 0.3812 | — |
| linear_baseline | 0.3489 | n-gram SVC 0.473 |
| stylometry | 0.3034 | — |
| ngram_lm | 0.1804 | — |

差距来源（按影响排序）：

1. **池子规模/质量**：我们 55,378 段（清洗后实际可用更少），公开 83,827 段；且缺"与 train 余弦 ≥0.3 剔除"这条。
2. **stage 1 覆盖率**：我们的 `_train` 在三元组批次上跑，35k 上限实际只前向 ~19k 样本（约 55%）；
   公开 recipe 的 stage 1 是**全量 plain CE**（100% 段落、class-weighted）。
3. **缺编码器成员**：没有 DeBERTa/ModernBERT，少了"保留排版"的视角。
4. **SVC 特征族太弱**：我们用 linear_baseline（0.349），公开是 4 视图 TF-IDF + LinearSVC（0.473）。

## 八、对提交失败的直接启示

公开高分 notebook 里有两处我们缺的防护：

1. `TINY = len(TE) < 50`：**交互态 3 行时，用训练行跑通 group / PRISM / twin 路径**，
   避免"隐藏重跑才第一次走到这段代码"。我们的 pipeline 没有这个开关。
2. T4 推理 OOM 时**逐段重试**（一个 passage 一个 passage 地过），并且**不给成员包 try/except**
   （早前有人这么干，OOM 的成员被静默跳过，分数降了几点才发现）。

我们的 run：训练峰值 12.87/14.6 GiB，stage1 2.45h，三折各 ~30min，refit 40.7min，总 6.05h，exit 0。
隐藏重跑 4,000 行时最可能死在**预测阶段 OOM**（batch 16 × `output_hidden_states` 25 层 ≈ 2.4–4.5 GB 额外显存）。
