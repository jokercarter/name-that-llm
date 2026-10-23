# vendor/top1_fleongg —— 当前主提交方案（Public 0.636）

## 这是什么

公开 Notebook **`fleongg/ntl-part-two-complement`** 的原样复现（推理代码零修改），
公开榜得分 **0.636**（submission 57551442），是当前公开代码的天花板。

本项目自 2026-10-23 起把主提交切换为该方案；原 PACT-18 两阶段流水线保留为
fallback（`configs/kaggle_2stage_8b.yaml` + `scripts/train_parallel.py`），不再作为默认提交。

## 方法结构

```
主集成（Nishant Dahal two-stage LoRA part two，LB 0.620）
  ├─ n-gram SVC                     权重 1.00
  ├─ Qwen3-14B LoRA (27/40 层)      权重 0.40  + centred twin 0.10
  ├─ Qwen2.5-14B LoRA (32/48 层)    权重 0.20  + centred twin 0.10
  └─ DeBERTa-v3-large (全参)         权重 0.25
+ 15% 互补栈（comp_stack.py，12 个成员，全部公开权重重跑推理）
+ 10% ATLAS（仅对"不确定行"，source-conditioned likelihood）
→ Sinkhorn 均衡 18 类（300 iter；<50 行时跳过）
```

## 运行环境

| 项 | 值 |
|---|---|
| 加速器 | **GPU T4 ×2**（14B 单卡放不下，`device_map` 跨卡切分） |
| 互联网 | **关闭**（`HF_HUB_OFFLINE=1`） |
| 输入体量 | 约 **29 GB**（16 数据集 + 8 kernel 输出 + 7 个 Qwen 模型） |
| 预期耗时 | 主集成约 15 min；含 12 成员与 ATLAS 预计 60–90 min |

## 输入清单（与原 notebook 完全一致）

### 数据集（16，均可公开访问）

| id | 许可 | 大小 |
|---|---|---|
| `nishantdahal/ntl-handwriting-blend` | CC BY-SA 4.0 | 1281 MB |
| `nishantdahal/ntl-two-stage-lora` | CC BY-SA 4.0 | 318 MB |
| `dhruvtibarewal/nlm-l24-adapters` | MIT | 3731 MB |
| `dhruvtibarewal/nlm-matched-clean` | MIT | 29 MB |
| `dhruvtibarewal/nlm-offline-wheels` | Other (见描述) | 45 MB |
| `seyitkaangunes/ntl-extra-api-passages` | CC BY 4.0 | 5 MB |
| `abdullah1afify/ntl-generations` | CC BY 4.0 | 1 MB |
| `dophuchung/ntl-proxy-generations` | CC BY 4.0 | 1 MB |
| `lucasboesen/ntl-free-proxy-generations` | CC BY 4.0 | 4 MB |
| `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` | Apache 2.0 | 321 MB |
| `josephadamski91/ntl-atlas-qwen3-8b-v1` | Other (见描述) | 463 MB |
| `fleongg/ntl-legacy-qwen24-recovery` | Other (见描述) | 138 MB |
| `fleongg/ntl-quality-selected-public-svc-pool` | Other (见描述) | 23 MB |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | CC BY 4.0 | 97 MB |
| `bbucxi/qwen35-9b-safetensors` | **Unknown（谨慎）** | 19329 MB |
| `pjmathematician/ultrabert-380m-ntl` | Apache 2.0 | 3819 MB |

### Kernel 输出（8，均为公开 notebook 的产出）

`lucasboesen/name-that-llm-{glm4-9b-trunc25, extract-encoders2, electra-large-weights,
desklib-weights, train-lora24-s7, train-tokattn, train-tokattn2}`、
`vnton37nguyn/ntl-train-lora-qwen25-instyle`

### 基座模型（7）

`qwen-lm/qwen-3` 的 14b-base / 8b-base / 14b / 8b / 4b-base，
`qwen-lm/qwen2.5` 的 7b / 14b

## 如何运行

```bash
# 1) 推送（输入清单已在 kernel-metadata.json 里）
kaggle kernels push -p .tmp/kpush_top1

# 2) 看状态 / 收产物
kaggle kernels status jokercarter/ntl-top1-repro
kaggle kernels output jokercarter/ntl-top1-repro -p .tmp/ktop1_out

# 3) 提交到比赛（触发 ~4,000 行隐藏重跑）
kaggle competitions submit -c name-that-llm \
  -k jokercarter/ntl-top1-repro -v 1 -m "top1 reproduction"
```

## 已知风险

1. **`bbucxi/qwen35-9b-safetensors` 许可未知**（19.3 GB）；若合规审查严格，需替换或移除依赖它的 `tokattn_q35` 成员。
2. 12 个互补成员**必须全部成功**（作者用 "fail closed" 设计，不容错）；任一输入缺失会整轮失败。
3. 3 行交互态只验证格式，**隐藏重跑才是真考验**：`test.csv` 约 4,000 行，`len(test) >= 50` 才启用 Sinkhorn。
4. 原 notebook 作者注明"部分输入是维护在其账号下的辅助档案，可见性可能限制 fork 执行"。

## 署名（CC BY-SA / Apache 2.0 要求）

- 主方案：Nishant Dahal, *Two-stage LoRA, part two*
- 互补栈权重：Dhruv Tibarewal、Lucas Boesen、Vladislav Lavrentev、PJ Mathematician、vnton37nguyn
- ATLAS：Joseph Adamski
- Gaussian 共享 prompt 头：prvsiyan / kirilltym / sangam21
- 基座：Qwen team（Apache 2.0）
- 组装与发布：fleongg

本项目仅做原样复现与运行，未修改推理逻辑。
