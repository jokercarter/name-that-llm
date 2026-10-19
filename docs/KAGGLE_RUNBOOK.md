# Kaggle 运行手册：环境、依赖、数据挂载与实跑验证

本文记录 PACT-18 在 Kaggle Code Competition 环境中的**实际**运行方式。所有环境数字与报错都来自本机实测（`jokercarter/ntl-pact18-verify-run`），不是照抄文档。

## 0. 实测环境（2026-10，Kaggle 默认镜像）

| 项 | 实测值 | 影响 |
|---|---|---|
| Python | 3.13.15 | 与本地 3.13 一致，但 **joblib 跨版本仍不可靠**，提交必须重训 |
| torch | 2.11.0+cu128 | 需要 `libbitsandbytes_cuda128.so` 的 wheel |
| GPU | Tesla T4 × 2，**14.6 GiB**（非 16） | 显存预算按 14 GiB 算，不是 16 |
| 计算能力 | cc 7.5 | 无原生 bf16（见 §5.3） |
| `torch.cuda.is_bf16_supported()` | **True**（会误导） | 必须用 cc 判定 |
| 网络 | Off（提交态强制） | 依赖必须随数据集挂载 |
| 时限 / 磁盘 | 9 h / `/kaggle/working` 20 GB | 5 万条全量约 3–5 h，够用 |

## 0.1 实测结果（`jokercarter/ntl-pact18-verify-run`，3,600 行 × 2 折）

| 项 | 实测 |
|---|---|
| 总墙钟 | **34 分 16 秒**（含环境、安装、单测、训练、装配） |
| QLoRA 三段 | 492.6 s / 466.7 s / 626.1 s（fold1 / fold2 / refit） |
| 每步耗时 | **4.4 s / step**（32 个样本：8/卡 × 2 卡 × 累积 2） |
| 吞吐 | **0.216 s / 样本**（双卡合计，4.6 样本/s）；单卡 0.43 s |
| 有效 MFU | **0.055**（FLOP 模型按 0.18 估算，乐观了 3 倍） |
| 单卡峰值显存 | 7.51 GiB（refit）/ 5.01 GiB（fold），共 14.6 GiB |
| 模型加载 | 33 s / 次（4 GB fp32 checkpoint → 4-bit） |
| 可训练参数 | 52,199,706（与配置一致） |
| 训练健康度 | loss 3.33 → 2.75，内层 BA 0.056 → 0.149，早停在第 90 步回滚到第 60 步 |

> 3,600 行时内层 BA 只有 0.149 是正常的：每类仅 100 条、只跑了 45 步、学习率还在 warmup 段。这个数字**不能**用来判断最终分数。

### 0.2 由实测推出的 8 小时配置

| 方案 | 样本通过 | 预计（batch 8） | 预计（batch 12） |
|---|---:|---:|---:|
| 50k × 2 折 + 重训 | 100k | 6.0 h | 5.4 h |
| **50k × 3 折 + 重训** | 150k | 9.0 h | **7.8–8.2 h** |
| 50k × 4 折 + 重训 | 200k | 12.0 h | 10.4 h |
| 4B × 2 卡，3 折 | 150k | 21 h | 18 h |

结论：**5 万条 + 3 折在 batch 8 下是 9.0 h，正好卡在 9 h 上限之上**；把每卡 batch 提到 12（瞬时 24、累积后等效 48）可以拿回 10–15% 的 MFU，落在 7.8–8.2 h。如果机器比实测慢，`--deadline-hours 8.8` 会截断重训而不是让 kernel 被杀。想更稳就把折数降到 2（6.0 h）。

**参数规模不是这个任务的成本杠杆**：4B 的 FLOPs 是 1.7B 的 2.35 倍，3 折要 21 h。想加容量就加 LoRA rank（r32→r64，52M→~100M 可训练，耗时只 +3%）。

## 1. 依赖：镜像缺什么

| 包 | 来源 | 是否需要处理 |
|---|---|---|
| torch / transformers / peft / accelerate | 镜像自带 | 否 |
| sklearn / xgboost / scipy / pandas / pyarrow | 镜像自带 | 否 |
| **bitsandbytes** | **镜像没有** | **必须离线安装**，否则 4-bit 加载失败、Qwen 成员被静默关闭 |

wheel 的获取与打包（本地执行，需联网）：

```bash
pip download "bitsandbytes>=0.45" --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_24_x86_64 \
  --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --python-version 311 --implementation cp --abi cp311 -d wheels
# 实测拿到 bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl（42 MB，含 cuda118/121/124/126/128 运行库）
```

Kaggle 内安装（无网）：

```python
!pip install -q --no-index --find-links {DS} bitsandbytes
import bitsandbytes as bnb   # 必须断言能 import，不要靠 try/except 静默降级
```

> 教训：v1 提交时 `QwenMember` 在 `try/except` 里 `from transformers import BitsAndBytesConfig` 失败后**静默退回 fp16 + CPU offload**，实际跑成 4 成员栈、分数只有 0.52 左右。依赖缺失必须 fail-fast。

## 2. 数据挂载：Kaggle 的新布局

2026 版 Kaggle 把输入改成两层：

```
/kaggle/input/competitions/name-that-llm/{train.csv,test.csv,sample_submission.parquet}
/kaggle/input/datasets/<owner>/<slug>/...
```

**不再是** `/kaggle/input/<slug>`。实测 `/kaggle/input` 只有 `competitions` 和 `datasets` 两个目录，旧的扁平路径全部失效。代码里必须两种布局都兼容：

```python
def pick_dir(patterns, marker):
    for pat in patterns:
        for cand in sorted(glob.glob(pat)):
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, marker)):
                return cand
    return None

DS    = pick_dir(['/kaggle/input/datasets/*/ntl-pact18-verify',
                  '/kaggle/input/ntl-pact18-verify'], 'train_verify.csv')
MODEL = pick_dir(['/kaggle/input/datasets/*/qwen3-1-7b',
                  '/kaggle/input/qwen3-1-7b'], 'config.json')
COMP  = pick_dir(['/kaggle/input/competitions/name-that-llm',
                  '/kaggle/input/name-that-llm'], 'test.csv')
```

`src/ntl_pact18/runtime.py` 的 `find_competition_dir()` 已经同时支持这两种路径，但 `--model-path`、`--train-file` 这类命令行参数必须由调用方解析。

### 2.1 三个挂载对象

| 类型 | 引用 | 内容 | 用途 |
|---|---|---|---|
| Competition | `name-that-llm` | `train.csv`、`test.csv` | 竞赛数据 |
| Dataset | `jokercarter/ntl-pact18-verify` | `ntl-pact18-code/`（源码）、`bitsandbytes-*.whl`、`train_verify.csv`、`ntl_merged_external_large.parquet` | 代码、依赖、外部语料 |
| Dataset | `vodung020905/qwen3-1-7b` | `config.json`、2×safetensors、tokenizer | 骨干权重（Apache-2.0） |

> **坑**：数据集里的 `.zip` 会被 Kaggle **自动解压**，`ntl-pact18-code.zip` 挂载后变成目录 `ntl-pact18-code/ntl-pact18-code/`。所以 notebook 里不能假设 zip 还在，要先判断文件是否存在、再回退到目录。

## 3. Notebook 执行步骤与预期输出

| # | 动作 | 预期输出 | 失败含义 |
|---|---|---|---|
| 1 | 打印环境 | `devices : 2`、两行 `Tesla T4 14.6 GiB cc 7.5`、`input : ['competitions','datasets']` | 只有 1 张卡 → Accelerator 没选 T4×2 |
| 2 | 解析挂载 + 装 bnb | `DS/MODEL/COMP` 三个路径、`bitsandbytes 0.50.2` | 路径为 None → Add Input 漏了 |
| 3 | 解压/复制源码 | `package ok, dist world = 1` | 找不到 `pipeline.py` → 数据集版本未就绪 |
| 4 | 单元测试 | `59 passed` | 失败说明镜像版本与本地不兼容 |
| 5 | `train_parallel.py --stage all` | 见 §3.1 | 见 §5 |
| 6 | 汇总日志与元数据 | `trainable_parameters`、每折秒数 | — |
| 7 | 校验 parquet | 两列、无空值、标签 ⊆ 18 类 | 提交格式错误 |

### 3.1 第 5 步的关键日志（按出现顺序）

```
[launch] visible GPUs: 2 ['Tesla T4', 'Tesla T4']
[launch] rank 0 -> qwen_rank0.log
[qwen] rows=3600 prompts=... classes=18 world=2
[qwen] fold 1/2: train=1800 val=1800
[qwen] qlora adapter: r=32 on 7 projections, layers=[2, 5, 8, 12, 16, 20, 24, 28], trainable=52,199,706
[qwen] qlora split: train=1440 val=192 ... world=2 per_rank_batch=8 global_batch=16
[qwen] step 5/90 loss ... (…s)
[qwen] step 15: inner val BA 0.xxxx (best …)
[mem] rank0 fold1: peak=… GiB / 14.6 GiB
[qwen] fold 2/2 ...
[qwen] refit on the full training file
[qwen] folds+refit in 0.xx h ([...] min)
[classical] members: ['stylegram','stylometry','ngram_lm','linear_baseline']
[classical] done in x.x min
[assemble] stacking layer fitted on 1800 competition rows out of 3600
[fit] done in ... | oof blended balanced acc = 0.xxxx (evaluated on 1800 rows)
[predict] rows=3 clusters=... changed_by_prism=...
[assemble] submission -> /kaggle/working/submission.parquet (3 rows, ... labels used)
```

**读法**：`trainable=52,199,706` 必须与文档一致；`peak` 应远低于 14.6 GiB；`evaluated on 1800 rows` 证明外部行只参与训练、不参与融合层拟合；`folds+refit` 的秒数用来外推 5 万条（×13.9 倍样本）。

## 4. 生产运行（5 万条）

```bash
# 1) 生成 5 万条训练文件（2,699 竞赛 + 47,304 外部，其中 3,430 条为尾部类重复）
python scripts/build_external_mix.py --target-total 50000 --oversample \
    --base {COMP}/train.csv \
    --external {DS}/ntl_merged_external_large.parquet \
    --out /kaggle/working/trainval_plus_ext_50k.csv

# 2) 全量训练 + 装配 + 出提交
python scripts/train_parallel.py --stage all \
    --config configs/kaggle_9h_50k_2gpu.yaml \
    --train-file /kaggle/working/trainval_plus_ext_50k.csv \
    --model-path {MODEL} \
    --test-file {COMP}/test.csv \
    --submission /kaggle/working/submission.parquet
```

预期：3 折 + 1 次重训，每卡约 75k 样本通过，**3.0–5.4 h**（取决于实测 s/样本），单卡峰值 ~3 GB。首次必须先跑 20 步标定，用 `scripts/estimate_qlora_budget.py --sec-per-sample <实测>` 复核。

## 5. 错误目录（现象 → 原因 → 修复）

### 5.1 `AssertionError: Add Input -> Datasets`（本次实测，v1–v3 连续踩）

- **现象**：`/kaggle/input` 只有 `['competitions','datasets']`，`/kaggle/input/ntl-pact18-verify` 不存在。
- **原因**：Kaggle 2026 改为 `/kaggle/input/datasets/<owner>/<slug>`。
- **修复**：用 §2 的 `pick_dir` 兼容两种布局；不要硬编码扁平路径。

### 5.2 `unzip: cannot find ... ntl-pact18-code.zip`（本次实测）

- **原因**：数据集内的 zip 被自动解压成目录，且多一层同名嵌套。
- **修复**：先判断 zip 是否存在，否则用 `glob` 找 `**/src/ntl_pact18/pipeline.py` 反推源码根目录再 `copytree`。

### 5.3 T4 上 bf16 被误判为可用（本次实测）

- **现象**：`torch.cuda.is_bf16_supported()` 在 T4 / torch 2.11 返回 **True**，但 sm_75 没有原生 bf16 单元，4-bit kernel 会走 fp32 仿真。
- **修复**：用计算能力判定 —— `torch.cuda.get_device_capability() >= (8,0)` 才用 bf16，否则 fp16。已在 `QwenMember._compute_torch_dtype` 落地。

### 5.4 `ModuleNotFoundError: No module named 'bitsandbytes'`

- **原因**：镜像不带 bnb；旧代码在 `except` 里退回 fp16+CPU offload，训练慢 20 倍且分数掉一档。
- **修复**：挂 wheel 并 `import bitsandbytes` 断言；`QwenMember` 的 4-bit 分支不要再静默回退。

### 5.5 数据集版本未就绪导致挂载为空

- **现象**：刚 `datasets version` 完立刻 `kernels push`，容器内看不到新版本。
- **修复**：等 1–2 分钟，用 `kaggle datasets files <ref>` 确认文件时间戳更新后再 push。

### 5.6 `CUDA out of memory`

- **原因**：`batch_size`/`max_len` 过大，或两折残留显存。
- **修复**：先降 `features.qwen.batch_size`（8→4），再降 `max_len`（384→320）；`release()` 已在每折后释放。

### 5.7 NCCL 挂死 / `no usable interface`

- **原因**：容器内 PCIe P2P 被禁、或网卡不可用。
- **修复**：启动器已默认 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`；仍挂死时加 `NCCL_SOCKET_IFNAME=eth0`，并把 `WORLD_SIZE` 降到 1 做单卡对照。

### 5.8 `AttributeError: Can't get attribute '...' on <module '__main__'>`

- **原因**：Web 端 import 的 notebook 把类定义在 `__main__`，`joblib.load` 时找不到。
- **修复**：类要放在可导入模块里（本项目的成员都在 `ntl_pact18.*` 下，只有自造测试替身会遇到）。

### 5.9 `submission.csv` 被拒

- **原因**：赛题要 `submission.parquet`，两列 `id/model`，且 id 必须覆盖测试集。
- **修复**：`train_parallel.py --stage assemble` 已直接写 parquet，并断言列名、行数、id 唯一、标签 ⊆ 18 类。

### 5.10 joblib 跨版本

- **原因**：本地 sklearn 1.9.1 / torch 2.14.1 存的模型，Kaggle 是 sklearn 1.6 / torch 2.11。
- **修复**：**在 notebook 内重训**（本文流程就是重训），或固定镜像版本；不要上传本地 `.joblib`。

## 6. 验收清单

- [ ] `/kaggle/input` 解析出 3 个路径
- [ ] `bitsandbytes` 可 import，版本号打印
- [ ] `devices : 2` 且两张都是 T4
- [ ] 单测 59 passed
- [ ] `trainable=52,199,706`
- [ ] 单卡 `peak` < 8 GiB
- [ ] `evaluated on <竞赛行数> rows`（不是全量）
- [ ] `submission.parquet`：2 列、无空值、无重复 id、18 类内
- [ ] 日志里没有 `falling back`、`not usable`、`Traceback`
