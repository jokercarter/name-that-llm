# 运行环境适配：一套算法，两个宿主

> 代码：`src/ntl_pact18/runtime.py`、`scripts/run_local.py`、`scripts/run_kaggle.py`
> 配置入口：`load_config(..., profile=...)`、`run.py --profile {auto,local,kaggle,off}`

## 1. 设计原则

本地（Windows + RTX 4060 Laptop 8 GB）与 Kaggle（T4 ×2、只读输入、断网）只允许在**四个操作项**上不同：

| 操作项 | 本地 | Kaggle | 为什么必须不同 |
|---|---|---|---|
| `paths.data_dir` | `<repo>/data/real` | `/kaggle/input/competitions/name-that-llm` | Kaggle 输入只读且路径固定 |
| `paths.artifacts_dir` | `<repo>/artifacts/real` | `/kaggle/working/artifacts` | Kaggle 只有 `/kaggle/working` 可写 |
| `features.qwen.model_id` | `Qwen/Qwen3-8B-Base`（HF 缓存） | 挂载目录（自动探测 `config.json` 的 `model_type=qwen3`） | 断网，不能从 Hub 拉取 |
| `device.require_gpu` / `preference` | `false` / `auto` | `true` / `cuda` | 无 GPU 时应直接失败而不是静默降级 |

**以下内容两个 profile 完全相同**，由 `build_profile()` 的返回值直接保证：成员列表、TF-IDF 四路的 n-gram 与 `max_features`、`stylegram.C`、`stylometry` 树数与学习率、`ngram_lm.alpha`、`n_folds`、温度缩放、融合权重、家族专家、护盾惩罚、聚类阈值、`prism.lam`、`max_len`、`batch_size`。

也就是说：profile 只回答“数据在哪、模型在哪、能不能写、要不要强制 GPU”，不回答“模型怎么训练”。

## 2. 检测规则

```python
detect_platform()
  ├─ KAGGLE_KERNEL_RUN_TYPE 或 KAGGLE_URL_BASE 存在 → kaggle
  ├─ /kaggle/input 存在                              → kaggle
  └─ 否则                                            → local
```

可以用 `--profile` 强制覆盖，避免“检测变了导致结果不可复现”：

- `auto`（默认）：按上面的规则判断；
- `local` / `kaggle`：强制指定，日志里会写 `forced by profile=...`；
- `off`：不注入任何 profile，完全按 YAML 走（`synth` / `smoke` / `split` 默认走这条）。

**合并优先级**（低 → 高）：`default.yaml` < `--config` 文件 < **profile** < `--set k=v` < `NTL_*` 环境变量。因此显式 `--set paths.data_dir=...` 永远能盖掉 profile。

## 3. 使用方式

### 3.1 自动识别（推荐）

```bash
# 本机：自动解析为 local
python run.py --config configs/real.yaml train

# Kaggle notebook 里：自动解析为 kaggle
!python /kaggle/input/<code-dataset>/run.py --config configs/real.yaml train
```

### 3.2 两个显式入口（“两份版本”）

算法完全共用，入口只是把 profile 固定下来：

```bash
# 本地
python scripts/run_local.py --config configs/real.yaml train
python scripts/run_local.py --config configs/real.yaml predict \
    --model artifacts/real/model.joblib --test-file data/real/test.csv \
    --truth-file data/real/test_truth.csv

# Kaggle
!python /kaggle/input/<code-dataset>/scripts/run_kaggle.py train
!python /kaggle/input/<code-dataset>/scripts/run_kaggle.py predict \
    --test-file /kaggle/input/competitions/name-that-llm/test.csv
```

两个脚本做的事只有一行：在参数最前面插入 `--profile local` 或 `--profile kaggle`，其余全部委托给同一个 `cli.main`。所以不存在“两份代码逐渐漂移”的问题。

### 3.3 查看当前解析结果

```bash
python run.py device            # 会先打印 runtime profile，再打印设备报告
python run.py --profile kaggle device   # 在本地预演 Kaggle profile（会提示无 /kaggle/input）
```

## 4. 实测差异（2026-10-18）

| | 本地 RTX 4060 8 GB | Kaggle T4 ×2 |
|---|---|---|
| 数据 | 2,303 行（trainval） | 2,699 行（全量 train.csv） |
| 成员 | 5 个（含 qwen_probe） | **4 个**（Qwen 被关闭，见下） |
| 耗时 | 1,182.7 s（19.7 min） | **约 9 min** |
| 加速器 | RTX 4060（cc 8.9，bf16 支持） | Tesla T4（cc 7.5，无 bf16），只用到 1 张 |
| 产物 | `artifacts/real/model_run2.joblib` | `submission.parquet`（3 行样例） |

**Kaggle 上 Qwen 被关闭的原因**：镜像里没有 `bitsandbytes`，`import bitsandbytes` 抛 `ModuleNotFoundError`，预检把 `features.qwen.enabled` 置为 `False`。模型本身挂载成功了（`/kaggle/input/models/manojkumarcs28/qwen3-8b/...`），缺的只是 4-bit 量化依赖。

要让 Kaggle 也跑 5 成员，有两条路：

1. **把 wheel 放进数据集**（推荐）：本地 `pip download bitsandbytes --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.11`，把 `.whl` 传进源码数据集，在 notebook 里 `pip install --no-index --find-links /kaggle/input/... bitsandbytes`；
2. **不用 4-bit，改 fp16 + 双卡分片**：需要把 `QwenMember.load()` 的 `device_map={"": 0}` 改成 `"auto"`，并让池化在每层输出所在设备上计算。改动更大，但能真正用上第二张 T4。

## 5. 对现有 notebook 的接入

`kaggle_submit.ipynb` 的第二个单元已经把手工 `cfg.set(...)` 换成：

```python
profile = build_profile("kaggle")
cfg = load_config(str(CODE / "configs" / "real.yaml"), profile="kaggle")
```

并保留依赖预检：`bitsandbytes` 缺失时打印告警并关闭 Qwen，而不是让整次运行崩掉。这样本地与 Kaggle 的差异只剩下 profile，算法参数不可能被某一侧悄悄改掉。

## 6. 已知边界

- `synth` / `smoke` / `split` 不套用 profile（它们的输入输出本来就是本地临时文件），这是刻意的，避免 `smoke` 把合成数据写进 `data/real`。
- Kaggle 上 `data_dir` 用 `find_competition_dir()` 兼容两种挂载布局（`/kaggle/input/name-that-llm` 与 `/kaggle/input/competitions/name-that-llm`）。
- Qwen 探测按 `config.json` 的 `model_type` 前缀匹配，不写死镜像路径；换一个 Qwen3 镜像也能工作。
