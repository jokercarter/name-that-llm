# 两阶段 LoRA 在 Kaggle 上的运行说明

对应代码：`configs/kaggle_2stage_8b.yaml`、`scripts/train_parallel.py --stage all`、
`src/ntl_pact18/features/{qwen_member,qlora_head}.py`。

## 1. 运行环境

| 项 | 要求 | 说明 |
|---|---|---|
| Accelerator | **GPU T4 × 2** | 只选单卡会在 `_launch_ranks` 直接退出并打印可见 GPU 数 |
| Internet | **Off** | 所有依赖与权重都通过挂载获得 |
| 镜像 | Kaggle 最新容器 | 实测 torch 2.11+cu128 / Python 3.13 / transformers 5.x |
| 主机内存 | ~29 GB | 两个 rank 各持一份 8B 的 4-bit 副本 |
| 单卡显存 | 16 GB | 4-bit 权重 ~4.5 GB（截断 24 层）+ 激活 ~4 GB |

镜像**不自带 `bitsandbytes`**（这是首版 4 成员回退的直接原因）。仓库数据集里放了一个
cp311/manylinux 的 wheel，notebook 用离线方式安装：

```bash
pip install -q --no-index --find-links /kaggle/input/datasets/<owner>/ntl-pact18-verify bitsandbytes
```

装完必须 `import bitsandbytes`，失败就让 kernel 报错，不要静默退回 fp16（那会在 T4 上显存翻倍）。

## 2. 数据路径

Kaggle 现在把输入挂在 `/kaggle/input/{competitions,datasets,models}/<owner>/<slug>`，
旧的扁平路径 `/kaggle/input/name-that-llm` 已失效。notebook 用 `pick_dir()` 同时兼容两种布局。

| 用途 | 路径 |
|---|---|
| 竞赛数据 | `/kaggle/input/competitions/name-that-llm/{train.csv,test.csv}` |
| 代码 + wheel + 外部语料 | `/kaggle/input/datasets/jokercarter/ntl-pact18-verify/` |
| 8B 权重 | `/kaggle/input/models/manojkumarcs28/qwen3-8b/...`（用 `config.json` 的 `model_type=qwen3 & hidden_size>=4096` 自动发现） |
| 工作目录 | `/kaggle/working/ntl-pact18-code` |

数据集里的 `ntl-pact18-code.zip` 会被 Kaggle **自动解压**成
`ntl-pact18-code/ntl-pact18-code/`，所以 `unzip` 分支通常不会命中，走的是
`glob('**/src/ntl_pact18/pipeline.py')` + `copytree` 的兜底分支。

## 3. 必要配置

`configs/kaggle_2stage_8b.yaml` 的关键项：

```yaml
features.qwen:
  mode: two_stage
  truncate_layers: 24                  # 36 -> 24，后 12 层不实例化
  layers: [2,4,6,8,12,16,20,24]        # 8 层池化
  pool_mode: mean_concat               # 每层 LayerNorm + 掩码均值
  lora_r: 16 / alpha: 32 / dropout: 0.05
  triplet_batch: 3                     # 每卡每步 3 个三元组，全局 6
  grad_accum: 2
  stage1_epochs: 2 / stage1_lr: 1e-4   # 只用外部语料
  stage2_epochs: 2 / stage2_lr: 5e-5   # 只用主办方数据
  head_lr: 1e-3 / stage2_head_lr: 5e-4
  centred_weight: 1.0
  label_smoothing: 0.0 / supcon_weight: 0.0
```

`data.n_folds: 3`、`classical.max_fit_rows: 0`（经典成员用全量）。

PRISM 降到 `min_size: 4 / lam: 0.5`：隐藏测试集每个 prompt 只有约 2–3 段，
容量约束几乎没有可约束的空间。`experts.enabled: false`（公开实测 family gates 无收益）。

## 4. 已经处理的环境差异

| 差异 | 处理 |
|---|---|
| T4 不支持原生 bf16，但 `torch.cuda.is_bf16_supported()` 在 sm_75 上返回 True | 改用 `get_device_capability() >= (8,0)` 判定，T4 走 fp16 |
| 镜像无 bitsandbytes | 数据集内 wheel + `--no-index` 安装，并断言导入成功 |
| 输入挂载布局变更 | `pick_dir()` 兼容 `competitions/datasets` 新布局与旧扁平布局 |
| 数据集 zip 被自动解压 | 先试 `unzip`，再退回 `glob` + `copytree` |
| 两个 rank 同时流式加载 16 GB 权重 | rank 1 在 `load()` 前等 barrier，rank 0 加载完才放行（串行加载） |
| 交互态 `test.csv` 只有 3 行 | Sinkhorn 在 `len(test) < 50` 时跳过；PRISM 的 `min_size` 会自然跳过 |
| 9 小时硬上限 | `--deadline-hours` 把每次 fit 的时间预算按剩余时间夹紧，宁可截断也不让 kernel 被杀 |

## 5. 运行

```bash
python scripts/build_external_mix.py \
  --base /kaggle/input/competitions/name-that-llm/train.csv \
  --external /kaggle/input/datasets/*/ntl-pact18-verify/ntl_merged_external_large.parquet \
  --target-total 50000 --out /kaggle/working/train_2stage.csv

python scripts/train_parallel.py --stage all \
  --config configs/kaggle_2stage_8b.yaml \
  --train-file /kaggle/working/train_2stage.csv \
  --test-file /kaggle/input/competitions/name-that-llm/test.csv \
  --submission /kaggle/working/submission.parquet \
  --deadline-hours 8.0 --reserve-hours 0.5 \
  --set features.qwen.model_id=/kaggle/input/models/.../qwen3-8b
```

数据构建结果是 **46,573 行**：2,699 主办方 + 43,874 外部（每类上限 2,628，无重复补齐）。
其中可组成三元组的外部行约 30,950（4,810 个 prompt 组含 ≥3 个不同模型）。

## 6. 预期日志

```
[qwen] two-stage: pool=43874 rows / ... prompts | host=2699 rows / 150 prompts | folds=3
[qwen] rows=46573 ... mode=two_stage truncate=24
[qwen] loading ... (4bit=True)
[qwen] truncated trunk: 36 -> 24 blocks (upper blocks never materialised)
[qwen] stage1: 30950 pool passages -> 2 epoch(s)
[qwen] stage1: ... optimizer steps ... in ...s
[qwen] fold 1/3: train=1799 val=900
[qwen] stage2: train=1799 groups=100 world=2 triplets/rank=3 epochs=2 lr=5e-05/0.0005
...
[qwen] total X.XX h
[classical] ... [assemble] ... submission -> .../submission.parquet
```

## 7. 失败模式

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: ntl_pact18` | 在 kernel 进程里直接 import，`src` 不在 `sys.path` | 每个非子进程的 cell 首行 `sys.path.insert(0, f'{WORK}/src')` |
| `No module named bitsandbytes` | 镜像不自带 | 离线 wheel；装不上就显式报错 |
| `torch.OutOfMemoryError` during load | 两个 rank 同时加载 | 已串行化；若仍 OOM，把 `load_in_4bit` 保持 true 并降低 `max_len` |
| 日志只有 `loading` 没有 `truncated` | `truncate_layers` 未生效 | 检查 `config` 是否被 `--set` 覆盖 |
| `stage1: no triplet batch` | 外部数据里没有 ≥3 个不同模型同组 | 检查 `origin` 列与语料分组 |
| 训练 loss 不降 | lr 过高 / 三元组标签重复 | 看 `stage1` 前 50 步；确认 `_triplet_units` 的输出 |
