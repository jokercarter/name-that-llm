"""Build the "top1 + ModernBERT" submission kernel from the vendored baseline.

The baseline (fleongg, Public 0.636) has 18 members but no ModernBERT. The public
0.626 stack (nuridudak) shows a two-stage fine-tuned ModernBERT-large is worth
0.569 OOF on its own and is an orthogonal view: a bidirectional encoder keeps
case / punctuation / spacing exactly, which the decoder tokenizers discard.

This script inserts one additive cell right after the DeBERTa member:

  * mean-pooled ModernBERT embeddings at max_len 512, fp32 on cuda:0
  * an 18-way head fitted on the 2,699 host rows (StandardScaler + LogisticRegression),
    the same pattern comp_stack uses for its encoder members
  * vote("modernbert", p, 0.20) and a group-centred twin at 0.10
  * interactive-only: prints the head's prompt-grouped 5-fold OOF balanced accuracy

Nothing else changes: existing members, weights, ATLAS and Sinkhorn are untouched,
and all votes go through the notebook's own ``vote()`` so the final
``sum(MEMBER_EVIDENCE) == LOGIT`` assertion still holds.

Usage
-----
    python scripts/add_modernbert_member.py              # build + push
    python scripts/add_modernbert_member.py --build-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_NB = ROOT / "vendor" / "top1_fleongg" / "ntl-part-two-complement.ipynb"
BASE_META = ROOT / ".tmp" / "kpush_top1" / "kernel-metadata.json"
STAGE = ROOT / ".tmp" / "kpush_mb"
KAGGLE_PY = ROOT / ".tmp" / "kaggle-cli" / "Scripts" / "python.exe"
ENC_DATASET = "nuridudak/ntl-two-stage-encoders"   # Apache 2.0

MEMBER_CELL = '''# ---- complementary member: ModernBERT-large (two-stage fine-tuned, Apache-2.0) ----
# Borrowed from the public 0.626 stack (nuridudak): a bidirectional encoder keeps
# typography exactly (case, punctuation, spacing) -- a view the decoder LoRAs lose.
# We fit our own 18-way head on the 2,699 host rows, exactly like comp_stack does
# for its encoder members, so no head format assumptions are made.
MB_W = 0.20
MB_TWIN_W = 0.10
MB_MAXLEN = 512

mb_dir = os.path.dirname(find("modernbert/config.json"))
print(f"[modernbert] checkpoint {mb_dir}", flush=True)
mtok = AutoTokenizer.from_pretrained(mb_dir)
menc = AutoModel.from_pretrained(mb_dir, dtype=torch.float32).to("cuda:0").eval()


def mb_embed(texts, bs=16):
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), bs):
            batch = texts[s:s + bs]
            bt = mtok(batch, padding=True, truncation=True,
                      max_length=MB_MAXLEN, return_tensors="pt").to("cuda:0")
            h = menc(**bt).last_hidden_state
            m = bt["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.vstack(out)


t_mb = time.time()
Ztr_mb = mb_embed(train.text.astype(str).tolist())
Zte_mb = mb_embed(test.text.fillna("").astype(str).tolist())
print(f"[modernbert] embeddings train {Ztr_mb.shape} test {Zte_mb.shape} "
      f"in {time.time() - t_mb:.0f}s", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

mbs = StandardScaler().fit(Ztr_mb)
clf_mb = LogisticRegression(C=0.005, max_iter=5000).fit(mbs.transform(Ztr_mb), y)
P_mb = clf_mb.predict_proba(mbs.transform(Zte_mb))
full_mb = np.full((len(Zte_mb), K), 1e-12)
full_mb[:, clf_mb.classes_] = P_mb
full_mb /= full_mb.sum(1, keepdims=True)
vote("modernbert", full_mb, MB_W)


def _centre(Z, g):
    Zc = Z.copy()
    for _g in np.unique(g):
        m = g == _g
        if m.sum() >= 2:
            Zc[m] -= Zc[m].mean(0)
    return Zc


if twin_rows.any():
    p_tr = train["prompt_id"].to_numpy()
    mbs2 = StandardScaler().fit(_centre(Ztr_mb, p_tr))
    clf_mb2 = LogisticRegression(C=0.005, max_iter=5000).fit(
        mbs2.transform(_centre(Ztr_mb, p_tr)), y)
    Pc = clf_mb2.predict_proba(mbs2.transform(_centre(Zte_mb, groups)[twin_rows]))
    full_c = np.full((int(twin_rows.sum()), K), 1e-12)
    full_c[:, clf_mb2.classes_] = Pc
    full_c /= full_c.sum(1, keepdims=True)
    vote("modernbert twin", full_c, MB_TWIN_W, rows=twin_rows)
else:
    print("[modernbert] twin skipped: no clusters of 2-18 in this test set", flush=True)

# interactive-only evidence: solo prompt-grouped OOF of the head we just fitted
if len(test) < 50:
    def _bal(y_true, y_pred, k=K):
        rec = [(y_pred[y_true == c] == c).mean() for c in range(k) if (y_true == c).any()]
        return float(np.mean(rec))
    oof = np.zeros(len(y), dtype=int)
    for tr_i, va_i in GroupKFold(n_splits=5).split(Ztr_mb, y, groups=train["prompt_id"]):
        sc_ = StandardScaler().fit(Ztr_mb[tr_i])
        cl_ = LogisticRegression(C=0.005, max_iter=5000).fit(sc_.transform(Ztr_mb[tr_i]), y[tr_i])
        oof[va_i] = cl_.predict(sc_.transform(Ztr_mb[va_i]))
    print(f"[modernbert] solo prompt-grouped 5-fold OOF balanced acc = {_bal(y, oof):.4f}", flush=True)

del menc
gc.collect()
torch.cuda.empty_cache()
'''

HEADER = [
    "# 复现 + 增量：NTL Part Two Complement（Public 0.636）\n",
    "\n",
    "基线：[fleongg/ntl-part-two-complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement)（Public **0.636**）。\n",
    "基线部分为原样复现；**唯一改动是在主集成之后追加一个 ModernBERT-large 成员**（含簇内居中 twin），\n",
    "思路取自公开 0.626 方案（nuridudak）：双向编码器原样保留大小写/标点/空格/排版，是解码器 LoRA 看不到的正交视角。\n",
    "\n",
    "- 编码器权重：`nuridudak/ntl-two-stage-encoders`（Apache 2.0，ModernBERT + desklib 两阶段微调）\n",
    "- 分类头：在主办方 2,699 行上用标准化 + 逻辑回归**现场拟合**（与 comp_stack 对其编码器成员的做法一致）\n",
    "- 权重：plain 0.20 / centred twin 0.10；总计约 +8% 的对数概率质量\n",
    "- 其余 18 个成员、权重、ATLAS、Sinkhorn 全部不动\n",
]


def build(slug: str, owner: str) -> Path:
    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    idx = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and 'vote("deberta"' in "".join(c["source"]))
    nb["cells"].insert(idx + 1, {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": MEMBER_CELL.splitlines(keepends=True),
    })
    nb["cells"][0]["source"] = HEADER

    meta = json.loads(BASE_META.read_text(encoding="utf-8"))
    meta.update(id=f"{owner}/{slug}", title=slug)
    meta["dataset_sources"] = list(meta["dataset_sources"]) + [ENC_DATASET]

    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "ntl-part-two-complement.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (STAGE / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"staged {STAGE} | cells {len(nb['cells'])} "
          f"| datasets {len(meta['dataset_sources'])}")
    return STAGE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="ntl-top1-plus-modernbert")
    ap.add_argument("--owner", default="jokercarter")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    stage = build(args.slug, args.owner)
    if args.build_only:
        return 0
    return subprocess.call([str(KAGGLE_PY), "-m", "kaggle", "kernels", "push",
                            "-p", str(stage)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
