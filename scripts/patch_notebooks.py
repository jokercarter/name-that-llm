"""Patch the two-stage Kaggle notebooks: robust model-source detection.

The recursive ``/kaggle/input/**/config.json`` scan silently fails for some
model mounts (the PyTorch-framework ``manojkumarcs28/qwen3-8b`` instance never
appeared, which is what killed the first smoke run). The public 0.55x kernels
prove ``qwen-lm/qwen-3/Transformers/8b-base/1`` mounts where the recursive scan
finds it, so the detection now tries explicit single-level paths first (single
``*`` globs do match symlinked mount levels) and keeps the recursive scan as a
fallback.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELL2_TEMPLATE = '''import glob, os, sys, time
import json

import torch

print('python  :', sys.version.split()[0])
print('torch   :', torch.__version__, '| cuda', torch.version.cuda)
print('devices :', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  gpu{i}: {p.name} | {p.total_memory / 2**30:.1f} GiB | cc {p.major}.{p.minor}')
print('input   :', sorted(os.listdir('/kaggle/input')))


def pick_dir(patterns, marker):
    for pat in patterns:
        for cand in sorted(glob.glob(pat)):
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, marker)):
                return cand
    return None

DS = pick_dir(['/kaggle/input/datasets/*/ntl-pact18-verify',
               '/kaggle/input/ntl-pact18-verify',
               '/kaggle/input/*/ntl-pact18-verify'], 'ntl_merged_external_large.parquet')
COMP = pick_dir(['/kaggle/input/competitions/name-that-llm',
                 '/kaggle/input/name-that-llm'], 'test.csv')

# Model source qwen-lm/qwen-3/Transformers/8b-base/1. Explicit single-level
# paths first: each '*' matches one symlinked mount level (a recursive '**'
# scan does not reliably descend into the mount symlinks). The recursive scan
# stays as the fallback - it is what the public 0.55x kernels use.
MODEL = pick_dir([
    '/kaggle/input/models/qwen-lm/qwen-3/Transformers/8b-base/1',
    '/kaggle/input/models/*/qwen-3/Transformers/8b-base/1',
    '/kaggle/input/models/*/*/Transformers/8b-base/1',
    '/kaggle/input/qwen-3',
    '/kaggle/input/*/qwen-3',
], 'config.json')
if MODEL is None:
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            meta = json.loads(open(cfg, encoding='utf-8').read())
        except Exception:
            continue
        if str(meta.get('model_type', '')).startswith('qwen3') and int(meta.get('hidden_size', 0)) >= 4096:
            MODEL = os.path.dirname(cfg)
            break
print('DS   :', DS)
print('COMP :', COMP)
print('MODEL:', MODEL)
assert DS and COMP and MODEL, 'Add Input: competition + ntl-pact18-verify + qwen-lm/qwen-3 (Transformers/8b-base/1)'
print('dataset files:', sorted(os.listdir(DS))[:8])
'''


def patch(notebook: Path) -> None:
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    cells = nb['cells']
    target = None
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code' and 'pick_dir' in ''.join(c['source']):
            target = i
            break
    assert target is not None, f'{notebook.name}: no detection cell found'
    lines = CELL2_TEMPLATE.rstrip('\n').split('\n')
    cells[target]['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells[target].setdefault('id', 'env-detect')
    cells[target]['outputs'] = []
    cells[target]['execution_count'] = None
    notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'patched {notebook.name} (cell {target})')


if __name__ == '__main__':
    for name in ('kaggle_2stage_smoke.ipynb', 'kaggle_2stage_full.ipynb'):
        patch(ROOT / name)
