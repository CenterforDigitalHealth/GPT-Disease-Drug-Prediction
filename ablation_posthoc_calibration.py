#!/usr/bin/env python3
"""Ablation runner for post-hoc calibration only (single checkpoint, multiple evals)."""
from __future__ import annotations

import argparse, csv, json, os, shlex, subprocess, sys, time
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open('w', encoding='utf-8') as f:
        f.write('$ ' + ' '.join(shlex.quote(c) for c in cmd) + '\n\n')
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} | see {log_path}")
    return time.time() - t0


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(',') if x.strip()]


def _load_metrics(eval_out: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for fp in sorted(eval_out.glob('*_composite_metrics.json')):
        out[fp.name.replace('_composite_metrics.json', '')] = json.loads(fp.read_text(encoding='utf-8'))
    return out


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Ablation: post-hoc calibration only')
    p.add_argument('--gpu_ids', type=str, default='0')
    p.add_argument('--input_path', type=str, default='../data')
    p.add_argument('--output_root', type=str, default='ablation/posthoc_calibration')
    p.add_argument('--run_name', type=str, default='default')
    p.add_argument('--model_ckpt_path', type=str, required=True)
    p.add_argument('--posthoc_calibrations', type=str, default='none,affine')
    p.add_argument('--dataset_subset_size', type=int, default=10000)
    p.add_argument('--eval_batch_size', type=int, default=64)
    p.add_argument('--data_files', type=str, default='kr_val.bin,kr_test.bin,JMDC_extval.bin')
    p.add_argument('--extra_eval_args', type=str, default='')
    args = p.parse_args()

    calibrations = _parse_list(args.posthoc_calibrations)
    bad = [x for x in calibrations if x not in {'none', 'affine'}]
    if bad:
        raise ValueError(f'Unknown calibration methods: {bad}')

    repo = Path(__file__).resolve().parent
    run_root = (repo / args.output_root / args.run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / 'ablation_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')

    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    extra_eval = shlex.split(args.extra_eval_args) if args.extra_eval_args.strip() else []

    ckpt = Path(args.model_ckpt_path).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')

    rows: List[Dict[str, object]] = []
    for i, cal in enumerate(calibrations, 1):
        trial_name = f't{i:02d}_calibration-{cal}'
        root = run_root / trial_name
        eval_out, logs = root / 'eval_out', root / 'logs'

        eval_cmd = [
            sys.executable, '-m', 'evaluate_auc_v6', f'--input_path={args.input_path}',
            f'--model_ckpt_path={ckpt}', f'--output_path={eval_out}',
            f'--dataset_subset_size={args.dataset_subset_size}', f'--eval_batch_size={args.eval_batch_size}',
            f'--data_files={args.data_files}', f'--posthoc_calibration={cal}'
        ] + extra_eval
        eval_sec = _run(eval_cmd, repo, env, logs / 'eval.log')

        metrics = _load_metrics(eval_out)
        row: Dict[str, object] = {'trial_id': i, 'trial_name': trial_name, 'posthoc_calibration': cal, 'eval_seconds': round(eval_sec, 2)}
        for prefix, m in metrics.items():
            for k in ('auc_mean', 'shift_rmse_drug_cond', 'shift_r2_drug_cond', 'total_rmse_drug_cond', 'total_r2_drug_cond'):
                if k in m:
                    row[f'{prefix}.{k}'] = m[k]
        rows.append(row)
        (root / 'trial_summary.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        print(f'[done] {trial_name}')

    _write_csv(rows, run_root / 'summary.csv')
    (run_root / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
