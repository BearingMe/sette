#!/usr/bin/env python3
"""M11: reconstruct the quality baselines and attribute error to stages.

Reads an M10 sequence directory (FP16 and INT8 frames produced from identical
inputs) and reports, without changing any arithmetic:

  * the three distinct measurements (reference-vs-FP16, reference-vs-INT8,
    FP16-vs-INT8) recomputed from the saved frames and checked against the
    accepted EXP-029 values;
  * per-boundary INT8-vs-FP16 error for every exported tensor, decoded in its
    own storage domain (E4M3, UNORM8, SNORM8, f16, f32, packed 11-11-10);
  * a ranked list by relative RMS, aggregated across frames.

Ranking is by measured boundary error and by the already published cumulative
ablations (EXP-028).  It is an attribution of *where* the executed error is
largest, not a causal leave-one-out experiment.

  m11_quality.py --sequence DIR [--out JSON] [--reference RAW_FRAME]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m_b6_frame as b6
import m09_cumulative as cu

#: boundary -> (path inside a frame directory, storage kind)
BOUNDARIES = {
    'enc0.tokens': ('enc0-native/out-tokens.bin', 'e4m3'),
    'chain.enc1': ('chain-native-results/out-enc1-out.bin', 'e4m3'),
    'chain.enc1.merge': ('chain-native-results/out-enc1-merge.bin', 'e4m3'),
    'chain.enc2': ('chain-native-results/out-enc2-out.bin', 'e4m3'),
    'chain.enc2.merge': ('chain-native-results/out-enc2-merge.bin', 'e4m3'),
    'chain.tube-1': ('chain-native-results/out-tube-1-out.bin', 'e4m3'),
    'chain.tube-2': ('chain-native-results/out-tube-2-out.bin', 'e4m3'),
    'chain.tube-3': ('chain-native-results/out-tube-3-out.bin', 'e4m3'),
    'chain.tube-4': ('chain-native-results/out-tube-4-out.bin', 'e4m3'),
    'chain.tube-5': ('chain-native-results/out-tube-5-out.bin', 'e4m3'),
    'chain.tube-6': ('chain-native-results/out-tube-6-out.bin', 'e4m3'),
    'chain.dec2': ('chain-native-results/out-dec2-out.bin', 'e4m3'),
    'chain.dec1': ('chain-native-results/out-dec1.bin', 'e4m3'),
    'dec0.gate': ('dec0-native/results/out-gate.bin', 'unorm8'),
    'dec0.mix': ('dec0-native/results/out-mix.bin', 'unorm8'),
    'dec0.feature': ('dec0-native/results/out-feat.bin', 'snorm8'),
    'dec0.covariance': ('dec0-native/results/out-cov.bin', 'f16'),
    'post.packed': ('post-native/results/out-packed.bin', 'packed'),
    'post.history': ('post-native/results/out-history.bin', 'f16'),
    'post.half': ('post-native/results/out-half.bin', 'f32'),
    'downsample.frame': ('downsample-native/results/out-out.bin', 'f16'),
}

#: Accepted EXP-029 values for the canonical frame, recomputed here from the raw
#: frames rather than trusted from the report.
ACCEPTED = {
    'reference_vs_fp16_db': 47.05138949457013,
    'reference_vs_int8_db': 44.13928631943929,
    'fp16_vs_int8_db': 44.48752927480509,
}


def decode(path, kind):
    raw = np.fromfile(path, np.uint8)
    if kind == 'e4m3':
        return cu.mw.e4m3_decode(raw).astype(np.float64)
    if kind == 'unorm8':
        return raw.astype(np.float64) / 255.0
    if kind == 'snorm8':
        signed = np.where(raw < 128, raw.astype(np.int16), raw.astype(np.int16) - 256)
        return np.clip(signed, -127, 127).astype(np.float64) / 127.0
    if kind == 'f16':
        return np.frombuffer(raw.tobytes(), '<f2').astype(np.float64)
    if kind == 'f32':
        return np.frombuffer(raw.tobytes(), '<f4').astype(np.float64)
    if kind == 'packed':
        word = np.frombuffer(raw.tobytes(), '<u4').astype(np.uint32)
        rgb = np.empty((word.size, 3), np.float64)
        rgb[:, 0] = (word & 2047) / 2047.0
        rgb[:, 1] = ((word >> 11) & 2047) / 2047.0
        rgb[:, 2] = ((word >> 22) & 1023) / 1023.0
        return rgb.reshape(-1)
    raise ValueError(kind)


def compare(a, b):
    """INT8 (`b`) against FP16 (`a`) in the tensor's own decoded domain."""
    d = b - a
    finite = np.isfinite(d)
    ref_rms = float(np.sqrt(np.mean(a[finite] ** 2))) if finite.any() else 0.0
    rms = float(np.sqrt(np.mean(d[finite] ** 2))) if finite.any() else None
    absd = np.abs(d[finite])
    peak = float(np.max(np.abs(a[finite]))) if finite.any() else 0.0
    return {
        'elements': int(a.size),
        'non_finite': int((~finite).sum()),
        'mean_abs': float(absd.mean()) if absd.size else None,
        'rms': rms,
        'relative_rms': (rms / ref_rms) if rms is not None and ref_rms else None,
        'reference_peak': peak,
        'rms_over_peak': (rms / peak) if rms is not None and peak else None,
        'max_abs': float(absd.max()) if absd.size else None,
        'p99_abs': float(np.percentile(absd, 99)) if absd.size else None,
        'exact_fraction': float(np.count_nonzero(d == 0) / d.size),
    }


def frame_metrics(ref, test):
    row = b6.metrics(ref, test)
    return {k: row[k] for k in ('psnr_peak1_db', 'ssim_windowed_luma', 'rms',
                                'mean_abs', 'p99_abs', 'max_abs', 'signed_mean',
                                'nan_or_inf_pairs', 'saturated_test')}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sequence', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--reference', type=Path, default=None,
                    help='canonical raw RGBA16F reference frame')
    ap.add_argument('--frames', default='',
                    help='comma-separated frame indices (default: every frame present)')
    a = ap.parse_args()
    root = a.sequence.resolve()
    sequence = json.loads((root / 'sequence.json').read_text())
    indices = ([int(x) for x in a.frames.split(',') if x.strip()]
               if a.frames else sorted(int(k) for k in sequence['per_frame']))
    reference = a.reference or (Path(sequence['fixtures']) / 'downsample' / 'ref-out.bin')

    report = {'sequence': str(root), 'frames': indices, 'reference': str(reference),
              'baselines': {}, 'boundaries': {}, 'ranking': [], 'status': 'running'}

    # ---- baselines: the three measurements are not interchangeable ----------
    ref = b6.frame(reference)
    fp16 = b6.frame(root / 'fp16' / f'frame-{indices[0]:02d}' /
                    'downsample-native' / 'results' / 'out-out.bin')
    int8 = b6.frame(root / 'int8' / f'frame-{indices[0]:02d}' /
                    'downsample-native' / 'results' / 'out-out.bin')
    baselines = {
        'reference_vs_fp16': frame_metrics(ref, fp16),
        'reference_vs_int8': frame_metrics(ref, int8),
        'fp16_vs_int8': frame_metrics(fp16, int8),
    }
    checks = {
        'reference_vs_fp16_db': baselines['reference_vs_fp16']['psnr_peak1_db'],
        'reference_vs_int8_db': baselines['reference_vs_int8']['psnr_peak1_db'],
        'fp16_vs_int8_db': baselines['fp16_vs_int8']['psnr_peak1_db'],
    }
    report['baselines'] = baselines
    report['baseline_reproduction'] = {
        name: {'measured': checks[name], 'accepted': ACCEPTED[name],
               'matches': abs(checks[name] - ACCEPTED[name]) < 1e-9}
        for name in ACCEPTED}
    report['measurement_contract'] = {
        'psnr': baselines['reference_vs_int8']['psnr_peak1_db'] is not None and
                'peak 1.0, linear RGB f16, finite pairs only',
        'alpha_separate': True,
        'reference_sha256': b6.sha(reference),
        'reference_provenance': 'canonical d4r capture for the same frame (B6)',
        'frame_pair': [indices[0], indices[0]],
    }

    # ---- per-boundary attribution across every frame ------------------------
    per_boundary = {name: [] for name in BOUNDARIES}
    for index in indices:
        left = root / 'fp16' / f'frame-{index:02d}'
        right = root / 'int8' / f'frame-{index:02d}'
        row = {'frame': index,
               'frame_metrics': frame_metrics(
                   b6.frame(left / 'downsample-native/results/out-out.bin'),
                   b6.frame(right / 'downsample-native/results/out-out.bin'))}
        for name, (relative, kind) in BOUNDARIES.items():
            lp, rp = left / relative, right / relative
            if not lp.is_file() or not rp.is_file():
                raise SystemExit(f'missing boundary {name}: {lp} / {rp}')
            cell = compare(decode(lp, kind), decode(rp, kind))
            cell['kind'] = kind
            per_boundary[name].append({'frame': index, **cell})
        report['boundaries'][str(index)] = row

    summary = {}
    for name, rows in per_boundary.items():
        rel = [r['relative_rms'] for r in rows if r['relative_rms'] is not None]
        over = [r['rms_over_peak'] for r in rows if r['rms_over_peak'] is not None]
        summary[name] = {
            'kind': rows[0]['kind'], 'frames': len(rows),
            'relative_rms_median': float(np.median(rel)),
            'relative_rms_min': float(np.min(rel)),
            'relative_rms_max': float(np.max(rel)),
            'rms_over_peak_median': float(np.median(over)),
            'reference_peak_median': float(np.median([r['reference_peak'] for r in rows])),
            'max_abs_max': float(max(r['max_abs'] for r in rows)),
            'exact_fraction_median': float(np.median([r['exact_fraction'] for r in rows])),
            'non_finite': int(sum(r['non_finite'] for r in rows)),
        }
    report['boundary_summary'] = summary
    report['ranking'] = sorted(summary, key=lambda n: summary[n]['rms_over_peak_median'],
                               reverse=True)
    report['per_frame'] = {str(i): next(r for r in per_boundary['downsample.frame']
                                       if r['frame'] == i) for i in indices}

    ok = (all(v['matches'] for v in report['baseline_reproduction'].values())
          and all(v['non_finite'] == 0 for v in summary.values())
          and all(report['boundaries'][str(i)]['frame_metrics']['nan_or_inf_pairs'] == 0
                  for i in indices))
    report['checks'] = {
        'baselines_reproduce': all(v['matches'] for v in report['baseline_reproduction'].values()),
        'all_boundaries_present': len(summary) == len(BOUNDARIES),
        'boundaries_finite': all(v['non_finite'] == 0 for v in summary.values()),
        'frames_finite': all(report['boundaries'][str(i)]['frame_metrics']['nan_or_inf_pairs'] == 0
                             for i in indices),
    }
    report['status'] = 'PASS' if ok else 'FAIL'
    out = a.out or (root / 'm11-quality.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'],
                      'baselines': {k: round(v['psnr_peak1_db'], 6)
                                    for k, v in baselines.items()},
                      'ranking': [(n, round(summary[n]['rms_over_peak_median'], 6))
                                  for n in report['ranking'][:10]]}, indent=2))
    print('out:', out)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
