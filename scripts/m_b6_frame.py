#!/usr/bin/env python3
"""B6: the complete native frame, with execution and fidelity judged separately.

Runs enc0 -> ten native layers -> dec0 -> post -> downsample with no captured
tensor substituted between neural stages.  The only captured inputs that remain
are the stage inputs the network does not produce (colour and motion).

Three things are kept apart on purpose:

  execution   did every declared stage run, produce its declared output at the
              declared size and format, stay finite and unsaturated, and repeat
              byte for byte?  This gate is declared in `EXECUTION` and fails
              closed.
  fidelity    how close is the frame to a reference?  Floors in `REGRESSION` are
              regression bounds around the measured control paths, not a claim
              of fidelity.  A fidelity verdict of PASS needs a predeclared
              contract and an independent validation set; with none available
              the verdict is INCONCLUSIVE and nothing above is reported as an
              oracle result.
  verdict     PASS only when both hold.  EXECUTORY is the honest label for a
              clean execution whose fidelity cannot be established.

Paths measured against the canonical captured frame:

  ref        canonical captured frame
  image      image stages fed the captured dec1 field (image-stage error alone)
  native     image stages fed the native ten-layer chain (this reconstruction)
  translated image stages fed the translated-PTX chain (independent chain)

  m_b6_frame.py --out NEW_DIR [--fixtures DIR] [--layers DIR]
                [--translated-dec1-capture REPLAY_DIR] [--reuse]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIXTURES = Path(os.environ.get(
    'SETTE_FIXTURES',
    Path(os.path.expanduser('~')) / 'AppData/Local/Temp/opencode/m-capture-run/fixtures'))
TEMP = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))) / 'Temp/opencode'
LAYERS = Path(os.environ.get('SETTE_LAYERS', TEMP / 'm-capture-run/layers'))
W, H = 1920, 1080
DEBUG = shutil.ignore_patterns('results*', 'dbg*', 'p[0-9]*', 'q[0-9]*', 'r[0-9]*',
                               's[0-9]*', 'd[0-9]*')

#: Bytes per texel for every format a declared stage output uses.
FORMAT_BYTES = {'R16G16B16A16_FLOAT': 8, 'R32G32B32A32_FLOAT': 16, 'R8G8B8A8_UNORM': 4,
                'R32_UINT': 4, 'R16G16_FLOAT': 4, 'R16_FLOAT': 2, 'R8_UNORM': 1}

#: Declared before the run.  A gated run needs all of it, and reuse cannot
#: supply the repeat half of it.
EXECUTION = {
    'stages': ['enc0', 'chain', 'dec0', 'post', 'downsample'],
    'frame': {'height': H, 'width': W, 'channels': 4, 'dtype': 'float16'},
    'nan_or_inf_max': 0,
    'saturated_max': 0,
    'repeat_required_for_pass': True,
    'boundary_hashes_required': True,
}

#: Regression bounds on the canonical frame.  These are *not* a fidelity claim:
#: the image-stage control (`image`, i.e. every stage except the neural chain fed
#: its captured input) already sits at 46.73 dB / 0.99791 SSIM, so a 45 dB / 0.99
#: floor cannot be met by a reconstruction defect and is only there to catch one.
REGRESSION = {
    'psnr_peak1_db_min': 45.0,
    'ssim_windowed_luma_min': 0.99,
    'mean_abs_max': 0.005,
    'max_abs_max': 0.5,
    'chain_gap_psnr_peak1_db_min': 42.0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(probe, stage, out, shaders, seed=None, timing=None, vram=None):
    if Path(out).is_dir():
        shutil.rmtree(out)
    command = [str(probe), '--fixture', str(stage), '--out', str(out), '--shaders', str(shaders)]
    if seed is not None:
        command += ['--seed', str(seed)]
    if timing is not None:
        command += ['--timing', str(timing)]
    if vram is not None:
        command += ['--vram', str(vram)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'{stage}: {done.stdout[-600:]}{done.stderr[-600:]}')
    (Path(out) / 'execution.log').write_text(done.stdout + done.stderr)
    return Path(out)


def clone_fixture(name, dst):
    if dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(FIXTURES / name, dst, ignore=DEBUG)
    return dst


def declared_outputs(stage_txt):
    """{role: bytes} for every output a fixture declares, read from its manifest."""
    out = {}
    for line in Path(stage_txt).read_text().splitlines():
        words = line.split()
        if words and words[0] == 'dec0_int8':
            import m_stage_fixture as sf
            out.update(sf.DEC0_DIAGNOSTICS)
        if words and words[0] == 'enc0_int8':
            import m09_complete as complete
            out.update(complete.ENC0_DIAGNOSTICS)
        if not words or words[0] != 'uav':
            continue
        if words[1] == 'buffer':
            name, size = words[3], int(words[4])
        elif words[1] == 'tex2d':
            fmt, w, h = words[3], int(words[4]), int(words[5])
            name, size = words[6], w * h * FORMAT_BYTES[fmt]
        else:
            raise RuntimeError(f'{stage_txt}: unknown uav kind {words[1]}')
        if not (name.startswith('out-') and name.endswith('.bin')):
            raise RuntimeError(f'{stage_txt}: unexpected output name {name}')
        out[name[len('out-'):-len('.bin')]] = size
    return out


def build_chain(layers, out, enc1_input, q_int8=False, enc1_int8=None,
                mlp_int8=False, enc1_mlp=None, mlp_mode=None):
    sys.path.insert(0, str(HERE))
    import m_layer_suite as suite
    info = suite.build_chain(str(layers), str(out), q_int8=q_int8, enc1_int8=enc1_int8,
                             mlp_int8=mlp_int8, enc1_mlp=enc1_mlp, mlp_mode=mlp_mode)
    graph = out / 'graph.txt'
    lines = graph.read_text().splitlines()
    replaced = False
    for i, line in enumerate(lines):
        words = line.split()
        if words[:2] == ['resource', 'enc1_in']:
            lines[i] = ' '.join(words[:3] + [str(enc1_input)])
            replaced = True
    if not replaced:
        raise RuntimeError('chain graph has no enc1_in resource')
    graph.write_text('\n'.join(lines) + '\n')
    return info


def run_chain(network_probe, chain_dir, out, shaders, seed=0, timing=None, vram=None,
              mode='diagnostic', arena=False):
    command = [str(network_probe), '--graph', str(chain_dir / 'graph.txt'),
               '--out', str(out), '--shaders', str(shaders), '--seed', str(seed),
               '--mode', mode]
    if arena:
        command += ['--arena']
    if timing is not None:
        command += ['--timing', str(timing)]
    if vram is not None:
        command += ['--vram', str(vram)]
        command += ['--arena-report', str(Path(vram).with_name(Path(vram).stem + '-arena.json'))]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'chain: {done.stdout[-600:]}{done.stderr[-600:]}')
    (Path(out) / 'execution.log').write_text(done.stdout + done.stderr)
    return Path(out) / 'out-dec1.bin'


def image_stages(dec1, tag, out, probe, shaders, dec0_mode='fp16', seed=None, timing_dir=None,
                 vram_dir=None):
    """dec0 -> post -> downsample, with the neural chain's dec1 as dec0's tokens."""
    dec0 = clone_fixture('dec0', out / f'dec0-{tag}')
    shutil.copyfile(dec1, dec0 / 'in-tokens.bin')
    if dec0_mode != 'fp16':
        import m_stage_fixture as sf
        sf.prepare_dec0_mode(dec0, dec0_mode)
    res0 = run(probe, dec0 / 'stage.txt', dec0 / 'results', shaders, seed=seed,
               timing=(timing_dir / f'dec0-{tag}.json') if timing_dir else None,
               vram=(vram_dir / f'dec0-{tag}.json') if vram_dir else None)

    post = clone_fixture('post', out / f'post-{tag}')
    for role in ('gate', 'mix', 'cov'):
        shutil.copyfile(res0 / f'out-{role}.bin', post / f'in-{role}.bin')
    res1 = run(probe, post / 'stage.txt', post / 'results', shaders,
               timing=(timing_dir / f'post-{tag}.json') if timing_dir else None,
               vram=(vram_dir / f'post-{tag}.json') if vram_dir else None)

    down = clone_fixture('downsample', out / f'downsample-{tag}')
    shutil.copyfile(res1 / 'out-packed.bin', down / 'in-src.bin')
    res2 = run(probe, down / 'stage.txt', down / 'results', shaders,
               timing=(timing_dir / f'downsample-{tag}.json') if timing_dir else None,
               vram=(vram_dir / f'downsample-{tag}.json') if vram_dir else None)
    return {'frame': res2 / 'out-out.bin',
            'stages': {'dec0': {'fixture': dec0, 'results': res0},
                       'post': {'fixture': post, 'results': res1},
                       'downsample': {'fixture': down, 'results': res2}}}


def boundary_hashes(stages):
    """Every declared output of every stage, keyed '<stage>/<role>'."""
    out = {}
    for name, info in stages.items():
        if 'file' in info:
            out[f'{name}/out-dec1'] = sha(info['file'])
            continue
        fixture, results = Path(info['fixture']), Path(info['results'])
        for role in declared_outputs(fixture / 'stage.txt'):
            out[f'{name}/out-{role}'] = sha(results / f'out-{role}.bin')
    return out


def frame(path):
    """The final RGBA16F frame, 1920x1080, in its native layout."""
    data = np.fromfile(path, np.float16)
    if data.size != H * W * 4:
        raise RuntimeError(f'{path}: {data.size} f16 values, expected {H * W * 4}')
    return data.reshape(H, W, 4)


GAUSS_SIGMA, GAUSS_RADIUS = 1.5, 5
GAUSS = np.exp(-0.5 * (np.arange(-GAUSS_RADIUS, GAUSS_RADIUS + 1) / GAUSS_SIGMA) ** 2)
GAUSS /= GAUSS.sum()


def _conv1d(a, kernel, axis, slab=128):
    """Valid-shape-free, zero-padded separable convolution along one axis."""
    n = kernel.size
    pad = n // 2
    pad_width = [(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)]
    padded = np.pad(a, pad_width, mode='reflect')
    view = np.lib.stride_tricks.sliding_window_view(padded, n, axis=axis)
    if axis == 0:
        pieces = [np.tensordot(view[i:i + slab], kernel, axes=([-1], [0]))
                  for i in range(0, view.shape[0], slab)]
        return np.concatenate(pieces, axis=0)
    return np.tensordot(view, kernel, axes=([-1], [0]))


def _blur(a):
    return _conv1d(_conv1d(a, GAUSS, 0), GAUSS, 1)


def luma(rgb):
    rgb = np.asarray(rgb, np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def ssim_windowed_luma(ref, test, data_range=1.0):
    """Standard SSIM: 11x11 Gaussian window (sigma 1.5), K1=0.01, K2=0.03, L=data_range.

    Means, variances and covariance are the Gaussian-weighted moments of the
    window (no N/(N-1) correction, which is what the reference definition uses),
    applied to BT.709 luma of the RGB channels.  Identical frames give exactly 1.
    """
    a, b = luma(ref[..., :3]), luma(test[..., :3])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    mu_a, mu_b = _blur(a), _blur(b)
    va = _blur(a * a) - mu_a * mu_a
    vb = _blur(b * b) - mu_b * mu_b
    cov = _blur(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(np.mean(num / den))


def ssim_block8_luma(ref, test):
    """The original non-overlapping 8x8 block metric, kept as a diagnostic only."""
    block = 8
    a, b = luma(ref[..., :3]), luma(test[..., :3])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    gh, gw = a.shape[0] // block, a.shape[1] // block
    a = a[:gh * block, :gw * block].reshape(gh, block, gw, block).transpose(0, 2, 1, 3)
    b = b[:gh * block, :gw * block].reshape(gh, block, gw, block).transpose(0, 2, 1, 3)
    mu_a, mu_b = a.mean(axis=(2, 3)), b.mean(axis=(2, 3))
    va = a.var(axis=(2, 3), ddof=0)
    vb = b.var(axis=(2, 3), ddof=0)
    cov = ((a - mu_a[..., None, None]) * (b - mu_b[..., None, None])).mean(axis=(2, 3))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                  ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())


def metrics(ref, test, border=32):
    """Error of `test` against `ref` over finite RGB pairs, with alpha kept separate."""
    a = ref[..., :3].astype(np.float64)
    b = test[..., :3].astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    excluded = int((~finite).sum())
    # Non-finite pairs are excluded from every statistic and counted, never
    # replaced by zero: a zero would hide exactly the cases worth failing on.
    d = np.where(finite, b - a, np.nan)
    absd = np.abs(d)
    mse = float(np.nanmean(d ** 2)) if excluded < d.size else None
    peak = float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = absd / (np.abs(a) + 1e-6)
    luma_abs = 0.2126 * absd[..., 0] + 0.7152 * absd[..., 1] + 0.0722 * absd[..., 2]
    block = 16
    gh, gw = luma_abs.shape[0] // block, luma_abs.shape[1] // block
    per_block = np.nanmean(luma_abs[:gh * block, :gw * block]
                           .reshape(gh, block, gw, block), axis=(1, 3))
    worst = np.dstack(np.unravel_index(np.argsort(per_block.ravel())[-5:][::-1],
                                       per_block.shape))[0]
    inner = luma_abs[border:-border, border:-border]
    edge = np.ones(luma_abs.shape, bool)
    edge[border:-border, border:-border] = False
    alpha_a = ref[..., 3].astype(np.float64)
    alpha_b = test[..., 3].astype(np.float64)
    alpha_d = np.abs(alpha_b - alpha_a)
    return {
        'data_range': 1.0,
        'psnr_basis': 'linear RGB f16, peak 1.0, finite pairs only',
        'samples_rgb': int(a.size),
        'excluded_non_finite_pairs': excluded,
        'finite_pairs': int(finite.sum()),
        'psnr_peak1_db': float(10 * np.log10(1.0 / mse)) if mse else None,
        'psnr_peak_reference_db': float(10 * np.log10(peak ** 2 / mse)) if mse else None,
        'reference_peak': peak,
        'ssim_windowed_luma': ssim_windowed_luma(ref, test),
        'ssim_algorithm': 'windowed, 11x11 Gaussian sigma 1.5, K1 0.01, K2 0.03, L 1.0, '
                          'BT.709 luma',
        'ssim_block8_luma_legacy': ssim_block8_luma(ref, test),
        'mean_abs': float(np.nanmean(absd)),
        'rms': float(np.sqrt(mse)) if mse is not None else None,
        'p50_abs': float(np.nanpercentile(absd, 50)),
        'p99_abs': float(np.nanpercentile(absd, 99)),
        'max_abs': float(np.nanmax(absd)),
        'mean_relative': float(np.nanmean(rel)),
        'signed_mean': float(np.nanmean(d)),
        'block_mean_abs_min': float(np.nanmin(per_block)),
        'block_mean_abs_median': float(np.nanmedian(per_block)),
        'block_mean_abs_max': float(np.nanmax(per_block)),
        'worst_blocks_yx': np.asarray(worst).astype(int).tolist(),
        'border_margin': border,
        'border_mean_abs': float(np.nanmean(luma_abs[edge])),
        'interior_mean_abs': float(np.nanmean(inner)) if inner.size else None,
        'alpha_identical': int((alpha_a == alpha_b).sum()),
        'alpha_samples': int(alpha_a.size),
        'alpha_mean_abs': float(alpha_d.mean()),
        'alpha_max_abs': float(alpha_d.max()),
        'nan_or_inf_pairs': excluded,
        'nan_or_inf_reference': int((~np.isfinite(a)).sum()),
        'nan_or_inf_test': int((~np.isfinite(b)).sum()),
        'saturated_reference': int((np.abs(a) >= 65504).sum()),
        'saturated_test': int((np.abs(b) >= 65504).sum()),
    }


def export(prefix, ref, test):
    def tonemap(x):
        y = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return (np.power(y, 1 / 2.2) * 255 + 0.5).astype(np.uint8)

    diff = np.abs(test.astype(np.float64) - ref.astype(np.float64))
    scale = float(np.nanpercentile(diff[..., :3], 99.9)) or 1.0
    alpha_scale = float(np.nanpercentile(diff[..., 3], 99.9)) or 1.0
    reference = np.dstack([tonemap(ref[..., :3]), tonemap(ref[..., 3:4])])
    reconstructed = np.dstack([tonemap(test[..., :3]), tonemap(test[..., 3:4])])
    difference = np.dstack([tonemap(np.clip(diff[..., :3] / scale, 0, 1)),
                            tonemap(np.clip(diff[..., 3:4] / alpha_scale, 0, 1))])
    paths = {}
    for name, image in (('reference', reference), ('reconstructed', reconstructed),
                        ('difference', difference)):
        path = prefix.with_name(prefix.name + '-' + name + '.png')
        Image.fromarray(image, 'RGBA').save(path)
        paths[name] = str(path)
    return {'images': paths, 'rgb_difference_scale': scale,
            'alpha_difference_scale': alpha_scale,
            'note': '2.2-gamma display mapping for viewing only; metrics use raw f16'}


def input_manifest(a):
    """Everything whose change would invalidate a reused result."""
    fixtures = {}
    for stage in ('enc0', 'dec0', 'post', 'downsample'):
        directory = FIXTURES / stage
        fixtures[stage] = {p.name: sha(p) for p in sorted(directory.iterdir())
                           if p.is_file()}
    layers = {}
    for directory in sorted(LAYERS.iterdir()):
        if not directory.is_dir():
            continue
        layers[directory.name] = {p.name: sha(p) for p in sorted(directory.iterdir())
                                  if p.is_file()}
    overrides = {}
    for name in ('enc1_int8', 'enc1_mlp', 'cumulative_layers'):
        directory = getattr(a, name, None)
        if directory:
            files = directory.rglob('*') if name == 'cumulative_layers' else directory.iterdir()
            overrides[name] = {str(p.relative_to(directory)): sha(p)
                               for p in sorted(files) if p.is_file()}
    return {'probe_sha256': sha(a.probe), 'network_probe_sha256': sha(a.network_probe),
             'enc1_overrides': overrides,
             'colour': (str(a.colour) if getattr(a, 'colour', None) else None),
             'colour_sha256': (sha(a.colour) if getattr(a, 'colour', None) else None),
             'mlp_mode': getattr(a, 'mlp_mode', 'fp16'),
             'cumulative_mode': getattr(a, 'cumulative_mode', 'none'),
             'dec0_mode': getattr(a, 'dec0_mode', 'fp16'),
             'enc0_mode': getattr(a, 'enc0_mode', 'fp16'),
             'attention_mode': getattr(a, 'attention_mode', 'fp16'),
             'q_mode': getattr(a, 'q_mode', 'fp16'),
            'shaders': {p.name: sha(p) for p in sorted(a.shaders.glob('*.cso'))},
            'fixtures': fixtures, 'layers': layers,
            'translated_dec1_capture': (str(a.translated_dec1_capture)
                                        if a.translated_dec1_capture else None)}


def judge_execution(report, contract=EXECUTION):
    """Pure decision over a finished report.  Every check must pass; nothing defaults."""
    checks, reasons = {}, []

    def check(name, ok, detail, performed=True):
        checks[name] = {'ok': bool(ok), 'detail': detail, 'performed': bool(performed)}
        if not ok:
            reasons.append(f'{name}: {detail}')

    stages = report.get('stages', {})
    check('stages_present',
          all(stages.get(name) for name in contract['stages']),
          f"declared {contract['stages']}, ran {sorted(k for k, v in stages.items() if v)}")

    missing = report.get('missing_outputs') or []
    check('declared_outputs', not missing, f'{len(missing)} missing or mis-sized: {missing}')

    frame = report.get('frame', {})
    want = contract['frame']
    check('frame_shape',
          frame.get('height') == want['height'] and frame.get('width') == want['width']
          and frame.get('channels') == want['channels'] and frame.get('dtype') == want['dtype'],
          f'got {frame.get("height")}x{frame.get("width")}x{frame.get("channels")} '
          f'{frame.get("dtype")}')

    bad = report.get('non_finite', {})
    check('finite_frame', bad.get('frame', 0) <= contract['nan_or_inf_max'],
          f'{bad.get("frame", 0)} non-finite frame values')
    check('finite_dec1', bad.get('dec1', 0) <= contract['nan_or_inf_max'],
          f'{bad.get("dec1", 0)} non-finite dec1 codes')
    check('no_unexpected_saturation',
          bad.get('saturated', 0) <= contract['saturated_max'],
          f'{bad.get("saturated", 0)} samples at the f16 limit')

    repeat = report.get('repeat', {})
    if contract['repeat_required_for_pass']:
        fresh = bool(repeat.get('fresh'))
        check('deterministic',
              bool(repeat.get('deterministic')) and fresh,
              ('a fresh repeat run was not performed' if not fresh else
               str(repeat.get('mismatched') or 'no boundary mismatch')),
              performed=fresh)

    if contract['boundary_hashes_required']:
        boundaries = report.get('boundaries', {})
        check('boundary_hashes', len(boundaries) >= 7,
              f'{len(boundaries)} boundary hashes recorded')

    unproven = [name for name, row in checks.items() if not row['performed']]
    status = 'PASS' if not reasons else ('FAIL' if not unproven else 'NOT-EXECUTED')
    return {'status': status, 'checks': checks, 'reasons': reasons,
            'not_performed': unproven}


def judge_regression(report, bounds=REGRESSION):
    """Regression bounds on the canonical frame.  Not a fidelity claim."""
    rows, reasons = {}, []
    for case in ('native', 'image'):
        row = report.get('comparisons', {}).get(case)
        if row is None:
            reasons.append(f'{case}: not measured')
            continue
        checks = {
            'psnr_peak1_db': (row.get('psnr_peak1_db') or 0.0) >= bounds['psnr_peak1_db_min'],
            'ssim_windowed_luma': row.get('ssim_windowed_luma', 0.0)
            >= bounds['ssim_windowed_luma_min'],
            'mean_abs': row.get('mean_abs', 1.0) <= bounds['mean_abs_max'],
            'max_abs': row.get('max_abs', 1.0) <= bounds['max_abs_max'],
            'finite': row.get('nan_or_inf_test', 1) == 0,
            'unsaturated': row.get('saturated_test', 1) == 0,
        }
        rows[case] = checks
        reasons += [f'{case}.{k}' for k, ok in checks.items() if not ok]
    gap = report.get('chain_gap', {}).get('image-vs-native', {})
    if (gap.get('psnr_peak1_db') or 0.0) < bounds['chain_gap_psnr_peak1_db_min']:
        reasons.append('chain_gap: the neural chain degrades the frame beyond the bound')
    return {'status': 'PASS' if not reasons else 'FAIL', 'bounds': bounds, 'cases': rows,
            'reasons': reasons}


def verify_reuse(previous, manifest, hashes):
    """Refuse to reuse outputs that a different execution could have produced."""
    if previous.get('inputs') != manifest:
        raise SystemExit('reuse refused: the recorded inputs do not match this '
                         'execution (probe, shaders, fixtures or layers changed)')
    recorded = (previous.get('repeat') or {}).get('hashes') or {}
    if not recorded:
        raise SystemExit('reuse refused: no recorded repeat hashes to verify against')
    differing = sorted(k for k in hashes if recorded.get(k) != hashes[k])
    if differing:
        raise SystemExit('reuse refused: outputs do not match the recorded run: '
                         + ', '.join(differing))
    return recorded


def validate_precision_modes(q_mode, dec0_mode, mlp_mode='fp16'):
    if sum(mode != 'fp16' for mode in (q_mode, dec0_mode, mlp_mode)) > 1:
        raise SystemExit('combining experimental INT8 modes is not authorized')


def main():
    global FIXTURES, LAYERS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fixtures', type=Path, default=FIXTURES)
    ap.add_argument('--layers', type=Path, default=LAYERS)
    ap.add_argument('--colour', type=Path, default=None,
                    help='override the enc0 960x540 RGBA16F colour input for this frame '
                         '(temporal sequences); the shared fixture is never modified')
    ap.add_argument('--translated-dec1-capture', type=Path, default=None)
    ap.add_argument('--probe', type=Path,
                    default=REPO / 'build/native/Release/m_image_stage.exe')
    ap.add_argument('--network-probe', type=Path,
                    default=REPO / 'build/native/Release/m_network_stage.exe')
    ap.add_argument('--shaders', type=Path, default=REPO / 'build/native')
    ap.add_argument('--no-repeat', action='store_true',
                    help='skip the second full run used for determinism (cannot pass)')
    ap.add_argument('--reuse', action='store_true',
                    help='reuse stage outputs already present in --out (cannot pass)')
    ap.add_argument('--q-mode', choices=['fp16', 'int8-q'], default='fp16',
                    help='enc1 Q projection: the FP16 dispatch, or the INT8 pair')
    ap.add_argument('--enc1-int8', type=Path, default=None,
                    help='enc1 fixture containing the INT8 Q projection')
    ap.add_argument('--dec0-mode', choices=['fp16', 'int8-expansion'], default='fp16')
    ap.add_argument('--enc0-mode', choices=['fp16', 'int8-embedding'], default='fp16')
    ap.add_argument('--attention-mode', choices=['fp16', 'int8'], default='fp16')
    ap.add_argument('--mlp-mode', choices=['fp16', 'int8-up', 'int8-pair'], default='fp16')
    ap.add_argument('--enc1-mlp', type=Path, help='enc1 INT8-up-only fixture')
    ap.add_argument('--dec0-frame', type=Path, help='dec0-only raw RGBA16F control')
    ap.add_argument('--mlpup-frame', type=Path, help='MLP-up-only raw RGBA16F control')
    ap.add_argument('--fp16-frame', type=Path, help='baseline raw RGBA16F frame for direct error')
    ap.add_argument('--enc1q-frame', type=Path, help='enc1-Q-only raw RGBA16F control')
    ap.add_argument('--cumulative-mode', choices=['none', 'qv', 'output', 'mlp', 'spatial', 'selective'], default='none')
    ap.add_argument('--cumulative-layers', type=Path, help='fixed ten-layer cumulative fixtures')
    ap.add_argument('--previous-frame', type=Path, help='previous accepted cumulative raw frame')
    ap.add_argument('--timing-dir', type=Path, default=None,
                    help='ask each probe for D3D12 GPU timestamps (requires the *_timed binaries)')
    ap.add_argument('--vram-dir', type=Path, default=None,
                    help='ask each probe for allocation/VRAM and host-phase JSON (requires the *_timed binaries)')
    ap.add_argument('--network-mode', choices=['diagnostic', 'production'], default='diagnostic',
                    help='neural graph mode: all dumps, or only the final tensor')
    ap.add_argument('--network-arena', action='store_true',
                    help='reuse transient neural buffers from a bounded placed-resource arena')
    a = ap.parse_args()
    validate_precision_modes(a.q_mode, a.dec0_mode, a.mlp_mode)
    if a.cumulative_mode == 'none' and a.cumulative_layers:
        ap.error('--cumulative-layers requires an explicit cumulative mode')
    if a.attention_mode!='fp16' and a.cumulative_mode not in ('spatial','selective'):
        ap.error('INT8 attention requires the cumulative spatial graph')
    if a.cumulative_mode != 'none':
        # `selective` explicitly combines the mixed layer graph with the two
        # image-stage precision flags; the layer graph itself is audited.
        flags = (a.q_mode, a.mlp_mode) if a.cumulative_mode == 'selective' else (
            a.q_mode, a.dec0_mode, a.mlp_mode)
        if any(m != 'fp16' for m in flags) or not a.cumulative_layers:
            ap.error('cumulative mode requires its own layer fixtures, not independent precision flags')
        import m_layer_suite as suite
        for label in suite.LABELS:
            record = json.loads((Path(suite.fixture(str(a.cumulative_layers), label)) / 'fixture.json').read_text())
            if record.get('int8_batch') != a.cumulative_mode:
                ap.error('cumulative fixture batch does not match requested mode')
            if bool(record.get('int8_attention')) != (a.attention_mode=='int8'):
                ap.error('attention mode and actual layer graph disagree')
        if a.cumulative_mode == 'selective':
            import m13_selective as selective
            checked = selective.audit(a.cumulative_layers)
            if checked['status'] != 'PASS':
                ap.error('selective fixture audit failed: ' + '; '.join(checked['problems'][:4]))
            # enc0/dec0 are independent image stages and stay explicit flags here.
        else:
            import m09_cumulative as cumulative
            cumulative.audit_config(a.cumulative_layers, a.cumulative_mode)
            if a.cumulative_mode == 'spatial': a.dec0_mode = 'int8-expansion'
    if a.mlp_mode != 'fp16' and a.enc1_mlp is None:
        ap.error('an INT8 MLP mode needs --enc1-mlp')
    FIXTURES, LAYERS = a.fixtures, a.layers
    for path in (a.probe, a.network_probe, a.shaders):
        if not path.exists():
            raise SystemExit(f'missing resource: {path}')
    for stage in ('enc0', 'dec0', 'post', 'downsample'):
        if not (FIXTURES / stage / 'stage.txt').is_file():
            raise SystemExit(f'missing fixture: {FIXTURES / stage / "stage.txt"}')

    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=a.reuse)
    if a.timing_dir:
        a.timing_dir.mkdir(parents=True, exist_ok=True)
    if a.vram_dir:
        a.vram_dir.mkdir(parents=True, exist_ok=True)
    manifest = input_manifest(a)
    if a.q_mode == 'int8-q' and a.enc1_int8 is None:
        raise SystemExit('--q-mode int8-q needs --enc1-int8')
    report = {'status': 'running', 'fixtures': str(FIXTURES), 'layers': str(LAYERS),
              'inputs': manifest, 'scripts': {'m_b6_frame.py': sha(__file__)},
              'q_mode': a.q_mode, 'enc1_int8': (str(a.enc1_int8) if a.enc1_int8 else None),
              'dec0_mode': a.dec0_mode,
              'enc0_mode': a.enc0_mode,
              'attention_mode': a.attention_mode,
              'colour': (str(a.colour) if a.colour else None),
              'mlp_mode': a.mlp_mode,
              'cumulative_mode': a.cumulative_mode,
              'reused': a.reuse, 'stages': {}, 'paths': {}, 'comparisons': {},
              'boundaries': {}, 'missing_outputs': []}

    def pipeline(tag):
        enc0_fixture=FIXTURES / 'enc0'
        if a.enc0_mode!='fp16' or a.colour is not None:
            import m09_complete as complete
            enc0_fixture=clone_fixture('enc0',a.out/f'enc0-fixture-{tag}')
            if a.colour is not None:
                expected=(FIXTURES/'enc0'/'in-colour.bin').stat().st_size
                data=Path(a.colour).read_bytes()
                if len(data)!=expected:
                    raise SystemExit(f'--colour {a.colour}: {len(data)} bytes, expected {expected}')
                (enc0_fixture/'in-colour.bin').write_bytes(data)
            if a.enc0_mode!='fp16':
                complete.prepare_enc0(enc0_fixture)
        enc0 = run(a.probe, enc0_fixture / 'stage.txt', a.out / f'enc0-{tag}',
                   a.shaders, seed=(1 if tag=='repeat' else 0) if a.enc0_mode!='fp16' else None,
                   timing=(a.timing_dir / f'enc0-{tag}.json') if a.timing_dir else None,
                   vram=(a.vram_dir / f'enc0-{tag}.json') if a.vram_dir else None)
        tokens = enc0 / 'out-tokens.bin'
        chain_dir = a.out / f'chain-{tag}'
        build_chain(a.cumulative_layers or LAYERS, chain_dir, tokens, q_int8=a.q_mode == 'int8-q',
                    enc1_int8=a.enc1_int8, mlp_int8=a.mlp_mode != 'fp16', enc1_mlp=a.enc1_mlp,
                    mlp_mode=a.mlp_mode)
        dec1 = run_chain(a.network_probe, chain_dir, a.out / f'chain-{tag}-results',
                          a.shaders, seed=1 if tag == 'repeat' and (a.mlp_mode != 'fp16' or a.cumulative_mode != 'none') else 0,
                          timing=(a.timing_dir / f'chain-{tag}.json') if a.timing_dir else None,
                          vram=(a.vram_dir / f'chain-{tag}.json') if a.vram_dir else None,
                          mode=a.network_mode, arena=a.network_arena)
        stages = image_stages(dec1, tag, a.out, a.probe, a.shaders,
                              dec0_mode=a.dec0_mode,
                              seed=(1 if tag == 'repeat' else 0) if a.dec0_mode != 'fp16' else None,
                              timing_dir=a.timing_dir, vram_dir=a.vram_dir)
        full = {'enc0': {'fixture': enc0_fixture, 'results': enc0},
                'chain': {'file': dec1}, **stages['stages']}
        hashes = boundary_hashes(full)
        if a.mlp_mode != 'fp16':
            for p in sorted(dec1.parent.glob('enc1-dbg-*.bin')):
                hashes['chain/' + p.stem] = sha(p)
        if a.cumulative_mode != 'none':
            for p in sorted(dec1.parent.glob('out-*.bin')):
                hashes['chain/' + p.stem] = sha(p)
        return {'tokens': tokens, 'dec1': dec1, 'frame': stages['frame'],
                'stages': full, 'hashes': hashes,
                'chain_graph_sha256': sha(chain_dir / 'graph.txt')}

    def reuse(tag):
        stages = {'enc0': {'fixture': FIXTURES / 'enc0',
                           'results': a.out / f'enc0-{tag}'},
                  'chain': {'file': a.out / f'chain-{tag}-results' / 'out-dec1.bin'},
                  'dec0': {'fixture': a.out / f'dec0-{tag}',
                           'results': a.out / f'dec0-{tag}' / 'results'},
                  'post': {'fixture': a.out / f'post-{tag}',
                           'results': a.out / f'post-{tag}' / 'results'},
                  'downsample': {'fixture': a.out / f'downsample-{tag}',
                                 'results': a.out / f'downsample-{tag}' / 'results'}}
        return {'tokens': a.out / f'enc0-{tag}' / 'out-tokens.bin',
                'dec1': stages['chain']['file'],
                'frame': stages['downsample']['results'] / 'out-out.bin',
                'stages': stages, 'hashes': boundary_hashes(stages),
                'chain_graph_sha256': None}

    native = reuse('native') if a.reuse else pipeline('native')
    report['stages'] = {k: str(v.get('results') or v.get('file'))
                        for k, v in native['stages'].items()}
    report['paths']['native'] = {k: str(v) for k, v in native.items()
                                 if k not in ('hashes', 'stages')}
    report['boundaries'] = native['hashes']

    repeat = {}
    if not a.reuse and not a.no_repeat:
        again = pipeline('repeat')
        differing = sorted(k for k in native['hashes']
                           if again['hashes'].get(k) != native['hashes'][k])
        repeat = {'fresh': True, 'deterministic': not differing,
                  'mismatched': differing, 'hashes': again['hashes'],
                  'chain_graph_sha256': again['chain_graph_sha256']}
    elif a.reuse:
        previous = {}
        if (a.out / 'b6.json').is_file():
            previous = json.loads((a.out / 'b6.json').read_text())
        recorded = verify_reuse(previous, manifest, native['hashes'])
        repeat = {'fresh': False, 'deterministic': bool(previous.get('repeat', {})
                                                        .get('deterministic')),
                  'inherited': True, 'mismatched': [], 'hashes': recorded}
    report['repeat'] = repeat

    ref = frame(FIXTURES / 'downsample' / 'ref-out.bin')
    report['reference'] = {'path': str(FIXTURES / 'downsample' / 'ref-out.bin'),
                           'sha256': sha(FIXTURES / 'downsample' / 'ref-out.bin'),
                           'provenance': 'canonical d4r capture for the same frame: '
                                         'native image stages with translated enc0/dec0/'
                                         'post/downsample and d4r native layer replacements'}
    dec1_fields = {'image': LAYERS / 'dec1' / 'ref-out.bin'}
    if a.translated_dec1_capture is not None:
        sys.path.insert(0, str(HERE))
        import m_layer_oracle as lo
        _, raw = lo.raw_layer(str(a.translated_dec1_capture), 'dec1')
        translated = a.out / 'translated-dec1.bin'
        translated.write_bytes(raw['output'].tobytes())
        dec1_fields['translated'] = translated
        report['translated_reference'] = {
            'path': str(a.translated_dec1_capture),
            'provenance': 'original Preset M PTX translated by ZLUDA, backend=translated, '
                          'dec1 layer only',
            'dec1_sha256': sha(translated)}
    paths = {name: image_stages(dec1, name, a.out, a.probe, a.shaders)
             for name, dec1 in dec1_fields.items()}

    cases = {'native': native, **paths}
    for name, path in cases.items():
        test = frame(path['frame'])
        report['comparisons'][name] = metrics(ref, test)
        report['comparisons'][name]['frame_sha256'] = sha(path['frame'])
        report['comparisons'][name]['exports'] = export(a.out / f'frame-{name}', ref, test)

    native_frame = frame(native['frame'])
    report['direct_comparisons'] = {}
    for label, source in [('hybrid-vs-fp16', a.fp16_frame), ('hybrid-vs-enc1q', a.enc1q_frame),
                          ('hybrid-vs-dec0', a.dec0_frame), ('hybrid-vs-mlpup', a.mlpup_frame),
                          ('hybrid-vs-previous', a.previous_frame)]:
        if source is None:
            continue
        base = frame(source)
        row = metrics(base, native_frame)
        row['reference_path'] = str(source)
        row['reference_sha256'] = sha(source)
        row['exports'] = export(a.out / label, base, native_frame)
        report['direct_comparisons'][label] = row
    report['chain_gap'] = {
        f'{name}-vs-native': metrics(native_frame, frame(path['frame']))
        for name, path in paths.items()}
    report['exports_reference'] = export(a.out / 'frame-reference', ref, ref)

    record = json.loads((LAYERS / 'dec1' / 'fixture.json').read_text())
    dec1_bytes = int(record['outputs']['out']['bytes'])
    frame_info = {'height': H, 'width': W, 'channels': 4, 'dtype': 'float16'}
    missing = []
    for name, path in (('native', native), *[(k, v) for k, v in paths.items()]):
        for stage, info in path['stages'].items():
            if 'file' in info:
                if Path(info['file']).stat().st_size != dec1_bytes:
                    missing.append(f'{name}/chain out-dec1')
                continue
            fixture, results = Path(info['fixture']), Path(info['results'])
            for role, size in declared_outputs(fixture / 'stage.txt').items():
                produced = results / f'out-{role}.bin'
                if not produced.is_file() or produced.stat().st_size != size:
                    missing.append(f'{name}/{stage}/out-{role}')
    report['missing_outputs'] = sorted(set(missing))
    report['frame'] = frame_info

    dec1 = np.fromfile(native['dec1'], np.uint8)
    test_frame = frame(native['frame'])
    report['non_finite'] = {
        'dec1': int(np.isin(dec1, (0x7F, 0xFF)).sum()),
        'frame': int((~np.isfinite(test_frame)).sum()),
        'reference_frame': int((~np.isfinite(ref)).sum()),
        'saturated': int((np.abs(test_frame) >= 65504).sum()),
        'reference_saturated': int((np.abs(ref) >= 65504).sum()),
    }

    report['execution'] = judge_execution(report)
    report['fidelity'] = {
        'status': 'INCONCLUSIVE',
        'reason': 'no independent static validation set (different content captured '
                  'through the same reference path) is available; the canonical frame is '
                  'a regression control, not a fidelity oracle',
        'measurement_contract': {
            'psnr': 'linear RGB f16, declared data range 1.0, finite pairs only',
            'ssim': 'windowed 11x11 Gaussian sigma 1.5 on BT.709 luma, K1 0.01, K2 0.03, '
                    'L 1.0',
            'error': 'mean/p50/p99/max absolute RGB, mean relative, signed bias',
            'spatial': '16x16 block means plus a 32-pixel border/interior split',
            'alpha': 'reported separately, never mixed into the RGB statistics',
            'reference_provenance': 'recorded per reference under `reference` and '
                                    '`translated_reference`',
        },
        'regression': judge_regression(report),
    }
    report['status'] = ('FAIL' if report['execution']['status'] == 'FAIL'
                        or report['fidelity']['regression']['status'] != 'PASS'
                        else 'EXPLORATORY')

    (a.out / 'b6.json').write_text(json.dumps(report, indent=2))
    summary = {k: {m: v[m] for m in ('psnr_peak1_db', 'ssim_windowed_luma', 'mean_abs',
                                     'max_abs', 'mean_relative', 'signed_mean',
                                     'nan_or_inf_pairs', 'saturated_test')}
               for k, v in report['comparisons'].items()}
    print(json.dumps(summary, indent=2))
    print('execution:', report['execution']['status'], report['execution']['reasons'])
    print('regression:', report['fidelity']['regression']['status'],
          report['fidelity']['regression']['reasons'])
    print('fidelity:', report['fidelity']['status'])
    print('status:', report['status'], '| reused:', a.reuse)
    print('out:', a.out)
    return 0 if report['status'] != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())


