#!/usr/bin/env python3
"""B6 (EXPLORATORY): the first complete native frame from real intermediates.

Runs enc0 -> ten native layers -> dec0 -> post -> downsample with no captured
tensor substituted between neural stages. The only captured inputs that remain are
the stage inputs the network does not produce (colour, motion, history).

It measures the frame against the canonical capture and against two controls that
separate the neural chain from the image stages:

  ref        canonical captured frame
  image      image stages fed the captured dec1 field (image-stage error alone)
  native     image stages fed the native ten-layer chain (this reconstruction)
  translated image stages fed the translated-PTX chain (independent chain)

Prints and records PSNR, SSIM, absolute/relative error, spatial distribution,
NaN/Inf, saturation and determinism, and exports viewable PNGs.

  m_b6_frame.py --out NEW_DIR [--fixtures DIR] [--layers DIR]
                [--translated-dec1-capture REPLAY_DIR]
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(probe, stage, out, shaders):
    if Path(out).is_dir():
        shutil.rmtree(out)
    done = subprocess.run([str(probe), '--fixture', str(stage), '--out', str(out),
                           '--shaders', str(shaders)], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'{stage}: {done.stdout[-600:]}{done.stderr[-600:]}')
    return Path(out)


def clone_fixture(name, dst):
    if dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(FIXTURES / name, dst, ignore=DEBUG)
    return dst


def build_chain(layers, out, enc1_input):
    sys.path.insert(0, str(HERE))
    import m_layer_suite as suite
    info = suite.build_chain(str(layers), str(out))
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


def run_chain(network_probe, chain_dir, out, shaders):
    done = subprocess.run([str(network_probe), '--graph', str(chain_dir / 'graph.txt'),
                           '--out', str(out), '--shaders', str(shaders), '--seed', '0'],
                          capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'chain: {done.stdout[-600:]}{done.stderr[-600:]}')
    return Path(out) / 'out-dec1.bin'


def image_stages(dec1, tag, out, probe, shaders):
    """dec0 -> post -> downsample, with the neural chain's dec1 as dec0's tokens."""
    dec0 = clone_fixture('dec0', out / f'dec0-{tag}')
    shutil.copyfile(dec1, dec0 / 'in-tokens.bin')
    res0 = run(probe, dec0 / 'stage.txt', dec0 / 'results', shaders)

    post = clone_fixture('post', out / f'post-{tag}')
    for role in ('gate', 'mix', 'cov'):
        shutil.copyfile(res0 / f'out-{role}.bin', post / f'in-{role}.bin')
    res1 = run(probe, post / 'stage.txt', post / 'results', shaders)

    down = clone_fixture('downsample', out / f'downsample-{tag}')
    shutil.copyfile(res1 / 'out-packed.bin', down / 'in-src.bin')
    res2 = run(probe, down / 'stage.txt', down / 'results', shaders)
    return {'frame': res2 / 'out-out.bin', 'dec0': res0, 'post': res1, 'downsample': res2,
            'hashes': {str(p.relative_to(out)): sha(p)
                       for p in (res0, res1, res2) for p in p.glob('*.bin')}}


def frame(path):
    data = np.fromfile(path, np.float16).reshape(H, W, 4)
    return data


def ssim_luma(ref, test):
    def luma(x):
        return (0.2126 * x[..., 0].astype(np.float64) + 0.7152 * x[..., 1].astype(np.float64)
                + 0.0722 * x[..., 2].astype(np.float64))
    luma_a, luma_b = luma(ref), luma(test)
    a, b = luma_a, luma_b
    block = 8
    gh, gw = luma_a.shape[0] // block, luma_a.shape[1] // block
    a = a[:gh * block, :gw * block].reshape(gh, block, gw, block).transpose(0, 2, 1, 3)
    b = b[:gh * block, :gw * block].reshape(gh, block, gw, block).transpose(0, 2, 1, 3)
    mu_a, mu_b = a.mean(axis=(2, 3)), b.mean(axis=(2, 3))
    va = a.var(axis=(2, 3), ddof=0)
    vb = b.var(axis=(2, 3), ddof=0)
    cov = ((a - mu_a[..., None, None]) * (b - mu_b[..., None, None])).mean(axis=(2, 3))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) *
                                                      (va + vb + c2))
    return float(s.mean())


def metrics(ref, test):
    a = ref[..., :3].astype(np.float64)
    b = test[..., :3].astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.where(finite, b - a, 0.0)
    absd = np.abs(d)
    mse = float((d ** 2).mean())
    peak = float(np.nanmax(np.abs(a))) or 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(finite, absd / (np.abs(a) + 1e-6), 0.0)
    luma = 0.2126 * absd[..., 0] + 0.7152 * absd[..., 1] + 0.0722 * absd[..., 2]
    block = 16
    gh, gw = luma.shape[0] // block, luma.shape[1] // block
    per_block = luma[:gh * block, :gw * block].reshape(gh, block, gw, block).mean(axis=(1, 3))
    worst = np.dstack(np.unravel_index(np.argsort(per_block.ravel())[-5:][::-1],
                                       per_block.shape))[0]
    return {
        'samples_rgb': int(a.size),
        'psnr_peak1_db': float(10 * np.log10(1.0 / mse)) if mse else None,
        'psnr_peak_reference_db': float(10 * np.log10(peak ** 2 / mse)) if mse else None,
        'ssim_luma': ssim_luma(ref, test),
        'mean_abs': float(absd.mean()),
        'p50_abs': float(np.percentile(absd, 50)),
        'p99_abs': float(np.percentile(absd, 99)),
        'max_abs': float(absd.max()),
        'mean_relative': float(rel.mean()),
        'signed_mean': float(d.mean()),
        'nan_or_inf_pairs': int((~finite).sum()),
        'nan_or_inf_reference': int((~np.isfinite(a)).sum()),
        'nan_or_inf_test': int((~np.isfinite(b)).sum()),
        'saturated_reference': int((np.abs(a) >= 65504).sum()),
        'saturated_test': int((np.abs(b) >= 65504).sum()),
        'block_mean_abs_min': float(per_block.min()),
        'block_mean_abs_median': float(np.median(per_block)),
        'block_mean_abs_max': float(per_block.max()),
        'worst_blocks_yx': np.asarray(worst).astype(int).tolist(),
    }


def export(prefix, ref, test):
    def tonemap(x):
        y = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return (np.power(y, 1 / 2.2) * 255 + 0.5).astype(np.uint8)
    diff = np.abs(test - ref)[..., :3].astype(np.float64)
    scale = float(np.percentile(diff, 99.9)) or 1.0
    paths = {}
    for name, image in (('reference', tonemap(ref[..., :3])),
                        ('reconstructed', tonemap(test[..., :3])),
                        ('difference', tonemap(np.clip(diff / scale, 0, 1)))):
        path = prefix.with_name(prefix.name + '-' + name + '.png')
        Image.fromarray(image, 'RGB').save(path)
        paths[name] = str(path)
    return {'images': paths, 'difference_scale': scale}


def main():
    global FIXTURES, LAYERS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fixtures', type=Path, default=FIXTURES)
    ap.add_argument('--layers', type=Path, default=LAYERS)
    ap.add_argument('--translated-dec1-capture', type=Path, default=None)
    ap.add_argument('--probe', type=Path,
                    default=REPO / 'build/native/Release/m_image_stage.exe')
    ap.add_argument('--network-probe', type=Path,
                    default=REPO / 'build/native/Release/m_network_stage.exe')
    ap.add_argument('--shaders', type=Path, default=REPO / 'build/native')
    ap.add_argument('--no-repeat', action='store_true',
                    help='skip the second full run used for determinism')
    ap.add_argument('--reuse', action='store_true',
                    help='reuse stage outputs already present in --out')
    a = ap.parse_args()
    FIXTURES, LAYERS = a.fixtures, a.layers
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=a.reuse)
    report = {'status': 'running', 'fixtures': str(FIXTURES), 'layers': str(LAYERS),
              'probe_sha256': sha(a.probe), 'network_probe_sha256': sha(a.network_probe),
              'shaders': {p.name: sha(p) for p in a.shaders.glob('*.cso')},
              'paths': {}, 'comparisons': {}, 'deterministic': None, 'reused': a.reuse}

    def pipeline(tag):
        enc0 = a.out / f'enc0-{tag}'
        dec1 = a.out / f'chain-{tag}-results' / 'out-dec1.bin'
        done = (a.out / f'downsample-{tag}' / 'results' / 'out-out.bin')
        if a.reuse and done.is_file():
            return {'enc0': enc0 / 'out-tokens.bin', 'dec1': dec1,
                    'frame': done,
                    'dec0': a.out / f'dec0-{tag}' / 'results',
                    'post': a.out / f'post-{tag}' / 'results',
                    'downsample': a.out / f'downsample-{tag}' / 'results',
                    'hashes': {}}
        enc0 = run(a.probe, FIXTURES / 'enc0' / 'stage.txt', a.out / f'enc0-{tag}',
                   a.shaders)
        tokens = enc0 / 'out-tokens.bin'
        chain_dir = a.out / f'chain-{tag}'
        build_chain(LAYERS, chain_dir, tokens)
        dec1 = run_chain(a.network_probe, chain_dir, a.out / f'chain-{tag}-results',
                         a.shaders)
        stages = image_stages(dec1, tag, a.out, a.probe, a.shaders)
        return {'enc0': enc0 / 'out-tokens.bin', 'dec1': dec1, **stages}

    native = pipeline('native')
    report['paths']['native'] = {k: str(v) for k, v in native.items() if k != 'hashes'}
    if not a.no_repeat:
        again = pipeline('repeat')
        report['deterministic'] = all(
            sha(native[k]) == sha(again[k]) for k in ('enc0', 'dec1', 'frame'))
        report['repeat_frame_sha256'] = sha(again['frame'])

    dec1_fields = {'image': LAYERS / 'dec1' / 'ref-out.bin'}
    if a.translated_dec1_capture is not None:
        sys.path.insert(0, str(HERE))
        import m_layer_oracle as lo
        _, raw = lo.raw_layer(str(a.translated_dec1_capture), 'dec1')
        translated = a.out / 'translated-dec1.bin'
        translated.write_bytes(raw['output'].tobytes())
        dec1_fields['translated'] = translated
    paths = {'image': image_stages(dec1_fields['image'], 'image', a.out, a.probe,
                                   a.shaders)}
    if 'translated' in dec1_fields:
        paths['translated'] = image_stages(dec1_fields['translated'], 'translated', a.out,
                                           a.probe, a.shaders)

    ref = frame(FIXTURES / 'downsample' / 'ref-out.bin')
    report['reference_sha256'] = sha(FIXTURES / 'downsample' / 'ref-out.bin')
    cases = {'native': native} | paths
    for name, path in cases.items():
        test = frame(path['frame'])
        report['comparisons'][name] = metrics(ref, test)
        report['comparisons'][name]['frame_sha256'] = sha(path['frame'])
        report['comparisons'][name]['exports'] = export(a.out / f'frame-{name}', ref, test)
    chain_gap = {}
    native_frame = frame(native['frame'])
    for name in paths:
        chain_gap[name + '-vs-native'] = metrics(native_frame, frame(paths[name]['frame']))
    report['chain_gap'] = chain_gap
    report['exports_reference'] = export(a.out / 'frame-reference', ref, ref)
    report['status'] = 'pass'
    (a.out / 'b6.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {m: v[m] for m in ('psnr_peak1_db', 'psnr_peak_reference_db',
                                            'ssim_luma', 'mean_abs', 'max_abs',
                                            'mean_relative', 'signed_mean',
                                            'nan_or_inf_pairs', 'saturated_test')}
                          for k, v in report['comparisons'].items()}, indent=2))
    print('deterministic:', report['deterministic'])
    print('out:', a.out)


if __name__ == '__main__':
    sys.exit(main())





