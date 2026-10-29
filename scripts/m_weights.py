#!/usr/bin/env python3
"""Preset M weight inventory: every known matrix of one captured frame, with its
FP8 codes and exact f16 values.

The capture is the Windows ZLUDA replay dump (`D4R_CUDA_REPLAY_DUMP_DIR`,
filter `rrlite_`).  NGX stores the front-end matrices in its tensor-core blocked
layout and the Swin layers in a blocked layout d4r's `plain_layer()` already
decodes; this module wraps both so that every matrix is a row-major `W[K][N]`
(y = x W, K the reduction axis) over natural channel order, plus the original
e4m3 codes so an INT8 conversion starts from NVIDIA's stored values rather than
from a re-encoding round trip.

Kernel names carry the variant NGX selected (motion resolution and range).  The
probe submits display-resolution motion, so this capture holds `mvhi`; nothing
here aliases `mvhi` to `mvlo`: `parse_kernel()` exposes the fields and
`inventory()` keeps them, because the two variants are different stages even
when their weight blocks are the same size (C17).

Independent checks this module performs on itself:
  * `e4m3_decode()` is a second, bit-level implementation of the OCP e4m3
    decode, compared against d4r's table by the tests;
  * `layout_coverage()` proves the blocked-layout formula is a bijection onto
    the blob's exact byte extent (no aliasing, no holes);
  * `self_check()` decodes every matrix, round-trips its codes and rejects
    non-finite values.

usage: m_weights.py CAPTURE [--json OUT] [--quiet]
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ENGINE = os.path.join(REPO, 'references', 'sources', 'd4r', 'engine')
sys.path.insert(0, ENGINE)          # also puts kernels/tools on the path
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools'))

import m_model as mm          # noqa: E402
import model_enc3 as me       # noqa: E402
import swin_model as sm       # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64
E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# kernel-name grammar


_ENC0 = re.compile(r'^enc0_4x4_(mvlo|mvhi)_(hdr|ldr)_folded$')
_POST = re.compile(r'^post_3_([12])_(mvlo|mvhi)_(hdr|ldr)_folded$')
_DOWN = re.compile(r'^downsample_kernel_(static|dynamic)_(hdr|ldr)$')
_DEC0 = re.compile(r'^dec0_4x4(_folded)?$')
_LAYER = re.compile(r'^(' + '|'.join(mm.LAYERS) + r')_4x4$')


def parse_kernel(kernel):
    """Explicit variant fields of a whitelisted `rrlite_*` kernel name.

    Returns None for kernels this project does not model.  The result always
    names the family and, where NGX has variants, the motion and range fields;
    callers select on those instead of matching a prefix.
    """
    if not kernel.startswith('rrlite_'):
        return None
    body = kernel[len('rrlite_'):]
    m = _ENC0.match(body)
    if m:
        return {'family': 'enc0', 'stage': 'enc0', 'motion': m.group(1),
                'range': m.group(2), 'folded': True}
    m = _POST.match(body)
    if m:
        return {'family': 'post', 'stage': f'post_3_{m.group(1)}', 'motion': m.group(2),
                'range': m.group(3), 'folded': True}
    m = _DOWN.match(body)
    if m:
        return {'family': 'downsample', 'stage': 'downsample', 'motion': m.group(1),
                'range': m.group(2), 'folded': False}
    m = _DEC0.match(body)
    if m:
        return {'family': 'dec0', 'stage': 'dec0', 'motion': None, 'range': None,
                'folded': bool(m.group(1))}
    m = _LAYER.match(body)
    if m:
        return {'family': 'layer', 'stage': m.group(1), 'motion': None, 'range': None,
                'folded': False}
    return None


# Network order of the captured stages; the tube blocks repeat.  post and
# downsample carry no matrices and appear so the graph inventory is complete.
STAGE_ORDER = (['enc0', 'enc1', 'enc2'] + ['enc3_tube'] * 6
               + ['dec2', 'dec1', 'dec0', 'post_3_1', 'downsample'])


# ---------------------------------------------------------------------------
# bit-level e4m3 (OCP, satfinite) - independent of d4r's table


def e4m3_decode(codes):
    """Decode uint8 e4m3 codes to float64 from the bit definition."""
    c = np.asarray(codes, np.uint8).astype(np.int64)
    sign = (c >> 7) & 1
    expo = (c >> 3) & 0xF
    mant = c & 0x7
    value = np.where(expo == 0, mant * 2.0 ** -9,
                     (1.0 + mant / 8.0) * 2.0 ** (np.maximum(expo, 1) - 7))
    value = np.where((expo == 0xF) & (mant == 0x7), np.nan, value)
    return np.where(sign == 1, -value, value)


def e4m3_encode(values):
    """Nearest finite e4m3 code (ties to even code, satfinite at 448)."""
    v = np.asarray(values, F64)
    sign = np.signbit(v).astype(np.uint8) << 7
    a = np.abs(v)
    pos = e4m3_decode(np.arange(127, dtype=np.uint8))      # codes 0..126, ascending
    idx = np.clip(np.searchsorted(pos, a), 1, 126)
    lo, hi = pos[idx - 1], pos[idx]
    pick_hi = (hi - a < a - lo) | ((hi - a == a - lo) & ((idx & 1) == 0))
    mag = np.where(pick_hi, idx, idx - 1).astype(np.uint8)
    mag = np.where(a >= pos[126], 126, mag)
    return (sign | mag).astype(np.uint8)


# ---------------------------------------------------------------------------
# blocked layouts


def front_end_layout(stage):
    """Byte geometry of an NGX front-end weight blob, as its own stage reads it.

    enc0: embedding [128][64] e4m3 at 4096, 64 f16 biases at 12288, and the
    8x16x16 dither table at 0 (enc0_m.comp bindings 10-12).
    dec0: expansion [64][128] e4m3 at 0, 128 f16 biases at 8192; the stage reads
    20 columns per sub-position, so 80 of the 128 columns are live
    (dec0_m.comp bindings 7-9).

    `live` is the column permutation that turns the stored column axis into the
    stage's own order: enc0's output channels are pair-permuted by `gp()`, and
    dec0 selects 20 of every 32 stored columns for its four sub-positions.
    """
    if stage == 'enc0':
        return {'K': 128, 'N': 64, 'base': 4096, 'ks': 512, 'ns': 2048,
                'bias': 12288, 'bias_count': 64, 'aux': (0, 4096),
                'live': sm.gp(np.arange(64))}
    if stage == 'dec0':
        return {'K': 64, 'N': 128, 'base': 0, 'ks': 512, 'ns': 1024,
                'bias': 8192, 'bias_count': 128, 'aux': None,
                'live': (np.arange(4)[:, None] * 32 + np.arange(20)[None, :]).ravel()}
    raise KeyError(stage)


def layout_offsets(spec):
    """The blocked byte offset of every (k, n) element of a front-end matrix."""
    K, N = spec['K'], spec['N']
    k = np.arange(K)[:, None]
    n = np.arange(N)[None, :]
    j = n // 8
    return (spec['base'] + spec['ks'] * (k // 32) + spec['ns'] * (j // 2)
            + 64 * (n % 8) + 16 * ((k % 16) // 4) + 8 * (j % 2)
            + 4 * ((k % 32) // 16) + k % 4)


def layout_coverage(spec):
    """Structural check: the offset table must cover the blob extent exactly.

    A blocked layout of a K x N byte matrix inside [base, base + K*N) has to be
    a permutation of those bytes.  Anything else means the formula aliases some
    elements and leaves others unreachable.
    """
    off = layout_offsets(spec)
    want = np.arange(spec['base'], spec['base'] + spec['K'] * spec['N'])
    got = np.sort(off.ravel())
    return {'bijective': bool(np.array_equal(got, want)),
            'extent': [int(want[0]), int(want[-1]) + 1],
            'units_64B': int(spec['K'] * spec['N'] // 64)}


# ---------------------------------------------------------------------------
# captures


def weight_blob(directory, ptr_offset=0):
    """The weight allocation bytes, starting at the kernel's weight pointer.

    Same rule `compile_m.py` uses: the argument block's declared pointer decides
    which dumped allocation holds the weights.
    """
    args = open(os.path.join(directory, 'args.bin'), 'rb').read()
    ptr = struct.unpack_from('<Q', args, ptr_offset)[0]
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if q and q[0] == 'alloc':
            base = int(q[2], 16)
            size = os.path.getsize(os.path.join(directory, f'alloc-{q[1]}.bin'))
            if base <= ptr < base + size:
                return np.fromfile(os.path.join(directory, f'alloc-{q[1]}.bin'), np.uint8)[ptr - base:]
    raise ValueError(f'{directory}: the weights allocation is missing')


def kernel_of(directory):
    with open(os.path.join(directory, 'manifest.txt')) as handle:
        return handle.readline().split()[1]


@dataclass
class Stage:
    position: int
    stage: str
    kernel: str
    variant: dict
    directory: str
    sequence: int
    matrices: list = field(default_factory=list)   # list of dict rows
    vectors: dict = field(default_factory=dict)
    blob_bytes: int = 0

    def as_dict(self):
        """JSON-safe description (the arrays stay in memory)."""
        return {
            'position': self.position, 'stage': self.stage, 'kernel': self.kernel,
            'variant': self.variant, 'sequence': self.sequence,
            'blob_bytes': self.blob_bytes,
            'matrices': [{'name': r['name'], 'shape': r['shape'],
                          'codes': int(r['codes'].size),
                          'bias': None if r['bias'] is None else int(r['bias'].size),
                          'layout': r['layout']} for r in self.matrices],
            'vectors': {k: (int(v.size) if isinstance(v, np.ndarray) else v)
                        for k, v in self.vectors.items()},
        }


def discover(capture):
    """Map the capture's launch directories onto the network's stage order.

    Launches are grouped by kernel name and ordered by the hook's per-kernel
    `replay-<seq>` counter, which is launch order for that kernel (the six tube
    blocks).  The network position comes from the name, never from directory
    sort order, because different kernels share `replay-000001`.
    """
    groups = {}
    for directory in sorted(glob.glob(os.path.join(capture, 'replay-*-rrlite_*'))):
        if not os.path.isdir(directory):
            continue
        kernel = kernel_of(directory)
        parsed = parse_kernel(kernel)
        if parsed is None:
            continue
        sequence = int(re.search(r'replay-(\d+)-', os.path.basename(directory)).group(1))
        groups.setdefault(kernel, []).append((sequence, parsed, directory))
    stages = []
    positions = {}
    for stage in STAGE_ORDER:
        positions[stage] = [i for i, s in enumerate(STAGE_ORDER) if s == stage]
    used = {stage: 0 for stage in STAGE_ORDER}
    for kernel, entries in sorted(groups.items()):
        entries.sort()
        parsed = entries[0][1]
        stage = parsed['stage']
        slots = positions.get(stage, [])
        for sequence, variant, directory in entries:
            if used[stage] >= len(slots):
                break
            stages.append(Stage(slots[used[stage]], stage, kernel, variant, directory,
                                sequence))
            used[stage] += 1
    stages.sort(key=lambda s: s.position)
    return stages


# ---------------------------------------------------------------------------
# extraction


def _front_end(stage):
    spec = front_end_layout(stage.stage)
    blob = weight_blob(stage.directory)
    codes = blob[layout_offsets(spec)][:, spec['live']]
    codes = np.ascontiguousarray(codes)
    bias_all = np.ascontiguousarray(blob[spec['bias']:spec['bias'] + 2 * spec['bias_count']]).view(F16)
    bias = np.ascontiguousarray(bias_all[spec['live']])
    row = {
        'name': f'{stage.stage}.embedding' if stage.stage == 'enc0' else f'{stage.stage}.expansion',
        'shape': list(codes.shape),
        'codes': codes,
        'W': np.ascontiguousarray(me.E4M3[codes]).astype(F16),
        'layout': {'K': spec['K'], 'N': spec['N'], 'base': spec['base'],
                   'ks': spec['ks'], 'ns': spec['ns'], 'bias_offset': spec['bias'],
                   'columns_live': int(codes.shape[1])},
        'bias': bias,
    }
    stage.matrices.append(row)
    stage.blob_bytes = int(blob.size)
    stage.vectors['bias'] = bias
    return spec, blob


def _layers(stage):
    P = me.Params(stage.directory)
    tensors = mm.plain_layer(P, stage.stage)
    C, heads, NPM, CIN = mm.LAYERS[stage.stage]
    named = []
    for h in range(heads):
        named.append((f'q{h}', tensors['q'][h]))
        named.append((f'v{h}', tensors['v'][h]))
    named.append(('wo', tensors['wo']))
    for i in range(len(tensors['w1'])):
        named.append((f'w1_{i}', tensors['w1'][i]))
        named.append((f'w2_{i}', tensors['w2'][i]))
    if NPM:
        named.append(('merge_w', tensors['merge_w']))
    if CIN:
        named.append(('expand_w', tensors['expand_w']))
    code_check = True
    for name, W in named:
        W16 = np.ascontiguousarray(W).astype(F16)
        codes = e4m3_encode(W16)
        back = e4m3_decode(codes).astype(F16)
        code_check &= bool(np.array_equal(back, W16, equal_nan=True))
        stage.matrices.append({
            'name': f'{stage.stage}.{name}',
            'shape': list(W16.shape),
            'codes': codes,
            'W': W16,
            'layout': {'source': 'plain_layer'},
            'bias': None,
        })
    stage.vectors = {'g1': tensors['g1'], 'bo': tensors['bo'], 'g2': tensors['g2'],
                     'b2': tensors['b2'],
                     'b1': np.concatenate(tensors['b1']) if tensors['b1'] else None,
                     'rel_bias': np.concatenate(tensors['bias']) if tensors['bias'] else None}
    if NPM:
        stage.vectors['merge_b'] = tensors['merge_b']
    if CIN:
        stage.vectors['expand_b'] = tensors['expand_b']
    stage.blob_bytes = int(P.wbuf.size)
    stage.vectors['codes_lossless'] = code_check
    return P


def inventory(capture):
    """Every matrix of the frame, in network order, with codes and f16 values."""
    stages = discover(capture)
    for stage in stages:
        if stage.stage in ('enc0', 'dec0'):
            _front_end(stage)
        elif stage.variant['family'] == 'layer':
            _layers(stage)
        # post and downsample carry no matrices; they are listed for the graph
    return stages


def pointer_addresses(directory):
    """(argument offset, absolute address) of every resolved pointer of a launch."""
    base_of = {}
    rows = []
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if not q:
            continue
        if q[0] == 'alloc':
            base_of[q[1]] = int(q[2], 16)
        elif q[0] == 'pointer':
            rows.append((int(q[1]), base_of[q[2]] + int(q[3])))
    return rows


def activations(stages):
    """The captured operand matrix of every stage that reads one from a buffer.

    The Swin layers and dec0 read FP8 tokens from the shared arena.  A layer's
    input offset comes from its own parameter block; dec0's comes from pointer
    identity with the layer that produced it (dec1's output allocation), not
    from a guessed offset.  enc0's matrix operand is built inside the kernel
    from texture samples and is therefore not captured - it is absent here
    rather than approximated.

    Returns one record per launch, so the six tube blocks keep their own range.
    """
    counts = {}
    for stage in stages:
        counts[stage.stage] = counts.get(stage.stage, 0) + 1
    out = []
    for stage in stages:
        if stage.variant['family'] != 'layer':
            continue
        P = me.Params(stage.directory)
        C, _, _, CIN = mm.LAYERS[stage.stage]
        if CIN:
            x = mm.tokens(P.arena, P.inp - P.abase, P.W // 2, P.H // 2, CIN)
        else:
            x = mm.tokens(P.arena, P.inp - P.abase, P.W, P.H, C)
        label = stage.stage if counts[stage.stage] == 1 else f'{stage.stage}#{stage.sequence}'
        out.append({'label': label, 'stage': stage.stage, 'tensor': x,
                    'W': P.W, 'H': P.H, 'C': C,
                    'source': f'{stage.kernel} input buffer'})

    dec0 = next((s for s in stages if s.stage == 'dec0'), None)
    producer = next((s for s in stages if s.stage == 'dec1'), None)
    if dec0 is not None and producer is not None:
        P = me.Params(producer.directory)
        target = P.out
        addresses = {address for _, address in pointer_addresses(dec0.directory)}
        if target in addresses:
            x = mm.tokens(P.arena, target - P.abase, P.W, P.H, mm.LAYERS['dec1'][0])
            alias = sorted(offset for offset, address in
                           pointer_addresses(dec0.directory) if address == target)
            out.append({'label': 'dec0', 'stage': 'dec0', 'tensor': x, 'W': P.W, 'H': P.H,
                        'C': mm.LAYERS['dec1'][0],
                        'source': f'dec1 output buffer (dec0 arg offsets {alias})'})
        else:
            out.append({'label': 'dec0', 'stage': 'dec0', 'tensor': None,
                        'source': 'dec0 input pointer does not alias any layer output'})
    return out


def matrices(stages):
    """Flat (stage, name, W, codes) rows over the matrix-bearing stages."""
    for stage in stages:
        for row in stage.matrices:
            yield stage, row


def self_check(stages):
    """Decode checks that must hold for a usable inventory."""
    problems = []
    for stage, row in matrices(stages):
        W, codes = row['W'], row['codes']
        if W.shape != codes.shape:
            problems.append(f'{row["name"]}: shape mismatch')
        if not np.isfinite(W).all():
            problems.append(f'{row["name"]}: non-finite weight')
        if not np.array_equal(e4m3_decode(codes).astype(F16), W, equal_nan=True):
            problems.append(f'{row["name"]}: code table does not reproduce the values')
        if row['layout'] and 'base' in row['layout']:
            spec = {'K': row['layout']['K'], 'N': row['layout']['N'],
                    'base': row['layout']['base'], 'ks': row['layout']['ks'],
                    'ns': row['layout']['ns']}
            if not layout_coverage(spec)['bijective']:
                problems.append(f'{row["name"]}: layout is not a bijection')
    return problems


def summary(stages):
    return [s.as_dict() for s in stages]


def digests(stages):
    """Stable identity of every matrix: stage label, shape and code bytes."""
    import hashlib
    rows = []
    for stage in stages:
        for row in stage.matrices:
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(row['codes']).tobytes())
            if row['bias'] is not None:
                h.update(np.ascontiguousarray(row['bias']).tobytes())
            rows.append((f'{stage.stage}#{stage.sequence}', row['name'],
                         list(row['shape']), h.hexdigest()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--compare', help='another capture; report whether its matrices are identical')
    a = ap.parse_args()

    stages = inventory(a.capture)
    problems = self_check(stages)
    rows = []
    total_codes = 0
    for stage in stages:
        variant = ''
        if stage.variant['motion']:
            variant = f"{stage.variant['motion']}/{stage.variant['range']}"
        elements = sum(int(r['W'].size) for r in stage.matrices)
        total_codes += elements
        if not a.quiet:
            print(f'{stage.position:2d} {stage.stage:10s} {stage.kernel:38s} {variant:10s} '
                  f'matrices={len(stage.matrices):2d} elements={elements:7d} blob={stage.blob_bytes:7d} B')
        for row in stage.matrices:
            W = row['W'].astype(F64)
            rows.append({'stage': stage.stage, 'name': row['name'], 'shape': row['shape'],
                         'absmax': float(np.abs(W).max()),
                         'rms': float(np.sqrt((W ** 2).mean())),
                         'zeros': int((W == 0).sum())})
    if not a.quiet:
        print(f'\n{len(rows)} matrices, {total_codes} e4m3 values, '
              f'checks={"OK" if not problems else "FAILED"}')
    for problem in problems:
        print(f'  FAIL {problem}')
    if a.json:
        json.dump({'capture': a.capture, 'stages': summary(stages), 'matrices': rows,
                   'problems': problems}, a.json, indent=2)

    if a.compare:
        other = digests(inventory(a.compare))
        mine = digests(stages)
        by_name = {(s, n): d for s, n, _, d in other}
        same = different = absent = 0
        for stage, name, shape, digest in mine:
            key = (stage, name)
            if key not in by_name:
                absent += 1
                print(f'  absent in {a.compare}: {stage} {name}')
                continue
            if by_name[key] == digest:
                same += 1
            else:
                different += 1
                print(f'  DIFFERENT {stage} {name} {shape}')
        print(f'compare {a.capture} vs {a.compare}: identical={same} different={different} '
              f'absent={absent}')
        return 0 if different == 0 else 1
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
