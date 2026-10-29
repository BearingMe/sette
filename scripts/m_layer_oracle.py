#!/usr/bin/env python3
"""B5.3: the ten-layer oracle and its measured contract.

The layer-level numpy model in d4r's `m_model.py` rounds each dot product once
(`h1 @ q`) and sums softmax rows naively; the *kernel* model in
`swin_model.py` rounds once per 32-product step, uses the six-stage f16 butterfly
for the row sum and the bit-level exponent trick.  Only the latter is a candidate
byte-exact contract, so this script drives it against the captured launches and
reports, per layer and per output, exact FP8-code agreement and the decoded error.

It also exports the operands a native fixture needs (input planes, skip planes,
the decoded matrices, vectors and references) and a `fixture.json` describing
every byte range - see `m_layer_fixture.py`.

usage:
  m_layer_oracle.py CAPTURE                     compare every layer
  m_layer_oracle.py CAPTURE --json OUT          also write the measurements
  m_layer_oracle.py CAPTURE --layer tube#3      one layer only
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ENGINE = os.path.join(REPO, 'references', 'sources', 'd4r', 'engine')
sys.path.insert(0, ENGINE)
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools'))
sys.path.insert(0, HERE)

import m_model as mm          # noqa: E402
import model_enc3 as me       # noqa: E402
import swin_model as sm       # noqa: E402

F16, F64 = np.float16, np.float64


def h(a):
    return np.asarray(a, F64).astype(F16).astype(F64)


def mirror(v, n):
    v = np.abs(v)
    return np.minimum(v, 2 * n - 2 - v)


def forward(t, name, x, skip, sx, sy, block=None):
    """m_model.forward's window/merge geometry, with `block` replaced.

    `block` maps the full (8*gy, 8*gx, C) mirrored token grid to the same shape,
    so a block implementation owns its own internal ordering.  The default is
    `m_model.block` over 4x4 windows; the kernel-exact contract is a
    `KernelBlock`.
    """
    C, heads, NPM, CIN = mm.LAYERS[name]
    f = lambda a: np.asarray(a, F64)
    if CIN:
        H, W = skip.shape[:2]
    else:
        H, W = x.shape[:2]
    gx, gy = (W + sx + 7) // 8, (H + sy + 7) // 8
    Y, X = np.meshgrid(np.arange(8 * gy) - sy, np.arange(8 * gx) - sx, indexing='ij')
    if CIN:
        xf = f(x).reshape(-1, CIN)
        ef = me.mma_chain(xf, f(t['expand_w']),
                          C0=np.broadcast_to(f(t['expand_b']), (xf.shape[0], t['expand_b'].size)))
        e = h(ef).reshape(x.shape[0], x.shape[1], 4, C)
        Yb, Xb = (Y + sy) // 8 * 8 - sy, (X + sx) // 8 * 8 - sx
        ly, lx = mirror(Yb // 2 + (Y - Yb) // 2, H // 2), mirror(Xb // 2 + (X - Xb) // 2, W // 2)
        q = ((Y - Yb) & 1) * 2 + ((X - Xb) & 1)
        x0 = h(mm.q8(e[ly, lx, q]) + skip[mirror(Y, H), mirror(X, W)])
    else:
        x0 = x[mirror(Y, H), mirror(X, W)]
    if block is None:
        win = x0.reshape(2 * gy, 4, 2 * gx, 4, C).transpose(0, 2, 1, 3, 4).reshape(-1, 16, C)
        y = mm.block(t, win, C, heads).reshape(2 * gy, 2 * gx, 4, 4, C)
        y = y.transpose(0, 2, 1, 3, 4).reshape(8 * gy, 8 * gx, C)
    else:
        y = block(x0)
    out = mm.q8(y)
    merged = None
    if NPM:
        A = out.reshape(4 * gy, 2, 4 * gx, 2, C).transpose(0, 2, 1, 3, 4).reshape(4 * gy, 4 * gx, 4 * C)
        # NVIDIA's encoders clamp the patch merge to +-2*pi (as f16) before the
        # e4m3 encoding (swin_common.h merge_clamp; absent from m_model.forward)
        lim = np.float16(6.2831855)
        m = mm.q8(np.clip(h(A @ f(t['merge_w']) + f(t['merge_b'])), -lim, lim))
        my = ((np.arange(4 * gy) // 4) * 8 - sy) // 2 + np.arange(4 * gy) % 4
        mx = ((np.arange(4 * gx) // 4) * 8 - sx) // 2 + np.arange(4 * gx) % 4
        merged = np.zeros((H // 2, W // 2, NPM))
        oky, okx = (my >= 0) & (my < H // 2), (mx >= 0) & (mx < W // 2)
        merged[np.ix_(my[oky], mx[okx])] = m[np.ix_(oky, okx)]
    oy, ox = (Y[:, 0] >= 0) & (Y[:, 0] < H), (X[0] >= 0) & (X[0] < W)
    return out[np.ix_(oy, ox)], merged


class KernelBlock:
    """The kernel-exact block, adapted to the full 8x8-token block grid.

    `swin_block` orders the 64 tokens of one block by `tok_x`/`tok_y`, i.e. four
    4x4 windows in a 2x2 arrangement - not four consecutive windows of the
    window raster.  This adapter performs that permutation explicitly.
    """

    def __init__(self, P, name):
        self.P, self.name = P, name
        self.C, self.heads, self.npm, self.cin = mm.LAYERS[name]
        self.pre = 4 * self.C * self.cin + 8 * self.C if self.cin else 0
        self.t = mm.plain_layer(P, name)

    def __call__(self, x0):
        gy8, gx8, C = x0.shape
        gy, gx = gy8 // 8, gx8 // 8
        # T -> flat (ty, tx) index inside one 8x8 block
        T = np.arange(64)
        perm = (4 * (T >> 5) + ((T >> 2) & 3)) * 8 + ((T & 3) + 4 * ((T >> 4) & 1))
        blocks = x0.reshape(gy, 8, gx, 8, C).transpose(0, 2, 1, 3, 4).reshape(-1, 64, C)
        out = np.empty((blocks.shape[0], 64, C))
        for b in range(blocks.shape[0]):
            x2, _, _ = sm.swin_block(self.P, h(blocks[b][perm]), C, self.heads, self.pre)
            out[b][perm] = np.asarray(sm.pair_to_nat(x2), F64)
        return out.reshape(gy, gx, 8, 8, C).transpose(0, 2, 1, 3, 4).reshape(gy8, gx8, C)


def tokens(buf, off, W, H, C):
    a = me.E4M3[buf[off:off + (C // 32) * H * W * 32]].reshape(C // 32, H, W, 32)
    return a.transpose(1, 2, 0, 3).reshape(H, W, C)


# ---------------------------------------------------------------------------
# the natural-channel form of the block
#
# The kernel works in the tensor-core fragment layout: 32-channel groups in
# "pair" order, the Q/V projection columns in fragment order, and the patch
# merge reading natural order back.  These helpers express exactly the same
# arithmetic on natural channel order by folding the permutation into the
# matrices.  `NaturalBlock` is the specification the HLSL follows; it is
# checked against `KernelBlock` (the kernel-order model) before it is trusted.


def gp(c):
    c = np.asarray(c)
    return 32 * (c // 32) + 2 * ((c % 32) // 4) + (c % 2) + 16 * ((c % 4) // 2)


def ginv(n):
    n = np.asarray(n)
    m = n & 31
    e = m >> 4
    a = (m & 15) >> 1
    b = m & 1
    return (n & ~31) + 4 * a + 2 * e + b


def sumsq_natural(x):
    """`sumsq_tokens` on natural channel order (planes of 32, lanes t and t+4)."""
    sq = h(x * x)
    planes = x.shape[1] // 32

    def word_sum(word):
        out = []
        for e in range(2):
            sums = []
            for p in range(planes):
                c = 32 * p + 4 * word
                sums.append(h(sq[:, c + e] + sq[:, c + 2 + e]))
            if planes == 4:
                s = h(h(sums[0] + sums[1]) + h(sums[2] + sums[3]))
            elif planes == 3:
                s = h(h(sums[0] + sums[1]) + sums[2])
            else:
                s = h(sums[0] + sums[1])
            out.append(s)
        return out

    lanes = []
    for t in range(4):
        a, b = word_sum(t), word_sum(t + 4)
        lanes.append((h(a[0] + b[0]), h(a[1] + b[1])))
    a = [(h(lanes[t][0] + lanes[t ^ 1][0]), h(lanes[t][1] + lanes[t ^ 1][1]))
         for t in range(4)]
    b = (h(a[0][0] + a[2][0]), h(a[0][1] + a[2][1]))
    return h(b[0] + b[1])[:, None]


def tree_norm2_natural(x):
    """`tree_norm2` over the pair-order columns of a natural-order row."""
    sq = h(x * x)
    c = x.shape[1]
    nt8 = c // 8
    sqp = sq[:, ginv(np.arange(c))]

    def add(a, b):
        return (h(a[0] + b[0]), h(a[1] + b[1]))

    lanes = []
    for t in range(4):
        tiles = [(sqp[:, 8 * j + 2 * t], sqp[:, 8 * j + 2 * t + 1])
                 for j in range(nt8)]

        def quad(js):
            return add(add(tiles[js[0]], tiles[js[1]]), add(tiles[js[2]], tiles[js[3]]))

        if nt8 == 16:
            even = add(quad([0, 2, 4, 6]), quad([8, 10, 12, 14]))
            odd = add(quad([1, 3, 5, 7]), quad([9, 11, 13, 15]))
        elif nt8 == 12:
            even = add(quad([0, 2, 4, 6]), add(tiles[8], tiles[10]))
            odd = add(quad([1, 3, 5, 7]), add(tiles[9], tiles[11]))
        else:
            even = quad([0, 2, 4, 6])
            odd = quad([1, 3, 5, 7])
        lanes.append(add(even, odd))
    a = [(h(lanes[t][0] + lanes[t ^ 1][0]), h(lanes[t][1] + lanes[t ^ 1][1]))
         for t in range(4)]
    b = (h(a[0][0] + a[2][0]), h(a[0][1] + a[2][1]))
    return h(b[0] + b[1])[:, None]


def rms_natural(x, s, gamma):
    s16 = np.float16(s + np.float16(2.0 ** -13))
    r = np.float16(np.float32(1.0) / np.sqrt(np.float32(s16)))
    return me.q8(h(x * h(np.float32(r) * gamma[None, :])))


def softmax_rows(S):
    t = np.clip(me.f16(S * me.C_SCALE), -float(me.CLAMP), float(me.CLAMP))
    poly = me.f16(t * me.f16(t * (-t) + me.P1) + me.P0)
    w = me.exp_trick(poly).astype(F64)
    s = me.row_sum16(w)
    r = me.f16(np.float32(1.0) / s.astype(np.float32))
    return me.f16(w * r)


def natural_matrices(P, name):
    """One layer's block matrices in natural input/output channel order.

    Every matrix is built from the captured weight blob with the kernel's own
    layout (`woff_table`), then the pair-order permutation is folded into the
    axes so the block can run entirely in natural channel order.  Deriving each
    axis here, rather than reusing `plain_layer`'s model-specific pairing, is
    what makes the form checkable against the kernel-order model.
    """
    C, heads, NPM, CIN = mm.LAYERS[name]
    t = mm.plain_layer(P, name)
    pre = 4 * C * CIN + 8 * C if CIN else 0
    L = sm.layout(C, heads, pre)
    perm32 = gp(np.arange(32))
    permc = gp(np.arange(C))
    fp8 = lambda off: me.E4M3[P.wbuf[off]]
    wo = np.concatenate(
        [fp8(sm.woff_table(L['head'][k] + 64 * C + 512, 0, 512, 32, C))
         for k in range(heads)], 0)
    return {
        't': t,
        'q': np.concatenate([fp8(sm.woff_table(L['head'][k], 1024, 512, C, 32))[:, perm32]
                             for k in range(heads)], 1),
        'v': np.concatenate([fp8(sm.woff_table(L['head'][k] + 32 * C, 1024, 512, C, 32))[:, perm32]
                             for k in range(heads)], 1),
        # x1 is the pair-order output of the raw Wo image
        'wo': wo[:, permc],
        'w1': [fp8(sm.woff_table(B1, 512, 16 * C, C, 32))[:, perm32] for B1 in L['b1']],
        'b1': [P.f16vec(B1 + 32 * C, 32)[perm32] for B1 in L['b1']],
        # the down-projection reads G in natural order and writes pair order
        'w2': [fp8(sm.woff_table(B1 + 32 * C + 64, 0, 512, 32, C))[:, permc]
               for B1 in L['b1']],
    }


class NaturalBlock:
    """The block in natural channel order - the shader's specification."""

    def __init__(self, P, name):
        self.P, self.name = P, name
        self.C, self.heads, self.npm, self.cin = mm.LAYERS[name]
        self.pre = 4 * self.C * self.cin + 8 * self.C if self.cin else 0
        self.m = natural_matrices(P, name)

    def __call__(self, x0):
        C, heads = self.C, self.heads
        t = self.m['t']
        x0 = h(x0)
        h1 = rms_natural(x0, sumsq_natural(x0), t['g1'])
        Q = me.mma_chain(h1, self.m['q'])
        V = me.mma_chain(h1, self.m['v'])
        O = np.zeros((64, 32 * heads))
        for w in range(4):
            sl = slice(16 * w, 16 * w + 16)
            Qq = me.q8(Q[sl])
            for k in range(heads):
                Qh = Qq[:, 32 * k:32 * k + 32]
                S = me.f16(Qh @ Qh.T + t['bias'][k])
                O[sl, 32 * k:32 * k + 32] = me.f16(softmax_rows(S)
                                                   @ V[sl, 32 * k:32 * k + 32])
        O = me.q8(O)
        x1 = me.mma_chain(O, self.m['wo'], C0=h(x0 + t['bo']))
        h2 = rms_natural(x1, tree_norm2_natural(x1), t['g2'])
        x2 = h(x1 + t['b2'])
        for k in range(len(self.m['w1'])):
            z = me.mma_chain(h2, self.m['w1'][k],
                             C0=np.broadcast_to(self.m['b1'][k], (64, 32)))
            g = me.q8(me.gelu_poly(z))
            x2 = h(me.mma_chain(g, self.m['w2'][k], C0=x2))
        return x2


class NaturalGridBlock:
    """`NaturalBlock` applied over the block grid (same grouping as KernelBlock)."""

    def __init__(self, P, name):
        self.block = NaturalBlock(P, name)
        self.t = self.block.m['t']

    def __call__(self, x0):
        gy8, gx8, C = x0.shape
        gy, gx = gy8 // 8, gx8 // 8
        T = np.arange(64)
        perm = (4 * (T >> 5) + ((T >> 2) & 3)) * 8 + ((T & 3) + 4 * ((T >> 4) & 1))
        blocks = x0.reshape(gy, 8, gx, 8, C).transpose(0, 2, 1, 3, 4).reshape(-1, 64, C)
        out = np.empty((blocks.shape[0], 64, C))
        for b in range(blocks.shape[0]):
            out[b][perm] = np.asarray(self.block(h(blocks[b][perm])), F64)
        return out.reshape(gy, gx, 8, 8, C).transpose(0, 2, 1, 3, 4).reshape(gy8, gx8, C)


def raw_layer(directory, name):
    """The captured byte fields of one layer, exactly as stored.

    Returns the e4m3 input field, the e4m3 skip field (decoders), and the
    post-launch output and merge fields.  No decoding: a fixture must carry the
    captured bytes, not a re-encoding of the model's values.
    """
    C, heads, NPM, CIN = mm.LAYERS[name]
    P = me.Params(directory)
    out_dir = os.path.join(directory, 'gpu-output')
    after = next(np.fromfile(os.path.join(out_dir, f), np.uint8)
                 for f in sorted(os.listdir(out_dir))
                 if os.path.getsize(os.path.join(out_dir, f)) == P.arena.size)
    W, H = P.W, P.H
    fields = {
        'input': np.ascontiguousarray(P.arena[P.inp - P.abase:P.inp - P.abase + W * H * C]),
        'output': np.ascontiguousarray(after[P.out - P.abase:P.out - P.abase + W * H * C]),
    }
    if CIN:
        fields['low'] = np.ascontiguousarray(
            P.arena[P.inp - P.abase:P.inp - P.abase + (W // 2) * (H // 2) * CIN])
        fields['skip'] = np.ascontiguousarray(P.arena[P.p32 - P.abase:P.p32 - P.abase + W * H * C])
    if NPM:
        m = (W // 2) * (H // 2) * NPM
        fields['merge'] = np.ascontiguousarray(after[P.p48 - P.abase:P.p48 - P.abase + m])
    return P, fields


def read_layer(directory, name):
    C, heads, NPM, CIN = mm.LAYERS[name]
    P = me.Params(directory)
    # the arena's post-launch image, identified by size rather than by name
    out_dir = os.path.join(directory, 'gpu-output')
    after = next(np.fromfile(os.path.join(out_dir, f), np.uint8)
                 for f in sorted(os.listdir(out_dir))
                 if os.path.getsize(os.path.join(out_dir, f)) == P.arena.size)
    W, H = P.W, P.H
    if CIN:
        x = tokens(P.arena, P.inp - P.abase, W // 2, H // 2, CIN)
        skip = tokens(P.arena, P.p32 - P.abase, W, H, C)
    else:
        x = tokens(P.arena, P.inp - P.abase, W, H, C)
        skip = None
    ref = tokens(after, P.out - P.abase, W, H, C)
    mref = None
    if NPM:
        mref = tokens(after, P.p48 - P.abase, W // 2, H // 2, NPM)
    return P, x, skip, ref, mref


def measure(native, reference):
    """Exact code fraction and decoded error of one output tensor."""
    a = np.asarray(native, F64)
    b = np.asarray(reference, F64)
    delta = a - b
    peak = float(np.abs(b).max()) or 1.0
    return {
        'samples': int(b.size),
        'identical': float((a == b).mean()),
        'differing': int((a != b).sum()),
        'max_abs': float(np.abs(delta).max()) if delta.size else 0.0,
        'rms': float(np.sqrt((delta ** 2).mean())) if delta.size else 0.0,
        'rel_rms': float(np.sqrt((delta ** 2).mean()) / peak) if delta.size else 0.0,
        'signed_mean': float(delta.mean()) if delta.size else 0.0,
        'nonfinite': int((~np.isfinite(b)).sum()),
    }


def layers_of(capture):
    """The ten launches in network order, never in directory-name order."""
    found = {}
    for d in os.listdir(capture):
        if '-rrlite_' not in d:
            continue
        for name in mm.LAYERS:
            if d.endswith(f'_{name}_4x4'):
                sequence = int(d.split('-')[1])
                found.setdefault(name, []).append((sequence, os.path.join(capture, d)))
    ordered, used = [], {name: 0 for name in mm.LAYERS}
    for name in mm.ORDER:
        entries = sorted(found.get(name, []))
        index = used[name]
        if index < len(entries):
            ordered.append(entries[index][1])
            used[name] += 1
    return ordered


def run(capture, only=None, cross_check=False):
    dirs = layers_of(capture)
    if len(dirs) < 10:
        raise SystemExit(f'{capture}: {len(dirs)} layer launches, expected 10')
    rows = []
    for index, (name, directory) in enumerate(zip(mm.ORDER, dirs)):
        label = name if name != 'enc3_tube' else f'tube#{index - 1}'
        if only and only != label:
            continue
        P, x, skip, ref, mref = read_layer(directory, name)
        block = NaturalGridBlock(P, name)
        out, merged = forward(block.t, name, x, skip, P.sx, P.sy, block=block)
        row = {'layer': label, 'order': index, 'shape': [P.W, P.H, mm.LAYERS[name][0]],
               'shift': [P.sx, P.sy], 'out': measure(out, ref)}
        if mref is not None:
            row['merge'] = measure(merged, mref)
        if cross_check:
            kernel = KernelBlock(P, name)
            kout, kmerged = forward(kernel.t, name, x, skip, P.sx, P.sy, block=kernel)
            row['kernel_form_identical'] = bool(np.array_equal(out, kout))
            if mref is not None:
                row['kernel_form_merge_identical'] = bool(np.array_equal(merged, kmerged))
        rows.append(row)
        line = (f'{label:9s} {P.W}x{P.H}x{mm.LAYERS[name][0]:3d} shift {P.sx},{P.sy}  '
                f'out {100 * row["out"]["identical"]:7.3f}%  '
                f'rms {row["out"]["rms"]:.4g}  mean {row["out"]["signed_mean"]:+.4g}')
        if 'merge' in row:
            line += (f'  | merge {100 * row["merge"]["identical"]:7.3f}%  '
                     f'rms {row["merge"]["rms"]:.4g}')
        if cross_check:
            line += f'  | kernel form {row["kernel_form_identical"]}'
        print(line, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--layer', default=None)
    ap.add_argument('--cross-check', action='store_true',
                    help='also run the kernel-order model and require both forms identical')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()
    rows = run(a.capture, a.layer, a.cross_check)
    if a.json:
        json.dump({'capture': a.capture, 'contract': 'natural-order k32 model '
                   '(bit-identical to the kernel-order model)',
                   'layers': rows}, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
