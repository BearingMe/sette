#!/usr/bin/env python3
"""post's CPU oracle: `rrlite_post_3_1_mvhi_ldr_folded` on the canonical frame.

Every step is transcribed from the kernel's own PTX (`dlss-0145-00.ptx` of
`nvngx_dlss.dll` 310.7.0).  EXP-016 records the instruction behind each one;
the numbers in the comments are PTX line numbers.

The kernel, in order:

 1. **The tile** (36-282).  A 16x16 block owns 16 reconstruction columns and 32
    rows; its first 160 threads (`r52 = tid.x + 16*tid.y`, `if r52 > 159` skip)
    each fetch one render texel and write four f16 into each of three 1280-byte
    shared arrays.  The tile index is `(row = te/10, col = te%10)` and the texel
    is `clamp((r1 + col - 2, r2 + row - 2))` with `r1 = trunc(scale_x*16*bx)`,
    `r2 = trunc(scale_y*32*by)`.  The arrays are, per texel,
        [colour.r, colour.g, colour.b, mix.x]
        [mix.y,    mix.z,    gate.x,   gate.y]
        [gate.z,   cov.x*k,  cov.y*k,  cov.z*2k]
    with `k = -0.7213249802589417` (0fBF38A8C1).
 2. **The 3x3 Gaussians** (298-1333).  Tap `(i, j)` samples render texel
    `base + (i-1, j-1)`, `base = floor(pos - 0.5)`, with an f16 exponent
        e = f16( f16(dx^2*cov.xx) + f16(dy^2*cov.yy) ) + f16( f16(dx*dy)*cov.xy )
    and `w = ex2(f32(e))`; the nine accumulators are f16 (`fma.rn.f16x2`, one
    rounding per tap) and the normaliser W is the f32 sum of the *unrounded*
    weights in tap order.  The odd row of a thread's pair keeps the even row's
    three texel rows and adjusts its own dy by -1 when the two `floor(pos-0.5)`
    differ (301-332).
 3. **The candidates** (1488-1547).  `param+176` (display motion) is sampled at
    `trunc(pos/reconSize*displaySize)`; `mvhi` gives all three candidates one
    vector, so the sort only orders the three mix values.  The history position
    is `fma(1.5, mv+offset, pos)` and the history is skipped when it leaves
    `[0,2880] x [0,1620]` or `reset != 0`; on this frame `reset = 1`, so the
    colour texel at `floor(pos)` is used.
 4. **The blend** (1843-1873).  `g0 = sat(gain*W*gate.x)`, gates
    `fma(g0, 1-gate.y, gate.y)` and `fma(g0, 1-gate.z, gate.z)`, outputs
    `fma(cur-hist, gate, hist)`.
 5. **The outputs** (1902-2373).  `outA` -> the 2880x1620 RGBA16F surface,
    `outB` -> `trunc(clamp(fma(c, 2047, 0.5), 0, 2047))` (1023 for blue) in the
    2880x1620 word buffer, and the 1440x810 RGBA32F surface gets the mean of
    `outA` over the row pair and the neighbouring column.

usage: m_post_model.py CAPTURE_DIR [--json OUT]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64

#: the kernel's own constants, by PTX bit pattern.
COV_XXYY = F32(-0.7213249802589417)        # 0fBF38A8C1
COV_XY = F32(-1.4426499605178833)          # 0fBFB8A8C1
SAT_MAX = F32(32752.0)                     # 0f46FFE000
RC_FLOOR = F32(1.000000013351432e-10)      # 0f2EDBE6FF
INV_255 = F32(0.003921568859368563)        # 0f3B808081
C_1_1 = F32(1.100000023841858)             # 0f3F8CCCCD
C_M_0_05 = F32(-0.05000000074505806)       # 0fBD4CCCCD


def find_launch(capture):
    for entry in sorted(os.listdir(capture)):
        if 'post' in entry and 'mvhi' in entry and 'ldr' in entry:
            return os.path.join(capture, entry)
    raise SystemExit(f'{capture}: no mvhi/ldr post launch')


def params(directory):
    a = open(os.path.join(directory, 'args.bin'), 'rb').read()
    f2 = lambda o: struct.unpack_from('<2f', a, o)
    u2 = lambda o: struct.unpack_from('<2I', a, o)
    out = {
        'colour_origin': u2(8),
        'scale_x': struct.unpack_from('<f', a, 124)[0],
        'scale_y': struct.unpack_from('<f', a, 128)[0],
        'offset_x': struct.unpack_from('<f', a, 132)[0],
        'offset_y': struct.unpack_from('<f', a, 136)[0],
        'recon': u2(140),
        'render': u2(148),
        'gain': struct.unpack_from('<f', a, 120)[0],
        'mv_scale': f2(200),
        'mv_offset': f2(208),
        'display': u2(192),
        'reset': a[168],
    }
    return out


def meta(directory, image_id):
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if q and q[0] == 'image' and int(q[1]) == image_id:
            return {'w': int(q[6]), 'h': int(q[7]), 'fmt': int(q[9], 16),
                    'channels': int(q[10])}
    raise SystemExit(f'{directory}: no image {image_id}')


def _decode(raw, m):
    if m['fmt'] == 0x10:
        return np.frombuffer(raw, np.float16).astype(F32)
    if m['fmt'] == 0x01:
        return (raw.astype(F32) / F32(255.0))
    if m['fmt'] == 0x20:
        return np.frombuffer(raw, np.float32).astype(F32)
    raise SystemExit(f'unsupported backing 0x{m["fmt"]:02x}')


def image(directory, image_id, post=False):
    """The captured image as (h, w, c) float32, in the shader's own terms."""
    m = meta(directory, image_id)
    name = os.path.join(directory, 'gpu-output' if post else '',
                        f'image-{image_id}.bin')
    data = _decode(np.fromfile(name, np.uint8), m)
    return data.reshape(m['h'], m['w'], m['channels'])


def reference(directory):
    raw = np.fromfile(os.path.join(directory, 'gpu-output', 'alloc-11.bin'), np.uint32)
    return raw.reshape(1620, 2880)


def reference_image(directory, image_id, channels=3):
    return image(directory, image_id, post=True)[..., :channels]


def trunc_f16(x):
    """The captured RGBA16F surface truncates toward zero (EXP-016)."""
    magnitude = np.abs(np.asarray(x, F32)).astype(F16)
    up = magnitude.astype(F32)
    bits = magnitude.view(np.uint16)
    down = np.where(up > np.abs(x), (bits - 1).view(F16).astype(F32), up)
    return np.where(np.asarray(x, F32) < 0.0, -down, down)


def unorm10_codes(x):
    return np.clip(np.floor(np.asarray(x, F32) * F32(1023.0) + F32(0.5)), 0.0, 1023.0)


def f16(x):
    return np.asarray(x, F32).astype(F16)


def f16_fma(a, b, c):
    """`fma.rn.f16`/`.f16x2` on f16 inputs: one rounding."""
    return (np.asarray(a, F64) * np.asarray(b, F64) + np.asarray(c, F64)).astype(F16)


def fma32(a, b, c):
    """`fma.rn.f32`: a single rounding of a*b + c."""
    return (np.asarray(a, F64) * np.asarray(b, F64) + np.asarray(c, F64)).astype(F32)


def tiles(colour, mix, gate, cov):
    """The three shared arrays per render texel, as the fill builds them."""
    h, w = colour.shape[:2]
    t0 = np.empty((h, w, 4), F16)
    t1 = np.empty((h, w, 4), F16)
    t2 = np.empty((h, w, 4), F16)

    c = np.array(colour[..., :3], F32)
    c[~np.isfinite(c)] = 0.0
    c = np.minimum(c, SAT_MAX)
    t0[..., :3] = f16(c)
    t0[..., 3] = f16(mix[..., 0])

    t1[..., 0] = f16(mix[..., 1])
    t1[..., 1] = f16(mix[..., 2])
    t1[..., 2] = f16(gate[..., 0])
    t1[..., 3] = f16(gate[..., 1])

    t2[..., 0] = f16(gate[..., 2])
    t2[..., 1] = f16(np.asarray(cov[..., 0] * COV_XXYY, F32))
    t2[..., 2] = f16(np.asarray(cov[..., 1] * COV_XXYY, F32))
    t2[..., 3] = f16(np.asarray(cov[..., 2] * COV_XY, F32))
    return t0, t1, t2


def clamp_index(v, size):
    return np.clip(v, 0, size - 1)


def gaussians(t0, t1, t2, base_x, base_y, dx0, dy0):
    """The nine f16 accumulators and the f32 weight sum for a row pair.

    Returns arrays shaped (2, rows, width, ...): index 0 is the even row.
    """
    rows, width = base_x.shape
    acc0 = np.zeros((2, rows, width, 4), F16)
    acc1 = np.zeros((2, rows, width, 4), F16)
    acc2 = np.zeros((2, rows, width, 4), F16)
    weight = np.zeros((2, rows, width), F32)

    dx = [f16(dx0 + F16(i - 1))[None] for i in range(3)]
    dy = [f16(dy0[r] + F16(j - 1))[None] for r in range(2) for j in range(3)]
    dy = np.array(dy, F16).reshape(2, 3, rows, width)

    for i in range(3):                       # column-major: the PTX loads
        for j in range(3):                   # [base], [base+80], [base+160], ...
            gy = clamp_index(base_y + (j - 1), t0.shape[0])
            gx = clamp_index(base_x + (i - 1), t0.shape[1])
            a = t0[gy, gx]
            b = t1[gy, gx]
            c = t2[gy, gx]

            dxj = dx[i]                                     # (1, rows, width)
            dyj = dy[:, j]                                  # (2, rows, width)
            xx = c[..., 1]
            yy = c[..., 2]
            xy = c[..., 3]

            e = f16(f16(f16(dxj * dxj) * xx) + f16(f16(dyj * dyj) * yy))
            e = f16(e + f16(f16(dxj * dyj) * xy))

            w = np.exp2(np.asarray(e, F32))
            wh = f16(w)
            acc0 = f16_fma(wh[..., None], a[None], acc0)
            acc1 = f16_fma(wh[..., None], b[None], acc1)
            acc2 = f16_fma(wh[..., None], c[None], acc2)
            weight = weight + w
    return acc0, acc1, acc2, weight


def rcp(w):
    """`rcp.approx.ftz.f32`: one reciprocal, rounded to f32."""
    return np.asarray(F32(1.0) / np.asarray(w, F32), F32)


def blend(cur, mix, gate, weight, colour, pos_x, pos_y, P):
    """One output row's two blended colours."""
    inv = rcp(weight)[..., None]
    cur = np.asarray(cur * inv, F32)
    mix = np.asarray(mix * inv, F32)
    gate = np.asarray(gate * inv, F32)

    # the three candidate weights, sorted; the display-motion vector is one
    lo = np.minimum(mix[..., 0], mix[..., 1])
    hi = np.maximum(mix[..., 0], mix[..., 1])
    med = np.where(lo < mix[..., 2], mix[..., 2], lo)
    big = np.maximum(hi, med)
    second = np.minimum(hi, med)

    k1 = np.clip(fma32(big, C_1_1, C_M_0_05), 0.0, 1.0)
    k2 = np.clip(fma32(second, C_1_1, C_M_0_05), 0.0, 1.0)
    den = np.maximum(np.asarray(k1 + k2, F32), RC_FLOOR)
    kn = rcp(den)
    b2 = np.asarray(k2 * kn, F32)
    b1 = np.asarray(k1 * kn, F32)
    tiny = b2 < INV_255
    b2 = np.where(tiny, F32(0.0), b2)
    b1 = np.where(tiny, F32(1.0), b1)

    # the history sample: reset or off-image falls back to the colour texel
    ox, oy = P['colour_origin']
    rw, rh = P['render']
    hx = clamp_index(np.floor(pos_x).astype(np.int64) + ox, rw)
    hy = clamp_index(np.floor(pos_y).astype(np.int64) + oy, rh)
    fallback = colour[hy, hx, :3]

    mvs, mvo = P['mv_scale'], P['mv_offset']
    qx = fma32(F32(mvs[0]), F32(0.0) + F32(mvo[0]), pos_x)
    qy = fma32(F32(mvs[1]), F32(0.0) + F32(mvo[1]), pos_y)
    recon = P['recon']
    off = ((qx < 0.0) | (qy < 0.0)
           | (qx > F32(recon[0])) | (qy > F32(recon[1])))
    if P['reset'] == 0:
        raise SystemExit('post: the history/Catmull-Rom path is not modelled '
                         '(reset = 0)')
    h1 = fallback
    h2 = np.where((b2 > 0.0)[..., None], h1, F32(0.0))
    # f525 = b2*h2 (rounded), then fma(b1, h1, f525)
    hist = fma32(b1[..., None], h1, np.asarray(b2[..., None] * h2, F32))

    gain = F32(P['gain'])
    g0 = np.clip(np.asarray((gain * weight) * gate[..., 0], F32), 0.0, 1.0)
    ga = fma32(g0, F32(1.0) - gate[..., 1], gate[..., 1])
    gb = fma32(g0, F32(1.0) - gate[..., 2], gate[..., 2])

    delta = np.asarray(cur - hist, F32)
    out_a = fma32(delta, ga[..., None], hist)
    out_b = fma32(delta, gb[..., None], hist)
    bad_a = ~np.isfinite(out_a[..., 0] + out_a[..., 1] + out_a[..., 2])
    bad_b = ~np.isfinite(out_b[..., 0] + out_b[..., 1] + out_b[..., 2])
    out_a = np.where(bad_a[..., None], cur, out_a)
    out_b = np.where(bad_b[..., None], cur, out_b)
    return out_a, out_b


def pack(out_b):
    """`trunc(clamp(fma(c, 2047, 0.5), 0, 2047))`, blue over 1023."""
    def code(v, top):
        x = fma32(v, F32(top), F32(0.5))
        return np.clip(x, 0.0, top).astype(np.uint32)
    r = code(out_b[..., 0], 2047.0)
    g = code(out_b[..., 1], 2047.0)
    b = code(out_b[..., 2], 1023.0)
    return r | (g << 11) | (b << 22)


def reconstruct(directory, P=None, crop=None):
    P = P or params(directory)
    colour = image(directory, 0)
    gate = image(directory, 3)
    mix = image(directory, 4)
    cov = image(directory, 5)
    t0, t1, t2 = tiles(colour, mix, gate, cov)

    rw, rh = P['recon']
    sx, sy = F32(P['scale_x']), F32(P['scale_y'])
    ox, oy = F32(P['offset_x']), F32(P['offset_y'])
    pairs = rh // 2
    k0, k1 = crop or (0, pairs)
    rh_view = 2 * (k1 - k0)

    x = np.arange(rw, dtype=F32) + F32(0.5)
    y0 = (np.arange(k0, k1, dtype=F32) * F32(2.0)) + F32(0.5)
    y1 = y0 + F32(1.0)

    x = np.arange(rw, dtype=F32) + F32(0.5)
    y0 = np.arange(pairs, dtype=F32) * F32(2.0) + F32(0.5)
    y1 = y0 + F32(1.0)
    pos_x = np.broadcast_to(np.asarray(sx * x + ox, F32), (k1 - k0, rw))
    pos_y = np.stack([np.broadcast_to(np.asarray(sy * y0 + oy, F32)[:, None], (k1 - k0, rw)),
                      np.broadcast_to(np.asarray(sy * y1 + oy, F32)[:, None], (k1 - k0, rw))])

    base_x = np.floor(pos_x - F32(0.5)).astype(np.int64)
    base_y = np.floor(pos_y - F32(0.5)).astype(np.int64)

    dx0 = f16((base_x.astype(F32) + F32(0.5)) - pos_x)
    dy0 = f16((base_y.astype(F32) + F32(0.5)) - pos_y)
    # the odd row adjusts its own residual when the two bases differ (301-332)
    dy0[1] = np.where(base_y[1] != base_y[0],
                      f16(dy0[1] - F16(1.0)), dy0[1])

    acc0, acc1, acc2, weight = gaussians(t0, t1, t2, base_x, base_y[0], dx0, dy0)

    cur = acc0[..., :3]
    mixv = np.concatenate([acc0[..., 3:4], acc1[..., 0:2]], axis=-1)
    gatev = np.concatenate([acc1[..., 2:4], acc2[..., 0:1]], axis=-1)

    pos_x_all = np.stack([pos_x, pos_x])
    out_a, out_b = blend(cur, mixv, gatev, weight, colour, pos_x_all, pos_y, P)

    # the half-resolution mean: own row pair + the neighbouring column's pair
    pair_sum = np.asarray(out_a[0] + out_a[1], F32)          # (pairs, rw, 3)
    mean = np.asarray((pair_sum[:, 0::2] + pair_sum[:, 1::2]) * F32(0.25), F32)

    # (row type, pair, x) -> (y, x)
    packed = np.transpose(pack(out_b), (1, 0, 2)).reshape(rh_view, rw)
    out_a = np.transpose(out_a, (1, 0, 2, 3)).reshape(rh_view, rw, 3)
    out_b = np.transpose(out_b, (1, 0, 2, 3)).reshape(rh_view, rw, 3)

    return {'out_a': out_a, 'out_b': out_b, 'packed': packed, 'half': mean}


def compare(native, ref):
    same = native == ref
    diff = int((~same).sum())
    out = {'identical': float(same.mean()), 'differing_items': diff,
           'items': int(same.size)}
    if diff:
        d = native[~same].astype(np.int64) - ref[~same].astype(np.int64)
        out['max_abs_delta'] = int(np.abs(d).max())
        if native.ndim == 1:
            rows = np.nonzero(~same)[0]
        else:
            rows = np.nonzero((~same).any(axis=tuple(range(1, same.ndim))))[0]
        out['rows'] = [int(rows.min()), int(rows.max())]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    directory = find_launch(a.capture)
    P = params(directory)
    got = reconstruct(directory, P)
    ref = reference(directory)
    ref10 = reference_image(directory, 10)
    ref11 = reference_image(directory, 11)
    history = trunc_f16(got['out_a'])
    half = unorm10_codes(got['half'])
    result = {
        'launch': directory, 'params': P,
        'packed': compare(got['packed'], ref),
        'history': compare(history, ref10),
        'half': {'identical': float((half == unorm10_codes(ref11)).mean()),
                 'codes_differing': int((half != unorm10_codes(ref11)).sum()),
                 'codes': int(half.size)},
    }
    print(f'{os.path.basename(directory)} reset {P["reset"]} recon {P["recon"]} '
          f'render {P["render"]}')
    for key in ('packed', 'history', 'half'):
        print(f'  {key:8s} {result[key]}')
    ok = (result['packed']['differing_items'] == 0
          and result['history']['differing_items'] == 0
          and result['half']['codes_differing'] == 0)
    if a.json:
        json.dump(result, a.json, indent=2)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
