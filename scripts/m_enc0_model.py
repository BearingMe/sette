#!/usr/bin/env python3
"""enc0's CPU oracle: `rrlite_enc0_4x4_mvhi_ldr_folded` on the canonical frame.

Reproduces the captured E4M3 token field byte for byte (8 355 840 / 8 355 840,
130 560 / 130 560 tokens fully identical).  Every step below is transcribed from
the kernel's PTX (module `dlss-0154-00.ptx` of `nvngx_dlss.dll` 310.7.0); EXP-015
records the instruction that establishes each one.

The pipeline, in the order the PTX performs it:

 1. **Which pixel.**  A field token (ty, tx) is the 2x2 pixel block at
    (2*ty + dy, 2*tx + dx).  The block's four pixels are the embedding's four
    k32 steps, in the order (0,0), (0,1), (1,0), (1,1): the mma's A fragment for
    step q reads the shared-memory word of pixel p + {0, 1, 8, 9}, and the
    shared tile is 8x8 pixels row-major (EXP-015 F3).
 2. **The mirrored texel.**  The pixel is folded into the image with
    `mirror(v) = v < size ? v : 2*size - v - 2` (the PTX's `not`/`add`/`selp`
    sequence), then point-sampled (`tex.base.2d`), so the value is one texel.
 3. **The 32 slots.**  Each pixel produces 32 feature slots; on a reset frame
    exactly 15 are nonzero and they carry the texel's channels in the cycle
    r, g, b, r, g, b, ... in ascending slot order (the other 2 slots of each
    colour triple are the history and previous-feature slots, which `reset`
    zeroes).  The slot order comes from the two `st.shared.v4.u32` stores and
    the dither registers that feed them, not from d4r's slot map - the two
    differ in 8 of the 15 positions.
 4. **The dither.**  `n = table[(frame<<8) | ((y&15)<<4) | (x&15)] >> 9`, then
    the PTX's own truncation
        t = f16(f16(h + (e|n)) - e);  r = bits(t) & 0xFF80
    with `h` the f16 of the value, `e = bits(h) & 0xFC00`.  The addition adds
    `(e|n)` to *h*, not to `e` - the earlier model added it to `e`, which
    discarded the mantissa and halved the number of distinct token vectors.
 5. **The embedding.**  128 features x 64 columns, e4m3 x e4m3, the bias seeding
    an f16 accumulator and one f16 rounding per 32 products (the mma chain).
 6. **The epilogue.**  A f16 clamp to +-2*pi, `cvt.rn.satfinite.e4m3x2`, and the
    channel permutation of the eight final accumulator chains (EXP-015).
 7. **The store.**  `(c/32)*4177920 + ty*15360 + tx*32 + (c%32)`, with both the
    row stride and the plane stride taken from the *padded* height (544).

usage: m_enc0_model.py CAPTURE_DIR [--json OUT]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'references', 'sources', 'd4r', 'kernels', 'tools'))

import m_weights as mw        # noqa: E402
import model_enc3 as me       # noqa: E402

F16, F32, F64 = np.float16, np.float32, np.float64

#: The 32 slots of one pixel, and the colour channel each carries on a reset
#: frame (-1 = exactly zero).  Derived in EXP-015 from the PTX: the two
#: `st.shared.v4.u32` stores fix the word order, the dither registers fix the
#: value order, and `reset` zeroes the history and exposure slots.
SLOT_CHANNEL = np.array(
    [0, 1, -1, -1, 2, 0, -1, -1, 1, 2, -1, -1, 0, 1, -1, -1,
     2, 0, -1, -1, 1, 2, -1, -1, 0, 1, -1, -1, 2, -1, -1, -1], np.int8)

#: `max.f16x2`/`min.f16x2` with `0f40C90FDB` / `0fC0C90FDB` (2*pi as f16).
TWO_PI = F32(6.28125)


def find_launch(capture):
    for entry in sorted(os.listdir(capture)):
        if 'enc0' in entry and 'mvhi' in entry and 'ldr' in entry:
            return os.path.join(capture, entry)
    raise SystemExit(f'{capture}: no mvhi/ldr enc0 launch')


def params(directory):
    a = open(os.path.join(directory, 'args.bin'), 'rb').read()
    out = {
        'origin': struct.unpack_from('<2i', a, 8),
        'padded': struct.unpack_from('<2i', a, 16),
        'visible': struct.unpack_from('<2i', a, 56),
        'render_inv': struct.unpack_from('<2f', a, 64),
        'frame': struct.unpack_from('<I', a, 312)[0],
        'reset': a[316],
    }
    for off, name in ((168, 'jitter'), (200, 'motion_scale'),
                      (208, 'motion_offset'), (224, 'display_motion_pixel'),
                      (232, 'colour_window_max'), (240, 'motion_window_max'),
                      (248, 'history_size'), (256, 'history_inv')):
        out[name] = struct.unpack_from('<2f', a, off)
    out['token_output'] = struct.unpack_from('<Q', a, 48)[0]
    return out


def image(directory, image_id):
    meta = None
    for line in open(os.path.join(directory, 'manifest.txt')):
        q = line.split()
        if q and q[0] == 'image' and int(q[1]) == image_id:
            meta = {'w': int(q[6]), 'h': int(q[7]), 'fmt': int(q[9], 16),
                    'channels': int(q[10])}
    if meta is None:
        raise SystemExit(f'no image {image_id}')
    raw = np.fromfile(os.path.join(directory, f'image-{image_id}.bin'), np.uint8)
    dtype = np.float16 if meta['fmt'] == 0x10 else np.float32
    return np.frombuffer(raw, dtype).reshape(
        meta['h'], meta['w'], meta['channels']).astype(np.float32)


def blob(directory, length=12416):
    raw = np.fromfile(os.path.join(directory, 'alloc-0.bin'), np.uint8)
    if raw.size < length:
        raise SystemExit(f'weight allocation is {raw.size} B, wanted {length}')
    return raw[:length]


def mirror(v, size):
    """`mirror(v) = v < size ? v : 2*size - v - 2`, the PTX's fold."""
    return np.where(v < size, v, 2 * size - v - 2)


def dither(values, n):
    """The PTX's dithered truncation to the FP8 grid.

    `and 0xFC00FC00` -> `or n` -> `add.f16x2 h, (e|n)` -> `sub.f16x2 -e`
    -> `and 0xFF80FF80`.  Both the add and the sub round to f16.
    """
    v = np.asarray(values, F32)
    v = np.where(np.isnan(v), 0.0, v)
    h = np.asarray(np.minimum(v, F32(1024.0)), F16)
    e = (h.view(np.uint16) & 0xFC00).astype(np.uint16)
    en = np.asarray((e | n.astype(np.uint16)).view(np.float16), F32)
    added = np.asarray(np.asarray(h, F32) + en, F16)
    t = np.asarray(np.asarray(added, F32) - np.asarray(e.view(np.float16), F32), F16)
    return np.asarray(np.uint16(t.view(np.uint16) & 0xFF80).view(np.float16), F32)


def features(colour, table, frame, padded, visible):
    """(TH, TW, 128) f32: the four pixels' slots, in k order."""
    th, tw = padded[1] // 2, padded[0] // 2
    out = np.zeros((th, tw, 4, 32), F32)
    ys, xs = np.arange(th), np.arange(tw)
    for sub, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        py = 2 * ys[:, None] + dy
        px = 2 * xs[None, :] + dx
        texel = colour[mirror(py, visible[1]), mirror(px, visible[0])]
        index = (frame << 8) | ((py & 15) << 4) | (px & 15)
        n = (table[index] >> 9).astype(np.uint32)
        v = np.zeros((th, tw, 32), F32)
        for slot in range(32):
            ch = SLOT_CHANNEL[slot]
            if ch >= 0:
                v[..., slot] = texel[..., ch]
        out[:, :, sub, :] = np.stack([dither(v[..., s], n) for s in range(32)], -1)
    return out.reshape(th, tw, 128)


def channel_columns():
    """Field channel -> mma column, from the epilogue's cvt/pack/store chain.

    The eight final accumulator chains are `mma 7, 8, 15, 16, 23, 24, 31, 32`
    (four k32 steps each).  The epilogue packs two chains into one stored word,
    so a token's channel `c` is not the column that produced it.
    """
    out = np.empty(64, np.int32)
    for c in range(64):
        plane, cc = divmod(c, 32)
        j = (cc >> 2) if cc < 16 else ((cc - 16) >> 2)
        r = cc & 3
        if cc < 16:
            base = (2 * j, 2 * j + 1, 16 + 2 * j, 17 + 2 * j)[r]
        else:
            base = (8 + 2 * j, 9 + 2 * j, 24 + 2 * j, 25 + 2 * j)[r]
        out[c] = plane * 32 + base
    return out


def embedding(codes, matrix, bias):
    """One f16 rounding per 32 products, the bias seeding the accumulator."""
    x = me.E4M3[codes.reshape(-1, 128)].astype(F64)
    w = me.E4M3[matrix].astype(F64)
    out = me.mma_chain(x, w, C0=np.broadcast_to(bias.astype(F64),
                                                (x.shape[0], w.shape[1])))
    return np.asarray(out, F16).reshape(codes.shape[0], codes.shape[1], 64)


def reference(directory, padded):
    """The captured field at the launch's own output pointer, as (TH, TW, 64)."""
    post = np.fromfile(os.path.join(directory, 'gpu-output', 'alloc-1.bin'), np.uint8)
    plane = (padded[1] // 2) * (padded[0] // 2) * 32
    th, tw = padded[1] // 2, padded[0] // 2
    field = post[0xFF0000:0xFF0000 + 2 * plane]
    codes = np.empty((th, tw, 64), np.uint8)
    for p in range(2):
        codes[:, :, p * 32:(p + 1) * 32] = field[p * plane:(p + 1) * plane].reshape(
            th, tw, 32)
    return codes


def stack(weights):
    table = np.frombuffer(weights[:4096], np.uint16)
    off = mw.layout_offsets(mw.front_end_layout('enc0'))
    matrix = np.ascontiguousarray(weights[off])
    bias = np.frombuffer(weights[12288:12416], np.float16)
    return table, matrix, bias


def tokens(directory, weights=None, params_=None):
    """The reconstructed token field: (TH, TW, 64) e4m3 codes."""
    P = params_ or params(directory)
    w = weights if weights is not None else blob(directory)
    table, matrix, bias = stack(w)
    colour = image(directory, 0)
    feat = features(colour, table, P['frame'], P['padded'], P['visible'])
    codes = mw.e4m3_encode(feat)
    acc = embedding(codes, matrix, bias)
    acc = np.clip(acc, -TWO_PI, TWO_PI)
    return mw.e4m3_encode(np.ascontiguousarray(acc[..., channel_columns()]))


def compare(native, ref):
    same = native == ref
    bad = np.nonzero(~same)
    result = {
        'identical': float(same.mean()),
        'tokens_fully_identical': float(same.all(-1).mean()),
        'differing_bytes': int(bad[0].size),
        'bytes': int(same.size),
        'vectors_native': int(np.unique(native.reshape(-1, 64), axis=0).shape[0]),
        'vectors_reference': int(np.unique(ref.reshape(-1, 64), axis=0).shape[0]),
        'value_range_native': [float(me.E4M3[native].min()),
                               float(me.E4M3[native].max())],
        'value_range_reference': [float(me.E4M3[ref].min()),
                                  float(me.E4M3[ref].max())],
    }
    if bad[0].size:
        a = np.asarray(me.E4M3[native[bad]], F64)
        b = np.asarray(me.E4M3[ref[bad]], F64)
        result['max_abs_error'] = float(np.abs(a - b).max())
        result['rows'] = [int(bad[0].min()), int(bad[0].max())]
        result['planes'] = sorted({int(c) // 32 for c in np.unique(bad[2])})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    directory = find_launch(a.capture)
    P = params(directory)
    ref = reference(directory, P['padded'])
    got = tokens(directory, params_=P)
    result = {'launch': directory,
              'params': {k: v for k, v in P.items() if k != 'token_output'},
              'reset': int(P['reset']), 'frame': int(P['frame']),
              'comparison': compare(got, ref)}
    print(f'{os.path.basename(directory)}: frame {P["frame"]} reset {P["reset"]} '
          f'padded {tuple(P["padded"])} visible {tuple(P["visible"])}')
    for key, value in result['comparison'].items():
        print(f'  {key:24s} {value}')
    print('\nbyte-exact' if result['comparison']['differing_bytes'] == 0
          else '\nNOT byte-exact')
    if a.json:
        json.dump(result, a.json, indent=2)
    return 0 if result['comparison']['differing_bytes'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
