#!/usr/bin/env python3
"""B4.4 instrument: a partial reconstruction of `rrlite_enc0_4x4_mvhi_ldr_folded`.

**This model does not reproduce the reference.**  It exists because it is the
instrument that bounded the remaining gap, and because three of its four parts
are established: the resource and parameter map, the blob layout (matrix, bias
and the dithered-truncation table), and the token store.  What is not yet right
is the map from a pixel's 32 feature slots to the 128 k channels of the
embedding: the model produces exactly as many distinct token vectors as the
capture (24) but none of them match, and its values are about a factor of two
too large.  See EXP-015.

Established from the PTX of `rrlite_enc0_4x4_mvhi_ldr_folded` (EXP-015):

  * a workgroup is 8x8 pixels with 32 threads in two passes of 8x4; each pixel
    contributes 32 feature slots, so a 2x2 pixel block (one token) supplies the
    128 channels of the embedding, one pixel per k32 step;
  * the depth texture (param+136), both 1x1 textures (param+112, param+120) and
    the pointer at param+40 are never loaded: this is the display-motion variant,
    which takes one motion vector per pixel and no depth;
  * frame 1 has `reset = 1`, so the history term (`halfTex`), the previous
    feature (`featTex`) and the motion-change scalar are exactly zero and every
    nonzero slot is one of the three colour channels;
  * the three RG16F surfaces at param+280/288/296 are written with the motion
    candidates (zero on this frame, which is why the capture shows them zero).

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
W, H = 960, 540                # visible render size (param+56)
TW, TH = 480, 272               # token grid from the padded size (param+16)
PLANE = TH * TW * 32            # 4 177 920

# The 32 feature slots of one pixel and the colour channel each carries on a
# reset frame.  d4r's pinned `enc0_features.glsl` documents the same 32 slots; on
# a reset frame the history and previous-feature slots are exactly zero, so every
# nonzero slot is a repeat of one of the three channels.
SLOT_CHANNEL = {0: 0, 1: 1, 4: 2,
                5: 0, 8: 0, 9: 0, 12: 1, 13: 1, 16: 1,
                17: 2, 20: 2, 21: 2, 24: 0, 25: 0, 28: 0}


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
                      (248, 'history_size'), (256, 'history_inv')):
        out[name] = struct.unpack_from('<2f', a, off)
    out['e_multiplier'] = struct.unpack_from('<f', a, 268)[0]
    out['exposure'] = struct.unpack_from('<f', a, 272)[0]
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


def blob(directory, length):
    raw = np.fromfile(os.path.join(directory, 'alloc-0.bin'), np.uint8)
    if raw.size < length:
        raise SystemExit(f'weight allocation is {raw.size} B, wanted {length}')
    return raw[:length]


def mirror(v, size):
    """The kernel's fold: |v| inside the image, mirrored outside it."""
    a = np.abs(v)
    return np.where(a < size, a, 2 * size - a - 2)


def half(value):
    return np.asarray(np.asarray(value, F32), F16)


def dither(values, n):
    """The kernel's dithered truncation to the FP8 grid.

    `and 0xFC00` | `or n` -> add -> subtract the exponent -> `and 0xFF80`.
    The result can overshoot 448; `cvt.rn.satfinite` is what saturates it.
    """
    h = np.float16(np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
                           0.0, 1024.0))
    bits = h.view(np.uint16).astype(np.uint32)
    e = bits & 0xFC00
    base = np.asarray(np.uint16(e).view(np.float16), F32)
    added = half(base + np.asarray(np.uint16(e | n).view(np.float16), F32))
    t = half(np.asarray(added, F32) - base)
    r = np.uint16(t.view(np.uint16) & 0xFF80).view(np.float16)
    return np.asarray(r, F32)


def features(colour, table, frame):
    """(TH, TW, 4, 32) float32: the slots of the four pixels of each token."""
    slots = np.zeros((TH, TW, 4, 32), F32)
    ys, xs = np.arange(TH), np.arange(TW)
    for sub, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        py = 2 * ys + dy
        px = 2 * xs + dx
        own = colour[mirror(py, H)[:, None], mirror(px, W)[None, :]]
        n = ((table[(frame << 8) | ((np.abs(py) & 15)[:, None] << 4)
                    | (np.abs(px) & 15)[None, :]] >> 9).astype(np.uint32))[..., None]
        v = np.zeros((TH, TW, 32), F32)
        for slot, channel in SLOT_CHANNEL.items():
            v[..., slot] = own[..., channel]
        slots[:, :, sub, :] = np.stack([dither(v[..., k], n[..., 0])
                                        for k in range(32)], -1)
    return slots


def embedding(feature_codes, matrix_codes, bias):
    """One f16 rounding per 32 products, the bias seeding the accumulator."""
    x = me.E4M3[feature_codes.reshape(-1, 128)].astype(F64)
    w = me.E4M3[matrix_codes].astype(F64)
    out = me.mma_chain(x, w, C0=np.broadcast_to(bias.astype(F64),
                                                (x.shape[0], w.shape[1])))
    return np.asarray(out, F16).reshape(feature_codes.shape[0],
                                        feature_codes.shape[1], 64)


def reference(directory):
    """The captured token field, which is also enc1's input."""
    post = np.fromfile(os.path.join(directory, 'gpu-output', 'alloc-1.bin'), np.uint8)
    field = post[0xFF0000:0xFF0000 + 2 * PLANE]
    codes = np.empty((TH, TW, 64), np.uint8)
    for plane in range(2):
        codes[:, :, plane * 32:(plane + 1) * 32] = \
            field[plane * PLANE:(plane + 1) * PLANE].reshape(TH, TW, 32)
    return codes


def compare(native, reference_codes):
    same = native == reference_codes
    return {
        'identical': float(same.mean()),
        'tokens_fully_identical': float(same.all(-1).mean()),
        'per_channel': [float(same[..., c].mean()) for c in range(64)],
        'native_unique_vectors': int(np.unique(native.reshape(-1, 64), axis=0).shape[0]),
        'reference_unique_vectors': int(np.unique(reference_codes.reshape(-1, 64),
                                                  axis=0).shape[0]),
        'native_value_range': [float(me.E4M3[native].min()),
                               float(me.E4M3[native].max())],
        'reference_value_range': [float(me.E4M3[reference_codes].min()),
                                  float(me.E4M3[reference_codes].max())],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--json', type=argparse.FileType('w'), default=None)
    a = ap.parse_args()

    directory = find_launch(a.capture)
    P = params(directory)
    colour = image(directory, 0)
    weights = blob(directory, 12416)
    table = np.frombuffer(weights[:4096], np.uint16)
    spec = mw.front_end_layout('enc0')
    offsets = mw.layout_offsets(spec)
    matrix = np.ascontiguousarray(weights[offsets])
    bias = np.frombuffer(weights[12288:12416], np.float16)

    slots = features(colour, table, P['frame'])
    codes = mw.e4m3_encode(slots.reshape(TH, TW, 128))
    tokens = mw.e4m3_encode(embedding(codes, matrix, bias))
    ref = reference(directory)
    result = {'launch': directory, 'params': {k: v for k, v in P.items()
                                              if k != 'token_output'},
              'reset': int(P['reset']), 'frame': int(P['frame']),
              'weight_layout': {'K': spec['K'], 'N': spec['N'],
                                'base': spec['base'], 'bias': spec['bias']},
              'comparison': compare(tokens, ref),
              'reproduces_reference': False}
    print(f'{os.path.basename(directory)}: frame {P["frame"]} reset {P["reset"]}')
    for key, value in result['comparison'].items():
        if key != 'per_channel':
            print(f'  {key:26s} {value}')
    print('\nThis model does not reproduce the reference: see EXP-015. The gap is the')
    print('map from a pixel\'s 32 feature slots to the 128 embedding channels, and the')
    print('byte order of the 64 output channels within the stored token.')
    if a.json:
        json.dump(result, a.json, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
