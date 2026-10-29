#!/usr/bin/env python3
"""Deterministic INT8 encoders for Preset M's weights and activations, plus the
integer GEMM arithmetic a GPU would execute on them.

Everything here is CPU-only and self-contained: no captured data, no d4r model
code, no GPU.  The point is to make the *arithmetic* explicit so it can be
checked against independent integer arithmetic and later reproduced in a kernel.

Conventions
-----------
* Symmetric signed INT8, no zero point: `q = clamp(round_half_even(w / s))`.
* A scale is stored as f32 (as d4r's `widenM` does) and quantization divides by
  that stored value, so encode and decode are self-consistent.  `d4r_split` also
  reproduces d4r's asymmetry (round against a float64 scale, store float32).
* Scopes: one scale per tensor, per output channel (column of W[K][N]), per
  input channel (row of W), or one per block of K rows.
* Activation quantization is per row (per token) or per tensor, as a separate
  choice from the weight scope.

`gemm_integer()` executes the product the way an INT8 kernel must: exact integer
products, int32 accumulation, and one floating-point rescale per scale group.
"""
import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64
QMAX = 127


def _f32(value):
    return np.float32(value)


def scales_for(W, scope, block=None):
    """The scale table for one scope, in float32, as an INT8 kernel would store it."""
    W = np.asarray(W, F64)
    if scope == 'per_tensor':
        top = np.abs(W).max()
        return np.array([top / QMAX if top else 1.0], F32)
    if scope == 'per_output_channel':
        top = np.abs(W).max(axis=0)
        return np.where(top > 0, top / QMAX, 1.0).astype(F32)
    if scope == 'per_input_channel':
        top = np.abs(W).max(axis=1)
        return np.where(top > 0, top / QMAX, 1.0).astype(F32)
    if scope == 'block_k':
        block = int(block)
        rows = -(-W.shape[0] // block)
        out = np.empty((rows, W.shape[1]), F32)
        for r in range(rows):
            tile = W[r * block:(r + 1) * block]
            top = np.abs(tile).max(axis=0)
            out[r] = np.where(top > 0, top / QMAX, 1.0)
        return out
    raise ValueError(scope)


def quantize(W, scope, block=None, d4r_split=False):
    """Symmetric INT8 encoding of W[K][N] plus the scale table it was built with."""
    W = np.asarray(W, F64)
    scales = scales_for(W, scope, block)
    if scope == 'per_tensor':
        divisor = np.float64(scales[0])
        ratio = W / divisor
    elif scope == 'per_output_channel':
        divisor = scales.astype(F64)
        ratio = W / divisor[None, :]
    elif scope == 'per_input_channel':
        divisor = scales.astype(F64)
        ratio = W / divisor[:, None]
    elif scope == 'block_k':
        block = int(block)
        divisor = np.repeat(scales, block, axis=0)[:W.shape[0]].astype(F64)
        ratio = W / divisor
    else:
        raise ValueError(scope)
    if d4r_split:
        # d4r rounds against a float64 scale but stores the float32 one; the
        # only difference is which side of a rounding tie a value falls on.
        exact = W / (np.asarray(divisor, F64))
        codes = np.clip(np.rint(exact), -QMAX, QMAX)
    else:
        codes = np.clip(np.rint(ratio), -QMAX, QMAX)
    return {'codes': codes.astype(np.int8), 'scales': scales, 'scope': scope,
            'block': block, 'ratio': ratio}


def dequantize(q):
    """Exact float64 reconstruction of an encoded matrix."""
    codes = q['codes'].astype(F64)
    scales = np.asarray(q['scales'], F64)
    if q['scope'] == 'per_tensor':
        return codes * scales[0]
    if q['scope'] == 'per_output_channel':
        return codes * scales[None, :]
    if q['scope'] == 'per_input_channel':
        return codes * scales[:, None]
    if q['scope'] == 'block_k':
        block = int(q['block'])
        return codes * np.repeat(scales, block, axis=0)[:codes.shape[0]]
    raise ValueError(q['scope'])


def scale_bytes(q):
    return int(np.asarray(q['scales']).size * 4)


def metrics(W, q):
    """Error and representation statistics of one encoding."""
    W = np.asarray(W, F64)
    Wq = dequantize(q)
    err = Wq - W
    norm = np.linalg.norm(W)
    ratio = q['ratio']
    codes = q['codes']
    data = int(codes.size)
    return {
        'scope': q['scope'] + (f'{q["block"]}' if q['scope'] == 'block_k' else ''),
        'max_abs': float(np.abs(err).max()),
        'mean_abs': float(np.abs(err).mean()),
        'rmse': float(np.sqrt((err ** 2).mean())),
        'rel_frobenius': float(np.linalg.norm(err) / norm) if norm else 0.0,
        'over_range': float(np.mean(np.abs(ratio) > QMAX + 0.5)),
        'at_endpoint': float(np.mean(np.abs(codes) == QMAX)),
        'distinct_codes': int(np.unique(codes).size),
        'code_rms_ratio': float(np.sqrt((codes.astype(F64) ** 2).mean()) / QMAX),
        'zeros_preserved': bool(np.array_equal(codes[W == 0], np.zeros(int((W == 0).sum()), np.int8))),
        'data_bytes': data,
        'scale_bytes': scale_bytes(q),
        'total_bytes': data + scale_bytes(q),
        'bytes_over_fp8': float((data + scale_bytes(q)) / data),
    }


def quantize_rows(x, scope='per_row'):
    """Activation encoding: one scale per row (token) or one for the tensor."""
    x = np.asarray(x, F64)
    if scope == 'per_row':
        top = np.abs(x).max(axis=1, keepdims=True)
        top = np.where(top > 0, top, 1.0)
        # d4r computes inv = 127/mx in f32 and ts = mx * (1/127) in f32
        inv = (np.float32(QMAX) / top.astype(F32)).astype(F64)
        ratio = x * inv
        scale = (top.astype(F32) * np.float32(1.0 / QMAX)).astype(F64)
    elif scope == 'per_tensor':
        top = np.abs(x).max()
        inv = np.float64(np.float32(QMAX) / np.float32(top if top else 1.0))
        ratio = x * inv
        scale = np.full((x.shape[0], 1), np.float64(np.float32(top if top else 1.0) * np.float32(1.0 / QMAX)))
    else:
        raise ValueError(scope)
    codes = np.clip(np.rint(ratio), -QMAX, QMAX).astype(np.int8)
    return {'codes': codes, 'scales': scale, 'scope': scope}


def dequantize_rows(a):
    return a['codes'].astype(F64) * np.asarray(a['scales'], F64)


def gemm_exact(x, w):
    """Float64 reference product, no intermediate rounding."""
    return np.asarray(x, F64) @ np.asarray(w, F64)


def gemm_integer(x_codes, w_codes, x_scales, w_scales, w_scope, block=None):
    """The product an INT8 kernel computes: int32 accumulation, explicit rescales.

    `x_scales` is one scale per row, `w_scales` follows the weight scope.  For a
    K-blocked weight scope the integer sum is restarted per block and rescaled
    into the float64 accumulator, which is the only way per-K scales can be
    applied after an integer reduction.
    """
    xq = np.asarray(x_codes, np.int64)
    wq = np.asarray(w_codes, np.int64)
    xs = np.asarray(x_scales, F64).reshape(-1, 1)
    K = xq.shape[1]
    if w_scope in ('per_tensor', 'per_output_channel'):
        acc = xq @ wq                                  # exact integers, no overflow here
        if w_scope == 'per_tensor':
            sw = np.full((wq.shape[1],), float(np.asarray(w_scales, F64)[0]))
        else:
            sw = np.asarray(w_scales, F64)
        return acc.astype(F64) * xs * sw[None, :]
    if w_scope == 'per_input_channel':
        # each reduction step carries its own scale: no single integer reduction
        out = np.zeros((xq.shape[0], wq.shape[1]), F64)
        sw = np.asarray(w_scales, F64)
        for k in range(K):
            out += (xq[:, k:k + 1] * wq[k:k + 1, :]).astype(F64) * sw[k]
        return out * xs
    if w_scope == 'block_k':
        block = int(block)
        sw = np.asarray(w_scales, F64)
        out = np.zeros((xq.shape[0], wq.shape[1]), F64)
        for start in range(0, K, block):
            stop = min(start + block, K)
            acc = xq[:, start:stop] @ wq[start:stop]
            row = start // block
            out += acc.astype(F64) * sw[row][None, :]
        return out * xs
    raise ValueError(w_scope)


def gemm_int8(x, w, w_scope, x_scope='per_row', block=None):
    """Quantize both operands and run the integer product; returns (y, stats)."""
    aq = quantize_rows(x, x_scope)
    wq = quantize(w, w_scope, block)
    y = gemm_integer(aq['codes'], wq['codes'], aq['scales'], wq['scales'], w_scope, block)
    reference = gemm_exact(x, w)
    norm = np.linalg.norm(reference)
    stats = {
        'activation_scope': x_scope,
        'weight_scope': wq['scope'] + (f'{block}' if block else ''),
        'rel_frobenius': float(np.linalg.norm(y - reference) / norm) if norm else 0.0,
        'max_abs': float(np.abs(y - reference).max()),
        'rmse': float(np.sqrt(((y - reference) ** 2).mean())),
        'activation_at_endpoint': float(np.mean(np.abs(aq['codes']) == QMAX)),
        'weight_only_rel': None,
    }
    # weight-only contribution, exact accumulation: separates the two sources
    wq_only = dequantize(wq)
    yw = gemm_exact(x, wq_only)
    stats['weight_only_rel'] = float(np.linalg.norm(yw - reference) / norm) if norm else 0.0
    return y, stats


def d4r_split_agreement(W):
    """How many codes change when the rounding divisor is float64 instead of f32."""
    plain = quantize(W, 'per_output_channel')['codes']
    split = quantize(W, 'per_output_channel', d4r_split=True)['codes']
    return int(np.count_nonzero(plain != split)), int(plain.size)


def determinism(W):
    """Same input, repeated encoding: identical codes and scale bits."""
    first = quantize(W, 'per_output_channel')
    second = quantize(W, 'per_output_channel')
    return (np.array_equal(first['codes'], second['codes'])
            and np.array_equal(np.asarray(first['scales']).view(np.uint32),
                               np.asarray(second['scales']).view(np.uint32)))


def independent_integer_product(x_codes, w_codes):
    """Int64 element-wise reference for one product, computed without BLAS."""
    xq = np.asarray(x_codes, np.int64)
    wq = np.asarray(w_codes, np.int64)
    out = np.zeros((xq.shape[0], wq.shape[1]), np.int64)
    for i in range(xq.shape[0]):
        for j in range(wq.shape[1]):
            total = 0
            for k in range(xq.shape[1]):
                total += int(xq[i, k]) * int(wq[k, j])
            out[i, j] = total
    return out
