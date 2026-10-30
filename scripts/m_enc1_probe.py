#!/usr/bin/env python3
"""Verified enc1 block/window diagnostic; captures and dumps remain local.

Select one mismatching and one exact interior 4x4 window from a native fixture.
Check raw allocation offsets against the decoded field and d4r's loader before
comparing any intermediates. Optionally dump the existing native graph and run
one-ULP sensitivity tests (diagnostic only, never a correction).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import m_layer_oracle as lo
import m_layer_fixture as lf
import m_weights as mw


def coordinates(P, bx, by):
    t = np.arange(64)
    return (8 * bx - P.sx + (t & 3) + 4 * ((t >> 4) & 1),
            8 * by - P.sy + ((t >> 2) & 3) + 4 * (t >> 5))


def field_offsets(P, x, y, pointer):
    c = np.arange(64)
    return (pointer - P.abase +
            (((c[None, :] // 32) * P.H + y[:, None]) * P.W + x[:, None]) * 32
            + c[None, :] % 32)


def verified_block(P, raw, decoded, bx, by):
    x, y = coordinates(P, bx, by)
    assert ((x >= 0) & (x < P.W) & (y >= 0) & (y < P.H)).all()
    offsets = field_offsets(P, x, y, P.inp)
    x0 = lo.me.E4M3[P.arena[offsets]]
    assert np.array_equal(x0, decoded[y, x])
    assert np.array_equal(x0, lo.sm.load_tokens(P, bx, by, 2))
    # Output uses unmirrored coordinates; compare bytes, not decoded +/- zero.
    output = raw['output'][field_offsets(P, x, y, P.out) - (P.out - P.abase)]
    return x0, output, x, y


def norm(x, s, gamma, delta=0, token=None):
    r = np.float16(np.float32(1) / np.sqrt(np.float32(lo.h(s + 2 ** -13))))
    if delta:
        r = r.copy()
        target = np.inf if delta > 0 else -np.inf
        if token is None:
            r = np.nextafter(r, np.float16(target))
        else:
            r[token] = np.nextafter(r[token], np.float16(target))
    return lo.me.q8(lo.h(x * lo.h(r.astype(np.float64) * gamma))), r


def trace_block(block, x0, perturb=None, pv='exact'):
    """NaturalBlock with named boundaries and one optional diagnostic override."""
    m, h, me = block.m, lo.h, lo.me
    t = m['t']
    op, token, delta = perturb or ('', None, 0)
    trace = {'x0': x0}
    trace['sum1'] = lo.sumsq_natural(x0)
    trace['h1'], trace['r1'] = norm(x0, trace['sum1'], t['g1'],
                                    delta if op == 'r1' else 0, token)
    trace['q'] = me.mma_chain(trace['h1'], m['q'])
    trace['v'] = me.mma_chain(trace['h1'], m['v'])
    O = np.zeros((64, 64))
    for w in range(4):
        sl = slice(16 * w, 16 * w + 16)
        Qq = me.q8(trace['q'][sl])
        for head in range(2):
            cols = slice(32 * head, 32 * head + 32)
            Q = Qq[:, cols]
            S = h(Q @ Q.T + t['bias'][head])
            scale = np.full((16, 1), float(me.C_SCALE))
            if op == 'scale' and token // 16 == w:
                scale[token % 16] = np.nextafter(np.float16(me.C_SCALE),
                                                np.float16(np.inf if delta > 0 else -np.inf))
            a = np.clip(h(S * scale), -float(me.CLAMP), float(me.CLAMP))
            poly = h(a * h(a * -a + me.P1) + me.P0)
            weights = me.exp_trick(poly).astype(np.float64)
            sums = me.row_sum16(weights)
            reciprocal = h(np.float32(1) / sums.astype(np.float32))
            if op == 'softrecip' and token // 16 == w:
                reciprocal[token % 16] = np.nextafter(
                    reciprocal[token % 16].astype(np.float16),
                    np.float16(np.inf if delta > 0 else -np.inf)).astype(np.float64)
            probs = h(weights * reciprocal)
            values = trace['v'][sl, cols]
            if pv == 'exact':
                result = probs @ values
            elif pv == 'final-f32':
                result = (probs @ values).astype(np.float32)
            elif pv == 'sequential-f32':
                result = np.zeros((16, 32), np.float32)
                for k in range(16):
                    result += (probs[:, k, None] * values[k]).astype(np.float32)
            elif pv == 'pairwise-f32':
                products = [(probs[:, k, None] * values[k]).astype(np.float32)
                            for k in range(16)]
                while len(products) > 1:
                    products = [products[i] + products[i + 1]
                                for i in range(0, len(products), 2)]
                result = products[0]
            else:
                raise ValueError(pv)
            O[sl, cols] = h(result)
    trace['o'] = me.q8(O)
    trace['x1'] = me.mma_chain(trace['o'], m['wo'], C0=h(x0 + t['bo']))
    trace['sum2'] = lo.tree_norm2_natural(trace['x1'])
    trace['h2'], trace['r2'] = norm(trace['x1'], trace['sum2'], t['g2'],
                                    delta if op == 'r2' else 0, token)
    trace['x2'] = h(trace['x1'] + t['b2'])
    out = trace['x2']
    for k in range(8):
        z = me.mma_chain(trace['h2'], m['w1'][k], C0=np.broadcast_to(m['b1'][k], (64, 32)))
        out = me.mma_chain(me.q8(me.gelu_poly(z)), m['w2'][k], C0=out)
    trace['x2b'] = out
    return trace


def codes(x):
    return mw.e4m3_encode(np.asarray(x, np.float64))


def select_windows(P, got, ref):
    selected = {}
    worst = -1
    for by in range(1, P.H // 8 - 1):
        for bx in range(1, P.W // 8 - 1):
            x, y = coordinates(P, bx, by)
            for w in range(4):
                sl = slice(16 * w, 16 * w + 16)
                errors = int((got[y[sl], x[sl]] != ref[y[sl], x[sl]]).sum())
                key = 'mismatch' if errors else 'control'
                if key == 'mismatch' and errors > worst:
                    worst = errors
                    selected[key] = (bx, by, w)
                elif key == 'control' and key not in selected:
                    selected[key] = (bx, by, w)
    if len(selected) == 2:
        return selected
    raise ValueError('could not find both an exact and mismatching interior window')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', type=Path)
    ap.add_argument('fixture', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--perturb', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    directory = str(a.capture / 'replay-000001-rrlite_enc1_4x4')
    P, raw = lo.raw_layer(directory, 'enc1')
    _, decoded, _, _, _ = lo.read_layer(directory, 'enc1')
    def field(data):
        return data.reshape(2, P.H, P.W, 32).transpose(1, 2, 0, 3).reshape(P.H, P.W, 64)
    ref = field(raw['output'])
    got = field(np.fromfile(a.fixture / 'results/out-out.bin', np.uint8))
    selected = select_windows(P, got, ref)
    record = json.loads((a.fixture / 'fixture.json').read_text())
    pack = lf.WeightPack(P, 'enc1')
    assert np.array_equal(np.fromfile(a.fixture / 'weights.bin', np.float32), pack.data)
    for key, expected in (('W', P.W), ('H', P.H), ('C', 64), ('sx', P.sx), ('sy', P.sy)):
        assert record['geometry'][key] == expected, key
    native = a.out / 'native'
    if a.native:
        text = (a.fixture / 'graph.txt').read_text()
        # Keep the graph in its fixture directory so all init/params resolve.
        sizes = dict(record['resources'])
        for name in ('x0', 'h1', 'q', 'v', 'o', 'x1', 'h2', 'x2', 'x2b'):
            text += f'dump {name} 0 {sizes[name]} dbg-{name}.bin\n'
        graph = a.fixture / 'graph-enc1-probe.txt'
        graph.write_text(text)
        import subprocess
        for seed, destination in ((0, native), (1, a.out / 'native-repeat')):
            subprocess.run([str(Path(lf.REPO) / 'build/native/Release/m_network_stage.exe'),
                            '--graph', str(graph), '--out', str(destination),
                            '--seed', str(seed),
                            '--shaders', str(Path(lf.REPO) / 'build/native')], check=True)
        for name in ('out-out.bin', 'out-merge.bin'):
            assert (native / name).read_bytes() == (a.out / 'native-repeat' / name).read_bytes()
    block = lo.NaturalBlock(P, 'enc1')
    report = {'capture': directory, 'weights_sha256': hashlib.sha256(P.wbuf).hexdigest(),
              'native_repeated_with_two_sentinel_seeds': a.native,
              'windows': {}}
    for label, (bx, by, w) in selected.items():
        x0, output, x, y = verified_block(P, raw, decoded, bx, by)
        trace = trace_block(block, x0)
        assert np.array_equal(trace['x2b'], block(x0))
        kernel, _, _ = lo.sm.swin_block(P, x0, 64, 2)
        assert np.array_equal(trace['x2b'], lo.sm.pair_to_nat(kernel))
        # Independent native baseline output must agree with traced reconstruction.
        assert np.array_equal(codes(trace['x2b']), got[y, x])
        sl = slice(16 * w, 16 * w + 16)
        base = codes(trace['x2b'])
        row = {'block': [bx, by], 'window': w, 'xy': np.stack([x[sl], y[sl]], 1).tolist(),
               'output_mismatches': int((base[sl] != output[sl]).sum()), 'native': {},
               'perturbations': [], 'pv_sensitivity': {}}
        for pv in ('exact', 'final-f32', 'sequential-f32', 'pairwise-f32'):
            variant = trace_block(block, x0, pv=pv)
            row['pv_sensitivity'][pv] = {
                'block_mismatches': int((codes(variant['x2b']) != output).sum()),
                'attention_codes_changed': int((variant['o'] != trace['o']).sum())}
        for name, values in trace.items():
            file = native / f'dbg-{name}.bin'
            if file.exists():
                gw = record['geometry']['GW']
                d = np.fromfile(file, np.float16).reshape(-1, values.shape[1])
                observed = d[(y + P.sy) * gw + x + P.sx]
                row['native'][name] = int((observed != values).sum())
                assert row['native'][name] == 0, name
        np.savez(a.out / f'{label}.npz', **trace, reference_codes=output, x=x, y=y)
        if a.perturb:
            for token in range(16 * w, 16 * w + 16):
                for op in ('r1', 'r2', 'scale', 'softrecip'):
                    for delta in (-1, 1):
                        changed = codes(trace_block(block, x0, (op, token, delta))['x2b'])
                        row['perturbations'].append({
                            'token': token, 'op': op, 'ulp': delta,
                            'changed_codes': int((changed[sl] != base[sl]).sum()),
                            'mismatches': int((changed[sl] != output[sl]).sum()),
                            'token_mismatches': int((changed[token] != output[token]).sum()),
                            'regressions': int(((base[sl] == output[sl]) &
                                                (changed[sl] != output[sl])).sum())})
        report['windows'][label] = row
        print(label, 'block', bx, by, 'window', w, 'mismatches', row['output_mismatches'],
              'native', row['native'])
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
