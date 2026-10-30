#!/usr/bin/env python3
"""Instrument a disposable copy of pinned conservative enc1; never the submodule.

Record x0, norm1, Q/V and attention output in verified natural token/channel
order. Require instrumented output to equal the uninstrumented conservative run.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

import m_enc1_contract as contract
import m_enc1_probe as probe
import m_layer_oracle as lo


def once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f'instrumentation anchor is not unique: {anchor}')
    return text.replace(anchor, replacement)


def instrument(text):
    text = once(text, '#include "swin_common.h"', '''#include "swin_common.h"
extern "C" {
__device__ int d4r_swin_debug_block[2] = {4, 1};
__device__ half_t d4r_swin_debug_values[11 * 64 * 128];
}
template <int C, int AST> __device__ void snapshot(const half_t* a, int s,
    int bx, int by, int tid, bool natural) {
    if (bx == d4r_swin_debug_block[0] && by == d4r_swin_debug_block[1] && tid < 64)
        for (int c = 0; c < C; ++c)
            d4r_swin_debug_values[(s * 64 + tid) * 128 + c] = a[tid * AST + (natural ? c : apos(c))];
    __syncthreads();
}
''')
    anchor = '    // residual accumulators of the output projection:'
    text = once(text, anchor, '    snapshot<C, AST>(X0, 0, bx, by, tid, true);\n' + anchor)
    anchor = '            const half_t rn = rsqrt16(a0 + a1);'
    text = once(text, anchor, anchor + '''
            if (bx == d4r_swin_debug_block[0] && by == d4r_swin_debug_block[1] && t == 0) {
                d4r_swin_debug_values[(5 * 64 + T) * 128] = a0 + a1;
                d4r_swin_debug_values[(6 * 64 + T) * 128] = rn;
            }
''')
    anchor = '    // ------------------------------------------------ stage 2: Q/V projection + attention (wave = head)'
    text = once(text, anchor, '    snapshot<C, AST>(A, 1, bx, by, tid, false);\n' + anchor)
    anchor = '                    b[s][q] = bload(w16, L::qv(h, q >> 1), 2, kc, s, q & 1, l16);'
    text = once(text, anchor, anchor + '''
            if (bx == d4r_swin_debug_block[0] && by == d4r_swin_debug_block[1] && wv < NH && hi == 0)
                for (int s = 0; s < 2; ++s)
                    for (int q = 0; q < 4; ++q)
                        for (int j = 0; j < 16; ++j) {
                            int c = 32 * h + ginv(16 * (q & 1) + l16);
                            int k = 32 * kc + kslot(s, j);
                            d4r_swin_debug_values[((9 + (q >> 1)) * 64 + k) * 128 + c] = b[s][q][j];
                        }
''')
    anchor = '                    k32(acc[m][q], a0, b[0][q], a1, b[1][q]);'
    text = once(text, anchor, '''                {
''' + anchor + '''
                    if (kc == 0 && bx == d4r_swin_debug_block[0] && by == d4r_swin_debug_block[1])
                        for (int i = 0; i < 8; ++i) {
                            int T = 16 * (m0 + m) + mrow(i);
                            int c = 32 * h + ginv(16 * (q & 1) + l16);
                            d4r_swin_debug_values[((7 + (q >> 1)) * 64 + T) * 128 + c] = acc[m][q].get(i);
                        }
                }
''')
    anchor = '        // pack: Q as q8 pairs, V as f16 pairs (all that attention needs), halving the live registers'
    text = once(text, anchor, '''        if (bx == d4r_swin_debug_block[0] && by == d4r_swin_debug_block[1])
            for (int m = 0; m < MW; ++m)
                for (int q = 0; q < 2; ++q)
                    for (int i = 0; i < 8; ++i) {
                        int T = 16 * (m0 + m) + mrow(i);
                        int c = 32 * h + ginv(16 * q + l16);
                        d4r_swin_debug_values[(2 * 64 + T) * 128 + c] = acc[m][q].get(i);
                        d4r_swin_debug_values[(3 * 64 + T) * 128 + c] = acc[m][2 + q].get(i);
                    }
''' + anchor)
    anchor = '    // ------------------------------------------------ stage 3:'
    # Stage marker suffix varies only outside this exact pinned marker prefix.
    text = once(text, anchor, '    snapshot<C, AST>(A, 4, bx, by, tid, false);\n' + anchor)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--block', default='4,1')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'references/sources/d4r'
    copied = a.out / 'kernels'
    for name in ('m', 'common'):
        shutil.copytree(source / 'kernels' / name, copied / name)
    block = copied / 'm/swin_block.h'
    block.write_text(instrument(block.read_text()))
    baseline = json.loads((a.experiment / 'report.json').read_text())
    flags = baseline['builds']['conservative']['command']
    index = flags.index(str(source / f'kernels/m/{contract.KERNEL}.hip'))
    module = a.out / f'{contract.KERNEL}.hsaco'
    cmd = flags[:index] + [copied / f'm/{contract.KERNEL}.hip', '-o', module, '--no-gpu-bundle-output']
    build_command = list(map(str, cmd))
    contract.command(cmd, a.out / 'build.log')
    original = baseline['runs']['conservative-first']['command']
    capture = Path(original[original.index('--fixture-dir') + 1])
    env = dict(os.environ, D4R_SWIN_DEBUG_BLOCK=a.block)
    cmd = original.copy()
    cmd[cmd.index('--module') + 1] = str(module)
    cmd[cmd.index('--output-dir') + 1] = str(a.out / 'gpu-output')
    text = contract.command(cmd, a.out / 'replay.log', env)
    contract.verify_replay(text)
    equal = all(contract.sha(a.out / 'gpu-output' / name) == digest for name, digest in
                baseline['runs']['conservative-first']['output_hashes'].items())
    if not equal:
        raise RuntimeError('instrumentation changed conservative output')
    P, raw = lo.raw_layer(str(capture), 'enc1')
    bx, by = map(int, a.block.split(','))
    _, decoded, _, _, _ = lo.read_layer(str(capture), 'enc1')
    x0, _, _, _ = probe.verified_block(P, raw, decoded, bx, by)
    cpu = probe.trace_block(lo.NaturalBlock(P, 'enc1'), x0)
    got = np.fromfile(a.out / 'gpu-output/swin-stages.f16', np.float16).reshape(11, 64, 128)
    rows = {}
    for i, key in enumerate(('x0', 'h1', 'q', 'v', 'o', 'sum1', 'r1')):
        actual = got[i, :, :64] if i < 5 else got[i, :, :1]
        ref = np.asarray(cpu[key], np.float16).reshape(actual.shape)
        diff = actual.view(np.uint16) != ref.view(np.uint16)
        rows[key] = {'differing_f16': int(diff.sum()), 'elements': actual.size,
                     'first_indices': np.argwhere(diff)[:8].tolist()}
        np.save(a.out / f'{key}-native.npy', actual)
        np.save(a.out / f'{key}-cpu.npy', ref)
    matrices = lo.natural_matrices(P, 'enc1')
    for i, key in enumerate(('q', 'v')):
        weights = got[9 + i, :, :64]
        expected = matrices[key]
        if not np.array_equal(weights, expected):
            raise RuntimeError(f'actual {key} weight operands differ from CPU mapping')
        partial = np.float16(cpu['h1'][:, :32] @ expected[:32])
        actual = got[7 + i, :, :64]
        rows[key + '-k32-first'] = {'differing_f16': int((actual.view(np.uint16) != partial.view(np.uint16)).sum()),
                                    'weights_exact': True}
        np.save(a.out / f'{key}-k32-first-native.npy', actual)
        np.save(a.out / f'{key}-weights-native.npy', weights)
    report = {'block': [bx, by], 'instrumented_output_equals_baseline': equal,
              'module_sha256': contract.sha(module), 'build_command': build_command,
              'baseline_report_sha256': contract.sha(a.experiment / 'report.json'),
              'replay_command': cmd, 'replay_log_sha256': contract.sha(a.out / 'replay.log'),
              'instrumented_source_sha256': contract.sha(block), 'boundaries': rows}
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
