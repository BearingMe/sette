#!/usr/bin/env python3
"""Trace the original enc1 PTX using the pinned interpreter (not GPU evidence).

The interpreter's approximate operations are correctly-rounded substitutes;
its limitations and any unsupported instruction are retained in the report.
Only local NPZ/JSON traces are written, never PTX or weights in the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import m_layer_oracle as lo
import ptxsim as sim
from m_enc1_probe import coordinates, field_offsets, trace_block


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ptx', type=Path)
    ap.add_argument('capture', type=Path)
    ap.add_argument('--block', nargs=2, type=int, default=[1, 1])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    directory = str(a.capture / 'replay-000001-rrlite_enc1_4x4')
    P, raw = lo.raw_layer(directory, 'enc1')
    x, y = coordinates(P, *a.block)
    x0 = lo.sm.load_tokens(P, *a.block, 2)
    trace = trace_block(lo.NaturalBlock(P, 'enc1'), x0)
    stmts, types, shared = sim.parse_kernel(str(a.ptx), 'rrlite_enc1_4x4')
    mem = sim.Memory(directory)
    block = sim.Block(stmts, types, shared, mem, 'rrlite_enc1_4x4_param_0',
                      (32, 1, 2), P.grid, (*a.block, 0))
    arrays, calls = {}, []
    def mma_hook(b, st, wave, A, B, C, D, pa, pb):
        key = f'line{st.line}_wave{wave}_call{len(calls)}'
        calls.append({'key': key, 'line': st.line, 'wave': wave, 'shape': list(A.shape)})
        for name, value in zip('ABCD', (A, B, C, D)):
            arrays[key + '_' + name] = value.copy()
    block.mma_hook = mma_hook
    error = None
    try:
        block.run()
    except NotImplementedError as exc:
        error = str(exc)
    arrays.update({'model_' + key: value for key, value in trace.items()})
    np.savez(a.out / 'ptx-trace.npz', **arrays)
    report = {'ptx_sha256': hashlib.sha256(a.ptx.read_bytes()).hexdigest(),
              'block': a.block, 'error': error, 'mma_calls': calls,
              'limitations': 'PTX interpreter, exact MMA; approximate instructions substituted'}
    idx = mem.bases.index(P.abase)
    offsets = field_offsets(P, x, y, P.out)
    result = mem.data[idx][offsets]
    written = set()
    arena_id = mem.ids[idx]
    for alloc_id, starts, payload, _ in mem.stores:
        if alloc_id == arena_id:
            for offset in range(payload.shape[1]):
                written.update((starts + offset).tolist())
    report['output_bytes_written'] = sum(int(i) in written for i in offsets.ravel())
    assert report['output_bytes_written'] == offsets.size
    from m_enc1_probe import codes
    reference = raw['output'][offsets - (P.out - P.abase)]
    report['output_vs_capture'] = int((result != reference).sum())
    report['output_vs_model'] = int((result != codes(trace['x2b'])).sum())
    print('PTX output vs capture/model', report['output_vs_capture'], report['output_vs_model'])
    (a.out / 'ptx-report.json').write_text(json.dumps(report, indent=2))
    print('mma calls', len(calls), 'error', error)
    for call in calls[:4]:
        print(call, 'A row0', arrays[call['key'] + '_A'][0, :8])
    print('model h1 row0', trace['h1'][0, :8])


if __name__ == '__main__':
    main()
